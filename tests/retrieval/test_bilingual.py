from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.retrieval.bilingual import (
    BilingualRetrievalService,
    fuse_language_routes,
)
from paper_research_agent.retrieval.config import (
    BilingualRetrievalConfig,
    RetrievalConfig,
)
from paper_research_agent.retrieval.query_rewrite import QueryRewriteResult
from paper_research_agent.retrieval.query_store import (
    CachedRewrite,
    CacheLookup,
    SQLiteQueryAuditLogger,
    SQLiteQueryRewriteCache,
)
from paper_research_agent.retrieval.rights import CorpusRightsMap
from tests.retrieval.test_bm25 import chunk


class StaticIndex:
    def __init__(self, rankings):
        self.rankings = rankings
        self.calls = []

    def search(self, query, top_k, *, filters=None):
        self.calls.append((query, filters))
        return list(self.rankings[query][:top_k])


class RecordingReranker:
    def __init__(self):
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return [10.0 if "english winner" in text else 0.0 for text in texts]


class FakeRewriter:
    model_id = "qwen3.7-plus"
    prompt_version = "query-rewrite-v1"

    def __init__(self, *, delay=0.0, error=None):
        self.delay = delay
        self.error = error
        self.calls = []

    async def rewrite(self, query):
        self.calls.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return QueryRewriteResult(
            english_query="English query",
            actual_model="qwen3.7-plus-2026-05-26",
            input_tokens=12,
            output_tokens=5,
        )


class FailingCache:
    def lookup(self, *args, **kwargs):
        raise sqlite3.OperationalError("cache unavailable")

    def put(self, *args, **kwargs):
        raise sqlite3.OperationalError("cache unavailable")


class StaleCache:
    def lookup(self, *args, **kwargs):
        return CacheLookup(
            stale=CachedRewrite(
                english_query="English query",
                actual_model="cached-model",
                created_at=datetime.now(UTC) - timedelta(days=100),
                latency_ms=100,
                input_tokens=99,
                output_tokens=10,
            )
        )

    def put(self, *args, **kwargs):
        raise AssertionError("stale provider fallback must not rewrite cache")


def retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        embedding_model="org/embed",
        embedding_revision="a" * 40,
        reranker_model="org/rerank",
        reranker_revision="b" * 40,
        sparse_candidates=3,
        vector_candidates=3,
        rerank_candidates=3,
        top_k=2,
    )


def bilingual_config(directory: Path, *, timeout=2.0) -> BilingualRetrievalConfig:
    return BilingualRetrievalConfig(
        rewrite_timeout_seconds=timeout,
        cache_path=Path("data/runtime/cache.sqlite3"),
        audit_path=Path("data/runtime/audit.sqlite3"),
    )


class BilingualRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        directory = Path(self.temp.name)
        self.cache = SQLiteQueryRewriteCache(directory / "cache.sqlite3")
        self.audit = SQLiteQueryAuditLogger(directory / "audit.sqlite3", plaintext_days=30)
        self.zh = chunk("zh", "中文相关")
        self.en = chunk("en", "english winner")
        self.other = chunk("other", "other evidence", "T001")
        self.sparse = StaticIndex(
            {
                "中文问题": [(self.zh, 4.0), (self.other, 1.0)],
                "English query": [(self.en, 5.0), (self.other, 2.0)],
            }
        )
        self.vector = StaticIndex(
            {
                "中文问题": [(self.zh, 0.9), (self.other, 0.2)],
                "English query": [(self.en, 0.95), (self.other, 0.3)],
            }
        )
        self.reranker = RecordingReranker()
        self.directory = directory
        self.services = []

    async def asyncTearDown(self) -> None:
        for service in self.services:
            service.close()
        self.temp.cleanup()

    def service(self, rewriter, *, timeout=2.0):
        service = BilingualRetrievalService(
            self.sparse,
            self.vector,
            self.reranker,
            rewriter,
            self.cache,
            self.audit,
            retrieval_config(),
            bilingual_config(self.directory, timeout=timeout),
            index_id="idx",
            rights=CorpusRightsMap({"C001": "redistributable", "T001": "internal_research_only"}),
        )
        self.services.append(service)
        return service

    async def test_success_runs_two_level_rrf_then_one_english_rerank(self) -> None:
        rewriter = FakeRewriter()
        run = await self.service(rewriter).search("中文问题")

        self.assertEqual(run.schema_version, "bilingual-retrieval-run-v1")
        self.assertEqual(run.original_query, "中文问题")
        self.assertEqual(run.rewrite.status, "success")
        self.assertEqual(run.hits[0].chunk_id, "en")
        self.assertIn("zh.route_rrf", run.hits[1].ranks)
        self.assertIn("en.route_rrf", run.hits[1].ranks)
        self.assertEqual(self.reranker.calls[0][0], "English query")
        self.assertEqual(len(self.reranker.calls), 1)
        self.assertTrue(run.audit_persisted)
        self.assertEqual(run.storage_classes["C001"], "redistributable")

        with closing(sqlite3.connect(self.directory / "audit.sqlite3")) as connection:
            stages = {row[0] for row in connection.execute("SELECT DISTINCT stage FROM rankings")}
        self.assertTrue({"zh.bm25", "en.vector", "cross_route_rrf", "final"} <= stages)

    async def test_timeout_returns_chinese_hybrid_without_english_reranker(self) -> None:
        run = await self.service(FakeRewriter(delay=0.05), timeout=0.005).search("中文问题")

        self.assertEqual(run.rewrite.status, "timeout")
        self.assertTrue(run.degraded)
        self.assertEqual(run.degraded_reason, "query_rewrite_timeout")
        self.assertEqual(run.hits[0].chunk_id, "zh")
        self.assertEqual(self.reranker.calls, [])
        self.assertEqual([call[0] for call in self.sparse.calls], ["中文问题"])

    async def test_successful_rewrite_is_reused_from_persistent_cache(self) -> None:
        first_rewriter = FakeRewriter()
        first = await self.service(first_rewriter).search("中文问题")
        self.assertEqual(first.rewrite.status, "success")

        second_rewriter = FakeRewriter(error=AssertionError("must not call provider"))
        second = await self.service(second_rewriter).search("中文问题")
        self.assertEqual(second.rewrite.status, "cache_hit")
        self.assertEqual(second_rewriter.calls, [])
        self.assertFalse(second.degraded)

    async def test_cache_failure_never_breaks_rewrite_or_local_fallback(self) -> None:
        service = BilingualRetrievalService(
            self.sparse,
            self.vector,
            self.reranker,
            FakeRewriter(),
            FailingCache(),
            self.audit,
            retrieval_config(),
            bilingual_config(self.directory),
            index_id="idx",
        )
        self.services.append(service)

        run = await service.search("中文问题")

        self.assertEqual(run.rewrite.status, "success")
        self.assertEqual(run.rewrite.cache_error_class, "OperationalError")

    async def test_stale_cache_is_visible_as_degraded_and_costs_zero_tokens(self) -> None:
        service = BilingualRetrievalService(
            self.sparse,
            self.vector,
            self.reranker,
            FakeRewriter(error=RuntimeError("provider failed")),
            StaleCache(),
            self.audit,
            retrieval_config(),
            bilingual_config(self.directory),
            index_id="idx",
        )
        self.services.append(service)

        run = await service.search("中文问题")

        self.assertEqual(run.rewrite.status, "stale_cache")
        self.assertTrue(run.degraded)
        self.assertEqual(run.degraded_reason, "query_rewrite_error_using_stale_cache")
        self.assertEqual(run.rewrite.input_tokens, 0)
        self.assertEqual(run.rewrite.output_tokens, 0)

    async def test_async_close_cancels_inflight_rewrite(self) -> None:
        service = self.service(FakeRewriter(delay=5))
        pending = asyncio.create_task(service.search("中文问题"))
        await asyncio.sleep(0.01)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

        await asyncio.wait_for(service.aclose(), timeout=1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await service.search("中文问题")

    def test_cross_route_rrf_does_not_double_count_internal_channels(self) -> None:
        a = chunk("a", "a")
        b = chunk("b", "b")
        candidate_a = (a, 0.5, {"zh.bm25": 1, "zh.vector": 1}, {"zh.route_rrf": 0.5})
        candidate_b = (b, 0.4, {"en.bm25": 1, "en.vector": 1}, {"en.route_rrf": 0.4})

        fused = fuse_language_routes({"zh": [candidate_a], "en": [candidate_b]}, rrf_k=60)

        self.assertEqual(fused[0][1], fused[1][1])
        self.assertEqual([item[0].chunk_id for item in fused], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
