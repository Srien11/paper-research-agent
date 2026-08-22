from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
import threading
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
        self.thread_names = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        self.thread_names.append(threading.current_thread().name)
        return [10.0 if "english winner" in text else 0.0 for text in texts]


class FakeRewriter:
    model_id = "qwen3.7-plus-2026-05-26"
    prompt_version = "query-rewrite-v3"

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


class QueryAwareRewriter(FakeRewriter):
    def __init__(self, *, delay=0.0, error=None):
        super().__init__(delay=delay, error=error)
        self.active = 0
        self.max_active = 0

    async def rewrite(self, query):
        self.calls.append(query)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return QueryRewriteResult(
                english_query=f"English {query}",
                actual_model=self.model_id,
                input_tokens=12,
                output_tokens=5,
            )
        finally:
            self.active -= 1


class BarrierIndex(StaticIndex):
    def __init__(self, rankings, barrier):
        super().__init__(rankings)
        self.barrier = barrier
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def search(self, query, top_k, *, filters=None):
        with self.lock:
            self.calls.append((query, filters))
            if query.startswith("query-"):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
        if query.startswith("query-"):
            self.barrier.wait(timeout=2)
        result = list(self.rankings[query][:top_k])
        if query.startswith("query-"):
            with self.lock:
                self.active -= 1
        return result


class BarrierReranker(RecordingReranker):
    def __init__(self, barrier):
        super().__init__()
        self.barrier = barrier
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def score(self, query, texts):
        with self.lock:
            self.calls.append((query, list(texts)))
            self.thread_names.append(threading.current_thread().name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.barrier.wait(timeout=2)
        with self.lock:
            self.active -= 1
        return [float(len(texts) - index) for index, _text in enumerate(texts)]


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

    def service(self, rewriter, *, timeout=2.0, local_workers=None):
        kwargs = {} if local_workers is None else {"local_workers": local_workers}
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
            **kwargs,
        )
        self.services.append(service)
        return service

    def test_local_worker_default_is_six_and_lower_override_is_supported(self) -> None:
        default_service = self.service(FakeRewriter())
        lower_service = self.service(FakeRewriter(), local_workers=2)

        self.assertEqual(default_service._local_executor._max_workers, 6)
        self.assertEqual(lower_service._local_executor._max_workers, 2)

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
        self.assertTrue(self.reranker.thread_names[0].startswith("paper-retrieval-local-ml"))
        self.assertTrue(run.audit_persisted)
        self.assertEqual(run.storage_classes["C001"], "redistributable")

        with closing(sqlite3.connect(self.directory / "audit.sqlite3")) as connection:
            stages = {row[0] for row in connection.execute("SELECT DISTINCT stage FROM rankings")}
        self.assertTrue({"zh.bm25", "en.vector", "cross_route_rrf", "final"} <= stages)

    def test_local_worker_count_rejects_unsafe_values(self) -> None:
        for workers in (0, 9):
            with (
                self.subTest(workers=workers),
                self.assertRaisesRegex(
                    ValueError,
                    "local retrieval workers must be between 1 and 8",
                ),
            ):
                self.service(FakeRewriter(), local_workers=workers)

    async def test_resolve_query_exposes_cached_rewrite_without_running_chunk_recall(self) -> None:
        service = self.service(FakeRewriter())

        first = await service.resolve_query("中文问题")
        second = await service.resolve_query("中文问题")

        self.assertEqual(first.english_query, "English query")
        self.assertEqual(second.status, "cache_hit")
        self.assertEqual(self.sparse.calls, [])
        self.assertEqual(self.vector.calls, [])

    async def test_request_can_expand_recall_and_skip_reranking_for_candidate_discovery(self) -> None:
        run = await self.service(FakeRewriter()).search(
            "中文问题",
            top_k=3,
            candidate_k=4,
            recall_k=4,
            rerank=False,
        )

        self.assertEqual(self.reranker.calls, [])
        self.assertTrue(all(call[0] in {"中文问题", "English query"} for call in self.sparse.calls))
        self.assertLessEqual(len(run.hits), 3)

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

    async def test_six_workers_keep_recall_rerank_filters_and_audit_isolated(self) -> None:
        queries = tuple(f"query-{index}" for index in range(6))
        chunks = {
            query: chunk(f"chunk-{index}", f"evidence {index}")
            for index, query in enumerate(queries)
        }
        rankings = {
            **{query: [(chunks[query], 1.0)] for query in queries},
            **{
                f"English {query}": [(chunks[query], 1.0)]
                for query in queries
            },
        }
        recall_barrier = threading.Barrier(6)
        rerank_barrier = threading.Barrier(6)
        sparse = StaticIndex(rankings)
        vector = BarrierIndex(rankings, recall_barrier)
        reranker = BarrierReranker(rerank_barrier)
        rewriter = QueryAwareRewriter(delay=0.01)
        service = BilingualRetrievalService(
            sparse,
            vector,
            reranker,
            rewriter,
            self.cache,
            self.audit,
            retrieval_config(),
            bilingual_config(self.directory),
            index_id="idx-six",
            local_workers=6,
        )
        self.services.append(service)
        filters = tuple({"corpus_id": f"C{index:03d}"} for index in range(6))

        runs = await asyncio.gather(
            *(
                service.search(query, filters=query_filters)
                for query, query_filters in zip(queries, filters, strict=True)
            )
        )

        self.assertEqual(vector.max_active, 6)
        self.assertEqual(reranker.max_active, 6)
        self.assertEqual(
            [run.hits[0].chunk_id for run in runs],
            [chunks[query].chunk_id for query in queries],
        )
        for query, query_filters in zip(queries, filters, strict=True):
            self.assertIn((query, query_filters), vector.calls)
            self.assertIn((f"English {query}", query_filters), vector.calls)
        self.assertTrue(all(run.audit_persisted for run in runs))
        with closing(sqlite3.connect(self.directory / "audit.sqlite3")) as connection:
            audit_queries = {
                row[0] for row in connection.execute("SELECT original_query FROM runs")
            }
            run_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        self.assertEqual(audit_queries, set(queries))
        self.assertEqual(run_count, 6)

        await service.aclose()
        self.assertTrue(service._local_executor._shutdown)

    async def test_six_cache_hits_skip_provider_and_rewrite_flights_stay_independent(
        self,
    ) -> None:
        cached_queries = tuple(f"cached-{index}" for index in range(6))
        priming_rewriter = QueryAwareRewriter()
        priming_service = self.service(priming_rewriter)
        await asyncio.gather(*(priming_service.resolve_query(query) for query in cached_queries))

        forbidden = QueryAwareRewriter(error=AssertionError("must not call provider"))
        cached_service = self.service(forbidden)
        cached = await asyncio.gather(
            *(cached_service.resolve_query(query) for query in cached_queries)
        )
        self.assertTrue(all(item.status == "cache_hit" for item in cached))
        self.assertEqual(forbidden.calls, [])

        uncached = tuple(f"uncached-{index}" for index in range(6))
        concurrent_rewriter = QueryAwareRewriter(delay=0.02)
        concurrent_service = self.service(concurrent_rewriter)
        rewritten = await asyncio.gather(
            *(concurrent_service.resolve_query(query) for query in uncached)
        )
        self.assertEqual(concurrent_rewriter.max_active, 6)
        self.assertEqual([item.english_query for item in rewritten], [f"English {q}" for q in uncached])

        same_query_results = await asyncio.gather(
            *(concurrent_service.resolve_query("one-flight") for _ in range(6))
        )
        self.assertEqual(concurrent_rewriter.calls.count("one-flight"), 1)
        self.assertEqual(
            {item.english_query for item in same_query_results},
            {"English one-flight"},
        )

    async def test_memory_aware_query_has_per_request_one_day_retention(self) -> None:
        await self.service(FakeRewriter()).search("中文问题", privacy_ttl_days=1)

        with closing(sqlite3.connect(self.directory / "cache.sqlite3")) as connection:
            cache_row = connection.execute("SELECT created_at, expires_at FROM rewrites").fetchone()
        with closing(sqlite3.connect(self.directory / "audit.sqlite3")) as connection:
            audit_row = connection.execute(
                "SELECT created_at, plaintext_expires_at FROM runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        self.assertIsNotNone(cache_row)
        self.assertIsNotNone(audit_row)
        assert cache_row is not None and audit_row is not None
        cache_retention = datetime.fromisoformat(cache_row[1]) - datetime.fromisoformat(
            cache_row[0]
        )
        audit_retention = datetime.fromisoformat(audit_row[1]) - datetime.fromisoformat(
            audit_row[0]
        )
        self.assertEqual(cache_retention, timedelta(days=1))
        self.assertEqual(audit_retention, timedelta(days=1))

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
