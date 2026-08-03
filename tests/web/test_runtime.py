from __future__ import annotations

import asyncio
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_research_agent.answering.models import AnswerRequest, GenerationResult
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.memory.config import ShortTermMemoryConfig
from paper_research_agent.memory.models import ShortTermMemoryTurn
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    SearchHit,
)
from paper_research_agent.web.runtime import (
    RAGRuntime,
    RuntimeBusyError,
    RuntimeClosedError,
    RuntimeDependencies,
    SafePaperMetadata,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk() -> EvidenceChunk:
    text = "Retrieval augmented generation improves grounded factual answering."
    return EvidenceChunk(
        chunk_id="chunk-1",
        asset_id="asset-1",
        corpus_id="C001",
        section_id="results",
        element_ids=("element-1",),
        page_start=3,
        page_end=3,
        token_start=0,
        token_end=8,
        text=text,
        text_sha256=_digest(text),
        config_sha256="a" * 64,
    )


class FakeRetriever:
    def __init__(self, chunk: EvidenceChunk):
        self.chunk = chunk
        self.queries: list[str] = []
        self.closed = 0

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        privacy_ttl_days: int | None = None,
    ) -> BilingualRetrievalRun:
        del privacy_ttl_days
        self.queries.append(query)
        return BilingualRetrievalRun(
            pipeline_id="test-pipeline",
            original_query=query,
            rewrite=QueryRewriteTrace(
                status="success",
                english_query="grounded RAG factual answering",
                requested_model="qwen-test",
                actual_model="qwen-test",
                prompt_version="query-rewrite-v2",
                latency_ms=4.0,
            ),
            degraded=False,
            top_k=top_k or 10,
            hits=(
                SearchHit(
                    chunk_id=self.chunk.chunk_id,
                    corpus_id=self.chunk.corpus_id,
                    asset_id=self.chunk.asset_id,
                    section_id=self.chunk.section_id,
                    page_start=self.chunk.page_start,
                    page_end=self.chunk.page_end,
                    text_sha256=self.chunk.text_sha256,
                    ranks={"en.vector": 1},
                    scores={"en.vector": 0.9},
                    final_score=0.9,
                    final_rank=1,
                ),
            ),
            index_id="idx-test",
            config_sha256="b" * 64,
            storage_classes={"C001": "internal_research_only"},
            rights_status="loaded",
            audit_persisted=True,
        )

    async def aclose(self) -> None:
        self.closed += 1


class FakeGenerator:
    model_id = "qwen-test"
    prompt_version = "rag-answer-json-v1"

    def __init__(self, gate: asyncio.Event | None = None):
        self.calls = 0
        self.closed = 0
        self.gate = gate

    async def generate(self, request: AnswerRequest) -> GenerationResult:
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        citation_id = request.context.citations[0].citation_id
        return GenerationResult(
            content=(
                '{"status":"answered","claims":[{"text":"RAG 可以改善事实性回答。",'
                f'"citation_ids":["{citation_id}"]}}],"insufficient_reason":null}}'
            ),
            requested_model=self.model_id,
            actual_model=self.model_id,
            prompt_version=self.prompt_version,
            input_tokens=120,
            output_tokens=20,
            latency_ms=8.0,
            attempts=1,
        )

    async def aclose(self) -> None:
        self.closed += 1


class FakeMemoryStore:
    def __init__(self):
        self.turns: list[ShortTermMemoryTurn] = []

    def recent(self, session_id: str, *, now=None) -> tuple[ShortTermMemoryTurn, ...]:
        del now
        return tuple(turn for turn in self.turns if turn.session_id == session_id)

    def append(self, turn: ShortTermMemoryTurn, *, now=None) -> bool:
        del now
        self.turns.append(turn)
        return True


def _runtime(*, gate: asyncio.Event | None = None) -> tuple[RAGRuntime, FakeRetriever, FakeGenerator]:
    chunk = _chunk()
    retriever = FakeRetriever(chunk)
    generator = FakeGenerator(gate)
    runtime = RAGRuntime(
        RuntimeDependencies(
            chunks=(chunk,),
            papers={
                "C001": SafePaperMetadata(
                    corpus_id="C001",
                    title="A Private Local Paper",
                    official_url="https://example.test/paper",
                    storage_class="internal_research_only",
                )
            },
            retriever=retriever,
            generator=generator,
            memory_store=FakeMemoryStore(),
            memory_config=ShortTermMemoryConfig(),
        ),
        excerpt_chars=48,
    )
    return runtime, retriever, generator


class RAGRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_from_environment_requires_corpus_and_forwards_optional_paths(self) -> None:
        with (
            patch.dict("os.environ", {"PRA_PROJECT_ROOT": "project-root"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "PRA_CORPUS_DIR"),
        ):
            RAGRuntime.from_environment()

        sentinel, _, _ = _runtime()
        environment = {
            "PRA_PROJECT_ROOT": "project-root",
            "PRA_CORPUS_DIR": "corpus",
            "PRA_CHUNKS_PATH": "private/chunks.jsonl",
            "PRA_RETRIEVAL_CONFIG": "private/retrieval.json",
            "PRA_BILINGUAL_CONFIG": "private/bilingual.json",
            "PRA_ANSWER_CONFIG": "private/answer.json",
            "PRA_MEMORY_CONFIG": "private/memory.json",
            "PRA_ANSWER_AUDIT_PATH": "private/answer-audit.sqlite3",
        }
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(RAGRuntime, "load", return_value=sentinel) as load,
        ):
            self.assertIs(RAGRuntime.from_environment(), sentinel)
        kwargs = load.call_args.kwargs
        self.assertEqual(kwargs["project_root"], Path("project-root"))
        self.assertEqual(kwargs["corpus_dir"], Path("project-root/corpus"))
        self.assertEqual(kwargs["chunks_path"], Path("private/chunks.jsonl"))
        self.assertEqual(kwargs["answer_audit_path"], Path("private/answer-audit.sqlite3"))

    async def test_reuses_dependencies_and_returns_only_safe_trace(self) -> None:
        runtime, retriever, generator = _runtime()

        first = await runtime.ask("RAG 如何改善事实性？", session_id="a" * 32)
        second = await runtime.ask("它的依据是什么？", session_id="a" * 32)

        self.assertEqual(generator.calls, 2)
        self.assertEqual(len(retriever.queries), 2)
        self.assertIn("上一轮研究问题", retriever.queries[1])
        self.assertEqual(first.retrieval.english_query, "grounded RAG factual answering")
        self.assertEqual(first.retrieval.index_id, "idx-test")
        self.assertTrue(first.retrieval.audit_persisted)
        self.assertEqual(second.context.included_memory_turn_count, 1)
        self.assertEqual(first.context.included_evidence_count, 1)
        self.assertGreater(first.context.estimated_tokens, 0)
        self.assertEqual(first.sources[0].title, "A Private Local Paper")
        self.assertEqual(first.sources[0].storage_class, "internal_research_only")
        self.assertLessEqual(len(first.sources[0].excerpt), 49)

        rendered = first.model_dump_json()
        self.assertNotIn("local_pdf_path", rendered)
        self.assertNotIn("image_path", rendered)
        self.assertNotIn("scores", rendered)

    async def test_rejects_concurrent_question_as_busy(self) -> None:
        gate = asyncio.Event()
        runtime, _, _ = _runtime(gate=gate)
        first = asyncio.create_task(runtime.ask("第一个问题", session_id="b" * 32))
        while not runtime.is_busy:
            await asyncio.sleep(0)

        with self.assertRaises(RuntimeBusyError):
            await runtime.ask("第二个问题", session_id="b" * 32)

        gate.set()
        await first
        self.assertFalse(runtime.is_busy)

    async def test_close_is_idempotent_and_prevents_new_work(self) -> None:
        runtime, retriever, generator = _runtime()

        await runtime.aclose()
        await runtime.aclose()

        self.assertEqual(retriever.closed, 1)
        self.assertEqual(generator.closed, 1)
        self.assertFalse(runtime.is_ready)
        with self.assertRaises(RuntimeClosedError):
            await runtime.ask("问题", session_id="c" * 32)


if __name__ == "__main__":
    unittest.main()
