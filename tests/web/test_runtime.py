from __future__ import annotations

import asyncio
import hashlib
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from paper_research_agent.agent.models import (
    EvidenceRecord,
    GetEvidenceResult,
    ResearchObservation,
    ResearchPlan,
    ResearchStep,
    SearchCorpusHit,
    SearchCorpusResult,
)
from paper_research_agent.agent.runtime import ResearchRuntimeResult
from paper_research_agent.answering.models import AnswerRequest, GenerationResult
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.models import ContextEvidence
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
        self.requests: list[AnswerRequest] = []

    async def generate(self, request: AnswerRequest) -> GenerationResult:
        self.calls += 1
        self.requests.append(request)
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


def _research_result(chunk: EvidenceChunk) -> ResearchRuntimeResult:
    record = EvidenceRecord(
        chunk_id=chunk.chunk_id,
        corpus_id=chunk.corpus_id,
        section_id=chunk.section_id,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        text_sha256=chunk.text_sha256,
        storage_class="internal_research_only",
    )
    search = SearchCorpusResult(
        query="grounded RAG",
        index_id="idx-agent",
        degraded=False,
        hits=(
            SearchCorpusHit(
                chunk_id=chunk.chunk_id,
                corpus_id=chunk.corpus_id,
                section_id=chunk.section_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text_sha256=chunk.text_sha256,
                storage_class="internal_research_only",
                final_rank=1,
            ),
        ),
    )
    return ResearchRuntimeResult(
        question="RAG 如何改善事实性？",
        plan=ResearchPlan(
            steps=(
                ResearchStep(
                    step_id="methods",
                    objective="查找方法",
                    query="grounded RAG",
                    top_k=4,
                ),
            )
        ),
        observations=(
            ResearchObservation(
                step_id="methods",
                objective="查找方法",
                search=search,
                evidence=GetEvidenceResult(records=(record,)),
            ),
        ),
        evidence=(
            ContextEvidence(
                chunk_id=chunk.chunk_id,
                corpus_id=chunk.corpus_id,
                asset_id=chunk.asset_id,
                section_id=chunk.section_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                text_sha256=chunk.text_sha256,
                storage_class="internal_research_only",
                final_score=1.0,
                final_rank=1,
            ),
        ),
        tool_call_count=2,
        task_state='{"kind":"untrusted_research_task_state","step_id":"methods"}',
    )


class FakeResearchAgent:
    def __init__(self, result: ResearchRuntimeResult):
        self.result = result
        self.calls: list[tuple[str, str]] = []
        self.clear_calls: list[str] = []
        self.closed = 0

    async def run(self, question: str, *, thread_id: str) -> ResearchRuntimeResult:
        self.calls.append((question, thread_id))
        return self.result.model_copy(update={"question": question})

    async def clear(self, thread_id: str) -> None:
        self.clear_calls.append(thread_id)

    async def aclose(self) -> None:
        self.closed += 1


def _runtime(
    *,
    gate: asyncio.Event | None = None,
    research_agent: FakeResearchAgent | None = None,
) -> tuple[RAGRuntime, FakeRetriever, FakeGenerator]:
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
            research_agent=research_agent,
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

    def test_agent_environment_flag_is_explicit_and_fail_closed(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(RAGRuntime.research_agent_enabled_from_environment())
        with patch.dict("os.environ", {"PRA_RESEARCH_AGENT_ENABLED": "true"}, clear=True):
            self.assertTrue(RAGRuntime.research_agent_enabled_from_environment())
        with (
            patch.dict(
                "os.environ",
                {"PRA_RESEARCH_AGENT_ENABLED": "sometimes"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "PRA_RESEARCH_AGENT_ENABLED"),
        ):
            RAGRuntime.research_agent_enabled_from_environment()

    async def test_from_environment_with_agent_maps_policy_and_checkpoint(self) -> None:
        runtime, _, _ = _runtime()
        answer_config = Mock(model="qwen-planner-2026-01-01")
        environment = {
            "PRA_PROJECT_ROOT": "project-root",
            "PRA_RESEARCH_AGENT_CHECKPOINT_PATH": "private/agent.sqlite3",
            "PRA_RESEARCH_AGENT_MAX_STEPS": "2",
            "PRA_RESEARCH_AGENT_EVIDENCE_PER_STEP": "3",
            "PRA_RESEARCH_AGENT_MAX_TOOL_CALLS": "4",
            "PRA_RESEARCH_AGENT_TIMEOUT_SECONDS": "45",
        }
        with (
            patch.dict("os.environ", environment, clear=True),
            patch.object(RAGRuntime, "from_environment", return_value=runtime),
            patch(
                "paper_research_agent.web.runtime.load_answering_config",
                return_value=answer_config,
            ),
            patch.object(runtime, "enable_research_agent", new=AsyncMock()) as enable,
        ):
            result = await RAGRuntime.from_environment_with_agent()

        self.assertIs(result, runtime)
        kwargs = enable.await_args.kwargs
        self.assertEqual(kwargs["model_id"], "qwen-planner-2026-01-01")
        self.assertEqual(
            kwargs["checkpoint_path"],
            Path("project-root/private/agent.sqlite3"),
        )
        self.assertEqual(kwargs["policy"].max_steps, 2)
        self.assertEqual(kwargs["policy"].evidence_per_step, 3)
        self.assertEqual(kwargs["policy"].max_tool_calls, 4)
        self.assertEqual(kwargs["policy"].timeout_seconds, 45)

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

    async def test_uses_agent_evidence_with_existing_answer_validation(self) -> None:
        chunk = _chunk()
        agent = FakeResearchAgent(_research_result(chunk))
        runtime, retriever, generator = _runtime(research_agent=agent)

        result = await runtime.ask("RAG 如何改善事实性？", session_id="d" * 32)

        self.assertEqual(retriever.queries, [])
        self.assertEqual(agent.calls, [("RAG 如何改善事实性？", "d" * 32)])
        self.assertEqual(result.retrieval.rewrite_status, "agent")
        self.assertEqual(result.retrieval.index_id, "idx-agent")
        self.assertEqual(result.retrieval.hits[0].route_ranks, {"agent": 1})
        self.assertEqual(result.answer.citations[0].chunk_id, "chunk-1")
        prompt = "\n".join(
            message.content for message in generator.requests[0].context.messages
        )
        self.assertIn("untrusted_research_task_state", prompt)

        await runtime.clear_conversation("d" * 32)
        self.assertEqual(agent.clear_calls, ["d" * 32])
        await runtime.aclose()
        self.assertEqual(agent.closed, 1)

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
