from __future__ import annotations

import asyncio
import hashlib
import unittest

from pydantic import ValidationError

from paper_research_agent.agent.policy import ResearchRuntimePolicy
from paper_research_agent.agent.runtime import ResearchAgentRuntime
from paper_research_agent.chunking.models import EvidenceChunk


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chunk() -> EvidenceChunk:
    text = "Retrieval augmented generation improves grounded answers."
    return EvidenceChunk(
        chunk_id="chunk-1",
        asset_id="asset-1",
        corpus_id="C001",
        section_id="results",
        element_ids=("element-1",),
        page_start=3,
        page_end=3,
        token_start=0,
        token_end=7,
        text=text,
        text_sha256=_digest(text),
        config_sha256="a" * 64,
    )


def _state(chunk: EvidenceChunk, *, text: str | None = None) -> dict[str, object]:
    evidence_text = chunk.text if text is None else text
    record = {
        "chunk_id": chunk.chunk_id,
        "corpus_id": chunk.corpus_id,
        "section_id": chunk.section_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text": evidence_text,
        "text_sha256": chunk.text_sha256,
        "evidence_type": chunk.evidence_type,
        "storage_class": "internal_research_only",
    }
    search = {
        "query": "grounded RAG",
        "index_id": "idx-test",
        "degraded": False,
        "degraded_reason": None,
        "hits": [
            {
                "chunk_id": chunk.chunk_id,
                "corpus_id": chunk.corpus_id,
                "section_id": chunk.section_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text_sha256": chunk.text_sha256,
                "evidence_type": chunk.evidence_type,
                "storage_class": "internal_research_only",
                "final_rank": 1,
            }
        ],
    }
    return {
        "question": "比较 RAG 方法",
        "plan": {
            "steps": [
                {
                    "step_id": "methods",
                    "objective": "查找方法",
                    "query": "grounded RAG",
                    "top_k": 4,
                }
            ]
        },
        "current_step": 1,
        "tool_call_count": 2,
        "observations": [
            {
                "step_id": "methods",
                "objective": "查找方法",
                "search": search,
                "evidence": {"records": [record], "missing_chunk_ids": []},
            }
        ],
        "evidence_records": [record],
    }


class FakeGraph:
    def __init__(self, state: dict[str, object], *, delay: float = 0):
        self.state = state
        self.delay = delay
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    async def ainvoke(
        self,
        value: dict[str, object],
        config: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((value, config))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.state


class ResearchRuntimePolicyTests(unittest.TestCase):
    def test_rejects_unknown_or_empty_tool_allowlist(self) -> None:
        with self.assertRaises(ValidationError):
            ResearchRuntimePolicy(allowed_tools=frozenset({"run_shell"}))
        with self.assertRaises(ValidationError):
            ResearchRuntimePolicy(allowed_tools=frozenset())


class ResearchAgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_maps_thread_and_revalidates_graph_evidence(self) -> None:
        chunk = _chunk()
        graph = FakeGraph(_state(chunk))
        runtime = ResearchAgentRuntime(
            graph=graph,
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
            policy=ResearchRuntimePolicy(max_steps=2, max_tool_calls=4),
        )

        result = await runtime.run("  比较 RAG 方法  ", thread_id="session-1")

        self.assertEqual(result.question, "比较 RAG 方法")
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(result.evidence[0].asset_id, "asset-1")
        self.assertEqual(result.evidence[0].final_rank, 1)
        self.assertEqual(graph.calls[0][0], {"question": "比较 RAG 方法"})
        self.assertEqual(
            graph.calls[0][1]["configurable"],
            {"thread_id": "session-1"},
        )
        self.assertIn('"step_id":"methods"', result.task_state)
        self.assertNotIn(chunk.text, result.task_state)

    async def test_rejects_evidence_changed_after_tool_execution(self) -> None:
        chunk = _chunk()
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(_state(chunk, text="tampered evidence")),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
        )

        with self.assertRaisesRegex(ValueError, "immutable chunk"):
            await runtime.run("比较 RAG 方法", thread_id="session-1")

    async def test_enforces_total_timeout(self) -> None:
        chunk = _chunk()
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(_state(chunk), delay=0.05),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
            policy=ResearchRuntimePolicy(timeout_seconds=0.01),
        )

        with self.assertRaisesRegex(TimeoutError, "deadline"):
            await runtime.run("比较 RAG 方法", thread_id="session-1")

    async def test_rejects_blank_thread_id(self) -> None:
        chunk = _chunk()
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(_state(chunk)),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
        )

        with self.assertRaisesRegex(ValueError, "thread_id"):
            await runtime.run("比较 RAG 方法", thread_id=" ")


if __name__ == "__main__":
    unittest.main()
