from __future__ import annotations

import asyncio
import hashlib
import unittest
from collections.abc import Sequence

from pydantic import ValidationError

from paper_research_agent.agent.observability import AgentEvent
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
        "assessments": [
            {
                "evidence_sufficient": True,
                "status": "sufficient",
                "next_query": None,
                "next_objective": None,
            }
        ],
        "action_history": [
            {
                "sequence": 1,
                "action": "search_corpus",
                "step_id": "methods",
                "query": "grounded RAG",
                "chunk_ids": [],
                "outcome": None,
            },
            {
                "sequence": 2,
                "action": "get_evidence",
                "step_id": "methods",
                "query": None,
                "chunk_ids": [chunk.chunk_id],
                "outcome": None,
            },
            {
                "sequence": 3,
                "action": "assess_evidence",
                "step_id": "methods",
                "query": None,
                "chunk_ids": [],
                "outcome": "sufficient",
            },
            {
                "sequence": 4,
                "action": "finish",
                "step_id": None,
                "query": None,
                "chunk_ids": [],
                "outcome": "evidence_sufficient",
            },
        ],
        "replan_count": 0,
        "consecutive_no_new_evidence": 0,
        "active_step": None,
        "next_action": "finish",
        "evidence_sufficient": True,
        "termination_reason": "evidence_sufficient",
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


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def write(self, event: AgentEvent) -> bool:
        self.events.append(event)
        return True

    def types(self) -> Sequence[str]:
        return [event.event_type for event in self.events]


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
            event_sink=(events := EventRecorder()),
        )

        result = await runtime.run("  比较 RAG 方法  ", thread_id="session-1")

        self.assertEqual(result.question, "比较 RAG 方法")
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(result.termination_reason, "evidence_sufficient")
        self.assertTrue(result.evidence_sufficient)
        self.assertEqual(result.action_history[-1].action, "finish")
        self.assertEqual(result.evidence[0].asset_id, "asset-1")
        self.assertEqual(result.evidence[0].final_rank, 1)
        self.assertEqual(graph.calls[0][0]["question"], "比较 RAG 方法")
        self.assertRegex(str(graph.calls[0][0]["run_id"]), r"^[0-9a-f]{32}$")
        self.assertEqual(
            graph.calls[0][1]["configurable"],
            {"thread_id": "session-1"},
        )
        self.assertIn('"step_id":"methods"', result.task_state)
        self.assertNotIn(chunk.text, result.task_state)
        self.assertEqual(result.run_id, graph.calls[0][0]["run_id"])
        self.assertEqual(events.types(), ["run_started", "run_completed"])
        serialized = "".join(event.model_dump_json() for event in events.events)
        self.assertNotIn("比较 RAG 方法", serialized)
        self.assertNotIn("session-1", serialized)
        self.assertEqual(events.events[-1].tool_call_count, 2)
        self.assertEqual(events.events[-1].termination_reason, "evidence_sufficient")

    async def test_rejects_tampered_react_terminal_state(self) -> None:
        chunk = _chunk()
        invalid_sequence = _state(chunk)
        invalid_sequence["action_history"][1]["sequence"] = 9  # type: ignore[index]
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(invalid_sequence),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
        )
        with self.assertRaisesRegex(ValueError, "action sequence"):
            await runtime.run("比较 RAG 方法", thread_id="session-1")

        mismatched_termination = _state(chunk)
        mismatched_termination["termination_reason"] = "plan_exhausted"
        mismatched_termination["action_history"][-1]["outcome"] = "plan_exhausted"  # type: ignore[index]
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(mismatched_termination),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
        )
        with self.assertRaisesRegex(ValueError, "termination"):
            await runtime.run("比较 RAG 方法", thread_id="session-1")

    async def test_rejects_replan_and_tool_count_mismatches(self) -> None:
        chunk = _chunk()
        replan_mismatch = _state(chunk)
        replan_mismatch["replan_count"] = 1
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(replan_mismatch),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
        )
        with self.assertRaisesRegex(ValueError, "replan"):
            await runtime.run("比较 RAG 方法", thread_id="session-1")

        tool_mismatch = _state(chunk)
        tool_mismatch["tool_call_count"] = 1
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(tool_mismatch),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
        )
        with self.assertRaisesRegex(ValueError, "tool action"):
            await runtime.run("比较 RAG 方法", thread_id="session-1")

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
        events = EventRecorder()
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(_state(chunk), delay=0.05),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
            policy=ResearchRuntimePolicy(timeout_seconds=0.01),
            event_sink=events,
        )

        with self.assertRaisesRegex(TimeoutError, "deadline"):
            await runtime.run("比较 RAG 方法", thread_id="session-1")
        self.assertEqual(events.types(), ["run_started", "runtime_intercepted"])
        self.assertEqual(events.events[-1].reason_code, "total_timeout")

    async def test_logs_rejected_graph_output_without_sensitive_bodies(self) -> None:
        chunk = _chunk()
        events = EventRecorder()
        state = _state(chunk, text="tampered private evidence")
        state["question"] = "private question"
        runtime = ResearchAgentRuntime(
            graph=FakeGraph(state),
            chunks=(chunk,),
            storage_classes={"C001": "internal_research_only"},
            event_sink=events,
        )

        with self.assertRaisesRegex(ValueError, "immutable chunk"):
            await runtime.run("private question", thread_id="private-session")

        self.assertEqual(events.types(), ["run_started", "output_rejected"])
        serialized = "".join(event.model_dump_json() for event in events.events)
        self.assertNotIn("private question", serialized)
        self.assertNotIn("private-session", serialized)
        self.assertNotIn("tampered private evidence", serialized)

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
