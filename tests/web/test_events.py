from __future__ import annotations

import unittest
from datetime import UTC, datetime

from pydantic import ValidationError

from paper_research_agent.agent.orchestrator.artifacts import (
    ChatArtifact,
    ChildExecutionMetrics,
)
from paper_research_agent.agent.orchestrator.models import ChildTaskResult, MainAgentResult
from paper_research_agent.web.events import (
    AgentEventProjector,
    AgentStreamEvent,
    SafeRunEventDetail,
)
from paper_research_agent.web.models import AgentRunRequest, SafeEvidenceSource


class AgentEventContractTests(unittest.TestCase):
    def test_answer_event_accepts_only_safe_citation_projection(self) -> None:
        citation = SafeEvidenceSource(
            citation_id="E1",
            chunk_id="chunk-1",
            corpus_id="C001",
            title="测试论文",
            official_url="https://example.com/paper",
            page_start=3,
            page_end=4,
            evidence_type="text",
            storage_class="internal_research_only",
            excerpt="受控长度的证据预览",
            final_rank=1,
        )

        detail = SafeRunEventDetail(citations=(citation,))

        self.assertEqual(detail.citations[0].citation_id, "E1")
        with self.assertRaises(ValidationError):
            SafeRunEventDetail(
                citations=(
                    {
                        **citation.model_dump(),
                        "local_path": "D:/private/paper.pdf",
                    },
                )
            )

    def test_v2_event_requires_stable_identity_and_safe_detail(self) -> None:
        event = AgentStreamEvent(
            event_id=1,
            type="tool_started",
            occurred_at=datetime.now(UTC),
            request_id="req_1234567890123456",
            run_id="run-1",
            turn_id="a" * 32,
            node_id="tool:task-1:search-corpus:1",
            parent_node_id="task:task-1",
            task_id="task-1",
            status="running",
            title="检索本地论文",
            summary="正在检索",
            detail=SafeRunEventDetail(
                tool_name="search_corpus",
                capability="local_rag",
            ),
        )

        self.assertEqual(event.schema_version, "main-agent-stream-v2")
        self.assertEqual(event.detail.tool_name, "search_corpus")
        with self.assertRaises(ValidationError):
            SafeRunEventDetail(tool_name="search_corpus", arguments={"secret": "x"})

    def test_only_answer_delta_may_contain_delta(self) -> None:
        common = {
            "event_id": 1,
            "occurred_at": datetime.now(UTC),
            "request_id": "req_1234567890123456",
            "run_id": "run-1",
            "turn_id": "a" * 32,
            "node_id": "answer:main",
        }
        event = AgentStreamEvent(type="answer_delta", delta="一段回答", **common)
        self.assertEqual(event.delta, "一段回答")
        with self.assertRaises(ValidationError):
            AgentStreamEvent(type="reasoning_summary", delta="隐藏推理", **common)
        with self.assertRaises(ValidationError):
            AgentStreamEvent(type="answer_delta", **common)

    def test_pause_and_approval_are_nonterminal_stream_boundaries(self) -> None:
        common = {
            "occurred_at": datetime.now(UTC),
            "request_id": "req_1234567890123456",
            "run_id": "run-1",
            "turn_id": "a" * 32,
            "node_id": "run:run-1",
        }
        paused = AgentStreamEvent(
            event_id=1,
            type="run_paused",
            status="paused",
            **common,
        )
        approval = AgentStreamEvent(
            event_id=2,
            type="run_waiting_approval",
            status="waiting_approval",
            **common,
        )
        completed = AgentStreamEvent(
            event_id=3,
            type="run_completed",
            status="completed",
            **common,
        )

        self.assertTrue(paused.closes_delivery_segment)
        self.assertTrue(approval.closes_delivery_segment)
        self.assertFalse(paused.is_terminal)
        self.assertTrue(completed.is_terminal)

    def test_agent_run_request_requires_client_request_id(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRunRequest(message="hello", rag_mode="disabled")
        with self.assertRaises(ValidationError):
            AgentRunRequest(request_id="short", message="hello", rag_mode="disabled")

        request = AgentRunRequest(
            request_id="req_1234567890123456",
            message=" hello ",
            rag_mode="disabled",
        )
        self.assertEqual(request.message, "hello")

    def test_event_sequence_is_monotonic_and_rejects_duplicate_done(self) -> None:
        projector = AgentEventProjector(
            request_id="req_1234567890123456",
            run_id="run-1",
        )
        started = projector.event("run_started")
        done = projector.done(status="completed", workspace_version=2)

        self.assertEqual(started.event_id, 1)
        self.assertEqual(done.event_id, 2)
        self.assertEqual(done.schema_version, "main-agent-stream-v1")
        with self.assertRaisesRegex(RuntimeError, "done event already emitted"):
            projector.done(status="completed", workspace_version=2)

    def test_done_requires_non_negative_workspace_version(self) -> None:
        projector = AgentEventProjector(
            request_id="req_1234567890123456",
            run_id="run-1",
        )
        with self.assertRaises(ValidationError):
            projector.done(status="completed", workspace_version=-1)

    def test_task_completed_projects_safe_timing_and_token_metrics(self) -> None:
        projector = AgentEventProjector(
            request_id="req_1234567890123456",
            run_id="run-1",
        )
        child = ChildTaskResult(
            child_run_id="child-1",
            task_id="task-1",
            capability="direct_chat",
            status="completed",
            artifact=ChatArtifact(
                text="answer",
                metrics=ChildExecutionMetrics(
                    elapsed_ms=1250,
                    input_tokens=240,
                    output_tokens=36,
                    total_tokens=276,
                ),
            ),
        )
        result = MainAgentResult(
            run_id="run-1",
            request_id="req_1234567890123456",
            conversation_id="conversation-1",
            status="completed",
            child_results=(child,),
        )

        event = next(
            item
            for item in projector.project_result(result)
            if item.type == "task_completed"
        )

        self.assertEqual(event.counts["elapsed_ms"], 1250)
        self.assertEqual(event.counts["total_tokens"], 276)


if __name__ == "__main__":
    unittest.main()
