from __future__ import annotations

import json
import os
import unittest
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from paper_research_agent.agent.observability import AgentEvent
from paper_research_agent.agent.orchestrator.artifacts import DynamicToolArtifact
from paper_research_agent.agent.orchestrator.models import (
    ChildTaskResult,
    MainAgentRequest,
    MainAgentResult,
)
from paper_research_agent.conversation.store import InMemoryConversationStore
from paper_research_agent.web.app import create_app
from paper_research_agent.web.config import OwnerCredentials, WebConfig

ORIGIN = "https://example.test"


class _LegacyRuntime:
    is_ready = True
    is_busy = False

    def __init__(self) -> None:
        self.ask = AsyncMock()
        self.run_tool_research = AsyncMock(
            return_value={
                "run_id": "d" * 32,
                "status": "completed",
                "observations": (),
                "final_summary": "legacy tool",
            }
        )
        self.resume_tool_research = AsyncMock()

    def stream_chat(
        self, question: str, *, session_id: str
    ) -> AsyncIterator[dict[str, object]]:
        del question, session_id
        raise AssertionError("legacy stream must not run in primary mode")

    async def aclose(self) -> None:
        return None


class _MainRuntime:
    def __init__(self) -> None:
        self.requests: list[MainAgentRequest] = []
        self.resume_calls: list[tuple[str, bool]] = []
        self.event_sink: object | None = None

    async def run(self, request: MainAgentRequest) -> MainAgentResult:
        self.requests.append(request)
        message = str(request.message)
        waiting = "审批" in message
        return MainAgentResult(
            run_id="a" * 32,
            request_id=str(request.request_id),
            conversation_id=str(request.conversation_id),
            status="waiting_approval" if waiting else "completed",
            answer="" if waiting else "main answer",
            child_results=(
                ChildTaskResult(
                    child_run_id="child-1",
                    task_id="task-1",
                    capability="dynamic_tools",
                    status="waiting_approval" if waiting else "completed",
                    pending_approval=(
                        {
                            "approval_request_id": "approval-1",
                            "tool_name": "write_file",
                            "purpose": "保存结果",
                            "arguments_sha256": "f" * 64,
                            "expires_at_epoch": 2_000_000_000.0,
                        }
                        if waiting
                        else None
                    ),
                    artifact=(
                        None
                        if waiting
                        else DynamicToolArtifact(
                            text="main answer", tool_names=("search_web",)
                        )
                    ),
                ),
            ),
            pending_approval=(
                {
                    "approval_request_id": "approval-1",
                    "tool_name": "write_file",
                    "purpose": "保存结果",
                    "arguments_sha256": "f" * 64,
                    "expires_at_epoch": 2_000_000_000.0,
                }
                if waiting
                else None
            ),
            workspace_version=1,
        )

    async def resume_approval(
        self, *, request_id: str, approved: bool
    ) -> MainAgentResult:
        self.resume_calls.append((request_id, approved))
        return MainAgentResult(
            run_id="b" * 32,
            request_id=request_id,
            conversation_id=str(self.requests[-1].conversation_id),
            status="completed",
            answer="approval completed",
            workspace_version=2,
        )


class _RecordingSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def write(self, event: AgentEvent) -> bool:
        self.events.append(event)
        return True


class CompatibilityRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = _LegacyRuntime()
        self.main = _MainRuntime()
        self.main.event_sink = _RecordingSink()
        self.mode = patch.dict(os.environ, {"PRA_MAIN_AGENT_MODE": "primary"})
        self.mode.start()
        config = WebConfig(
            credentials=OwnerCredentials(username="owner", password="password"),
            session_secret=b"s" * 32,
            allowed_origins=frozenset({ORIGIN}),
        )
        self.app = create_app(
            config=config,
            runtime=self.runtime,  # type: ignore[arg-type]
            chat_runtime=self.runtime,  # type: ignore[arg-type]
            serve_static=False,
            conversation_store=InMemoryConversationStore(),
            main_agent_runtime=self.main,
        )
        self.client_context = TestClient(self.app, base_url=ORIGIN)
        self.client = self.client_context.__enter__()
        self.client.post(
            "/paper-research/api/login",
            headers={"Origin": ORIGIN},
            json={"username": "owner", "password": "password"},
        )

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.mode.stop()

    def test_primary_ask_calls_only_main_and_fails_explicit_legacy_projection(self) -> None:
        with patch(
            "paper_research_agent.web.app._prepare_turn", new=AsyncMock()
        ) as prepare:
            response = self.client.post(
                "/paper-research/api/ask",
                headers={"Origin": ORIGIN},
                json={"question": "paper question", "request_id": "req_1234567890123456"},
            )

        self.assertEqual(response.status_code, 409, response.text)
        self.assertIn("统一接口", response.json()["detail"])
        self.assertEqual(len(self.main.requests), 1)
        self.runtime.ask.assert_not_awaited()
        prepare.assert_not_awaited()

    def test_primary_chat_stream_is_a_main_agent_protocol_adapter(self) -> None:
        response = self.client.post(
            "/paper-research/api/chat/stream",
            headers={"Origin": ORIGIN},
            json={"question": "hello", "request_id": "req_1234567890123456"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        events = [json.loads(line) for line in response.text.splitlines() if line]
        self.assertEqual(events[0]["type"], "run_started")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(len(self.main.requests), 1)

    def test_primary_tool_routes_call_main_and_resume_same_request(self) -> None:
        run = self.client.post(
            "/paper-research/api/tools/run",
            headers={"Origin": ORIGIN},
            json={"question": "需要审批", "request_id": "req_1234567890123456"},
        )
        approval = self.client.post(
            "/paper-research/api/tools/approval",
            headers={"Origin": ORIGIN},
            json={"approved": True},
        )

        self.assertEqual(run.status_code, 200, run.text)
        self.assertEqual(run.json()["status"], "approval_required")
        self.assertEqual(approval.status_code, 200, approval.text)
        self.assertEqual(approval.json()["final_summary"], "approval completed")
        self.assertEqual(
            self.main.resume_calls,
            [("req_1234567890123456", True)],
        )
        self.runtime.run_tool_research.assert_not_awaited()
        self.runtime.resume_tool_research.assert_not_awaited()

    def test_deprecated_route_count_contains_no_request_body(self) -> None:
        self.client.post(
            "/paper-research/api/chat/stream",
            headers={"Origin": ORIGIN},
            json={"question": "sensitive-body", "request_id": "req_1234567890123456"},
        )
        counters = self.app.state.compatibility.deprecated_counts
        self.assertEqual(counters["chat_stream"], 1)
        self.assertNotIn("sensitive-body", repr(counters))
        sink = self.main.event_sink
        self.assertIsInstance(sink, _RecordingSink)
        self.assertEqual(sink.events[-1].event_type, "deprecated_endpoint_used")

    def test_deprecated_route_emits_body_free_observability_event(self) -> None:
        from paper_research_agent.web.compat import CompatibilityAdapter

        sink = _RecordingSink()
        adapter = CompatibilityAdapter(event_sink=sink)

        adapter.mark("chat_stream")

        event = sink.events[0]
        self.assertEqual(event.event_type, "deprecated_endpoint_used")
        self.assertEqual(event.endpoint, "chat_stream")
        self.assertEqual(event.requested_count, 1)


if __name__ == "__main__":
    unittest.main()
