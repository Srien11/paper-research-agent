from __future__ import annotations

import json
import os
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from paper_research_agent.agent.orchestrator.artifacts import ChatArtifact, FileArtifact
from paper_research_agent.agent.orchestrator.control import (
    AgentRunControl,
    apply_plan_edit,
    transition_run_control,
)
from paper_research_agent.agent.orchestrator.models import (
    AgentTask,
    ChildTaskResult,
    ConversationWorkspace,
    GoalState,
    MainAgentResult,
    TaskPlan,
)
from paper_research_agent.agent.orchestrator.runtime import MainAgentRuntime
from paper_research_agent.conversation.models import ConversationResolution
from paper_research_agent.conversation.store import InMemoryConversationStore
from paper_research_agent.web.app import create_app
from paper_research_agent.web.config import OwnerCredentials, WebConfig
from paper_research_agent.web.run_event_bus import RunEventBus

ORIGIN = "https://example.test"
REQUEST_ID = "req_1234567890123456"


class _LegacyRuntime:
    is_ready = True
    is_busy = False

    def __init__(self) -> None:
        self.clear_calls: list[str] = []

    async def clear_conversation(self, conversation_id: str) -> int:
        self.clear_calls.append(conversation_id)
        return 0

    async def aclose(self) -> None:
        return None


class _RecordingStore(InMemoryConversationStore):
    def __init__(self) -> None:
        super().__init__()
        self.results: dict[str, MainAgentResult] = {}

    def load_agent_run(self, request_id: str) -> MainAgentResult | None:
        return self.results.get(request_id)


class _MainRuntime:
    def __init__(self, store: _RecordingStore) -> None:
        self.store = store
        self.requests: list[object] = []
        self.resume_calls: list[tuple[str, bool]] = []
        self.error: Exception | None = None
        self.clear_calls: list[str] = []
        now = datetime(2026, 8, 11, tzinfo=UTC)
        goal_id = "d" * 32
        self.control = AgentRunControl(
            request_id=REQUEST_ID,
            run_id="a" * 32,
            conversation_id="placeholder",
            updated_at=now,
        )
        self.workspace = ConversationWorkspace(
            conversation_id="placeholder",
            active_goal=GoalState(
                goal_id=goal_id,
                objective="完成研究计划",
                origin_turn_id="e" * 32,
                created_at=now,
                updated_at=now,
            ),
            task_plan=TaskPlan(
                plan_id="f" * 32,
                goal_id=goal_id,
                tasks=(
                    AgentTask(
                        task_id="step-1",
                        goal_id=goal_id,
                        title="检索证据",
                        objective="找到证据",
                        success_criteria=("找到来源",),
                        capability="local_rag",
                        execution_reason="先建立证据基础",
                    ),
                ),
                created_at=now,
                updated_at=now,
            ),
            updated_at=now,
        )

    async def run(self, request: object) -> MainAgentResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        request_id = str(request.request_id)
        cached = self.store.results.get(request_id)
        if cached is not None:
            return cached
        result = MainAgentResult(
            run_id="a" * 32,
            request_id=request_id,
            conversation_id=str(request.conversation_id),
            status="completed",
            answer="统一回答",
            route_trace=("direct_chat",),
            child_results=(
                ChildTaskResult(
                    child_run_id="child-1",
                    task_id="task-1",
                    capability="direct_chat",
                    status="completed",
                    summary="内部摘要不得出现在事件中",
                    artifact=ChatArtifact(text="统一回答"),
                ),
                ChildTaskResult(
                    child_run_id="child-2",
                    task_id="task-2",
                    capability="file_edit",
                    status="completed",
                    artifact=FileArtifact(
                        text="已生成文件",
                        output_attachment_ids=("f" * 32,),
                    ),
                ),
            ),
            workspace_version=3,
        )
        self.store.results[request_id] = result
        return result

    async def resume_approval(self, *, request_id: str, approved: bool) -> MainAgentResult:
        self.resume_calls.append((request_id, approved))
        result = MainAgentResult(
            run_id="b" * 32,
            request_id=request_id,
            conversation_id=next(iter(self.store.results.values())).conversation_id,
            status="completed",
            answer="审批后完成",
            workspace_version=4,
        )
        self.store.results[request_id] = result
        return result

    async def clear(self, conversation_id: str) -> None:
        self.clear_calls.append(conversation_id)

    async def load_workspace_for_run(self, request_id: str):
        if request_id != self.control.request_id:
            return None
        return self.control, self.workspace

    async def load_control(self, request_id: str):
        return self.control if request_id == self.control.request_id else None

    async def command_run(self, *, request_id: str, command):
        assert request_id == self.control.request_id
        self.control = transition_run_control(self.control, command)
        return self.control

    async def edit_plan(self, *, request_id: str, edit):
        assert request_id == self.control.request_id
        self.workspace = apply_plan_edit(self.workspace, edit)
        return self.workspace


class _ProductGraph:
    async def ainvoke(self, value: dict[str, object], config: object = None) -> dict[str, object]:
        del config
        start = dict(value["run_start"])
        return {
            "run_id": start["run_id"],
            "base_workspace_version": 0,
            "final_answer": "实时回答",
            "termination_reason": "completed",
            "child_results": [],
        }


def _events(response: object) -> list[dict[str, object]]:
    text = str(response.text)
    return [json.loads(line) for line in text.splitlines() if line]


class MainAgentApiTests(unittest.TestCase):
    def test_product_runtime_uses_v2_durable_stream_and_resume_endpoint(self) -> None:
        store = InMemoryConversationStore()
        bus = RunEventBus(store)
        runtime = MainAgentRuntime(
            graph=_ProductGraph(),
            repository=store,
            run_event_publisher=bus.publisher,
        )
        config = WebConfig(
            credentials=OwnerCredentials(username="owner", password="correct-password"),
            session_secret=b"p" * 32,
            allowed_origins=frozenset({ORIGIN}),
        )
        with TestClient(
            create_app(
                config=config,
                runtime=self.legacy,  # type: ignore[arg-type]
                serve_static=False,
                conversation_store=store,
                main_agent_runtime=runtime,
                run_event_bus=bus,
            ),
            base_url=ORIGIN,
        ) as client:
            login = client.post(
                "/paper-research/api/login",
                headers={"Origin": ORIGIN},
                json={"username": "owner", "password": "correct-password"},
            )
            response = client.post(
                "/paper-research/api/agent/runs",
                headers={"Origin": ORIGIN},
                json={"request_id": "req_product_api_12345", "message": "hello"},
            )
            events = _events(response)

            self.assertEqual(events[0]["schema_version"], "main-agent-stream-v2")
            self.assertEqual(events[0]["type"], "run_started")
            self.assertEqual(events[-1]["type"], "run_completed")
            self.assertIn("answer_delta", [item["type"] for item in events])

            replay = client.get(
                "/paper-research/api/agent/runs/req_product_api_12345/events",
                params={"after_event_id": events[-2]["event_id"]},
            )
            replayed = _events(replay)
            self.assertEqual([item["type"] for item in replayed], ["run_completed"])

            history = client.get(
                f"/paper-research/api/conversations/{login.json()['conversation_id']}"
            ).json()
            assistant = next(
                item for item in history["messages"] if item["role"] == "assistant"
            )
            self.assertEqual(assistant["request_id"], "req_product_api_12345")
            self.assertEqual(assistant["events"][-1]["type"], "run_completed")

    def setUp(self) -> None:
        self.store = _RecordingStore()
        self.main = _MainRuntime(self.store)
        self.legacy = _LegacyRuntime()
        config = WebConfig(
            credentials=OwnerCredentials(username="owner", password="correct-password"),
            session_secret=b"s" * 32,
            allowed_origins=frozenset({ORIGIN}),
            max_question_chars=100,
        )
        self.environment = patch.dict(os.environ, {"PRA_MAIN_AGENT_MODE": "primary"})
        self.environment.start()
        self.client_context = TestClient(
            create_app(
                config=config,
                runtime=self.legacy,  # type: ignore[arg-type]
                serve_static=False,
                conversation_store=self.store,
                main_agent_runtime=self.main,  # type: ignore[arg-type]
            ),
            base_url=ORIGIN,
        )
        self.client = self.client_context.__enter__()
        login = self.client.post(
            "/paper-research/api/login",
            headers={"Origin": ORIGIN},
            json={"username": "owner", "password": "correct-password"},
        )
        self.assertEqual(login.status_code, 200)
        conversation_id = self._conversation_id()
        self.main.control = self.main.control.model_copy(
            update={"conversation_id": conversation_id}
        )
        self.main.workspace = self.main.workspace.model_copy(
            update={"conversation_id": conversation_id}
        )

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.environment.stop()

    def test_agent_run_stream_has_ordered_single_done_and_safe_artifacts(self) -> None:
        response = self.client.post(
            "/paper-research/api/agent/runs",
            headers={"Origin": ORIGIN},
            json={
                "request_id": REQUEST_ID,
                "message": "hello",
                "rag_mode": "disabled",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        events = _events(response)
        self.assertEqual(
            [item["event_id"] for item in events], list(range(1, len(events) + 1))
        )
        self.assertEqual(sum(item["type"] == "done" for item in events), 1)
        self.assertEqual(events[0]["type"], "run_started")
        self.assertEqual(events[-1]["status"], "completed")
        self.assertIn("file_result", [item["type"] for item in events])
        self.assertIn("f" * 32, response.text)
        self.assertNotIn("内部摘要", response.text)
        request = self.main.requests[0]
        self.assertEqual(request.request_id, REQUEST_ID)

    def test_plan_control_edit_and_explanation_endpoints(self) -> None:
        plan = self.client.get(
            f"/paper-research/api/agent/runs/{REQUEST_ID}/plan"
        )
        self.assertEqual(plan.status_code, 200, plan.text)
        self.assertEqual(plan.json()["tasks"][0]["execution_reason"], "先建立证据基础")

        paused = self.client.post(
            f"/paper-research/api/agent/runs/{REQUEST_ID}/control",
            headers={"Origin": ORIGIN},
            json={"action": "pause", "expected_revision": 0},
        )
        self.assertEqual(paused.status_code, 200, paused.text)
        self.assertEqual(paused.json()["status"], "pause_requested")
        self.main.control = self.main.control.model_copy(update={"status": "paused"})

        edited = self.client.patch(
            f"/paper-research/api/agent/runs/{REQUEST_ID}/plan",
            headers={"Origin": ORIGIN},
            json={
                "expected_revision": 1,
                "objective": "调整后的研究计划",
                "task_edits": [
                    {
                        "task_id": "step-1",
                        "budget": {"max_seconds": 45, "max_calls": 2},
                    }
                ],
            },
        )
        self.assertEqual(edited.status_code, 200, edited.text)
        self.assertEqual(edited.json()["objective"], "调整后的研究计划")
        self.assertEqual(edited.json()["tasks"][0]["max_calls"], 2)

        explanation = self.client.get(
            f"/paper-research/api/agent/runs/{REQUEST_ID}/tasks/step-1/explanation"
        )
        self.assertEqual(explanation.status_code, 200, explanation.text)
        self.assertIn("先建立证据基础", explanation.json()["explanation"])

    def test_plan_endpoint_returns_running_placeholder_before_plan_exists(self) -> None:
        self.main.workspace = self.main.workspace.model_copy(
            update={"active_goal": None, "task_plan": None}
        )

        response = self.client.get(
            f"/paper-research/api/agent/runs/{REQUEST_ID}/plan"
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["control"]["status"], "running")
        self.assertEqual(response.json()["tasks"], [])

    def test_duplicate_request_is_projected_as_reused_and_status_is_queryable(self) -> None:
        payload = {
            "request_id": REQUEST_ID,
            "message": "hello",
            "rag_mode": "disabled",
        }
        first = self.client.post(
            "/paper-research/api/agent/runs", headers={"Origin": ORIGIN}, json=payload
        )
        second = self.client.post(
            "/paper-research/api/agent/runs", headers={"Origin": ORIGIN}, json=payload
        )
        status = self.client.get(f"/paper-research/api/agent/runs/{REQUEST_ID}")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(_events(second)[0]["type"], "run_reused")
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()["answer"], "统一回答")
        self.assertEqual(status.json()["workspace_version"], 3)

    def test_approval_endpoint_resumes_by_request_id(self) -> None:
        waiting = MainAgentResult(
            run_id="c" * 32,
            request_id=REQUEST_ID,
            conversation_id=self._conversation_id(),
            status="waiting_approval",
            pending_approval={
                "approval_request_id": "approval-1",
                "tool_name": "write_file",
                "purpose": "保存结果",
                "arguments_sha256": "d" * 64,
                "expires_at_epoch": 2_000_000_000.0,
            },
            workspace_version=3,
        )
        self.store.results[REQUEST_ID] = waiting

        response = self.client.post(
            f"/paper-research/api/agent/runs/{REQUEST_ID}/approval",
            headers={"Origin": ORIGIN},
            json={"approved": True},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.main.resume_calls, [(REQUEST_ID, True)])
        self.assertEqual(_events(response)[-1]["status"], "completed")
        repeated = self.client.post(
            f"/paper-research/api/agent/runs/{REQUEST_ID}/approval",
            headers={"Origin": ORIGIN},
            json={"approved": True},
        )
        self.assertEqual(repeated.status_code, 409)

    def test_waiting_status_projects_only_safe_approval_fields(self) -> None:
        self.store.results[REQUEST_ID] = MainAgentResult(
            run_id="c" * 32,
            request_id=REQUEST_ID,
            conversation_id=self._conversation_id(),
            status="waiting_approval",
            pending_approval={
                "approval_request_id": "private-approval-id",
                "tool_name": "write_file",
                "purpose": "保存结果",
                "arguments_sha256": "d" * 64,
                "expires_at_epoch": 2_000_000_000.0,
                "arguments": {"secret": "must-not-leak"},
            },
            workspace_version=3,
        )

        response = self.client.get(f"/paper-research/api/agent/runs/{REQUEST_ID}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "waiting_approval")
        self.assertNotIn("private-approval-id", response.text)
        self.assertNotIn("must-not-leak", response.text)

    def test_client_can_close_stream_without_losing_queryable_result(self) -> None:
        with self.client.stream(
            "POST",
            "/paper-research/api/agent/runs",
            headers={"Origin": ORIGIN},
            json={"request_id": REQUEST_ID, "message": "hello"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            first_line = next(response.iter_lines())
            self.assertIn("run_started", first_line)

        status_response = self.client.get(
            f"/paper-research/api/agent/runs/{REQUEST_ID}"
        )
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["status"], "completed")

    def test_missing_attachment_and_invalid_status_id_are_rejected(self) -> None:
        missing = self.client.post(
            "/paper-research/api/agent/runs",
            headers={"Origin": ORIGIN},
            json={
                "request_id": REQUEST_ID,
                "message": "read it",
                "attachment_ids": ["f" * 32],
            },
        )
        invalid_status = self.client.get("/paper-research/api/agent/runs/short")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(invalid_status.status_code, 422)

    def test_session_owned_attachment_can_be_downloaded(self) -> None:
        uploaded = self.client.post(
            "/paper-research/api/files?filename=generated-note.md",
            headers={"Origin": ORIGIN, "Content-Type": "text/markdown"},
            content="生成内容".encode(),
        )
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        attachment_id = uploaded.json()["attachment_id"]

        downloaded = self.client.get(
            f"/paper-research/api/files/{attachment_id}/download"
        )

        self.assertEqual(downloaded.status_code, 200)
        self.assertEqual(downloaded.content.decode("utf-8"), "生成内容")
        self.assertIn("generated-note.md", downloaded.headers["content-disposition"])
        self.client.delete(
            f"/paper-research/api/files/{attachment_id}",
            headers={"Origin": ORIGIN},
        )

    def test_new_conversation_archives_server_history_without_deleting_it(self) -> None:
        previous = self._conversation_id()
        turn = self.store.begin_turn(previous, "持久化问题")
        self.store.complete_turn(
            turn.turn_id,
            route="normal_chat",
            status="completed",
            resolution=ConversationResolution(
                original_question="持久化问题",
                standalone_question="持久化问题",
                chinese_query="持久化问题",
                confidence=1,
            ),
            assistant_summary="持久化回答",
        )

        response = self.client.delete(
            "/paper-research/api/conversation",
            headers={"Origin": ORIGIN},
        )
        archive = self.client.get("/paper-research/api/conversations")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotEqual(response.json()["conversation_id"], previous)
        self.assertEqual(self.main.clear_calls, [])
        self.assertEqual(self.legacy.clear_calls, [])
        self.assertEqual(archive.status_code, 200, archive.text)
        saved = next(
            item
            for item in archive.json()["conversations"]
            if item["conversation_id"] == previous
        )
        self.assertEqual(saved["messages"], [])
        self.assertFalse(saved["messages_loaded"])

        detail = self.client.get(
            f"/paper-research/api/conversations/{previous}?message_limit=2"
        )
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertTrue(detail.json()["messages_loaded"])
        self.assertFalse(detail.json()["has_more_messages"])
        self.assertEqual(detail.json()["message_count"], 2)
        self.assertEqual(detail.json()["messages"][0]["text"], "持久化问题")
        self.assertEqual(detail.json()["messages"][1]["text"], "持久化回答")

        activated = self.client.post(
            f"/paper-research/api/conversations/{previous}/activate",
            headers={"Origin": ORIGIN},
            json={},
        )
        self.assertEqual(activated.status_code, 200, activated.text)
        self.assertEqual(self._conversation_id(), previous)

    def test_conversation_messages_are_hydrated_in_bounded_pages(self) -> None:
        conversation_id = self._conversation_id()
        for index in range(13):
            question = f"问题 {index}"
            turn = self.store.begin_turn(conversation_id, question)
            self.store.complete_turn(
                turn.turn_id,
                route="normal_chat",
                status="completed",
                resolution=ConversationResolution(
                    original_question=question,
                    standalone_question=question,
                    chinese_query=question,
                    confidence=1,
                ),
                assistant_summary=f"回答 {index}",
            )

        initial = self.client.get(
            f"/paper-research/api/conversations/{conversation_id}"
        )
        expanded = self.client.get(
            f"/paper-research/api/conversations/{conversation_id}?message_limit=48"
        )

        self.assertEqual(initial.status_code, 200, initial.text)
        self.assertEqual(len(initial.json()["messages"]), 24)
        self.assertEqual(initial.json()["message_count"], 26)
        self.assertTrue(initial.json()["has_more_messages"])
        self.assertEqual(expanded.status_code, 200, expanded.text)
        self.assertEqual(len(expanded.json()["messages"]), 26)
        self.assertFalse(expanded.json()["has_more_messages"])

    def test_auth_origin_request_id_and_runtime_errors_fail_closed(self) -> None:
        payload = {"request_id": REQUEST_ID, "message": "hello"}
        no_origin = self.client.post("/paper-research/api/agent/runs", json=payload)
        invalid = self.client.post(
            "/paper-research/api/agent/runs",
            headers={"Origin": ORIGIN},
            json={"request_id": "short", "message": "hello"},
        )
        self.main.error = RuntimeError("provider secret-token leaked")
        failed = self.client.post(
            "/paper-research/api/agent/runs",
            headers={"Origin": ORIGIN},
            json=payload,
        )

        self.assertEqual(no_origin.status_code, 403)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("secret-token", failed.text)
        self.client.post(
            "/paper-research/api/logout",
            headers={"Origin": ORIGIN},
        )
        anonymous = self.client.post(
            "/paper-research/api/agent/runs",
            headers={"Origin": ORIGIN},
            json=payload,
        )
        self.assertEqual(anonymous.status_code, 401)

    def test_internal_value_error_is_not_reported_as_invalid_user_request(self) -> None:
        self.main.error = ValueError("session_id contains unsafe private detail")

        response = self.client.post(
            "/paper-research/api/agent/runs",
            headers={"Origin": ORIGIN},
            json={"request_id": REQUEST_ID, "message": "hello"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "主 Agent 内部契约校验失败")
        self.assertNotIn("session_id", response.text)

    def _conversation_id(self) -> str:
        response = self.client.get("/paper-research/api/session")
        return str(response.json()["conversation_id"])


if __name__ == "__main__":
    unittest.main()
