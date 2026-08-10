from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from paper_research_agent.agent.orchestrator.artifacts import ChatArtifact, FileArtifact
from paper_research_agent.agent.orchestrator.models import ChildTaskResult, MainAgentResult
from paper_research_agent.conversation.store import InMemoryConversationStore
from paper_research_agent.web.app import create_app
from paper_research_agent.web.config import OwnerCredentials, WebConfig

ORIGIN = "https://example.test"
REQUEST_ID = "req_1234567890123456"


class _LegacyRuntime:
    is_ready = True
    is_busy = False

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


def _events(response: object) -> list[dict[str, object]]:
    text = str(response.text)
    return [json.loads(line) for line in text.splitlines() if line]


class MainAgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = _RecordingStore()
        self.main = _MainRuntime(self.store)
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
                runtime=_LegacyRuntime(),  # type: ignore[arg-type]
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

    def _conversation_id(self) -> str:
        response = self.client.get("/paper-research/api/session")
        return str(response.json()["conversation_id"])


if __name__ == "__main__":
    unittest.main()
