from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from paper_research_agent.agent.orchestrator.models import MainAgentResult
from paper_research_agent.answering.models import RAGAnswer
from paper_research_agent.web.app import create_app
from paper_research_agent.web.config import OwnerCredentials, WebConfig
from paper_research_agent.web.models import RecommendedQuestion

ORIGIN = "https://zhimoai.online"


class RuntimeBusyError(RuntimeError):
    pass


class FakeRuntime:
    def __init__(self) -> None:
        self.is_ready = True
        self.is_busy = False
        self.questions: list[tuple[str, str]] = []
        self.cleared: list[str] = []
        self.error: Exception | None = None
        self.tool_runs: list[tuple[str, str]] = []
        self.approvals: list[tuple[str, bool]] = []
        self.memory_lists = 0

    async def ask(self, question: str, *, session_id: str) -> object:
        if self.error is not None:
            raise self.error
        self.questions.append((question, session_id))
        answer = RAGAnswer(
            status="insufficient_evidence",
            answer_markdown="现有证据不足。",
            claims=(),
            citations=(),
            requested_model="qwen-test",
            prompt_version="answer-v1",
            latency_ms=1,
            attempts=0,
        )
        return SimpleNamespace(
            answer=answer,
            sources=(),
            retrieval=SimpleNamespace(
                original_question=question,
                resolved_question=question,
                english_query=None,
                rewrite_status="timeout",
                degraded=True,
                degraded_reason="query_rewrite_timeout",
                index_id="idx_test",
                audit_persisted=False,
                hits=(),
            ),
            context=SimpleNamespace(
                estimated_tokens=100,
                token_budget=8192,
                output_reserve_tokens=1200,
                included_memory_turn_count=0,
                omitted_memory_turn_count=0,
                included_evidence_count=0,
                omitted_evidence_count=0,
                evidence_insufficient=True,
            ),
            generation=SimpleNamespace(
                requested_model="qwen-test",
                actual_model=None,
                prompt_version="answer-v1",
                input_tokens=0,
                output_tokens=0,
                latency_ms=1,
                attempts=0,
                audit_persisted=False,
            ),
        )

    async def clear_conversation(self, session_id: str) -> int:
        self.cleared.append(session_id)
        return 1

    async def run_tool_research(self, question: str, *, session_id: str) -> object:
        self.tool_runs.append((question, session_id))
        return SimpleNamespace(
            run_id="a" * 32,
            status="approval_required",
            observations=(),
            final_summary=None,
            termination_reason=None,
            pending_approval=SimpleNamespace(
                tool_name="save_research_note",
                purpose="Save a confirmed finding",
                arguments={"content": "must not cross Web"},
                approval_request_id="b" * 32,
                arguments_sha256="c" * 64,
                expires_at_epoch=2_000_000_000,
            ),
        )

    async def resume_tool_research(self, *, session_id: str, approved: bool) -> object:
        self.approvals.append((session_id, approved))
        return SimpleNamespace(
            run_id="a" * 32,
            status="completed",
            observations=(
                SimpleNamespace(
                    sequence=1,
                    tool_name="save_research_note",
                    purpose="Save a confirmed finding",
                    result=SimpleNamespace(
                        status="denied" if not approved else "ok",
                        trust="side_effect",
                        items=(),
                    ),
                ),
            ),
            final_summary="Sensitive tool request was denied." if not approved else "Saved.",
            termination_reason="approval_denied" if not approved else "router_finished",
            pending_approval=None,
        )

    async def list_long_term_memories(self, *, limit: int = 20) -> object:
        self.memory_lists += 1
        return SimpleNamespace(
            items=(
                {
                    "memory_id": "d" * 32,
                    "kind": "preference",
                    "content": "优先使用中文回答",
                    "source_chunk_ids": (),
                    "version": 2,
                    "created_at": "2026-08-01T00:00:00+00:00",
                    "updated_at": "2026-08-04T00:00:00+00:00",
                    "expires_at": None,
                    "supersedes_memory_id": "e" * 32,
                    "content_sha256": "f" * 64,
                    "scope_id": "global",
                    "status": "active",
                    "internal_path": "must-not-leak",
                },
            )[:limit]
        )

    async def aclose(self) -> None:
        return None


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = FakeRuntime()
        config = WebConfig(
            credentials=OwnerCredentials(username="owner", password="correct-password"),
            session_secret=b"s" * 32,
            allowed_origins=frozenset({ORIGIN}),
            max_question_chars=100,
        )
        self.client_context = TestClient(
            create_app(
                config=config,
                runtime=self.runtime,
                serve_static=False,
                recommended_questions=(
                    RecommendedQuestion(
                        category="评测方法",
                        title="真实测试",
                        prompt="如何评估 RAG？",
                    ),
                ),
            ),
            base_url=ORIGIN,
        )
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def login(self) -> dict[str, object]:
        response = self.client.post(
            "/paper-research/api/login",
            headers={"Origin": ORIGIN},
            json={"username": "owner", "password": "correct-password"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_health_readiness_and_anonymous_session(self) -> None:
        health = self.client.get("/paper-research/healthz")
        ready = self.client.get("/paper-research/readyz")
        session = self.client.get("/paper-research/api/session")
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(ready.json(), {"status": "ready"})
        self.assertEqual(session.json(), {"authenticated": False})
        self.assertEqual(health.headers["cache-control"], "no-store, max-age=0")
        self.assertIn("default-src 'self'", health.headers["content-security-policy"])
        self.assertEqual(
            health.headers["permissions-policy"],
            "camera=(), microphone=(), geolocation=()",
        )

    def test_login_cookie_and_authenticated_session(self) -> None:
        logged_in = self.login()
        self.assertTrue(logged_in["authenticated"])
        response = self.client.get("/paper-research/api/session")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conversation_id"], logged_in["conversation_id"])

        cookie_header = (
            self.client.post(
                "/paper-research/api/login",
                headers={"Origin": ORIGIN},
                json={"username": "owner", "password": "correct-password"},
            )
            .headers["set-cookie"]
            .lower()
        )
        self.assertIn("httponly", cookie_header)
        self.assertIn("secure", cookie_header)
        self.assertIn("samesite=strict", cookie_header)
        self.assertIn("path=/paper-research", cookie_header)

    def test_recommended_questions_require_auth_and_only_return_safe_fields(self) -> None:
        path = "/paper-research/api/recommended-questions"
        self.assertEqual(self.client.get(path).status_code, 401)
        self.login()
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [
                {
                    "category": "评测方法",
                    "title": "真实测试",
                    "prompt": "如何评估 RAG？",
                }
            ],
        )

    def test_login_rejects_wrong_origin_and_credentials(self) -> None:
        wrong_origin = self.client.post(
            "/paper-research/api/login",
            headers={"Origin": "https://evil.example"},
            json={"username": "owner", "password": "correct-password"},
        )
        wrong_password = self.client.post(
            "/paper-research/api/login",
            headers={"Origin": ORIGIN},
            json={"username": "owner", "password": "wrong"},
        )
        self.assertEqual(wrong_origin.status_code, 403)
        self.assertEqual(wrong_password.status_code, 401)

    def test_validation_response_does_not_echo_password(self) -> None:
        response = self.client.post(
            "/paper-research/api/login",
            headers={"Origin": ORIGIN},
            json={"username": "owner", "password": ""},
        )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("password", response.text.lower())

    def test_ask_requires_auth_and_passes_server_conversation_id(self) -> None:
        unauthorized = self.client.post(
            "/paper-research/api/ask",
            headers={"Origin": ORIGIN},
            json={"question": "什么是 RAG？"},
        )
        self.assertEqual(unauthorized.status_code, 401)

        session = self.login()
        answered = self.client.post(
            "/paper-research/api/ask",
            headers={"Origin": ORIGIN},
            json={"question": "  什么是 RAG？  "},
        )
        self.assertEqual(answered.status_code, 200, answered.text)
        self.assertEqual(
            self.runtime.questions,
            [("什么是 RAG？", session["conversation_id"])],
        )
        self.assertTrue(answered.json()["retrieval"]["degraded"])

    def test_question_limit_and_busy_error_are_sanitized(self) -> None:
        self.login()
        too_long = self.client.post(
            "/paper-research/api/ask",
            headers={"Origin": ORIGIN},
            json={"question": "问" * 101},
        )
        self.assertEqual(too_long.status_code, 422)

        self.runtime.error = RuntimeBusyError("must not leak")
        busy = self.client.post(
            "/paper-research/api/ask",
            headers={"Origin": ORIGIN},
            json={"question": "问题"},
        )
        self.assertEqual(busy.status_code, 409)
        self.assertNotIn("must not leak", busy.text)

    def test_dynamic_tool_approval_is_authenticated_and_redacted(self) -> None:
        unauthorized = self.client.post(
            "/paper-research/api/tools/run",
            headers={"Origin": ORIGIN},
            json={"question": "Save this finding"},
        )
        self.assertEqual(unauthorized.status_code, 401)
        session = self.login()

        paused = self.client.post(
            "/paper-research/api/tools/run",
            headers={"Origin": ORIGIN},
            json={"question": "Save this finding"},
        )

        self.assertEqual(paused.status_code, 200, paused.text)
        self.assertEqual(paused.json()["status"], "approval_required")
        self.assertEqual(paused.json()["pending_approval"]["tool_name"], "save_research_note")
        self.assertNotIn('"arguments":', paused.text)
        self.assertNotIn("must not cross Web", paused.text)
        self.assertNotIn("approval_request_id", paused.text)
        self.assertEqual(
            self.runtime.tool_runs,
            [("Save this finding", session["conversation_id"])],
        )

        denied = self.client.post(
            "/paper-research/api/tools/approval",
            headers={"Origin": ORIGIN},
            json={"approved": False},
        )
        self.assertEqual(denied.status_code, 200, denied.text)
        self.assertEqual(denied.json()["termination_reason"], "approval_denied")
        self.assertEqual(
            self.runtime.approvals,
            [(session["conversation_id"], False)],
        )

    def test_long_term_memory_list_requires_auth_and_whitelists_fields(self) -> None:
        path = "/paper-research/api/memories"
        self.assertEqual(self.client.get(path).status_code, 401)
        self.login()

        response = self.client.get(path)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["memories"][0]["content"], "优先使用中文回答")
        self.assertEqual(response.json()["memories"][0]["version"], 2)
        self.assertNotIn("content_sha256", response.text)
        self.assertNotIn("scope_id", response.text)
        self.assertNotIn("internal_path", response.text)
        self.assertEqual(self.runtime.memory_lists, 1)

    def test_new_conversation_clears_memory_and_rotates_id(self) -> None:
        initial = self.login()
        response = self.client.delete(
            "/paper-research/api/conversation",
            headers={"Origin": ORIGIN},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.runtime.cleared, [initial["conversation_id"]])
        self.assertNotEqual(response.json()["conversation_id"], initial["conversation_id"])

    def test_logout_revokes_session(self) -> None:
        self.login()
        response = self.client.post(
            "/paper-research/api/logout",
            headers={"Origin": ORIGIN},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            self.client.get("/paper-research/api/session").json(),
            {"authenticated": False},
        )

    def test_main_agent_flag_routes_ask_and_preserves_request_id(self) -> None:
        class FakeMainAgent:
            def __init__(self) -> None:
                self.requests: list[object] = []

            async def run(self, request: object) -> MainAgentResult:
                self.requests.append(request)
                return MainAgentResult(
                    run_id="r" * 32,
                    request_id=request.request_id,  # type: ignore[attr-defined]
                    conversation_id=request.conversation_id,  # type: ignore[attr-defined]
                    status="completed",
                    answer="主 Agent 回答",
                    workspace_version=1,
                )

        main_agent = FakeMainAgent()
        config = WebConfig(
            credentials=OwnerCredentials(username="owner", password="correct-password"),
            session_secret=b"s" * 32,
            allowed_origins=frozenset({ORIGIN}),
            max_question_chars=100,
        )
        with patch.dict(os.environ, {"PRA_MAIN_AGENT_ENABLED": "true"}):
            app = create_app(
                config=config,
                runtime=self.runtime,
                serve_static=False,
                main_agent_runtime=main_agent,  # type: ignore[arg-type]
            )
            with TestClient(app, base_url=ORIGIN) as client:
                client.post(
                    "/paper-research/api/login",
                    headers={"Origin": ORIGIN},
                    json={"username": "owner", "password": "correct-password"},
                )
                response = client.post(
                    "/paper-research/api/ask",
                    headers={"Origin": ORIGIN},
                    json={
                        "question": "比较 RAG 与 GraphRAG",
                        "rag_mode": "preferred",
                        "request_id": "req-123",
                    },
                )
        self.assertEqual(response.status_code, 200, response.text)
        events = [json.loads(line) for line in response.text.strip().split("\n")]
        self.assertEqual(events[0]["type"], "run_started")
        self.assertEqual(events[0]["request_id"], "req-123")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["status"], "completed")
        self.assertEqual(main_agent.requests[0].request_id, "req-123")

    def test_main_agent_flag_off_keeps_legacy_ask(self) -> None:
        self.login()
        with patch.dict(os.environ, {"PRA_MAIN_AGENT_ENABLED": "false"}):
            response = self.client.post(
                "/paper-research/api/ask",
                headers={"Origin": ORIGIN},
                json={"question": "比较 RAG 与 GraphRAG", "rag_mode": "required"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("answer", payload)


if __name__ == "__main__":
    unittest.main()
