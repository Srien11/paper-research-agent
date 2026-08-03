from __future__ import annotations

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

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

        cookie_header = self.client.post(
            "/paper-research/api/login",
            headers={"Origin": ORIGIN},
            json={"username": "owner", "password": "correct-password"},
        ).headers["set-cookie"].lower()
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


if __name__ == "__main__":
    unittest.main()
