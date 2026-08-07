from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from paper_research_agent.agent.orchestrator.models import MainAgentResult
from paper_research_agent.answering.models import RAGAnswer
from paper_research_agent.conversation.models import (
    ConversationContextSnapshot,
    ConversationResolution,
    TurnInterpretation,
)
from paper_research_agent.conversation.store import InMemoryConversationStore
from paper_research_agent.web.app import create_app
from paper_research_agent.web.config import OwnerCredentials, WebConfig
from paper_research_agent.web.routing import RouteDecision

ORIGIN = "https://zhimoai.online"


class SeparateChatRuntime:
    is_ready = True
    is_busy = False

    def __init__(self) -> None:
        self.cleared: list[str] = []

    async def interpret_turn(
        self,
        snapshot: ConversationContextSnapshot,
        *,
        has_attachments: bool,
        rag_mode: str,
    ) -> TurnInterpretation:
        question = snapshot.original_question
        if question == "回到之前的测评方法":
            return TurnInterpretation(
                depends_on_history=True,
                standalone_question=question,
                chinese_query=question,
                confidence=0.4,
                needs_clarification=True,
                clarification_question="你指的是“大模型测评方法”，还是“RAG 测评方法”？",
                route="normal_chat",
                reason="多个历史主题相关度接近",
            )
        selected = ()
        standalone = question
        depends = False
        if "知识库" in question and snapshot.recent_turns:
            anchor = snapshot.recent_turns[-1]
            selected = (anchor.turn_id,)
            standalone = f"请基于本地论文知识库重新分析{anchor.user_question}。"
            depends = True
        if has_attachments:
            route = "attachment_qa"
        elif question.startswith("联网"):
            route = "web_research"
        elif rag_mode == "required" or "知识库" in question:
            route = "local_rag"
        else:
            route = "normal_chat"
        return TurnInterpretation(
            depends_on_history=depends,
            selected_history_turn_ids=selected,
            standalone_question=standalone,
            chinese_query=standalone,
            confidence=0.96,
            route=route,
            use_local_papers=(
                rag_mode == "required" or "知识库" in question
            ),
            use_web_research=route == "web_research",
            use_dynamic_tools=route == "web_research",
            use_attachments=has_attachments,
            research_mode="planned" if route == "web_research" else "single",
            reason="测试统一解释",
        )

    async def classify_route(
        self,
        question: str,
        *,
        has_attachments: bool,
        rag_mode: str,
        standalone_question: str | None = None,
        selected_history_turn_ids: tuple[str, ...] = (),
    ) -> RouteDecision:
        del standalone_question, selected_history_turn_ids
        if has_attachments:
            route = "attachment_qa"
        elif question.startswith("联网"):
            route = "web_research"
        else:
            route = "local_rag" if rag_mode == "required" else "normal_chat"
        return RouteDecision(
            route=route,
            confidence=1,
            reason="测试路由",
        )

    async def stream_chat(self, question: str, *, session_id: str):
        del session_id
        answer = "普通聊天已经讨论大模型测评。" if question == "大模型测评" else "继续回答。"
        yield {"type": "delta", "text": answer}
        yield {"type": "done", "metrics": None}

    async def clear_conversation(self, session_id: str) -> int:
        self.cleared.append(session_id)
        return 1

    async def stream_attachment_chat(
        self,
        question: str,
        *,
        attachment_texts: tuple[str, ...],
        session_id: str,
    ):
        del attachment_texts, session_id
        yield {"type": "delta", "text": f"附件回答：{question}"}
        yield {"type": "done", "metrics": None}

    async def aclose(self) -> None:
        return None


class SeparateRAGRuntime:
    is_ready = True
    is_busy = False
    rag_available = True
    agent_available = True
    research_planning_available = False

    def __init__(self) -> None:
        self.contexts: list[ConversationResolution] = []
        self.cleared: list[str] = []
        self.dynamic_questions: list[str] = []

    async def ask(
        self,
        question: str,
        *,
        session_id: str,
        research_mode: str = "single",
        conversation_context: ConversationResolution | None = None,
    ) -> object:
        del session_id, research_mode
        assert conversation_context is not None
        self.contexts.append(conversation_context)
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
                resolved_question=conversation_context.standalone_question,
                standalone_question=conversation_context.standalone_question,
                chinese_query=conversation_context.chinese_query,
                english_query="LLM evaluation",
                rewrite_status="success",
                degraded=False,
                degraded_reason=None,
                index_id="idx-test",
                audit_persisted=False,
                conversation_memory_hit_count=len(conversation_context.candidates),
                selected_history_turn_ids=conversation_context.selected_turn_ids,
                selected_history_questions=tuple(
                    item.user_question for item in conversation_context.selected_candidates
                ),
                inherited_across_route=conversation_context.inherited_across_route,
                rewrite_confidence=conversation_context.confidence,
                needs_clarification=False,
                hits=(),
            ),
            context=SimpleNamespace(
                estimated_tokens=100,
                token_budget=8192,
                output_reserve_tokens=1200,
                included_memory_turn_count=len(conversation_context.selected_turn_ids),
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
        del session_id
        self.dynamic_questions.append(question)
        return SimpleNamespace(
            run_id="b" * 32,
            status="completed",
            observations=(),
            final_summary=f"联网研究：{question}",
            termination_reason="router_finished",
            pending_approval=None,
        )

    async def aclose(self) -> None:
        return None


class CrossRouteConversationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryConversationStore()
        self.chat = SeparateChatRuntime()
        self.rag = SeparateRAGRuntime()
        config = WebConfig(
            credentials=OwnerCredentials(username="owner", password="password"),
            session_secret=b"s" * 32,
            allowed_origins=frozenset({ORIGIN}),
            max_question_chars=2_000,
        )
        self.context = TestClient(
            create_app(
                config=config,
                runtime=self.rag,
                chat_runtime=self.chat,
                conversation_store=self.store,
                serve_static=False,
            ),
            base_url=ORIGIN,
        )
        self.client = self.context.__enter__()
        response = self.client.post(
            "/paper-research/api/login",
            headers={"Origin": ORIGIN},
            json={"username": "owner", "password": "password"},
        )
        self.conversation_id = response.json()["conversation_id"]

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    def stream(self, question: str, *, rag_mode: str) -> list[dict[str, object]]:
        response = self.client.post(
            "/paper-research/api/chat/stream",
            headers={"Origin": ORIGIN},
            json={"question": question, "rag_mode": rag_mode},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]

    def test_normal_chat_to_local_rag_inherits_shared_topic(self) -> None:
        self.stream("大模型测评", rag_mode="disabled")
        events = self.stream("结合一下知识库", rag_mode="required")

        context = self.rag.contexts[-1]
        self.assertIn("大模型测评", context.standalone_question)
        self.assertTrue(context.inherited_across_route)
        rag_payload = next(item["payload"] for item in events if item["type"] == "rag_result")
        self.assertEqual(
            rag_payload["retrieval"]["selected_history_questions"], ["大模型测评"]
        )
        self.assertTrue(rag_payload["retrieval"]["inherited_across_route"])

    def test_preferred_mode_keeps_greeting_in_normal_chat(self) -> None:
        contexts_before = len(self.rag.contexts)

        events = self.stream("你好", rag_mode="preferred")

        self.assertEqual(len(self.rag.contexts), contexts_before)
        route = next(item for item in events if item["type"] == "route")
        self.assertEqual(route["route"], "normal_chat")
        self.assertFalse(route["capabilities"]["local_papers"])
        self.assertNotIn("rag_result", [item["type"] for item in events])

    def test_reference_knowledge_base_repeat_uses_model_interpretation(self) -> None:
        self.stream("大模型测评", rag_mode="disabled")
        events = self.stream("参考一下知识库再说一次", rag_mode="preferred")

        context = self.rag.contexts[-1]
        self.assertIn("大模型测评", context.standalone_question)
        self.assertTrue(context.selected_turn_ids)
        route = next(item for item in events if item["type"] == "route")
        self.assertTrue(route["capabilities"]["local_papers"])
        self.assertIn("rag_context", [item["type"] for item in events])
        self.assertIn("delta", [item["type"] for item in events])
        self.assertNotIn("rag_result", [item["type"] for item in events])

    def test_preferred_web_research_executes_local_rag_and_dynamic_tools(self) -> None:
        events = self.stream("联网研究大模型测评最新进展", rag_mode="preferred")

        event_types = [item["type"] for item in events]
        self.assertIn("rag_result", event_types)
        self.assertIn("tool_result", event_types)
        self.assertEqual(self.rag.dynamic_questions, ["联网研究大模型测评最新进展"])
        route = next(item for item in events if item["type"] == "route")
        self.assertEqual(
            route["capabilities"],
            {
                "local_papers": True,
                "web_research": True,
                "dynamic_tools": True,
                "attachments": False,
            },
        )

    def test_new_conversation_clears_both_runtimes_and_shared_index(self) -> None:
        self.stream("大模型测评", rag_mode="disabled")
        response = self.client.delete(
            "/paper-research/api/conversation", headers={"Origin": ORIGIN}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.store.history(self.conversation_id), ())
        self.assertEqual(self.rag.cleared, [self.conversation_id])
        self.assertEqual(self.chat.cleared, [self.conversation_id])

    def test_ambiguous_historical_topics_request_clarification_without_rag(self) -> None:
        self.stream("大模型测评方法", rag_mode="disabled")
        self.stream("RAG 测评方法", rag_mode="disabled")
        contexts_before = len(self.rag.contexts)

        events = self.stream("回到之前的测评方法", rag_mode="required")

        self.assertEqual(len(self.rag.contexts), contexts_before)
        route = next(item for item in events if item["type"] == "route")
        self.assertEqual(route["label"], "需要澄清")
        clarification = "".join(
            str(item.get("text", "")) for item in events if item["type"] == "delta"
        )
        self.assertIn("还是", clarification)
        latest = self.store.history(self.conversation_id)[-1]
        self.assertEqual(latest.status, "clarification_required")
        self.assertEqual(latest.user_question, "回到之前的测评方法")

    def test_main_agent_flag_routes_ask_with_request_id(self) -> None:
        class FakeMainAgent:
            async def run(self, request: object) -> MainAgentResult:
                return MainAgentResult(
                    run_id="r" * 32,
                    request_id=request.request_id,  # type: ignore[attr-defined]
                    conversation_id=request.conversation_id,  # type: ignore[attr-defined]
                    status="completed",
                    answer="主 Agent 集成回答",
                    workspace_version=1,
                )

        main_agent = FakeMainAgent()
        config = WebConfig(
            credentials=OwnerCredentials(username="owner", password="password"),
            session_secret=b"s" * 32,
            allowed_origins=frozenset({ORIGIN}),
            max_question_chars=2_000,
        )
        with patch.dict(os.environ, {"PRA_MAIN_AGENT_ENABLED": "true"}), TestClient(
            create_app(
                config=config,
                runtime=self.rag,
                chat_runtime=self.chat,
                conversation_store=self.store,
                serve_static=False,
                main_agent_runtime=main_agent,  # type: ignore[arg-type]
            ),
            base_url=ORIGIN,
        ) as client:
            login = client.post(
                "/paper-research/api/login",
                headers={"Origin": ORIGIN},
                json={"username": "owner", "password": "password"},
            )
            del login
            response = client.post(
                "/paper-research/api/ask",
                headers={"Origin": ORIGIN},
                json={
                    "question": "比较 RAG 与 GraphRAG",
                    "rag_mode": "preferred",
                    "request_id": "req-integration-1",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        events = [json.loads(line) for line in response.text.strip().split("\n")]
        self.assertEqual(events[0]["request_id"], "req-integration-1")
        self.assertEqual(events[-1]["status"], "completed")
        self.assertIn("主 Agent 集成回答", response.text)
        self.assertEqual(self.rag.contexts, [])

    def test_attachment_and_web_routes_store_original_user_questions(self) -> None:
        self.client.app.state.attachments = SimpleNamespace(
            extract=lambda session_id, attachment_ids: ("附件：a.md\n内部正文",)
        )
        attachment_response = self.client.post(
            "/paper-research/api/chat/stream",
            headers={"Origin": ORIGIN},
            json={
                "question": "总结附件",
                "rag_mode": "preferred",
                "attachment_ids": ["a" * 32],
            },
        )
        self.assertEqual(attachment_response.status_code, 200, attachment_response.text)
        self.stream("联网研究最新论文", rag_mode="preferred")

        history = self.store.history(self.conversation_id)
        self.assertEqual([item.route for item in history], ["attachment_qa", "web_research"])
        self.assertEqual(history[0].user_question, "总结附件")
        self.assertNotIn("内部正文", history[0].user_question)


if __name__ == "__main__":
    unittest.main()
