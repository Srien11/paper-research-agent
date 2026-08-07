from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime

import httpx

from paper_research_agent.agent.orchestrator.models import ContextMessage
from paper_research_agent.conversation.models import (
    ConversationCandidate,
    ConversationContextSnapshot,
    ConversationResolution,
)
from paper_research_agent.conversation.store import InMemoryConversationStore
from paper_research_agent.web.chat_runtime import (
    ConversationRuntime,
    DirectResponseRequest,
    RAGUnavailableError,
    RouteOutputError,
)


class ConversationRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_interpreter_receives_bounded_shared_context_and_returns_plan(self) -> None:
        requests: list[dict[str, object]] = []
        turn_id = "a" * 32

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            content = json.dumps(
                {
                    "depends_on_history": True,
                    "selected_history_turn_ids": [turn_id],
                    "standalone_question": "请基于本地论文知识库重新分析大模型测评。",
                    "chinese_query": "大模型测评 方法 指标 基准 安全性",
                    "confidence": 0.96,
                    "needs_clarification": False,
                    "clarification_question": None,
                    "route": "web_research",
                    "use_local_papers": True,
                    "use_web_research": True,
                    "use_dynamic_tools": True,
                    "use_attachments": False,
                    "research_mode": "planned",
                    "reason": "同时使用本地论文与动态研究",
                },
                ensure_ascii=False,
            )
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        candidate = ConversationCandidate(
            turn_id=turn_id,
            sequence=1,
            user_question="大模型测评",
            standalone_question="大模型测评",
            route="normal_chat",
            assistant_summary="讨论了评测指标和安全性。",
            status="completed",
            episode_id="b" * 16,
            relevance=1,
        )
        snapshot = ConversationContextSnapshot(
            original_question="参考一下知识库再说一次",
            recent_turns=(candidate,),
            recalled_turns=(),
            episodes=(),
            prepared_at=datetime.now(UTC),
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        interpretation = await runtime.interpret_turn(
            snapshot,
            has_attachments=False,
            rag_mode="preferred",
        )

        self.assertEqual(interpretation.selected_history_turn_ids, (turn_id,))
        self.assertTrue(interpretation.use_local_papers)
        self.assertTrue(interpretation.use_dynamic_tools)
        model_input = json.loads(requests[0]["messages"][-1]["content"])
        self.assertEqual(model_input["recent_turns"][0]["user_question"], "大模型测评")
        self.assertIn("评测指标和安全性", model_input["recent_turns"][0]["assistant_summary"])
        self.assertEqual(model_input["rag_mode"], "preferred")
        await client.aclose()

    async def test_turn_interpreter_rejects_unknown_selected_turn(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            content = json.dumps(
                {
                    "depends_on_history": True,
                    "selected_history_turn_ids": ["f" * 32],
                    "standalone_question": "错误主题",
                    "chinese_query": "错误主题",
                    "confidence": 0.9,
                    "route": "local_rag",
                    "use_local_papers": True,
                    "reason": "错误选择",
                }
            )
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        snapshot = ConversationContextSnapshot(
            original_question="继续",
            recent_turns=(),
            recalled_turns=(),
            episodes=(),
            prepared_at=datetime.now(UTC),
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        with self.assertRaises(RouteOutputError):
            await runtime.interpret_turn(
                snapshot,
                has_attachments=False,
                rag_mode="preferred",
            )

        self.assertEqual(calls, 2)
        await client.aclose()

    async def test_turn_interpreter_repairs_known_research_mode_alias(self) -> None:
        content = json.dumps(
            {
                "depends_on_history": False,
                "selected_history_turn_ids": [],
                "standalone_question": "大模型测评",
                "chinese_query": "大模型测评",
                "confidence": 0.9,
                "route": "local_rag",
                "use_local_papers": True,
                "research_mode": "rag",
                "reason": "使用本地论文",
            },
            ensure_ascii=False,
        )
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"choices": [{"message": {"content": content}}]}
                )
            )
        )
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)
        snapshot = ConversationContextSnapshot(
            original_question="大模型测评",
            recent_turns=(),
            recalled_turns=(),
            episodes=(),
            prepared_at=datetime.now(UTC),
        )

        interpretation = await runtime.interpret_turn(
            snapshot,
            has_attachments=False,
            rag_mode="preferred",
        )

        self.assertEqual(interpretation.research_mode, "single")
        await client.aclose()

    async def test_model_router_returns_strict_structured_decision(self) -> None:
        requests: list[dict[str, object]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "route": "file_edit",
                                        "confidence": 0.96,
                                        "reason": "用户明确要求改写附件",
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        decision = await runtime.classify_route(
            "把第二段改短", has_attachments=True, rag_mode="preferred"
        )

        self.assertEqual(decision.route, "file_edit")
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertIn("has_attachments", requests[0]["messages"][-1]["content"])
        self.assertIn('"rag_mode": "preferred"', requests[0]["messages"][-1]["content"])
        await client.aclose()

    async def test_model_router_repairs_code_fence_and_long_reason(self) -> None:
        reason = "说明" * 100

        def handler(request: httpx.Request) -> httpx.Response:
            content = "```json\n" + json.dumps(
                {"route": "normal_chat", "confidence": 0.8, "reason": reason},
                ensure_ascii=False,
            ) + "\n```"
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        decision = await runtime.classify_route(
            "大模型测评方法", has_attachments=False, rag_mode="disabled"
        )

        self.assertEqual(decision.route, "normal_chat")
        self.assertEqual(len(decision.reason), 160)
        await client.aclose()

    async def test_model_router_retries_invalid_contract_once(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            content = "not-json" if calls == 1 else json.dumps(
                {"route": "normal_chat", "confidence": 0.9, "reason": "普通知识问题"}
            )
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        decision = await runtime.classify_route(
            "大模型测评方法", has_attachments=False, rag_mode="disabled"
        )

        self.assertEqual(decision.route, "normal_chat")
        self.assertEqual(calls, 2)
        await client.aclose()

    async def test_model_router_raises_distinct_error_after_retry(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200, json={"choices": [{"message": {"content": '{"route":"unknown"}'}}]}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        with self.assertRaises(RouteOutputError):
            await runtime.classify_route(
                "大模型测评方法", has_attachments=False, rag_mode="disabled"
            )

        self.assertEqual(calls, 2)
        await client.aclose()

    async def test_calls_chat_contract_and_keeps_context(self) -> None:
        requests: list[dict[str, object]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            answer = "你好，我在。" if len(requests) == 1 else "我记得你刚才打了招呼。"
            return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        first = await runtime.run_tool_research("你好", session_id="session-1")
        second = await runtime.run_tool_research("我刚才说了什么？", session_id="session-1")

        self.assertEqual(first.final_summary, "你好，我在。")
        self.assertEqual(second.final_summary, "我记得你刚才打了招呼。")
        messages = requests[1]["messages"]
        self.assertTrue(any(item["role"] == "assistant" for item in messages))
        await client.aclose()

    async def test_contextual_chat_uses_explicit_projection_without_store_read(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            sse = (
                'data: {"choices": [{"delta": {"content": "继续回答。"}}]}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=sse.encode())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)
        store = InMemoryConversationStore()
        runtime.set_conversation_store(store)
        store.begin_turn("session-1", "store 中的旧问题")
        request = DirectResponseRequest(
            session_id="session-1",
            current_message="继续",
            recent_messages=(
                ContextMessage(
                    turn_id="t1",
                    sequence=1,
                    role="user",
                    content="来自显式上下文的旧问题",
                ),
            ),
            summary="讨论模型评测",
            active_goal="比较 RAG 与 GraphRAG",
            active_task="收集论文证据",
        )
        events: list[dict[str, object]] = []
        async for event in runtime.stream_contextual_chat(request):
            events.append(event)
        self.assertTrue(any(item["type"] == "done" for item in events))
        contents = [item["content"] for item in captured["body"]["messages"]]
        self.assertEqual(contents.count("继续"), 1)
        self.assertIn("来自显式上下文的旧问题", contents)
        self.assertNotIn("store 中的旧问题", contents)
        self.assertIn("活动目标：比较 RAG 与 GraphRAG", contents)
        self.assertIn("当前任务：收集论文证据", contents)
        await client.aclose()

    async def test_contextual_chat_rejects_blank_message(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=b"data: [DONE]\n\n"
                )
            )
        )
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)
        request = DirectResponseRequest(session_id="s", current_message="   ")
        with self.assertRaises(ValueError):
            async for _event in runtime.stream_contextual_chat(request):
                pass
        await client.aclose()

    async def test_rag_mode_is_unavailable_without_corpus(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(500))
        )
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)
        with self.assertRaises(RAGUnavailableError):
            await runtime.ask("论文问题", session_id="session-1")
        await client.aclose()

    async def test_streams_text_and_finishes_with_usage_metrics(self) -> None:
        body = (
            'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"，我在。"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5,'
            '"total_tokens":16}}\n\ndata: [DONE]'
        )
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, content=(body + "\n\n").encode(), headers={"content-type": "text/event-stream"}
                )
            )
        )
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        events = [event async for event in runtime.stream_chat("你好", session_id="session-1")]

        self.assertEqual("".join(event.get("text", "") for event in events), "你好，我在。")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["metrics"]["total_tokens"], 16)
        await client.aclose()

    async def test_attachment_question_is_not_treated_as_file_edit(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            body = 'data: {"choices":[{"delta":{"content":"这是摘要。"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=body.encode())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(api_key="test", model="qwen-test", client=client)

        events = [
            event
            async for event in runtime.stream_attachment_chat(
                "总结一下",
                attachment_texts=("附件：a.md\n完整正文",),
                session_id="session-1",
            )
        ]

        self.assertEqual(events[0]["text"], "这是摘要。")
        user_prompt = requests[0]["messages"][-1]["content"]
        self.assertIn("不要复述或输出完整附件", user_prompt)
        await client.aclose()

    async def test_local_rag_turn_is_visible_to_following_normal_chat(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            body = 'data: {"choices":[{"delta":{"content":"我记得该主题。"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=body.encode())

        store = InMemoryConversationStore()
        previous = store.begin_turn("session-1", "大模型测评")
        resolution = ConversationResolution(
            original_question="大模型测评",
            standalone_question="大模型测评",
            chinese_query="大模型测评",
            confidence=1,
            episode_id="a" * 16,
        )
        store.complete_turn(
            previous.turn_id,
            route="local_rag",
            status="completed",
            resolution=resolution,
            assistant_summary="论文证据覆盖方法、指标和安全性。",
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        runtime = ConversationRuntime(
            api_key="test",
            model="qwen-test",
            client=client,
            conversation_store=store,
        )

        _events = [
            event async for event in runtime.stream_chat("继续说说", session_id="session-1")
        ]

        messages = requests[0]["messages"]
        self.assertTrue(
            any(
                item["role"] == "assistant" and "方法、指标和安全性" in item["content"]
                for item in messages
            )
        )
        await client.aclose()
