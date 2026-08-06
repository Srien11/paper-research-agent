from __future__ import annotations

import json
import unittest

import httpx

from paper_research_agent.web.chat_runtime import (
    ConversationRuntime,
    RAGUnavailableError,
    RouteOutputError,
)


class ConversationRuntimeTests(unittest.IsolatedAsyncioTestCase):
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
