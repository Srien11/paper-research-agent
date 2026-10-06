from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import unittest
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.config import AnsweringConfig
from paper_research_agent.answering.dashscope import (
    AnswerGenerationError,
    DashScopeAnswerGenerator,
)
from paper_research_agent.answering.models import AnswerRequest
from paper_research_agent.context.models import AssembledContext, CitationRef, PromptMessage


def answer_request(*, output_reserve_tokens: int = 1200) -> AnswerRequest:
    return AnswerRequest(
        context=AssembledContext(
            messages=(
                PromptMessage(
                    role="system",
                    content=(
                        "Return exact claims JSON. Each claim uses citation_ids and its text "
                        "must not contain inline citation markers."
                    ),
                ),
                PromptMessage(role="user", content="UNTRUSTED evidence sentinel"),
            ),
            citations=(
                CitationRef(
                    citation_id="E1",
                    chunk_id="chunk-1",
                    corpus_id="C001",
                    asset_id="asset-1",
                    page_start=1,
                    page_end=1,
                    text_sha256=hashlib.sha256(b"evidence").hexdigest(),
                    storage_class="internal_research_only",
                ),
            ),
            estimated_tokens=100,
            token_budget=2000,
            output_reserve_tokens=output_reserve_tokens,
            omitted_evidence_count=0,
        )
    )


async def no_sleep(_seconds: float) -> None:
    return None


class DashScopeAnswerGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_uses_fixed_factual_parameters_and_returns_usage(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                request=request,
                json={
                    "model": "qwen3.7-plus-2026-05-26",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "status": "answered",
                                        "claims": [
                                            {"text": "证据支持该结论。", "citation_ids": ["E1"]}
                                        ],
                                        "insufficient_reason": None,
                                    },
                                    ensure_ascii=False,
                                )
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 101, "completion_tokens": 22},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(
                AnsweringConfig(),
                base_url="https://example.invalid/v1",
                client=client,
                sleep=no_sleep,
            )
            result = await generator.generate(answer_request())

        self.assertEqual(len(requests), 1)
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["model"], "qwen3.7-plus-2026-05-26")
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["top_p"], 0.7)
        self.assertEqual(payload["max_tokens"], 1200)
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["messages"][1]["content"], "UNTRUSTED evidence sentinel")
        self.assertNotIn("Authorization", payload)
        self.assertEqual((result.input_tokens, result.output_tokens), (101, 22))
        self.assertEqual(result.attempts, 1)

    async def test_retryable_invalid_response_is_retried_and_usage_is_accumulated(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            content = (
                "not-json"
                if calls == 1
                else '{"status":"answered","claims":[{"text":"结论。","citation_ids":["E1"]}],"insufficient_reason":null}'
            )
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(
                AnsweringConfig(),
                base_url="https://example.invalid/v1",
                client=client,
                sleep=no_sleep,
            )
            result = await generator.generate(answer_request())

        self.assertEqual(calls, 2)
        self.assertEqual(result.attempts, 2)
        self.assertEqual((result.input_tokens, result.output_tokens), (20, 4))

    async def test_http_error_is_sanitized_and_non_retryable_400_stops(self) -> None:
        secret = "EVIDENCE_AND_KEY_MUST_NOT_APPEAR"
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                400,
                request=request,
                json={"error": {"code": "InvalidParameter", "message": secret}},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(
                AnsweringConfig(),
                base_url="https://example.invalid/v1",
                client=client,
                sleep=no_sleep,
            )
            with self.assertRaises(AnswerGenerationError) as raised:
                await generator.generate(answer_request())

        self.assertEqual(calls, 1)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.error_code, "InvalidParameter")
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn("EVIDENCE", str(raised.exception))

    async def test_truncated_response_fails_closed_without_partial_answer(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"status":"answered"'},
                        }
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(
                AnsweringConfig(max_retries=0),
                base_url="https://example.invalid/v1",
                client=client,
            )
            with self.assertRaisesRegex(AnswerGenerationError, "invalid response"):
                await generator.generate(answer_request())

    async def test_output_limit_cannot_exceed_reserved_context_budget(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(
                AnsweringConfig(),
                base_url="https://example.invalid/v1",
                client=client,
            )
            with self.assertRaisesRegex(ValueError, "output reserve"):
                await generator.generate(answer_request(output_reserve_tokens=1199))
        self.assertEqual(calls, 0)

    async def test_timeout_is_sanitized(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("PRIVATE_CONTEXT_SENTINEL", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(
                AnsweringConfig(max_retries=0),
                base_url="https://example.invalid/v1",
                client=client,
            )
            with self.assertRaises(AnswerGenerationError) as raised:
                await generator.generate(answer_request())
        self.assertNotIn("PRIVATE_CONTEXT_SENTINEL", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()


class _GatedAnswerStream(httpx.AsyncByteStream):
    def __init__(self, first, tail, gate):
        self.first, self.tail, self.gate = first, tail, gate

    async def __aiter__(self):
        yield self.first
        await self.gate.wait()
        yield self.tail


def _sse(piece="", finish=None, **extra):
    body = {"choices": [{"index": 0, "delta": {"content": piece, "reasoning_content": "PRIVATE"},
                         "finish_reason": finish}], **extra}
    return ("data: " + json.dumps(body, ensure_ascii=False) + "\n\n").encode()


class StreamingAnswerGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_arrives_before_provider_finishes_and_preserves_usage(self):
        from paper_research_agent.answering.streaming import preview_scope
        gate, first_seen = asyncio.Event(), asyncio.Event()
        previews, payloads = [], []
        prefix = '{"status":"answered","claims":[{"text":"正在生成'
        suffix = '正文。","citation_ids":["E1"]}],"insufficient_reason":null}'

        async def preview(text):
            previews.append(text)
            if text:
                first_seen.set()

        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, stream=_GatedAnswerStream(
                _sse(prefix), _sse(suffix, "stop", model="actual-model") +
                _sse(usage={"prompt_tokens": 10, "completion_tokens": 20}) + b"data: [DONE]\n\n", gate))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(AnsweringConfig(max_retries=0), client=client)
            with preview_scope(preview):
                task = asyncio.create_task(generator.generate(answer_request()))
                try:
                    await asyncio.wait_for(first_seen.wait(), 1)
                    self.assertFalse(task.done())
                    self.assertEqual(previews[-1], "正在生成")
                finally:
                    gate.set()
                result = await task
        self.assertEqual(json.loads(result.content)["claims"][0]["text"], "正在生成正文。")
        self.assertEqual((result.input_tokens, result.output_tokens), (10, 20))
        self.assertEqual(result.actual_model, "actual-model")
        self.assertTrue(payloads[0]["stream"])
        self.assertNotIn("PRIVATE", "".join(previews))
        self.assertEqual(previews[-1], "正在生成正文。")

    async def test_truncated_attempt_is_cleared_then_retried(self):
        from paper_research_agent.answering.streaming import preview_scope
        previews, calls = [], []
        valid = json.dumps({"status": "answered", "claims": [{"text": "valid", "citation_ids": ["E1"]}],
                            "insufficient_reason": None})

        async def preview(text):
            previews.append(text)

        def handler(request):
            calls.append(request)
            body = _sse('{"claims":[{"text":"old') if len(calls) == 1 else _sse(valid, "stop") + b"data: [DONE]\n\n"
            return httpx.Response(200, content=body)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(AnsweringConfig(max_retries=1), client=client, sleep=no_sleep)
            with preview_scope(preview):
                result = await generator.generate(answer_request())
        self.assertEqual(result.attempts, 2)
        self.assertIn("old", previews)
        self.assertEqual(previews[previews.index("old") + 1], "")
        self.assertEqual(previews[-1], "valid")

    async def test_non_stop_error_invalid_schema_and_missing_done_fail_closed(self):
        from paper_research_agent.answering.streaming import preview_scope
        cases = [
            _sse('{"claims":[{"text":"draft"}]}', "length") + b"data: [DONE]\n\n",
            b'data: {"error":{"message":"PRIVATE"}}\n\n',
            _sse('{"claims":[{"text":"draft"}]}', "stop") + b"data: [DONE]\n\n",
            _sse('{"claims":[{"text":"draft"}]}', "stop"),
        ]
        for body in cases:
            with self.subTest(body_length=len(body)):
                previews = []
                async def preview(text, previews=previews):
                    previews.append(text)
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request, body=body: httpx.Response(200, content=body))) as client:
                    generator = DashScopeAnswerGenerator(AnsweringConfig(max_retries=0), client=client)
                    with preview_scope(preview), self.assertRaises(AnswerGenerationError) as error:
                        await generator.generate(answer_request())
                self.assertNotIn("PRIVATE", str(error.exception))
                self.assertEqual(previews[-1], "")


    async def test_stream_deadline_and_cancellation_are_not_swallowed(self):
        from paper_research_agent.answering.streaming import answer_preview, preview_scope
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                gate, first_seen = asyncio.Event(), asyncio.Event()
                async def preview(text, first_seen=first_seen):
                    if text:
                        first_seen.set()
                def handler(request, gate=gate):
                    return httpx.Response(200, stream=_GatedAnswerStream(
                        _sse('{"claims":[{"text":"draft'), b"", gate))
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    generator = DashScopeAnswerGenerator(
                        AnsweringConfig(max_retries=0, timeout_seconds=0.05 if not cancel else 5), client=client)
                    with preview_scope(preview):
                        task = asyncio.create_task(generator.generate(answer_request()))
                        await asyncio.wait_for(first_seen.wait(), 1)
                        if cancel:
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await task
                        else:
                            with self.assertRaisesRegex(AnswerGenerationError, "deadline"):
                                await task
                    self.assertIsNone(answer_preview.get())


    async def test_streaming_keeps_citation_validation_and_replaces_bad_draft(self):
        from paper_research_agent.answering.service import answer_context
        from paper_research_agent.answering.streaming import preview_scope
        previews, calls = [], []
        async def preview(text):
            previews.append(text)
        def handler(request):
            calls.append(request)
            citation = "E999" if len(calls) == 1 else "E1"
            content = json.dumps({"status": "answered", "claims": [
                {"text": "bad" if len(calls) == 1 else "valid", "citation_ids": [citation]}],
                "insufficient_reason": None})
            return httpx.Response(200, content=_sse(content, "stop") + b"data: [DONE]\n\n")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = DashScopeAnswerGenerator(AnsweringConfig(max_retries=0), client=client)
            with preview_scope(preview):
                answer = await answer_context(answer_request(), generator)
        self.assertEqual(answer.attempts, 2)
        self.assertEqual(answer.citations[0].citation_id, "E1")
        self.assertNotIn("bad", answer.answer_markdown)
        self.assertEqual(previews, ["", "bad", "", "valid"])
