from __future__ import annotations

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
