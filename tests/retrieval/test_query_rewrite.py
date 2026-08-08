from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.retrieval.query_rewrite import (
    SYSTEM_PROMPT,
    DashScopeQueryRewriter,
    QueryRewriteError,
    _parse_rewrite_response,
)


class QueryRewriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_contains_only_fixed_prompt_and_query_without_evidence(self) -> None:
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
                            "message": {
                                "content": json.dumps(
                                    {
                                        "english_query": (
                                            "Llama-3.1 TruthfulQA accuracy not below 80%"
                                        )
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 31, "completion_tokens": 9},
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rewriter = DashScopeQueryRewriter(
                "qwen3.7-plus",
                base_url="https://example.invalid/v1",
                client=client,
            )
            result = await rewriter.rewrite(
                "  Llama-3.1 在 TruthfulQA 上的 accuracy 是否不低于 80%？  "
            )

        self.assertEqual(len(requests), 1)
        payload = json.loads(requests[0].content)
        self.assertEqual(
            set(payload),
            {
                "model",
                "messages",
                "response_format",
                "temperature",
                "top_p",
                "enable_thinking",
                "max_tokens",
            },
        )
        self.assertEqual(
            payload["messages"],
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Llama-3.1 在 TruthfulQA 上的 accuracy 是否不低于 80%？",
                },
            ],
        )
        self.assertNotIn("evidence", payload)
        self.assertNotIn("chunks", payload)
        self.assertNotIn("context", payload)
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["top_p"], 0.7)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertFalse(payload["enable_thinking"])
        for required_constraint in (
            "model names",
            "dataset names",
            "abbreviations",
            "metrics",
            "numbers",
            "negations",
            "scope restrictions",
            "comparison direction",
            "Do not summarize",
            "Do not add",
            "Do not omit",
            "relationships",
            "question intent",
        ):
            self.assertIn(required_constraint, payload["messages"][0]["content"])
        self.assertEqual(
            result.english_query,
            "Llama-3.1 TruthfulQA accuracy not below 80%",
        )
        self.assertEqual(result.actual_model, "qwen3.7-plus-2026-05-26")
        self.assertEqual((result.input_tokens, result.output_tokens), (31, 9))

    async def test_provider_error_is_structured_and_does_not_echo_response_message(self) -> None:
        secret = "INTERNAL_EVIDENCE_MUST_NOT_BE_LOGGED"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                request=request,
                json={"error": {"code": "RateLimitExceeded", "message": secret}},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            rewriter = DashScopeQueryRewriter(
                "qwen3.7-plus",
                base_url="https://example.invalid/v1",
                client=client,
            )
            with self.assertRaises(QueryRewriteError) as raised:
                await rewriter.rewrite("测试查询")

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.error_code, "RateLimitExceeded")
        self.assertNotIn(secret, str(raised.exception))

    def test_response_parser_accepts_exact_schema_and_falls_back_to_requested_model(self) -> None:
        result = _parse_rewrite_response(
            {
                "choices": [
                    {"message": {"content": '{"english_query":"  robust RAG evaluation  "}'}}
                ],
                "usage": {"input_tokens": 12, "output_tokens": 4},
            },
            requested_model="qwen3.7-plus",
        )

        self.assertEqual(result.english_query, "robust RAG evaluation")
        self.assertEqual(result.actual_model, "qwen3.7-plus")
        self.assertEqual((result.input_tokens, result.output_tokens), (12, 4))

    def test_response_parser_rejects_malformed_or_overlong_content(self) -> None:
        invalid_responses = (
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {"content": "not-json"}}]},
            {"choices": [{"message": {"content": '{"english_query":"valid","unexpected":true}'}}]},
            {"choices": [{"message": {"content": '{"english_query":"   "}'}}]},
            {"choices": [{"message": {"content": json.dumps({"english_query": "x" * 4001})}}]},
        )
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(QueryRewriteError):
                _parse_rewrite_response(response, requested_model="qwen3.7-plus")


if __name__ == "__main__":
    unittest.main()
