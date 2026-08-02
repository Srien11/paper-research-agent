from __future__ import annotations

import sys
import tempfile
import unittest
import urllib.error
from email.message import Message
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.figures.dashscope import (
    FREE_QUOTA_ERROR_CODE,
    DashScopeRequestError,
    DashScopeVisionSummarizer,
    NoAvailableVisionModelError,
    _http_error,
)


def response(
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    model: str | None = None,
):
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"figure_type":"曲线图","summary":"性能随规模提升",'
                        '"key_findings":["A 高于 B"],"recognition_confidence":0.9}'
                    )
                }
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
        },
    }
    if model is not None:
        payload["model"] = model
    return payload


class DashScopeVisionSummarizerTests(unittest.TestCase):
    def test_sends_local_image_as_data_url_and_tracks_usage(self) -> None:
        requests = []

        def requester(model_id, payload):
            requests.append((model_id, payload))
            return response()

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            image_path.write_bytes(b"PNG fixture")
            summarizer = DashScopeVisionSummarizer(
                model_ids=("qwen3.7-plus",),
                requester=requester,
                sleep=lambda _: None,
                jitter=lambda _left, _right: 0,
            )
            result = summarizer.summarize(
                image_path,
                figure_name="Figure 1",
                caption="Results.",
            )

        self.assertEqual(result.summary.figure_type, "曲线图")
        self.assertEqual(result.model_id, "qwen3.7-plus")
        self.assertEqual(requests[0][0], "qwen3.7-plus")
        content = requests[0][1]["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertFalse(requests[0][1]["enable_thinking"])
        self.assertEqual(
            summarizer.usage_report()["qwen3.7-plus"]["total_tokens"],
            120,
        )

    def test_switches_model_only_after_explicit_free_quota_error(self) -> None:
        called = []

        def requester(model_id, _payload):
            called.append(model_id)
            if model_id == "qwen3.7-plus":
                raise DashScopeRequestError(
                    "quota exhausted",
                    status_code=403,
                    error_code=FREE_QUOTA_ERROR_CODE,
                )
            return response(
                input_tokens=80,
                output_tokens=10,
                model="qwen3.7-plus-2026-05-26",
            )

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            image_path.write_bytes(b"PNG fixture")
            summarizer = DashScopeVisionSummarizer(
                model_ids=("qwen3.7-plus", "qwen3.7-plus-2026-05-26"),
                requester=requester,
                sleep=lambda _: None,
                jitter=lambda _left, _right: 0,
            )
            result = summarizer.summarize(
                image_path,
                figure_name="Figure 1",
                caption="Results.",
            )

        self.assertEqual(called, ["qwen3.7-plus", "qwen3.7-plus-2026-05-26"])
        self.assertEqual(result.model_id, "qwen3.7-plus-2026-05-26")

    def test_stops_when_all_free_quotas_are_exhausted(self) -> None:
        def requester(_model_id, _payload):
            raise DashScopeRequestError(
                "quota exhausted",
                status_code=403,
                error_code=FREE_QUOTA_ERROR_CODE,
            )

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            image_path.write_bytes(b"PNG fixture")
            summarizer = DashScopeVisionSummarizer(
                model_ids=("first", "second"),
                requester=requester,
                sleep=lambda _: None,
                jitter=lambda _left, _right: 0,
            )
            with self.assertRaises(NoAvailableVisionModelError):
                summarizer.summarize(image_path, figure_name="Figure 1", caption="Results.")

    def test_retries_invalid_content_without_switching_models(self) -> None:
        calls = 0

        def requester(_model_id, _payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"choices": [{"message": {"content": "not json"}}]}
            return response()

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "figure.png"
            image_path.write_bytes(b"PNG fixture")
            summarizer = DashScopeVisionSummarizer(
                model_ids=("qwen3.7-plus",),
                requester=requester,
                max_retries=1,
                sleep=lambda _: None,
                jitter=lambda _left, _right: 0,
            )
            summarizer.summarize(image_path, figure_name="Figure 1", caption="Results.")

        self.assertEqual(calls, 2)
        self.assertEqual(
            summarizer.usage_report()["qwen3.7-plus"]["total_tokens"],
            120,
        )

    def test_parses_nested_quota_error_and_retry_after(self) -> None:
        headers = Message()
        headers["Retry-After"] = "30"
        error = urllib.error.HTTPError(
            "https://example.invalid",
            429,
            "rate limited",
            headers,
            BytesIO(
                b'{"error":{"code":"AllocationQuota.FreeTierOnly",'
                b'"message":"quota exhausted"}}'
            ),
        )

        try:
            parsed = _http_error(error)
        finally:
            error.close()

        self.assertEqual(parsed.error_code, FREE_QUOTA_ERROR_CODE)
        self.assertEqual(parsed.retry_after_seconds, 30)
        self.assertTrue(parsed.retryable)


if __name__ == "__main__":
    unittest.main()
