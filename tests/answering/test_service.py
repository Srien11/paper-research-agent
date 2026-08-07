from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.answering.models import AnswerRequest, GenerationResult
from paper_research_agent.answering.service import answer_context
from paper_research_agent.context.models import AssembledContext, CitationRef, PromptMessage


def request(*, insufficient: bool = False) -> AnswerRequest:
    citations = ()
    if not insufficient:
        citations = (
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
        )
    return AnswerRequest(
        context=AssembledContext(
            messages=(
                PromptMessage(role="system", content="trusted"),
                PromptMessage(role="user", content="question with evidence"),
            ),
            citations=citations,
            estimated_tokens=100,
            token_budget=2000,
            output_reserve_tokens=1200,
            omitted_evidence_count=0,
            evidence_insufficient=insufficient,
        )
    )


class FakeGenerator:
    model_id = "qwen3.7-plus-2026-05-26"
    prompt_version = "rag-answer-json-v1"

    def __init__(self, content: str | tuple[str, ...]):
        self.contents = (content,) if isinstance(content, str) else content
        self.calls = 0

    async def generate(self, answer_request: AnswerRequest) -> GenerationResult:
        del answer_request
        self.calls += 1
        content = self.contents[min(self.calls - 1, len(self.contents) - 1)]
        return GenerationResult(
            content=content,
            requested_model=self.model_id,
            actual_model=self.model_id,
            prompt_version=self.prompt_version,
            input_tokens=90,
            output_tokens=20,
            latency_ms=5,
            attempts=1,
        )


class FakeAudit:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def log(self, result) -> bool:
        del result
        self.calls += 1
        if self.fail:
            raise OSError("audit unavailable")
        return True


class AnswerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_generation_is_parsed_validated_and_audited(self) -> None:
        generator = FakeGenerator(
            json.dumps(
                {
                    "status": "answered",
                    "claims": [{"text": "得到一个结论。", "citation_ids": ["E1"]}],
                    "insufficient_reason": None,
                },
                ensure_ascii=False,
            )
        )
        audit = FakeAudit()
        result = await answer_context(request(), generator, audit=audit)
        self.assertEqual(result.answer_markdown, "得到一个结论。[E1]")
        self.assertEqual(generator.calls, 1)
        self.assertEqual(audit.calls, 1)
        self.assertTrue(result.audit_persisted)

    async def test_empty_context_returns_deterministic_answer_without_provider(self) -> None:
        generator = FakeGenerator("must not be used")
        result = await answer_context(request(insufficient=True), generator)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(generator.calls, 0)
        self.assertEqual(result.attempts, 0)
        self.assertIsNone(result.actual_model)

    async def test_audit_failure_does_not_discard_a_valid_answer(self) -> None:
        generator = FakeGenerator(
            '{"status":"answered","claims":[{"text":"结论。","citation_ids":["E1"]}],"insufficient_reason":null}'
        )
        result = await answer_context(request(), generator, audit=FakeAudit(fail=True))
        self.assertEqual(result.status, "answered")
        self.assertFalse(result.audit_persisted)

    async def test_retries_unknown_citation_and_accumulates_usage(self) -> None:
        generator = FakeGenerator(
            (
                '{"status":"answered","claims":[{"text":"结论。","citation_ids":["E2"]}],"insufficient_reason":null}',
                '{"status":"answered","claims":[{"text":"结论。","citation_ids":["E1"]}],"insufficient_reason":null}',
            )
        )

        result = await answer_context(request(), generator)

        self.assertEqual(generator.calls, 2)
        self.assertEqual(result.input_tokens, 180)
        self.assertEqual(result.output_tokens, 40)
        self.assertEqual(result.attempts, 2)


if __name__ == "__main__":
    unittest.main()
