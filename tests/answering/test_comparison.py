from __future__ import annotations

import hashlib
import json
import unittest

from paper_research_agent.answering.comparison import (
    answer_comparison,
    compiler_failed_comparison_answer,
)
from paper_research_agent.answering.dashscope import AnswerGenerationError
from paper_research_agent.answering.models import (
    ComparisonAnswerRequest,
    ComparisonDimension,
    ComparisonFact,
    ComparisonTarget,
    GenerationResult,
)
from paper_research_agent.context.models import AssembledContext, CitationRef, PromptMessage


def _context() -> AssembledContext:
    citations = tuple(
        CitationRef(
            citation_id=f"E{index}",
            chunk_id=f"chunk-{index}",
            corpus_id="C001" if index == 1 else "T001",
            asset_id=f"asset-{index}",
            page_start=1,
            page_end=1,
            text_sha256=hashlib.sha256(f"evidence-{index}".encode()).hexdigest(),
            storage_class="internal_research_only",
        )
        for index in (1, 2)
    )
    return AssembledContext(
        messages=(
            PromptMessage(role="system", content="trusted"),
            PromptMessage(role="user", content="compiled ledger"),
        ),
        citations=citations,
        estimated_tokens=100,
        token_budget=2000,
        output_reserve_tokens=1200,
        omitted_evidence_count=0,
        evidence_insufficient=False,
    )


def _request() -> ComparisonAnswerRequest:
    return ComparisonAnswerRequest(
        question="比较 A 与 B 的方法",
        context=_context(),
        targets=(
            ComparisonTarget(target_id="a", label="论文 A", corpus_id="C001"),
            ComparisonTarget(target_id="b", label="论文 B", corpus_id="T001"),
        ),
        dimensions=(ComparisonDimension(dimension_id="method", label="方法"),),
        facts=(
            ComparisonFact(
                fact_id="a-method-f1",
                requirement_id="a-method",
                target_id="a",
                dimension_id="method",
                statement="使用方法 A。",
                citation_ids=("E1",),
            ),
            ComparisonFact(
                fact_id="b-method-f1",
                requirement_id="b-method",
                target_id="b",
                dimension_id="method",
                statement="使用方法 B。",
                citation_ids=("E2",),
            ),
        ),
    )


class FakeGenerator:
    model_id = "qwen-test"
    prompt_version = "comparison-v1"

    def __init__(self, payloads: tuple[dict[str, object], ...]):
        self.payloads = payloads
        self.calls = 0

    async def generate(self, request) -> GenerationResult:
        del request
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return GenerationResult(
            content=json.dumps(payload, ensure_ascii=False),
            requested_model=self.model_id,
            actual_model=self.model_id,
            prompt_version=self.prompt_version,
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            attempts=1,
        )


class FailingGenerator:
    model_id = "qwen-test"
    prompt_version = "comparison-v1"

    def __init__(self):
        self.calls = 0

    async def generate(self, request) -> GenerationResult:
        del request
        self.calls += 1
        raise AnswerGenerationError(
            "answer generation returned an invalid response",
            attempts=2,
            input_tokens=20,
            output_tokens=10,
        )


class ComparisonAnswerTests(unittest.IsolatedAsyncioTestCase):
    def test_compiler_failure_is_not_reported_as_insufficient_evidence(self) -> None:
        generator = FakeGenerator(({},))

        result = compiler_failed_comparison_answer(generator)

        self.assertEqual(result.status, "compiler_failed")
        self.assertIn("不表示论文证据不足", result.answer_markdown)
        self.assertFalse(result.claims)
        self.assertFalse(result.citations)
        self.assertEqual(generator.calls, 0)

    async def test_renders_every_ledger_fact_and_trusted_citation(self) -> None:
        generator = FakeGenerator(
            (
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "draft A",
                            "citation_ids": ["E1"],
                            "fact_ids": ["a-method-f1"],
                        },
                        {
                            "text": "draft B",
                            "citation_ids": ["E2"],
                            "fact_ids": ["b-method-f1"],
                        },
                    ],
                    "insufficient_reason": None,
                },
            )
        )

        result = await answer_comparison(_request(), generator)

        self.assertEqual(generator.calls, 1)
        self.assertIn("使用方法 A。[E1]", result.answer_markdown)
        self.assertIn("使用方法 B。[E2]", result.answer_markdown)
        self.assertNotIn("draft A", result.answer_markdown)
        self.assertEqual(
            {fact_id for claim in result.claims for fact_id in claim.fact_ids},
            {"a-method-f1", "b-method-f1"},
        )

    async def test_repairs_only_the_invalid_dimension_draft_once(self) -> None:
        invalid = {
            "status": "answered",
            "claims": [
                {
                    "text": "incomplete",
                    "citation_ids": ["E1"],
                    "fact_ids": ["a-method-f1"],
                }
            ],
            "insufficient_reason": None,
        }
        valid = {
            "status": "answered",
            "claims": [
                {
                    "text": "complete",
                    "citation_ids": ["E1", "E2"],
                    "fact_ids": ["a-method-f1", "b-method-f1"],
                }
            ],
            "insufficient_reason": None,
        }
        generator = FakeGenerator((invalid, valid))

        result = await answer_comparison(_request(), generator)

        self.assertEqual(generator.calls, 2)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.attempts, 2)

    async def test_provider_failure_repairs_dimension_from_trusted_ledger(self) -> None:
        generator = FailingGenerator()

        result = await answer_comparison(_request(), generator)

        self.assertEqual(generator.calls, 1)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.input_tokens, 20)
        self.assertEqual(
            {fact_id for claim in result.claims for fact_id in claim.fact_ids},
            {"a-method-f1", "b-method-f1"},
        )


if __name__ == "__main__":
    unittest.main()
