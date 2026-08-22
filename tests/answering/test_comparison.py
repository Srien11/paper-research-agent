from __future__ import annotations

import hashlib
import unittest

from paper_research_agent.answering.comparison import (
    answer_comparison,
    compiler_failed_comparison_answer,
)
from paper_research_agent.answering.models import (
    ComparisonAnswerRequest,
    ComparisonDimension,
    ComparisonFact,
    ComparisonTarget,
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


class ForbiddenGenerator:
    model_id = "qwen-test"
    prompt_version = "comparison-v1"

    def __init__(self):
        self.calls = 0

    async def generate(self, request):
        del request
        self.calls += 1
        raise AssertionError("comparison renderer must not call Provider")


def _multi_dimension_request() -> ComparisonAnswerRequest:
    return ComparisonAnswerRequest(
        question="比较 A 与 B",
        context=_context(),
        targets=_request().targets,
        dimensions=(
            ComparisonDimension(dimension_id="method", label="方法"),
            ComparisonDimension(dimension_id="dataset", label="数据集"),
        ),
        facts=(
            ComparisonFact(
                fact_id="b-dataset-f1",
                requirement_id="b-dataset",
                target_id="b",
                dimension_id="dataset",
                statement="B 数据集事实。",
                citation_ids=("E2",),
            ),
            ComparisonFact(
                fact_id="a-method-f2",
                requirement_id="a-method",
                target_id="a",
                dimension_id="method",
                statement="A 方法事实二。",
                citation_ids=("E1",),
            ),
            ComparisonFact(
                fact_id="a-method-f1",
                requirement_id="a-method",
                target_id="a",
                dimension_id="method",
                statement="A 方法事实一。",
                citation_ids=("E1",),
            ),
            ComparisonFact(
                fact_id="b-method-f1",
                requirement_id="b-method",
                target_id="b",
                dimension_id="method",
                statement="B 方法事实。",
                citation_ids=("E2",),
            ),
            ComparisonFact(
                fact_id="a-dataset-f1",
                requirement_id="a-dataset",
                target_id="a",
                dimension_id="dataset",
                statement="A 数据集事实。",
                citation_ids=("E1",),
            ),
        ),
    )


class ComparisonAnswerTests(unittest.IsolatedAsyncioTestCase):
    def test_compiler_failure_is_not_reported_as_insufficient_evidence(self) -> None:
        generator = ForbiddenGenerator()

        result = compiler_failed_comparison_answer(generator)

        self.assertEqual(result.status, "compiler_failed")
        self.assertIn("不表示论文证据不足", result.answer_markdown)
        self.assertFalse(result.claims)
        self.assertFalse(result.citations)
        self.assertEqual(generator.calls, 0)

    async def test_renders_every_ledger_fact_and_trusted_citation(self) -> None:
        generator = ForbiddenGenerator()

        result = await answer_comparison(_request(), generator)

        self.assertEqual(generator.calls, 0)
        self.assertIn("使用方法 A。[E1]", result.answer_markdown)
        self.assertIn("使用方法 B。[E2]", result.answer_markdown)
        self.assertEqual(
            [claim.text for claim in result.claims],
            ["使用方法 A。", "使用方法 B。"],
        )
        self.assertEqual(
            {fact_id for claim in result.claims for fact_id in claim.fact_ids},
            {"a-method-f1", "b-method-f1"},
        )
        self.assertIsNone(result.actual_model)
        self.assertEqual(result.prompt_version, "comparison-ledger-render-v2")
        self.assertEqual(result.input_tokens, 0)
        self.assertEqual(result.output_tokens, 0)
        self.assertEqual(result.latency_ms, 0)
        self.assertEqual(result.attempts, 0)

    async def test_rendering_is_stable_and_orders_dimensions_targets_then_facts(self) -> None:
        generator = ForbiddenGenerator()
        results = [
            await answer_comparison(_multi_dimension_request(), generator)
            for _ in range(20)
        ]

        hashes = {
            hashlib.sha256(item.answer_markdown.encode("utf-8")).hexdigest()
            for item in results
        }
        self.assertEqual(len(hashes), 1)
        self.assertEqual(generator.calls, 0)
        self.assertEqual(
            [claim.fact_ids[0] for claim in results[0].claims],
            [
                "a-method-f2",
                "a-method-f1",
                "b-method-f1",
                "a-dataset-f1",
                "b-dataset-f1",
            ],
        )
        self.assertTrue(all(len(claim.fact_ids) == 1 for claim in results[0].claims))
        self.assertEqual(
            [claim.citation_ids for claim in results[0].claims],
            [("E1",), ("E1",), ("E2",), ("E1",), ("E2",)],
        )

    async def test_empty_dimension_cells_are_explicit_without_fabricated_claims(self) -> None:
        request = _request().model_copy(
            update={
                "dimensions": (
                    *_request().dimensions,
                    ComparisonDimension(dimension_id="dataset", label="数据集"),
                )
            }
        )
        result = await answer_comparison(request, ForbiddenGenerator())

        self.assertEqual(result.answer_markdown.count("暂无可靠事实。"), 2)
        self.assertEqual(len(result.claims), 2)


if __name__ == "__main__":
    unittest.main()
