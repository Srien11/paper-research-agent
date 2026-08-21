from __future__ import annotations

import pytest

from paper_research_agent.evaluation.comparison_end_to_end import (
    ComparisonEndToEndGold,
    ComparisonGoldCitationRelation,
    ComparisonGoldClaim,
)
from paper_research_agent.evaluation.gold_scope import (
    ComparisonQuestionScope,
    OptionalClaimScope,
    RequiredClaimScope,
    apply_comparison_gold_scope,
)


def _gold() -> ComparisonEndToEndGold:
    return ComparisonEndToEndGold(
        question_id="CPG001",
        split="dev",
        relevant_paper_ids=("C001", "C002"),
        expected_dimensions=("method", "unasked metric"),
        must_have_claims=(
            ComparisonGoldClaim(claim_id="method", corpus_id="C001", normalized_fact="fact"),
            ComparisonGoldClaim(claim_id="metric", corpus_id="C002", normalized_fact="detail"),
        ),
        evidence_chunk_ids=("chunk-method", "chunk-metric"),
        citation_relations=(
            ComparisonGoldCitationRelation(claim_id="method", chunk_ids=("chunk-method",)),
            ComparisonGoldCitationRelation(claim_id="metric", chunk_ids=("chunk-metric",)),
        ),
    )


def test_scope_keeps_only_explicitly_requested_primary_claims() -> None:
    result = apply_comparison_gold_scope(
        question="比较两种方法的核心机制",
        gold=_gold(),
        scope=ComparisonQuestionScope(
            question_id="CPG001",
            expected_dimensions=("核心机制",),
            required_claims=(
                RequiredClaimScope(claim_id="method", request_spans=("核心机制",)),
            ),
            optional_claims=(
                OptionalClaimScope(claim_id="metric", reason="not_explicit"),
            ),
        ),
    )

    assert [item.claim_id for item in result.must_have_claims] == ["method"]
    assert result.expected_dimensions == ("核心机制",)
    assert result.evidence_chunk_ids == ("chunk-method",)
    assert [item.claim_id for item in result.citation_relations] == ["method"]


def test_scope_rejects_unaccounted_source_claims() -> None:
    with pytest.raises(ValueError, match="account for every source claim"):
        apply_comparison_gold_scope(
            question="比较核心机制",
            gold=_gold(),
            scope=ComparisonQuestionScope(
                question_id="CPG001",
                expected_dimensions=("核心机制",),
                required_claims=(
                    RequiredClaimScope(claim_id="method", request_spans=("核心机制",)),
                ),
            ),
        )


def test_scope_rejects_required_claim_without_verbatim_request_span() -> None:
    with pytest.raises(ValueError, match="request span"):
        apply_comparison_gold_scope(
            question="比较核心机制",
            gold=_gold(),
            scope=ComparisonQuestionScope(
                question_id="CPG001",
                expected_dimensions=("核心机制",),
                required_claims=(
                    RequiredClaimScope(claim_id="method", request_spans=("评估指标",)),
                ),
                optional_claims=(
                    OptionalClaimScope(claim_id="metric", reason="not_explicit"),
                ),
            ),
        )
