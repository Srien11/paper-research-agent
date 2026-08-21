from __future__ import annotations

import pytest

from paper_research_agent.evaluation.reranker_diagnostics import (
    RerankerFactDiagnostic,
    summarize_reranker_causes,
    token_coverage,
)


def _fact(fact_id: str, *, delta: int, overtakers: int, gap: float) -> RerankerFactDiagnostic:
    return RerankerFactDiagnostic(
        fact_id=fact_id,
        final_rank=6 + delta,
        pre_rerank_rank=6,
        rank_delta=delta,
        search_occurrences=2,
        reranker_score=0.2,
        top4_score_floor=0.3,
        score_margin_to_top4_floor=-0.1,
        top4_overtaker_count=overtakers,
        same_page_top4=True,
        same_section_top4=False,
        same_page_overtaker_count=overtakers,
        gold_chunk_chars=800,
        top4_median_chunk_chars=600,
        chunk_length_ratio=4 / 3,
        query_fact_token_coverage=0.25,
        gold_query_token_coverage=0.2,
        top4_max_query_token_coverage=0.2 - gap,
        query_overlap_gap=gap,
        broad_query_reproduced=True,
        atomic_query_eligible=True,
        atomic_query_rank=3,
        atomic_rank_improvement=3 + delta,
        atomic_query_reaches_top4=True,
    )


def test_summarizes_reranker_demotion_causes_without_bodies() -> None:
    summary = summarize_reranker_causes(
        (
            _fact("Q1:F1", delta=2, overtakers=2, gap=-0.2),
            _fact("Q2:F1", delta=-1, overtakers=0, gap=0.1),
            _fact("Q3:F1", delta=0, overtakers=0, gap=0),
        )
    )

    assert summary.fact_count == 3
    assert summary.reranker_demoted_count == 1
    assert summary.reranker_promoted_count == 1
    assert summary.reranker_unchanged_count == 1
    assert summary.facts_with_top4_overtakers == 1
    assert summary.total_top4_overtaker_count == 2
    assert summary.same_page_overtaker_count == 2
    assert summary.gold_longer_than_top4_median_count == 3
    assert summary.query_overlap_disadvantage_count == 1
    assert summary.broad_query_reproduction_count == 3
    assert summary.atomic_query_eligible_count == 3
    assert summary.atomic_query_top4_count == 3


def test_rejects_duplicate_fact_ids() -> None:
    fact = _fact("Q1:F1", delta=1, overtakers=1, gap=-0.1)

    with pytest.raises(ValueError, match="unique"):
        summarize_reranker_causes((fact, fact))


def test_bilingual_token_coverage_is_body_free_and_deterministic() -> None:
    assert token_coverage("retrieval metric", "metric for retrieval evaluation") == 1
    assert token_coverage("原子事实", "查询缺少原子事实约束") == 1
    assert token_coverage("", "anything") is None
