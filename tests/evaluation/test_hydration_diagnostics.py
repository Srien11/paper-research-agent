from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_research_agent.evaluation.hydration_diagnostics import (
    FactRankingDiagnostic,
    summarize_hydration_cutoffs,
)


def test_summarizes_true_hydration_losses_without_semantic_alternatives() -> None:
    facts = (
        FactRankingDiagnostic(
            fact_id="F1",
            loss_stage="not_hydrated",
            semantic_alternative=False,
            best_final_rank=6,
            stage_ranks={"vector": 4, "cross_route_rrf": 5, "final": 6},
            search_occurrences=2,
            same_page_top4=True,
        ),
        FactRankingDiagnostic(
            fact_id="F2",
            loss_stage="complete",
            semantic_alternative=True,
            best_final_rank=5,
            stage_ranks={"final": 5},
            search_occurrences=1,
            same_page_top4=False,
        ),
    )

    summary = summarize_hydration_cutoffs(facts, cutoffs=(4, 6, 8, 10))

    assert summary.true_hydration_loss_count == 1
    assert summary.remaining_by_cutoff == {4: 1, 6: 0, 8: 0, 10: 0}
    assert summary.final_rank_median == 6
    assert summary.repeated_search_count == 1
    assert summary.same_page_top4_rate == 1
    assert summary.reranker_demoted_count == 1


def test_summarizes_stage_medians_and_reranker_deltas() -> None:
    facts = (
        FactRankingDiagnostic(
            fact_id="F1",
            loss_stage="not_hydrated",
            best_final_rank=6,
            stage_ranks={"vector": 4, "cross_route_rrf": 5, "final": 6},
            search_occurrences=3,
            same_section_top4=True,
        ),
        FactRankingDiagnostic(
            fact_id="F2",
            loss_stage="not_hydrated",
            best_final_rank=9,
            stage_ranks={"vector": 8, "cross_route_rrf": 10, "final": 9},
            search_occurrences=1,
        ),
        FactRankingDiagnostic(
            fact_id="F3",
            loss_stage="not_hydrated",
            best_final_rank=8,
            stage_ranks={"cross_route_rrf": 8, "final": 8},
            search_occurrences=2,
        ),
    )

    summary = summarize_hydration_cutoffs(facts, cutoffs=(4, 6, 8, 10))

    assert summary.stage_rank_medians == {
        "cross_route_rrf": 8,
        "final": 8,
        "vector": 6,
    }
    assert summary.reranker_delta_median == 0
    assert summary.reranker_promoted_count == 1
    assert summary.reranker_demoted_count == 1
    assert summary.reranker_unchanged_count == 1
    assert summary.same_section_top4_rate == pytest.approx(1 / 3)


def test_not_hydrated_fact_requires_a_ranked_search_occurrence() -> None:
    with pytest.raises(ValidationError):
        FactRankingDiagnostic(
            fact_id="F1",
            loss_stage="not_hydrated",
            best_final_rank=None,
            search_occurrences=1,
        )

    with pytest.raises(ValidationError):
        FactRankingDiagnostic(
            fact_id="F1",
            loss_stage="not_hydrated",
            best_final_rank=5,
            search_occurrences=0,
        )


def test_summary_rejects_duplicate_fact_ids_and_invalid_cutoffs() -> None:
    fact = FactRankingDiagnostic(
        fact_id="F1",
        loss_stage="not_hydrated",
        best_final_rank=5,
        stage_ranks={"final": 5},
        search_occurrences=1,
    )

    with pytest.raises(ValueError, match="fact IDs must be unique"):
        summarize_hydration_cutoffs((fact, fact), cutoffs=(4, 6))

    with pytest.raises(ValueError, match="cutoffs"):
        summarize_hydration_cutoffs((fact,), cutoffs=(4, 4))
