"""Body-free ranking diagnostics for evidence hydration experiments."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence

from pydantic import Field, model_validator

from paper_research_agent.evaluation.comparison_end_to_end import (
    FactLossStage,
    FrozenEvaluationModel,
)


class FactRankingDiagnostic(FrozenEvaluationModel):
    """Safe rank lineage for one required fact across repeated searches."""

    fact_id: str = Field(min_length=1)
    loss_stage: FactLossStage
    semantic_alternative: bool = False
    best_final_rank: int | None = Field(default=None, ge=1)
    stage_ranks: dict[str, int] = Field(default_factory=dict)
    search_occurrences: int = Field(default=0, ge=0)
    same_page_top4: bool = False
    same_section_top4: bool = False

    @model_validator(mode="after")
    def validate_rank_lineage(self) -> FactRankingDiagnostic:
        if any(not stage.strip() or rank <= 0 for stage, rank in self.stage_ranks.items()):
            raise ValueError("stage names must be non-empty and ranks must be positive")
        if self.best_final_rank is not None and self.stage_ranks.get(
            "final", self.best_final_rank
        ) != self.best_final_rank:
            raise ValueError("best final rank must match the selected final stage rank")
        if self.loss_stage == "not_hydrated" and (
            self.best_final_rank is None or self.search_occurrences == 0
        ):
            raise ValueError("not_hydrated facts require a ranked search occurrence")
        if self.search_occurrences == 0 and self.best_final_rank is not None:
            raise ValueError("ranked facts require at least one search occurrence")
        return self


class HydrationCutoffSummary(FrozenEvaluationModel):
    """Aggregate projection for a fixed set of body-free fact diagnostics."""

    true_hydration_loss_count: int = Field(ge=0)
    remaining_by_cutoff: dict[int, int]
    final_rank_median: float | None = Field(default=None, ge=1)
    stage_rank_medians: dict[str, float] = Field(default_factory=dict)
    repeated_search_count: int = Field(default=0, ge=0)
    same_page_top4_rate: float | None = Field(default=None, ge=0, le=1)
    same_section_top4_rate: float | None = Field(default=None, ge=0, le=1)
    reranker_promoted_count: int = Field(default=0, ge=0)
    reranker_demoted_count: int = Field(default=0, ge=0)
    reranker_unchanged_count: int = Field(default=0, ge=0)
    reranker_delta_median: float | None = None


def summarize_hydration_cutoffs(
    facts: Iterable[FactRankingDiagnostic],
    *,
    cutoffs: Sequence[int] = (4, 6, 8, 10),
) -> HydrationCutoffSummary:
    """Project true hydration losses at each cutoff without exposing fact text."""
    materialized = tuple(facts)
    fact_ids = tuple(item.fact_id for item in materialized)
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("fact IDs must be unique")
    normalized_cutoffs = tuple(cutoffs)
    if (
        not normalized_cutoffs
        or any(cutoff <= 0 for cutoff in normalized_cutoffs)
        or tuple(sorted(set(normalized_cutoffs))) != normalized_cutoffs
    ):
        raise ValueError("cutoffs must be positive, unique, and increasing")

    true_losses = tuple(
        item
        for item in materialized
        if item.loss_stage == "not_hydrated" and not item.semantic_alternative
    )
    final_ranks = tuple(
        item.best_final_rank
        for item in true_losses
        if item.best_final_rank is not None
    )
    if len(final_ranks) != len(true_losses):
        raise ValueError("true hydration losses require final ranks")

    stages = sorted({stage for item in true_losses for stage in item.stage_ranks})
    stage_rank_medians = {
        stage: float(
            statistics.median(
                item.stage_ranks[stage]
                for item in true_losses
                if stage in item.stage_ranks
            )
        )
        for stage in stages
    }
    reranker_deltas = tuple(
        item.stage_ranks["final"] - item.stage_ranks["cross_route_rrf"]
        for item in true_losses
        if "final" in item.stage_ranks and "cross_route_rrf" in item.stage_ranks
    )
    count = len(true_losses)
    return HydrationCutoffSummary(
        true_hydration_loss_count=count,
        remaining_by_cutoff={
            cutoff: sum(rank > cutoff for rank in final_ranks)
            for cutoff in normalized_cutoffs
        },
        final_rank_median=(float(statistics.median(final_ranks)) if final_ranks else None),
        stage_rank_medians=stage_rank_medians,
        repeated_search_count=sum(item.search_occurrences > 1 for item in true_losses),
        same_page_top4_rate=(
            sum(item.same_page_top4 for item in true_losses) / count if count else None
        ),
        same_section_top4_rate=(
            sum(item.same_section_top4 for item in true_losses) / count if count else None
        ),
        reranker_promoted_count=sum(delta < 0 for delta in reranker_deltas),
        reranker_demoted_count=sum(delta > 0 for delta in reranker_deltas),
        reranker_unchanged_count=sum(delta == 0 for delta in reranker_deltas),
        reranker_delta_median=(
            float(statistics.median(reranker_deltas)) if reranker_deltas else None
        ),
    )
