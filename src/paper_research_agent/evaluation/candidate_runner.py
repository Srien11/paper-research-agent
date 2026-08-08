"""Aggregate candidate-paper retrieval metrics without exposing sealed questions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from paper_research_agent.evaluation.candidate_gold import (
    CandidateClueScope,
    CandidateGoldSplit,
)


class CandidatePaperEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str = Field(pattern=r"^CPG\d{3,}$")
    split: CandidateGoldSplit
    clue_scope: CandidateClueScope
    relevant_paper_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    candidate_paper_ids: tuple[str, ...]
    rewrite_status: Literal["success", "cache_hit", "stale_cache", "fallback_original"]
    rewrite_latency_ms: float = Field(default=0.0, ge=0)

    @field_validator("relevant_paper_ids", "candidate_paper_ids")
    @classmethod
    def require_unique_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("candidate paper IDs must be unique")
        return values


def summarize_candidate_paper_cases(
    cases: Iterable[CandidatePaperEvaluationCase],
    *,
    cutoffs: Sequence[int] = (5, 8),
) -> dict[str, object]:
    materialized = tuple(cases)
    normalized_cutoffs = tuple(dict.fromkeys(cutoffs))
    if not normalized_cutoffs or any(cutoff <= 0 for cutoff in normalized_cutoffs):
        raise ValueError("candidate metric cutoffs must be positive")
    primary = tuple(case for case in materialized if case.clue_scope == "title_abstract")
    diagnostic = tuple(case for case in materialized if case.clue_scope == "full_text_detail")
    latencies = sorted(case.rewrite_latency_ms for case in materialized)
    return {
        "schema_version": "candidate-paper-evaluation-summary-v1",
        "question_count": len(materialized),
        "primary_question_count": len(primary),
        "diagnostic_full_text_question_count": len(diagnostic),
        "rewrite_fallback_count": sum(
            case.rewrite_status == "fallback_original" for case in materialized
        ),
        "rewrite_latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": latencies[-1] if latencies else None,
        },
        "primary": _metrics(primary, normalized_cutoffs),
        "rewrite_success": _metrics(
            tuple(case for case in primary if case.rewrite_status != "fallback_original"),
            normalized_cutoffs,
        ),
        "rewrite_fallback": _metrics(
            tuple(case for case in primary if case.rewrite_status == "fallback_original"),
            normalized_cutoffs,
        ),
        "dev": _metrics(
            tuple(case for case in primary if case.split == "dev"),
            normalized_cutoffs,
        ),
        "sealed_test": _metrics(
            tuple(case for case in primary if case.split == "sealed_test"),
            normalized_cutoffs,
        ),
        "full_text_diagnostic": _metrics(diagnostic, normalized_cutoffs),
    }


def _metrics(
    cases: Sequence[CandidatePaperEvaluationCase],
    cutoffs: Sequence[int],
) -> dict[str, int | float | None]:
    metrics: dict[str, int | float | None] = {"question_count": len(cases)}
    for cutoff in cutoffs:
        recalls = []
        all_target_hits = []
        for case in cases:
            selected = set(case.candidate_paper_ids[:cutoff])
            relevant = set(case.relevant_paper_ids)
            recalls.append(len(selected & relevant) / len(relevant))
            all_target_hits.append(relevant <= selected)
        metrics[f"recall_at_{cutoff}_macro"] = (
            sum(recalls) / len(recalls) if recalls else None
        )
        metrics[f"all_target_hit_at_{cutoff}"] = (
            sum(all_target_hits) / len(all_target_hits) if all_target_hits else None
        )
    return metrics


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return values[max(0, math.ceil(len(values) * quantile) - 1)]
