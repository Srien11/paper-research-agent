"""Privacy-safe diagnostics for reranker demotion and local chunk competition."""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable

from pydantic import Field

from paper_research_agent.evaluation.comparison_end_to_end import FrozenEvaluationModel


class RerankerFactDiagnostic(FrozenEvaluationModel):
    """Body-free reranking signals for one required fact on its best search path."""

    fact_id: str = Field(min_length=1)
    final_rank: int = Field(ge=1)
    pre_rerank_rank: int = Field(ge=1)
    rank_delta: int
    search_occurrences: int = Field(ge=1)
    reranker_score: float
    top4_score_floor: float
    score_margin_to_top4_floor: float
    top4_overtaker_count: int = Field(ge=0, le=4)
    same_page_top4: bool = False
    same_section_top4: bool = False
    same_page_overtaker_count: int = Field(default=0, ge=0, le=4)
    same_section_overtaker_count: int = Field(default=0, ge=0, le=4)
    gold_chunk_chars: int = Field(ge=0)
    top4_median_chunk_chars: float = Field(ge=0)
    chunk_length_ratio: float | None = Field(default=None, ge=0)
    query_fact_token_coverage: float | None = Field(default=None, ge=0, le=1)
    gold_query_token_coverage: float | None = Field(default=None, ge=0, le=1)
    top4_max_query_token_coverage: float | None = Field(default=None, ge=0, le=1)
    query_overlap_gap: float | None = Field(default=None, ge=-1, le=1)
    broad_query_reproduced: bool | None = None
    atomic_query_eligible: bool = False
    atomic_query_rank: int | None = Field(default=None, ge=1)
    atomic_rank_improvement: int | None = None
    atomic_query_reaches_top4: bool | None = None


class RerankerCauseSummary(FrozenEvaluationModel):
    fact_count: int = Field(ge=0)
    reranker_promoted_count: int = Field(ge=0)
    reranker_demoted_count: int = Field(ge=0)
    reranker_unchanged_count: int = Field(ge=0)
    median_rank_delta: float | None = None
    median_score_margin_to_top4_floor: float | None = None
    facts_with_top4_overtakers: int = Field(ge=0)
    total_top4_overtaker_count: int = Field(ge=0)
    same_page_top4_count: int = Field(ge=0)
    same_section_top4_count: int = Field(ge=0)
    same_page_overtaker_count: int = Field(ge=0)
    same_section_overtaker_count: int = Field(ge=0)
    gold_longer_than_top4_median_count: int = Field(ge=0)
    median_chunk_length_ratio: float | None = Field(default=None, ge=0)
    query_overlap_disadvantage_count: int = Field(ge=0)
    median_query_fact_token_coverage: float | None = Field(default=None, ge=0, le=1)
    median_query_overlap_gap: float | None = Field(default=None, ge=-1, le=1)
    broad_query_reproduction_count: int = Field(ge=0)
    atomic_query_eligible_count: int = Field(ge=0)
    atomic_query_top4_count: int = Field(ge=0)
    median_atomic_rank_improvement: float | None = None


def summarize_reranker_causes(
    diagnostics: Iterable[RerankerFactDiagnostic],
) -> RerankerCauseSummary:
    facts = tuple(diagnostics)
    fact_ids = tuple(item.fact_id for item in facts)
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("reranker diagnostic fact IDs must be unique")

    def median(values: Iterable[float]) -> float | None:
        materialized = tuple(values)
        return float(statistics.median(materialized)) if materialized else None

    return RerankerCauseSummary(
        fact_count=len(facts),
        reranker_promoted_count=sum(item.rank_delta < 0 for item in facts),
        reranker_demoted_count=sum(item.rank_delta > 0 for item in facts),
        reranker_unchanged_count=sum(item.rank_delta == 0 for item in facts),
        median_rank_delta=median(item.rank_delta for item in facts),
        median_score_margin_to_top4_floor=median(
            item.score_margin_to_top4_floor for item in facts
        ),
        facts_with_top4_overtakers=sum(item.top4_overtaker_count > 0 for item in facts),
        total_top4_overtaker_count=sum(item.top4_overtaker_count for item in facts),
        same_page_top4_count=sum(item.same_page_top4 for item in facts),
        same_section_top4_count=sum(item.same_section_top4 for item in facts),
        same_page_overtaker_count=sum(item.same_page_overtaker_count for item in facts),
        same_section_overtaker_count=sum(
            item.same_section_overtaker_count for item in facts
        ),
        gold_longer_than_top4_median_count=sum(
            item.gold_chunk_chars > item.top4_median_chunk_chars for item in facts
        ),
        median_chunk_length_ratio=median(
            item.chunk_length_ratio
            for item in facts
            if item.chunk_length_ratio is not None
        ),
        query_overlap_disadvantage_count=sum(
            item.query_overlap_gap is not None and item.query_overlap_gap < 0
            for item in facts
        ),
        median_query_fact_token_coverage=median(
            item.query_fact_token_coverage
            for item in facts
            if item.query_fact_token_coverage is not None
        ),
        median_query_overlap_gap=median(
            item.query_overlap_gap
            for item in facts
            if item.query_overlap_gap is not None
        ),
        broad_query_reproduction_count=sum(
            item.broad_query_reproduced is True for item in facts
        ),
        atomic_query_eligible_count=sum(item.atomic_query_eligible for item in facts),
        atomic_query_top4_count=sum(
            item.atomic_query_reaches_top4 is True for item in facts
        ),
        median_atomic_rank_improvement=median(
            item.atomic_rank_improvement
            for item in facts
            if item.atomic_rank_improvement is not None
        ),
    )


_ASCII_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.+/-]*", flags=re.IGNORECASE)
_CJK_SEQUENCE = re.compile(r"[\u3400-\u9fff]+")


def lexical_tokens(text: str | None) -> frozenset[str]:
    """Return coarse bilingual tokens for aggregate overlap diagnostics only."""
    if not text:
        return frozenset()
    normalized = text.casefold()
    tokens = {match.group(0) for match in _ASCII_TOKEN.finditer(normalized)}
    for match in _CJK_SEQUENCE.finditer(normalized):
        sequence = match.group(0)
        if len(sequence) == 1:
            tokens.add(sequence)
        else:
            tokens.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return frozenset(tokens)


def token_coverage(required_text: str | None, available_text: str | None) -> float | None:
    """Measure how much of the first text's token intent appears in the second."""
    required = lexical_tokens(required_text)
    if not required:
        return None
    return len(required & lexical_tokens(available_text)) / len(required)
