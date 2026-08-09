"""Privacy-safe contracts and deterministic checks for comparison E2E evaluation."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonGoldClaim(FrozenEvaluationModel):
    claim_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    normalized_fact: str = Field(min_length=1)


class ComparisonGoldCitationRelation(FrozenEvaluationModel):
    claim_id: str = Field(min_length=1)
    chunk_ids: tuple[str, ...] = Field(default=(), min_length=1)
    span_hashes: tuple[str, ...] = ()


class ComparisonEndToEndGold(FrozenEvaluationModel):
    schema_version: Literal["comparison-end-to-end-gold-v1"] = (
        "comparison-end-to-end-gold-v1"
    )
    question_id: str = Field(pattern=r"^CPG\d{3}$")
    split: Literal["dev", "sealed_test"]
    relevant_paper_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    expected_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    must_have_claims: tuple[ComparisonGoldClaim, ...] = Field(min_length=1)
    evidence_chunk_ids: tuple[str, ...] = ()
    evidence_span_hashes: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    citation_relations: tuple[ComparisonGoldCitationRelation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_relations(self) -> ComparisonEndToEndGold:
        claim_ids = {claim.claim_id for claim in self.must_have_claims}
        relation_ids = {relation.claim_id for relation in self.citation_relations}
        if claim_ids != relation_ids:
            raise ValueError("every must-have claim requires exactly one citation relation")
        if len(relation_ids) != len(self.citation_relations):
            raise ValueError("citation relation claim IDs must be unique")
        if not set(self.evidence_chunk_ids).issuperset(
            chunk_id for item in self.citation_relations for chunk_id in item.chunk_ids
        ):
            raise ValueError("citation relation chunks must belong to the evidence gold set")
        return self


class RetrievalDiagnostic(FrozenEvaluationModel):
    step_id_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_ids: tuple[str, ...]
    dimension_ids: tuple[str, ...]
    corpus_id_filter: str | None = Field(default=None, pattern=r"^[CT]\d{3}$")
    search_count: int = Field(default=1, ge=1)
    search_hit_chunk_ids: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]


class CitationDiagnostic(FrozenEvaluationModel):
    claim_index: int = Field(ge=0)
    citation_id: str = Field(pattern=r"^E[1-9]\d*$")
    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")


FactLossStage = Literal[
    "not_retrieved",
    "not_hydrated",
    "not_visible_to_compiler",
    "not_compiled",
    "not_in_generation_input",
    "not_expressed",
    "citation_incorrect",
    "complete",
]


class FactLineageDiagnostic(FrozenEvaluationModel):
    """Body-free attribution for the earliest stage that lost one gold fact."""

    claim_id: str = Field(min_length=1)
    retrieved: bool
    hydrated: bool
    visible_to_compiler: bool
    exact_gold_chunk_compiled: bool
    same_paper_alternative_chunk_compiled: bool
    semantic_fact_compiled: bool
    compiled: bool
    in_generation_input: bool
    expressed: bool
    citation_correct: bool
    loss_stage: FactLossStage

    @model_validator(mode="before")
    @classmethod
    def derive_legacy_visibility(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        compiled = bool(value.get("compiled"))
        return {
            **value,
            "visible_to_compiler": value.get(
                "visible_to_compiler", bool(value.get("hydrated"))
            ),
            "exact_gold_chunk_compiled": value.get(
                "exact_gold_chunk_compiled", compiled
            ),
            "same_paper_alternative_chunk_compiled": value.get(
                "same_paper_alternative_chunk_compiled", False
            ),
            "semantic_fact_compiled": value.get("semantic_fact_compiled", compiled),
        }

    @model_validator(mode="after")
    def validate_stage(self) -> FactLineageDiagnostic:
        expected = _earliest_fact_loss(
            retrieved=self.retrieved,
            hydrated=self.hydrated,
            visible_to_compiler=self.visible_to_compiler,
            compiled=self.compiled,
            in_generation_input=self.in_generation_input,
            expressed=self.expressed,
            citation_correct=self.citation_correct,
        )
        if self.loss_stage != expected:
            raise ValueError("fact lineage loss stage is inconsistent with stage flags")
        return self


class ComparisonDeterministicScore(FrozenEvaluationModel):
    final_target_correct: bool
    evidence_hit: int = Field(ge=0)
    evidence_total: int = Field(ge=0)
    evidence_recall_at_k: float | None = Field(default=None, ge=0, le=1)
    evidence_k: int = Field(gt=0)
    evidence_corpus_purity: float | None = Field(default=None, ge=0, le=1)
    citation_gold_chunk_rate: float | None = Field(default=None, ge=0, le=1)


class SemanticFactMatchDiagnostic(FrozenEvaluationModel):
    claim_id: str = Field(min_length=1)
    answer_claim_indexes: tuple[int, ...] = ()
    citation_supported: bool = False


class ComparisonModelJudgeScore(FrozenEvaluationModel):
    dimension_hit: int = Field(ge=0)
    dimension_total: int = Field(ge=0)
    must_have_hit: int = Field(ge=0)
    must_have_total: int = Field(ge=0)
    citation_supported_hit: int = Field(ge=0)
    forbidden_present: int = Field(ge=0)
    forbidden_total: int = Field(ge=0)
    answer_complete: bool
    supported_answer_claim_count: int = Field(ge=0)
    answer_claim_count: int = Field(ge=0)
    semantic_fact_matches: tuple[SemanticFactMatchDiagnostic, ...] = ()
    judge_error_type: str | None = None


class CompilationAttemptDiagnostic(FrozenEvaluationModel):
    attempt: int = Field(ge=1, le=2)
    outcome: Literal["validated", "schema_invalid", "contract_invalid"]
    failure_code: str | None = None
    raw_ledger_cell_count: int | None = Field(default=None, ge=0)
    raw_fact_count: int | None = Field(default=None, ge=0)
    requested_requirement_ids: tuple[str, ...] = ()
    accepted_requirement_ids: tuple[str, ...] = ()
    failed_requirement_ids: tuple[str, ...] = ()


class CompilationRepairDiagnostic(FrozenEvaluationModel):
    applied: bool
    source_assessment_available: bool
    input_fact_count: int = Field(ge=0)
    retained_fact_count: int = Field(ge=0)
    dropped_chunk_scope_count: int = Field(ge=0)
    dropped_fact_mapping_count: int = Field(ge=0)
    missing_ledger_cell_count: int = Field(ge=0)
    fallback_empty_used: bool


class CompilationAuditDiagnostic(FrozenEvaluationModel):
    attempts: tuple[CompilationAttemptDiagnostic, ...] = Field(max_length=2)
    repair: CompilationRepairDiagnostic


class ComparisonCaseDiagnostic(FrozenEvaluationModel):
    schema_version: Literal["comparison-e2e-diagnostic-v1"] = (
        "comparison-e2e-diagnostic-v1"
    )
    question_id: str = Field(pattern=r"^CPG\d{3}$")
    split: Literal["dev", "sealed_test"]
    original_question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_question_preserved: bool
    rewrite_status: str
    rewrite_latency_ms: float | None = Field(default=None, ge=0)
    rewrite_information_retained: bool | None = None
    rewrite_required_token_count: int = Field(default=0, ge=0)
    rewrite_retained_token_count: int = Field(default=0, ge=0)
    candidate_paper_ids_top8: tuple[str, ...] = Field(max_length=8)
    final_paper_ids: tuple[str, ...]
    planned_dimensions: tuple[str, ...]
    retrievals: tuple[RetrievalDiagnostic, ...]
    step_budget: int | None = Field(default=None, ge=1)
    assessment_count: int = Field(default=0, ge=0)
    compilation_audit: CompilationAuditDiagnostic | None = None
    tool_call_count: int = Field(default=0, ge=0)
    tool_call_budget: int | None = Field(default=None, ge=1)
    citations: tuple[CitationDiagnostic, ...]
    fact_lineage: tuple[FactLineageDiagnostic, ...] = ()
    answer_status: Literal[
        "answered", "insufficient_evidence", "compiler_failed"
    ] | None = None
    deterministic_score: ComparisonDeterministicScore | None = None
    model_judge_score: ComparisonModelJudgeScore | None = None
    total_latency_ms: float = Field(ge=0)
    error_type: str | None = None
    error_reason_code: str | None = None
    error_stage: str | None = None

    @field_validator("candidate_paper_ids_top8", "final_paper_ids")
    @classmethod
    def unique_papers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("paper IDs must be unique")
        return values


def question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def classify_fact_lineage(
    claim_id: str,
    *,
    gold_chunk_ids: Iterable[str],
    retrieved_chunk_ids: Iterable[str],
    hydrated_chunk_ids: Iterable[str],
    compiled_chunk_ids: Iterable[str],
    semantic_compiled_chunk_ids: Iterable[str] | None = None,
    same_paper_alternative_chunk_ids: Iterable[str] | None = None,
    visible_chunk_ids: Iterable[str] | None = None,
    in_generation_input: bool,
    expressed: bool,
    citation_correct: bool,
) -> FactLineageDiagnostic:
    """Classify one gold fact without copying gold or evidence text into diagnostics."""
    expected = set(gold_chunk_ids)
    retrieved = bool(expected & set(retrieved_chunk_ids))
    hydrated = retrieved and bool(expected & set(hydrated_chunk_ids))
    visible = hydrated and bool(
        expected
        & set(hydrated_chunk_ids if visible_chunk_ids is None else visible_chunk_ids)
    )
    exact_compiled = visible and bool(expected & set(compiled_chunk_ids))
    semantic_compiled = (
        exact_compiled
        if semantic_compiled_chunk_ids is None
        else bool(tuple(semantic_compiled_chunk_ids))
    )
    alternative_compiled = (
        semantic_compiled
        and not exact_compiled
        and same_paper_alternative_chunk_ids is not None
        and bool(tuple(same_paper_alternative_chunk_ids))
    )
    compiled = semantic_compiled
    input_present = compiled and in_generation_input
    output_present = input_present and expressed
    citation_valid = output_present and citation_correct
    stage = _earliest_fact_loss(
        retrieved=retrieved,
        hydrated=hydrated,
        visible_to_compiler=visible,
        compiled=compiled,
        in_generation_input=input_present,
        expressed=output_present,
        citation_correct=citation_valid,
    )
    return FactLineageDiagnostic(
        claim_id=claim_id,
        retrieved=retrieved,
        hydrated=hydrated,
        visible_to_compiler=visible,
        exact_gold_chunk_compiled=exact_compiled,
        same_paper_alternative_chunk_compiled=alternative_compiled,
        semantic_fact_compiled=semantic_compiled,
        compiled=compiled,
        in_generation_input=input_present,
        expressed=output_present,
        citation_correct=citation_valid,
        loss_stage=stage,
    )


def aggregate_fact_lineage(
    diagnostics: Iterable[FactLineageDiagnostic],
) -> dict[str, object]:
    materialized = tuple(diagnostics)
    counts = Counter(item.loss_stage for item in materialized)

    def rate(field: str) -> float | None:
        if not materialized:
            return None
        return sum(bool(getattr(item, field)) for item in materialized) / len(materialized)

    stages: tuple[FactLossStage, ...] = (
        "not_retrieved",
        "not_hydrated",
        "not_visible_to_compiler",
        "not_compiled",
        "not_in_generation_input",
        "not_expressed",
        "citation_incorrect",
        "complete",
    )
    return {
        "fact_count": len(materialized),
        "exact_gold_chunk_recall": rate("exact_gold_chunk_compiled"),
        "same_paper_alternative_chunk_recall": rate(
            "same_paper_alternative_chunk_compiled"
        ),
        "semantic_fact_recall": rate("semantic_fact_compiled"),
        "stage_rates": {
            "retrieved": rate("retrieved"),
            "hydrated": rate("hydrated"),
            "visible_to_compiler": rate("visible_to_compiler"),
            "exact_gold_chunk_compiled": rate("exact_gold_chunk_compiled"),
            "same_paper_alternative_chunk_compiled": rate(
                "same_paper_alternative_chunk_compiled"
            ),
            "semantic_fact_compiled": rate("semantic_fact_compiled"),
            "compiled": rate("compiled"),
            "in_generation_input": rate("in_generation_input"),
            "expressed": rate("expressed"),
            "citation_correct": rate("citation_correct"),
        },
        "earliest_loss_counts": {stage: counts.get(stage, 0) for stage in stages},
    }


def aggregate_compilation_audits(
    cases: Iterable[ComparisonCaseDiagnostic],
) -> dict[str, int]:
    """Aggregate body-free transactional compiler unit counts."""
    attempts = tuple(
        attempt
        for case in cases
        if case.compilation_audit is not None
        for attempt in case.compilation_audit.attempts
    )
    return {
        "attempt_count": len(attempts),
        "requested_unit_count": sum(
            len(item.requested_requirement_ids) for item in attempts
        ),
        "accepted_unit_count": sum(
            len(item.accepted_requirement_ids) for item in attempts
        ),
        "failed_unit_count": sum(len(item.failed_requirement_ids) for item in attempts),
        "schema_failed_unit_count": sum(
            len(item.failed_requirement_ids)
            for item in attempts
            if item.outcome == "schema_invalid"
        ),
        "contract_failed_unit_count": sum(
            len(item.failed_requirement_ids)
            for item in attempts
            if item.outcome == "contract_invalid"
        ),
    }


def _earliest_fact_loss(
    *,
    retrieved: bool,
    hydrated: bool,
    visible_to_compiler: bool,
    compiled: bool,
    in_generation_input: bool,
    expressed: bool,
    citation_correct: bool,
) -> FactLossStage:
    stages: tuple[tuple[bool, FactLossStage], ...] = (
        (retrieved, "not_retrieved"),
        (hydrated, "not_hydrated"),
        (visible_to_compiler, "not_visible_to_compiler"),
        (compiled, "not_compiled"),
        (in_generation_input, "not_in_generation_input"),
        (expressed, "not_expressed"),
        (citation_correct, "citation_incorrect"),
    )
    for passed, loss_stage in stages:
        if not passed:
            return loss_stage
    return "complete"


_REQUIRED_ASCII_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:[CT]\d{3}|[A-Za-z][A-Za-z0-9_.+/-]*\d[A-Za-z0-9_.+/-]*|"
    r"\d+(?:\.\d+)?%?)(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


def deterministic_rewrite_retention(
    original_question: str, english_query: str | None
) -> tuple[bool | None, int, int]:
    """Conservatively verify identifiers/numbers that must survive translation."""
    if english_query is None:
        return None, 0, 0
    required = tuple(
        dict.fromkeys(match.group(0).casefold() for match in _REQUIRED_ASCII_TOKEN.finditer(original_question))
    )
    retained = sum(token in english_query.casefold() for token in required)
    return retained == len(required), len(required), retained


def validate_structural_guarantees(
    case: ComparisonCaseDiagnostic,
    *,
    chunk_corpus_ids: dict[str, str],
) -> tuple[str, ...]:
    failures: list[str] = []
    candidates = set(case.candidate_paper_ids_top8)
    if not set(case.final_paper_ids).issubset(candidates):
        failures.append("final_target_outside_candidates")
    for retrieval in case.retrievals:
        if len(retrieval.target_ids) != 1 or len(retrieval.dimension_ids) != 1:
            failures.append("non_atomic_target_dimension_retrieval")
        if retrieval.corpus_id_filter is None:
            failures.append("missing_corpus_id_filter")
        for chunk_id in (*retrieval.search_hit_chunk_ids, *retrieval.evidence_chunk_ids):
            if chunk_corpus_ids.get(chunk_id) != retrieval.corpus_id_filter:
                failures.append("cross_paper_evidence")
    for citation in case.citations:
        if chunk_corpus_ids.get(citation.chunk_id) != citation.corpus_id:
            failures.append("citation_corpus_mismatch")
    scoped_queries = {
        (item.corpus_id_filter, item.step_id_hash) for item in case.retrievals
    }
    if len(scoped_queries) != len(case.retrievals):
        failures.append("duplicate_scoped_retrieval")
    return tuple(dict.fromkeys(failures))


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def score_deterministic_case(
    case: ComparisonCaseDiagnostic,
    gold: ComparisonEndToEndGold,
    *,
    chunk_corpus_ids: dict[str, str],
    evidence_k: int = 50,
) -> ComparisonDeterministicScore:
    retrieved = tuple(
        dict.fromkeys(
            chunk_id
            for retrieval in case.retrievals
            for chunk_id in retrieval.search_hit_chunk_ids
        )
    )[:evidence_k]
    expected = set(gold.evidence_chunk_ids)
    evidence_hit = len(expected & set(retrieved))
    scoped_chunks = [
        (chunk_id, retrieval.corpus_id_filter)
        for retrieval in case.retrievals
        for chunk_id in (*retrieval.search_hit_chunk_ids, *retrieval.evidence_chunk_ids)
    ]
    pure = sum(
        corpus_id is not None and chunk_corpus_ids.get(chunk_id) == corpus_id
        for chunk_id, corpus_id in scoped_chunks
    )
    cited = tuple(case.citations)
    citation_gold = sum(
        item.chunk_id in expected and chunk_corpus_ids.get(item.chunk_id) == item.corpus_id
        for item in cited
    )
    return ComparisonDeterministicScore(
        final_target_correct=set(case.final_paper_ids) == set(gold.relevant_paper_ids),
        evidence_hit=evidence_hit,
        evidence_total=len(expected),
        evidence_recall_at_k=(evidence_hit / len(expected) if expected else None),
        evidence_k=evidence_k,
        evidence_corpus_purity=(pure / len(scoped_chunks) if scoped_chunks else None),
        citation_gold_chunk_rate=(citation_gold / len(cited) if cited else None),
    )


def aggregate_smoke_cases(
    cases: Iterable[ComparisonCaseDiagnostic],
    *,
    relevant_by_question: dict[str, tuple[str, ...]],
    chunk_corpus_ids: dict[str, str],
) -> dict[str, object]:
    materialized = tuple(cases)
    successes = tuple(case for case in materialized if case.error_type is None)
    latencies = [case.total_latency_ms for case in materialized]

    def recall_at(case: ComparisonCaseDiagnostic, cutoff: int) -> float:
        relevant = set(relevant_by_question[case.question_id])
        selected = set(case.candidate_paper_ids_top8[:cutoff])
        return len(relevant & selected) / len(relevant)

    structural_failures = {
        case.question_id: validate_structural_guarantees(
            case, chunk_corpus_ids=chunk_corpus_ids
        )
        for case in successes
    }
    return {
        "question_count": len(materialized),
        "success_count": len(successes),
        "failure_count": len(materialized) - len(successes),
        "candidate_recall_at_5": (
            sum(recall_at(case, 5) for case in materialized) / len(materialized)
            if materialized
            else None
        ),
        "candidate_recall_at_8": (
            sum(recall_at(case, 8) for case in materialized) / len(materialized)
            if materialized
            else None
        ),
        "final_target_accuracy": (
            sum(
                set(case.final_paper_ids) == set(relevant_by_question[case.question_id])
                for case in materialized
            )
            / len(materialized)
            if materialized
            else None
        ),
        "rewrite_retention_rate": (
            sum(case.rewrite_information_retained is True for case in materialized)
            / len(materialized)
            if materialized
            else None
        ),
        "structural_all_pass_rate": (
            sum(not failures for failures in structural_failures.values()) / len(successes)
            if successes
            else None
        ),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
        "timeout_rate": (
            sum(case.error_type == "TimeoutError" for case in materialized)
            / len(materialized)
            if materialized
            else None
        ),
    }


def aggregate_answer_scores(
    cases: Iterable[ComparisonCaseDiagnostic],
) -> dict[str, object]:
    materialized = tuple(cases)
    deterministic = tuple(
        deterministic_score
        for case in materialized
        if (deterministic_score := case.deterministic_score) is not None
    )
    attempted = tuple(
        judge_score
        for case in materialized
        if (judge_score := case.model_judge_score) is not None
    )
    judged = tuple(
        score
        for score in attempted
        if score.judge_error_type is None
    )

    def ratio(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator else None

    deterministic_metrics = {
        "final_target_accuracy": ratio(
            sum(item.final_target_correct for item in deterministic), len(materialized)
        ),
        "evidence_recall_at_50": ratio(
            sum(item.evidence_hit for item in deterministic),
            sum(item.evidence_total for item in deterministic),
        ),
        "evidence_corpus_purity": ratio(
            sum(
                item.evidence_corpus_purity
                for item in deterministic
                if item.evidence_corpus_purity is not None
            ),
            sum(item.evidence_corpus_purity is not None for item in deterministic),
        ),
        "citation_gold_chunk_rate": ratio(
            sum(
                item.citation_gold_chunk_rate
                for item in deterministic
                if item.citation_gold_chunk_rate is not None
            ),
            sum(item.citation_gold_chunk_rate is not None for item in deterministic),
        ),
    }
    model_metrics = {
        "judged_case_count": len(judged),
        "model_scored_case_rate": ratio(len(judged), len(materialized)),
        "judge_failure_rate": ratio(
            sum(item.judge_error_type is not None for item in attempted), len(attempted)
        ),
        "dimension_coverage": ratio(
            sum(item.dimension_hit for item in judged),
            sum(item.dimension_total for item in judged),
        ),
        "must_have_claim_recall": ratio(
            sum(item.must_have_hit for item in judged),
            sum(item.must_have_total for item in judged),
        ),
        "citation_correctness": ratio(
            sum(item.citation_supported_hit for item in judged),
            sum(item.must_have_hit for item in judged),
        ),
        "forbidden_claim_rate": ratio(
            sum(item.forbidden_present for item in judged),
            sum(item.forbidden_total for item in judged),
        ),
        "answer_completeness": ratio(
            sum(item.answer_complete for item in judged), len(judged)
        ),
    }
    all_correct = sum(
        case.error_type is None
        and case.deterministic_score is not None
        and case.deterministic_score.final_target_correct
        and case.deterministic_score.evidence_corpus_purity == 1
        and case.model_judge_score is not None
        and case.model_judge_score.judge_error_type is None
        and case.model_judge_score.dimension_hit == case.model_judge_score.dimension_total
        and case.model_judge_score.must_have_hit == case.model_judge_score.must_have_total
        and case.model_judge_score.citation_supported_hit
        == case.model_judge_score.must_have_total
        and case.model_judge_score.forbidden_present == 0
        and case.model_judge_score.answer_complete
        for case in materialized
    )
    return {
        "deterministic": deterministic_metrics,
        "model_judge": model_metrics,
        "end_to_end_all_correct_rate": ratio(all_correct, len(materialized)),
    }
