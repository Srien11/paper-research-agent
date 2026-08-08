"""Strict contracts for read-only research tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.ingestion.models import Sha256

StorageClass = Literal["redistributable", "internal_research_only"]
EvidenceType = Literal["text", "figure_summary"]
ResearchTaskType = Literal["direct", "comparison"]
AssessmentStatus = Literal[
    "sufficient",
    "missing_coverage",
    "conflicting_evidence",
    "no_hits",
]
TerminationReason = Literal[
    "evidence_sufficient",
    "tool_budget",
    "no_new_evidence",
    "plan_exhausted",
    "repeated_query",
]
ResearchActionName = Literal[
    "search_corpus",
    "get_evidence",
    "assess_evidence",
    "replan",
    "finish",
]

ASSESSMENT_STATUSES: frozenset[str] = frozenset(
    {"sufficient", "missing_coverage", "conflicting_evidence", "no_hits"}
)
TERMINATION_REASONS: frozenset[str] = frozenset(
    {
        "evidence_sufficient",
        "tool_budget",
        "no_new_evidence",
        "plan_exhausted",
        "repeated_query",
    }
)


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchCorpusInput(FrozenContract):
    """Bounded input accepted by the corpus search tool."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=20)
    corpus_id: str | None = Field(default=None, pattern=r"^[CT]\d{3}$")

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("search query must not be blank")
        return normalized


class SearchCorpusHit(FrozenContract):
    """Minimal search metadata exposed to an Agent before evidence hydration."""

    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    section_id: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text_sha256: Sha256
    evidence_type: EvidenceType = "text"
    storage_class: StorageClass
    final_rank: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_pages(self) -> SearchCorpusHit:
        if self.page_end < self.page_start:
            raise ValueError("search hit page range is reversed")
        return self


class SearchCorpusResult(FrozenContract):
    """Traceable search result without evidence bodies or provider payloads."""

    schema_version: Literal["research-search-tool-v1"] = "research-search-tool-v1"
    query: str = Field(min_length=1)
    corpus_id: str | None = Field(default=None, pattern=r"^[CT]\d{3}$")
    index_id: str = Field(min_length=1)
    degraded: bool
    degraded_reason: str | None = None
    hits: tuple[SearchCorpusHit, ...]

    @model_validator(mode="after")
    def validate_result(self) -> SearchCorpusResult:
        expected = list(range(1, len(self.hits) + 1))
        if [hit.final_rank for hit in self.hits] != expected:
            raise ValueError("search result ranks must be contiguous and ordered")
        if self.degraded != bool(self.degraded_reason):
            raise ValueError("degraded_reason must be present exactly for degraded results")
        chunk_ids = [hit.chunk_id for hit in self.hits]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("search result chunk IDs must be unique")
        if self.corpus_id is not None and any(
            hit.corpus_id != self.corpus_id for hit in self.hits
        ):
            raise ValueError("search result contains a hit outside the declared corpus scope")
        return self


class GetEvidenceInput(FrozenContract):
    """Explicit IDs accepted by the local evidence hydration tool."""

    chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=20)

    @field_validator("chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("chunk IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("chunk IDs must be unique")
        return normalized


class EvidenceRecord(FrozenContract):
    """One local evidence body with stable provenance and loaded rights."""

    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    section_id: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    text_sha256: Sha256
    evidence_type: EvidenceType = "text"
    storage_class: StorageClass

    @model_validator(mode="after")
    def validate_record(self) -> EvidenceRecord:
        if self.page_end < self.page_start:
            raise ValueError("evidence page range is reversed")
        if not self.text.strip():
            raise ValueError("evidence text must not be blank")
        return self


class GetEvidenceResult(FrozenContract):
    """Hydrated local evidence plus IDs that were not present in the catalog."""

    schema_version: Literal["research-evidence-tool-v1"] = "research-evidence-tool-v1"
    records: tuple[EvidenceRecord, ...]
    missing_chunk_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> GetEvidenceResult:
        record_ids = [record.chunk_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("evidence record chunk IDs must be unique")
        if len(self.missing_chunk_ids) != len(set(self.missing_chunk_ids)):
            raise ValueError("missing chunk IDs must be unique")
        if set(record_ids) & set(self.missing_chunk_ids):
            raise ValueError("a chunk ID cannot be both present and missing")
        return self


class ResearchTarget(FrozenContract):
    """One named paper, method, model, or study being compared."""

    target_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    label: str = Field(min_length=1, max_length=200)
    corpus_id: str | None = Field(default=None, pattern=r"^[CT]\d{3}$")

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("research target label must not be blank")
        return normalized


class ResearchDimension(FrozenContract):
    """One explicit comparison axis such as method, dataset, metric, or limitation."""

    dimension_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    label: str = Field(min_length=1, max_length=200)

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("research dimension label must not be blank")
        return normalized


class EvidenceRequirement(FrozenContract):
    """One required target-by-dimension evidence cell in a comparison plan."""

    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    target_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    dimension_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    description: str = Field(min_length=1, max_length=500)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("evidence requirement description must not be blank")
        return normalized


class EvidenceCoverage(FrozenContract):
    """Evidence IDs supporting one requirement without duplicating source text."""

    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    covered: bool
    chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @field_validator("chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("coverage chunk IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("coverage chunk IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_covered_state(self) -> EvidenceCoverage:
        if self.covered and not self.chunk_ids:
            raise ValueError("covered evidence requires chunk IDs")
        if not self.covered and self.chunk_ids:
            raise ValueError("uncovered evidence cannot include chunk IDs")
        return self


class ResearchStep(FrozenContract):
    """One bounded corpus-search objective produced by a research planner."""

    step_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    objective: str = Field(min_length=1, max_length=500)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=20)
    corpus_id: str | None = Field(default=None, pattern=r"^[CT]\d{3}$")
    target_ids: tuple[str, ...] = Field(default=(), max_length=4)
    dimension_ids: tuple[str, ...] = Field(default=(), max_length=5)

    @field_validator("objective", "query")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("research step text must not be blank")
        return normalized

    @field_validator("target_ids", "dimension_ids")
    @classmethod
    def validate_reference_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("research step reference IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("research step reference IDs must be unique")
        return normalized


class ResearchPlan(FrozenContract):
    """Ordered, auditable subquestions that stay within the read-only workflow."""

    schema_version: Literal["research-plan-v1"] = "research-plan-v1"
    task_type: ResearchTaskType = "direct"
    targets: tuple[ResearchTarget, ...] = Field(default=(), max_length=4)
    dimensions: tuple[ResearchDimension, ...] = Field(default=(), max_length=5)
    requirements: tuple[EvidenceRequirement, ...] = Field(default=(), max_length=20)
    steps: tuple[ResearchStep, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_steps(self) -> ResearchPlan:
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("research plan step IDs must be unique")
        target_ids = [target.target_id for target in self.targets]
        dimension_ids = [dimension.dimension_id for dimension in self.dimensions]
        requirement_ids = [item.requirement_id for item in self.requirements]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("research plan target IDs must be unique")
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError("research plan dimension IDs must be unique")
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("research plan requirement IDs must be unique")
        if self.task_type == "direct":
            if self.targets or self.dimensions or self.requirements:
                raise ValueError("direct research plan cannot declare comparison metadata")
            if any(step.target_ids or step.dimension_ids for step in self.steps):
                raise ValueError("direct research steps cannot reference comparison metadata")
            return self
        if len(self.targets) < 2:
            raise ValueError("comparison research plan requires at least two targets")
        if any(target.corpus_id is None for target in self.targets):
            raise ValueError("comparison research targets require resolved corpus IDs")
        corpus_ids = [target.corpus_id for target in self.targets]
        if len(corpus_ids) != len(set(corpus_ids)):
            raise ValueError("comparison research targets must use distinct corpus IDs")
        if not self.dimensions:
            raise ValueError("comparison research plan requires at least one dimension")
        known_targets = set(target_ids)
        known_dimensions = set(dimension_ids)
        if any(
            item.target_id not in known_targets or item.dimension_id not in known_dimensions
            for item in self.requirements
        ):
            raise ValueError("comparison requirement references an unknown target or dimension")
        required_pairs = {
            (target_id, dimension_id)
            for target_id in known_targets
            for dimension_id in known_dimensions
        }
        actual_pairs = {(item.target_id, item.dimension_id) for item in self.requirements}
        if actual_pairs != required_pairs or len(self.requirements) != len(required_pairs):
            raise ValueError("comparison requirements must form a complete target-dimension grid")
        if any(
            len(step.target_ids) != 1
            or len(step.dimension_ids) != 1
            or not set(step.target_ids) <= known_targets
            or not set(step.dimension_ids) <= known_dimensions
            for step in self.steps
        ):
            raise ValueError(
                "comparison research steps require one target and one dimension"
            )
        target_corpus_ids = {target.target_id: target.corpus_id for target in self.targets}
        for step in self.steps:
            expected_corpus_id = target_corpus_ids[step.target_ids[0]]
            if step.corpus_id != expected_corpus_id:
                raise ValueError("comparison research step corpus scope does not match its target")
        planned_pairs = {
            (target_id, dimension_id)
            for step in self.steps
            for target_id in step.target_ids
            for dimension_id in step.dimension_ids
        }
        if not required_pairs <= planned_pairs:
            raise ValueError("comparison research steps do not cover every requirement cell")
        return self


class ResearchObservation(FrozenContract):
    """One completed plan step and the exact tool results it produced."""

    step_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    objective: str = Field(min_length=1)
    search: SearchCorpusResult
    evidence: GetEvidenceResult


class EvidenceAssessment(FrozenContract):
    """One bounded reflection over accumulated local evidence."""

    schema_version: Literal["research-evidence-assessment-v1"] = "research-evidence-assessment-v1"
    evidence_sufficient: bool
    status: AssessmentStatus
    coverage: tuple[EvidenceCoverage, ...] = Field(default=(), max_length=20)
    next_query: str | None = Field(default=None, min_length=1, max_length=2000)
    next_objective: str | None = Field(default=None, min_length=1, max_length=500)
    next_requirement_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @field_validator("next_query", "next_objective")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("next search text must not be blank")
        return normalized

    @field_validator("next_requirement_ids")
    @classmethod
    def validate_next_requirement_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("next requirement IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("next requirement IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_decision(self) -> EvidenceAssessment:
        coverage_ids = [item.requirement_id for item in self.coverage]
        if len(coverage_ids) != len(set(coverage_ids)):
            raise ValueError("evidence coverage requirement IDs must be unique")
        has_next_query = self.next_query is not None
        has_next_objective = self.next_objective is not None
        if self.evidence_sufficient:
            if self.status != "sufficient":
                raise ValueError("sufficient evidence requires sufficient status")
            if has_next_query or has_next_objective or self.next_requirement_ids:
                raise ValueError("sufficient evidence cannot request another search")
        else:
            if self.status == "sufficient":
                raise ValueError("insufficient evidence cannot use sufficient status")
            if has_next_query != has_next_objective:
                raise ValueError("next query and objective must be present together")
            if not has_next_query and self.next_requirement_ids:
                raise ValueError("next requirement IDs require another search")
        return self


class ResearchActionRecord(FrozenContract):
    """Safe, body-free audit record for one ReAct transition."""

    sequence: int = Field(ge=1)
    action: ResearchActionName
    step_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
    )
    query: str | None = Field(default=None, min_length=1, max_length=2000)
    chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)
    outcome: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("query", "outcome")
    @classmethod
    def normalize_optional_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("action text must not be blank")
        return normalized

    @field_validator("chunk_ids")
    @classmethod
    def validate_action_chunk_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("action chunk IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("action chunk IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_action_fields(self) -> ResearchActionRecord:
        has_step = self.step_id is not None
        has_query = self.query is not None
        has_chunks = bool(self.chunk_ids)
        has_outcome = self.outcome is not None
        if self.action == "search_corpus":
            valid = has_step and has_query and not has_chunks and not has_outcome
        elif self.action == "get_evidence":
            valid = has_step and not has_query and has_chunks and not has_outcome
        elif self.action == "assess_evidence":
            valid = (
                has_step
                and not has_query
                and not has_chunks
                and self.outcome in ASSESSMENT_STATUSES
            )
        elif self.action == "replan":
            valid = has_step and has_query and not has_chunks and not has_outcome
        else:
            valid = (
                not has_step
                and not has_query
                and not has_chunks
                and self.outcome in TERMINATION_REASONS
            )
        if not valid:
            raise ValueError(f"invalid fields for research action: {self.action}")
        return self
