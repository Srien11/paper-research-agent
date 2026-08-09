"""Strict contracts for read-only research tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from paper_research_agent.ingestion.models import Sha256

StorageClass = Literal["redistributable", "internal_research_only"]
EvidenceType = Literal["text", "figure_summary"]
ResearchTaskType = Literal["direct", "comparison"]
AssessmentStatus = Literal[
    "sufficient",
    "missing_coverage",
    "conflicting_evidence",
    "no_hits",
    "compiler_failed",
]
EvidenceLedgerStatus = Literal["sufficient", "partial", "missing", "conflicting"]
EvidenceQualifierKind = Literal[
    "time",
    "dataset",
    "method",
    "metric",
    "scope",
    "condition",
    "other",
]
TerminationReason = Literal[
    "evidence_sufficient",
    "tool_budget",
    "no_new_evidence",
    "plan_exhausted",
    "repeated_query",
    "step_budget",
    "time_budget",
    "compiler_failed",
]
ResearchActionName = Literal[
    "search_corpus",
    "get_evidence",
    "assess_evidence",
    "replan",
    "finish",
]

ASSESSMENT_STATUSES: frozenset[str] = frozenset(
    {
        "sufficient",
        "missing_coverage",
        "conflicting_evidence",
        "no_hits",
        "compiler_failed",
    }
)
TERMINATION_REASONS: frozenset[str] = frozenset(
    {
        "evidence_sufficient",
        "tool_budget",
        "no_new_evidence",
        "plan_exhausted",
        "repeated_query",
        "step_budget",
        "time_budget",
        "compiler_failed",
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


class EvidenceFactRequirement(FrozenContract):
    """One question-derived atomic fact intent within a comparison cell."""

    fact_requirement_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$"
    )
    description: str = Field(min_length=1, max_length=500)
    required_qualifier_kinds: tuple[EvidenceQualifierKind, ...] = Field(
        default=(), max_length=7
    )
    origin: Literal["planned", "derived"] = "planned"

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("fact requirement description must not be blank")
        return normalized

    @field_validator("required_qualifier_kinds")
    @classmethod
    def unique_qualifier_kinds(
        cls, values: tuple[EvidenceQualifierKind, ...]
    ) -> tuple[EvidenceQualifierKind, ...]:
        if len(values) != len(set(values)):
            raise ValueError("required qualifier kinds must be unique")
        return values


class EvidenceRequirement(FrozenContract):
    """One required target-by-dimension evidence cell in a comparison plan."""

    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    target_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    dimension_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    description: str = Field(min_length=1, max_length=500)
    fact_requirements: tuple[EvidenceFactRequirement, ...] = Field(
        min_length=1, max_length=6
    )

    @model_validator(mode="before")
    @classmethod
    def derive_legacy_fact_requirement(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("fact_requirements"):
            return value
        requirement_id = value.get("requirement_id")
        description = value.get("description")
        if not isinstance(requirement_id, str) or not isinstance(description, str):
            return value
        return {
            **value,
            "fact_requirements": (
                {
                    "fact_requirement_id": f"{requirement_id}-primary",
                    "description": description,
                    "origin": "derived",
                },
            ),
        }

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("evidence requirement description must not be blank")
        return normalized

    @model_validator(mode="after")
    def unique_fact_requirements(self) -> EvidenceRequirement:
        identifiers = [item.fact_requirement_id for item in self.fact_requirements]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("fact requirement IDs must be unique within a cell")
        return self


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


class EvidenceQualifier(FrozenContract):
    """One explicit condition that limits how a compiled fact may be stated."""

    kind: EvidenceQualifierKind
    value: str = Field(min_length=1, max_length=500)

    @field_validator("value")
    @classmethod
    def normalize_value(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("evidence qualifier value must not be blank")
        return normalized


class EvidenceFactCompilation(FrozenContract):
    """Minimal model-authored fact before deterministic ledger projection."""

    statement: str = Field(min_length=1, max_length=1200)
    chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    fact_requirement_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    qualifiers: tuple[EvidenceQualifier, ...] = Field(default=(), max_length=12)

    @field_validator("statement")
    @classmethod
    def normalize_statement(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("compiled evidence statement must not be blank")
        return normalized

    @field_validator("chunk_ids", "fact_requirement_ids")
    @classmethod
    def validate_identifier_list(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("compiled evidence identifiers must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("compiled evidence identifiers must be unique")
        return normalized


class EvidenceCellCompilation(FrozenContract):
    """Minimal facts authored for one target-by-dimension requirement."""

    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    facts: tuple[EvidenceFactCompilation, ...] = Field(default=(), max_length=24)


class EvidenceCompilationBatch(FrozenContract):
    """Tolerant transport envelope whose cells are validated transactionally."""

    cells: tuple[dict[str, object], ...] = Field(default=(), max_length=20)


class CompiledEvidenceFact(FrozenContract):
    """One answer-ready atomic fact compiled from trusted local evidence IDs."""

    fact_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
    statement: str = Field(min_length=1, max_length=1200)
    chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=20)
    fact_requirement_ids: tuple[str, ...] = Field(default=(), max_length=6)
    qualifiers: tuple[EvidenceQualifier, ...] = Field(default=(), max_length=12)

    @field_validator("statement")
    @classmethod
    def normalize_statement(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("compiled evidence statement must not be blank")
        return normalized

    @field_validator("chunk_ids")
    @classmethod
    def validate_fact_chunk_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("compiled evidence chunk IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("compiled evidence chunk IDs must be unique")
        return normalized

    @field_validator("fact_requirement_ids")
    @classmethod
    def validate_fact_requirement_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("compiled fact requirement IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("compiled fact requirement IDs must be unique")
        return normalized


class EvidenceLedgerCell(FrozenContract):
    """Compiled facts and sufficiency state for one target-by-dimension cell."""

    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    status: EvidenceLedgerStatus
    facts: tuple[CompiledEvidenceFact, ...] = Field(default=(), max_length=24)
    missing_fact_requirement_ids: tuple[str, ...] = Field(default=(), max_length=6)

    @field_validator("missing_fact_requirement_ids")
    @classmethod
    def validate_missing_fact_requirement_ids(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("missing fact requirement IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("missing fact requirement IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_status(self) -> EvidenceLedgerCell:
        if self.status == "sufficient" and not self.facts:
            raise ValueError("sufficient evidence ledger cell requires facts")
        if self.status == "sufficient" and self.missing_fact_requirement_ids:
            raise ValueError("sufficient evidence ledger cell cannot contain missing facts")
        if self.status == "partial" and (
            not self.facts or not self.missing_fact_requirement_ids
        ):
            raise ValueError("partial evidence ledger cell requires facts and missing facts")
        if self.status == "missing" and self.facts:
            raise ValueError("missing evidence ledger cell cannot contain facts")
        fact_ids = [item.fact_id for item in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("evidence ledger fact IDs must be unique within a cell")
        return self


class EvidenceCompilationVisibility(FrozenContract):
    """Body-free audit of which hydrated chunks were visible to one ledger cell."""

    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    available_chunk_ids: tuple[str, ...] = Field(default=(), max_length=96)
    visible_chunk_ids: tuple[str, ...] = Field(default=(), max_length=96)
    truncated_chunk_ids: tuple[str, ...] = Field(default=(), max_length=96)

    @field_validator(
        "available_chunk_ids", "visible_chunk_ids", "truncated_chunk_ids"
    )
    @classmethod
    def validate_visibility_chunk_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("visibility chunk IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("visibility chunk IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_visibility_subsets(self) -> EvidenceCompilationVisibility:
        available = set(self.available_chunk_ids)
        if not set(self.visible_chunk_ids) <= available:
            raise ValueError("visible chunks must be available to the requirement")
        if not set(self.truncated_chunk_ids) <= set(self.visible_chunk_ids):
            raise ValueError("truncated chunks must be visible to the requirement")
        return self


class EvidenceCompilationAttemptAudit(FrozenContract):
    """Body-free result of one structured evidence-compilation attempt."""

    attempt: int = Field(ge=1, le=2)
    outcome: Literal["validated", "schema_invalid", "contract_invalid"]
    failure_code: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9_]{0,95}$"
    )
    raw_ledger_cell_count: int | None = Field(default=None, ge=0)
    raw_fact_count: int | None = Field(default=None, ge=0)
    requested_requirement_ids: tuple[str, ...] = Field(default=(), max_length=20)
    accepted_requirement_ids: tuple[str, ...] = Field(default=(), max_length=20)
    failed_requirement_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @field_validator(
        "requested_requirement_ids",
        "accepted_requirement_ids",
        "failed_requirement_ids",
    )
    @classmethod
    def validate_requirement_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("compilation audit requirement IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("compilation audit requirement IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_attempt_outcome(self) -> EvidenceCompilationAttemptAudit:
        if self.outcome == "validated" and self.failure_code is not None:
            raise ValueError("validated compilation attempt cannot have a failure code")
        if self.outcome != "validated" and self.failure_code is None:
            raise ValueError("failed compilation attempt requires a failure code")
        requested = set(self.requested_requirement_ids)
        accepted = set(self.accepted_requirement_ids)
        failed = set(self.failed_requirement_ids)
        if accepted & failed:
            raise ValueError("accepted and failed compilation units must be disjoint")
        if requested and accepted | failed != requested:
            raise ValueError("compilation attempt must account for every requested unit")
        if self.outcome == "validated" and failed:
            raise ValueError("validated compilation attempt cannot contain failed units")
        return self


class EvidenceCompilationRepairAudit(FrozenContract):
    """Body-free counts for conservative assessment repair."""

    applied: bool
    source_assessment_available: bool
    input_fact_count: int = Field(default=0, ge=0)
    retained_fact_count: int = Field(default=0, ge=0)
    dropped_chunk_scope_count: int = Field(default=0, ge=0)
    dropped_fact_mapping_count: int = Field(default=0, ge=0)
    missing_ledger_cell_count: int = Field(default=0, ge=0)
    fallback_empty_used: bool = False

    @model_validator(mode="after")
    def validate_repair_counts(self) -> EvidenceCompilationRepairAudit:
        accounted = (
            self.retained_fact_count
            + self.dropped_chunk_scope_count
            + self.dropped_fact_mapping_count
        )
        if accounted != self.input_fact_count:
            raise ValueError("repair fact counts must account for every input fact")
        if self.fallback_empty_used and self.source_assessment_available:
            raise ValueError("empty fallback cannot have a source assessment")
        return self


class EvidenceCompilationAudit(FrozenContract):
    """Local-only compilation diagnostics excluded from the provider schema."""

    attempts: tuple[EvidenceCompilationAttemptAudit, ...] = Field(
        default=(), max_length=2
    )
    repair: EvidenceCompilationRepairAudit

    @model_validator(mode="after")
    def validate_attempt_order(self) -> EvidenceCompilationAudit:
        attempts = [item.attempt for item in self.attempts]
        if attempts != list(range(1, len(attempts) + 1)):
            raise ValueError("compilation audit attempts must be consecutive")
        return self


class EvidenceFollowup(FrozenContract):
    """One atomic retry query for one uncovered comparison requirement."""

    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    query: str = Field(min_length=1, max_length=2000)
    objective: str = Field(min_length=1, max_length=500)

    @field_validator("query", "objective")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("follow-up text must not be blank")
        return normalized


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
    steps: tuple[ResearchStep, ...] = Field(min_length=1, max_length=24)

    @model_validator(mode="after")
    def validate_steps(self) -> ResearchPlan:
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("research plan step IDs must be unique")
        target_ids = [target.target_id for target in self.targets]
        dimension_ids = [dimension.dimension_id for dimension in self.dimensions]
        requirement_ids = [item.requirement_id for item in self.requirements]
        fact_requirement_ids = [
            fact_requirement.fact_requirement_id
            for requirement in self.requirements
            for fact_requirement in requirement.fact_requirements
        ]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("research plan target IDs must be unique")
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ValueError("research plan dimension IDs must be unique")
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("research plan requirement IDs must be unique")
        if len(fact_requirement_ids) != len(set(fact_requirement_ids)):
            raise ValueError("comparison fact requirement IDs must be globally unique")
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
    ledger: tuple[EvidenceLedgerCell, ...] = Field(default=(), max_length=20)
    compilation_visibility: tuple[EvidenceCompilationVisibility, ...] = Field(
        default=(), max_length=20
    )
    compilation_audit: SkipJsonSchema[EvidenceCompilationAudit | None] = None
    followups: tuple[EvidenceFollowup, ...] = Field(default=(), max_length=4)
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
        ledger_ids = [item.requirement_id for item in self.ledger]
        if len(ledger_ids) != len(set(ledger_ids)):
            raise ValueError("evidence ledger requirement IDs must be unique")
        visibility_ids = [
            item.requirement_id for item in self.compilation_visibility
        ]
        if len(visibility_ids) != len(set(visibility_ids)):
            raise ValueError("compilation visibility requirement IDs must be unique")
        fact_ids = [fact.fact_id for cell in self.ledger for fact in cell.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("compiled evidence fact IDs must be globally unique")
        has_next_query = self.next_query is not None
        has_next_objective = self.next_objective is not None
        if self.evidence_sufficient:
            if self.status != "sufficient":
                raise ValueError("sufficient evidence requires sufficient status")
            if (
                has_next_query
                or has_next_objective
                or self.next_requirement_ids
                or self.followups
            ):
                raise ValueError("sufficient evidence cannot request another search")
        else:
            if self.status == "sufficient":
                raise ValueError("insufficient evidence cannot use sufficient status")
            if has_next_query != has_next_objective:
                raise ValueError("next query and objective must be present together")
            if not has_next_query and self.next_requirement_ids:
                raise ValueError("next requirement IDs require another search")
            if self.followups and (
                has_next_query or has_next_objective or self.next_requirement_ids
            ):
                raise ValueError("followups cannot be mixed with legacy next-query fields")
            followup_ids = [item.requirement_id for item in self.followups]
            if len(followup_ids) != len(set(followup_ids)):
                raise ValueError("follow-up requirement IDs must be unique")
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
