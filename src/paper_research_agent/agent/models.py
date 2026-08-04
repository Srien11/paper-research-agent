"""Strict contracts for read-only research tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.ingestion.models import Sha256

StorageClass = Literal["redistributable", "internal_research_only"]
EvidenceType = Literal["text", "figure_summary"]


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchCorpusInput(FrozenContract):
    """Bounded input accepted by the corpus search tool."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=20)

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


class ResearchStep(FrozenContract):
    """One bounded corpus-search objective produced by a research planner."""

    step_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    objective: str = Field(min_length=1, max_length=500)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=20)

    @field_validator("objective", "query")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("research step text must not be blank")
        return normalized


class ResearchPlan(FrozenContract):
    """Ordered, auditable subquestions that stay within the read-only workflow."""

    schema_version: Literal["research-plan-v1"] = "research-plan-v1"
    steps: tuple[ResearchStep, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_steps(self) -> ResearchPlan:
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("research plan step IDs must be unique")
        return self


class ResearchObservation(FrozenContract):
    """One completed plan step and the exact tool results it produced."""

    step_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    objective: str = Field(min_length=1)
    search: SearchCorpusResult
    evidence: GetEvidenceResult
