"""Strict inputs and common outputs for all extended research tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    status: Literal["ok", "not_found", "insufficient", "approval_required", "denied"] = "ok"
    trust: Literal["citation_evidence", "research_context", "computed_result", "side_effect"] = (
        "research_context"
    )
    items: tuple[dict[str, Any], ...] = ()
    summary: dict[str, Any] = Field(default_factory=dict)


class AdjacentChunksInput(ToolInput):
    chunk_id: str = Field(min_length=1, max_length=256)
    before: int = Field(default=1, ge=0, le=3)
    after: int = Field(default=1, ge=0, le=3)


class PaperMetadataInput(ToolInput):
    corpus_ids: tuple[str, ...] = Field(min_length=1, max_length=20)


class ChunkIdsInput(ToolInput):
    chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=20)


class CorpusInput(ToolInput):
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")


class ScholarlySearchInput(ToolInput):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=20)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)

    @model_validator(mode="after")
    def validate_years(self) -> ScholarlySearchInput:
        if self.year_from and self.year_to and self.year_to < self.year_from:
            raise ValueError("year range is reversed")
        return self


class IdentifierInput(ToolInput):
    identifier: str = Field(min_length=1, max_length=500)


class CitationGraphInput(IdentifierInput):
    direction: Literal["references", "citations", "both"] = "both"
    limit: int = Field(default=20, ge=1, le=50)


class ElementLookupInput(CorpusInput):
    page: int | None = Field(default=None, ge=1)
    label: str | None = Field(default=None, min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=20)


class CalculateInput(ToolInput):
    expression: str = Field(min_length=1, max_length=500)


class AnalyzeExperimentDataInput(ToolInput):
    columns: tuple[str, ...] = Field(min_length=1, max_length=20)
    rows: tuple[tuple[float, ...], ...] = Field(min_length=1, max_length=1000)
    operations: tuple[Literal["count", "mean", "median", "stdev", "min", "max"], ...] = (
        "count",
        "mean",
    )

    @model_validator(mode="after")
    def validate_shape(self) -> AnalyzeExperimentDataInput:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("analysis columns must be unique")
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("analysis rows do not match columns")
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("analysis operations must be unique")
        return self


class VerifyClaimInput(ToolInput):
    claim: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=20)


class ApprovedWriteInput(ToolInput):
    approval_token: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SaveResearchNoteInput(ApprovedWriteInput):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=20_000)
    source_chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)


class ExportResearchReportInput(ApprovedWriteInput):
    relative_path: str = Field(min_length=1, max_length=240)
    format: Literal["markdown", "json"]
    content: str = Field(min_length=1, max_length=200_000)
    overwrite: bool = False


class ManageLongTermMemoryInput(ApprovedWriteInput):
    action: Literal["add", "search", "list", "update", "delete"]
    memory_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    kind: Literal["preference", "project_context", "confirmed_conclusion"] | None = None
    content: str | None = Field(default=None, min_length=1, max_length=3000)
    source_chunk_ids: tuple[str, ...] = Field(default=(), max_length=20)
    query: str | None = Field(default=None, min_length=1, max_length=500)
    expires_at: str | None = Field(default=None, max_length=64)
    scope_id: str = Field(default="global", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("content", "query")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory text cannot be blank")
        return normalized

    @field_validator("expires_at")
    @classmethod
    def validate_expiry(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("memory expiry must include a timezone")
        return parsed.isoformat()

    @model_validator(mode="after")
    def validate_action(self) -> ManageLongTermMemoryInput:
        if self.action == "add" and (self.kind is None or self.content is None):
            raise ValueError("add requires kind and content")
        if self.action == "update" and (self.memory_id is None or self.content is None):
            raise ValueError("update requires memory_id and content")
        if self.action in {"update", "delete"} and self.memory_id is None:
            raise ValueError(f"{self.action} requires memory_id")
        if self.action == "search" and self.query is None:
            raise ValueError("search requires query")
        if (
            self.action == "add"
            and self.kind == "confirmed_conclusion"
            and not self.source_chunk_ids
        ):
            raise ValueError("confirmed conclusions require source_chunk_ids")
        return self


TOOL_INPUT_SCHEMAS: dict[str, type[ToolInput]] = {
    "get_adjacent_chunks": AdjacentChunksInput,
    "get_paper_metadata": PaperMetadataInput,
    "trace_evidence_source": ChunkIdsInput,
    "get_paper_outline": CorpusInput,
    "search_scholarly_sources": ScholarlySearchInput,
    "resolve_paper_identifier": IdentifierInput,
    "get_citation_graph": CitationGraphInput,
    "check_paper_status": IdentifierInput,
    "extract_table": ElementLookupInput,
    "inspect_figure": ElementLookupInput,
    "extract_equation": ElementLookupInput,
    "calculate": CalculateInput,
    "analyze_experiment_data": AnalyzeExperimentDataInput,
    "verify_claim": VerifyClaimInput,
    "check_reproducibility": CorpusInput,
    "save_research_note": SaveResearchNoteInput,
    "export_research_report": ExportResearchReportInput,
    "manage_long_term_memory": ManageLongTermMemoryInput,
}
