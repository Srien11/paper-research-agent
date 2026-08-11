"""Strict, persistence-safe outputs returned by main-Agent child executors."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from paper_research_agent.answering.models import RAGAnswer


class FrozenArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChildExecutionMetrics(FrozenArtifact):
    """Safe timing, token, and context counters exposed by a child run."""

    elapsed_ms: int = Field(default=0, ge=0)
    first_token_ms: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_context_tokens: int = Field(default=0, ge=0)
    token_budget: int = Field(default=0, ge=0)
    output_reserve_tokens: int = Field(default=0, ge=0)


class ChildArtifactBase(FrozenArtifact):
    text: str = Field(default="", max_length=20_000)
    source_ids: tuple[str, ...] = Field(default=(), max_length=100)
    metrics: ChildExecutionMetrics = Field(default_factory=ChildExecutionMetrics)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("artifact source IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("artifact source IDs must be unique")
        return normalized


class ChatArtifact(ChildArtifactBase):
    kind: Literal["chat"] = "chat"


class LocalRAGTrace(FrozenArtifact):
    """Minimal retrieval metadata safe to persist above the RAG runtime."""

    index_id: str = Field(min_length=1, max_length=256)
    resolved_question_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    degraded: bool = False
    hit_count: int = Field(ge=0)


class LocalRAGArtifact(ChildArtifactBase):
    kind: Literal["local_rag"] = "local_rag"
    answer: RAGAnswer
    retrieval: LocalRAGTrace


class PendingApprovalArtifact(FrozenArtifact):
    tool_name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    purpose: str = Field(min_length=1, max_length=500)
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at_epoch: float = Field(gt=0)


class DynamicToolArtifact(ChildArtifactBase):
    kind: Literal["dynamic_tools"] = "dynamic_tools"
    tool_names: tuple[str, ...] = Field(default=(), max_length=20)
    pending_approval: PendingApprovalArtifact | None = None

    @field_validator("tool_names")
    @classmethod
    def validate_tool_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("artifact tool names must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("artifact tool names must be unique")
        return normalized


class AttachmentArtifact(ChildArtifactBase):
    kind: Literal["attachment_qa"] = "attachment_qa"
    source_attachment_ids: tuple[str, ...] = Field(default=(), max_length=5)

    @field_validator("source_attachment_ids")
    @classmethod
    def validate_attachment_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("source attachment IDs must be unique")
        return values


class FileArtifact(ChildArtifactBase):
    kind: Literal["file_edit"] = "file_edit"
    output_attachment_ids: tuple[str, ...] = Field(min_length=1, max_length=5)

    @field_validator("output_attachment_ids")
    @classmethod
    def validate_output_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("output attachment IDs must be unique")
        return values


ChildArtifact = Annotated[
    ChatArtifact
    | LocalRAGArtifact
    | DynamicToolArtifact
    | AttachmentArtifact
    | FileArtifact,
    Field(discriminator="kind"),
]
