"""Strict contracts for traceable, layered RAG context."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.ingestion.models import Sha256


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextEvidence(FrozenContract):
    """A retrieval hit joined to its immutable source text."""

    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    asset_id: str = Field(min_length=1)
    section_id: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str = Field(min_length=1)
    text_sha256: Sha256
    final_score: float
    final_rank: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> ContextEvidence:
        if self.page_end < self.page_start:
            raise ValueError("context evidence page range is reversed")
        actual_sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_sha256 != actual_sha256:
            raise ValueError("context evidence text_sha256 does not match text")
        return self


class CitationRef(FrozenContract):
    """Public citation metadata that maps an answer marker to one exact chunk."""

    citation_id: str = Field(pattern=r"^E[1-9]\d*$")
    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    asset_id: str = Field(min_length=1)
    section_id: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text_sha256: Sha256

    @model_validator(mode="after")
    def validate_pages(self) -> CitationRef:
        if self.page_end < self.page_start:
            raise ValueError("citation page range is reversed")
        return self


class PromptMessage(FrozenContract):
    """One model-facing message with an explicitly constrained role."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ContextRequest(FrozenContract):
    """Inputs to deterministic context assembly."""

    system_rules: str = Field(min_length=1)
    user_question: str = Field(min_length=1)
    evidence: tuple[ContextEvidence, ...]
    task_state: str | None = None
    conversation_history: tuple[PromptMessage, ...] = ()
    token_budget: int = Field(gt=0)
    output_reserve_tokens: int = Field(default=0, ge=0)

    @field_validator("system_rules", "user_question")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("required context text must not be blank")
        return value

    @field_validator("task_state")
    @classmethod
    def reject_blank_task_state(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("task_state must be omitted rather than blank")
        return value

    @model_validator(mode="after")
    def validate_traceability(self) -> ContextRequest:
        chunk_ids = [item.chunk_id for item in self.evidence]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("evidence chunk_ids must be unique")
        final_ranks = [item.final_rank for item in self.evidence]
        if len(set(final_ranks)) != len(final_ranks):
            raise ValueError("evidence final_ranks must be unique")
        if any(message.role == "system" for message in self.conversation_history):
            raise ValueError("conversation_history cannot contain system messages")
        if self.output_reserve_tokens >= self.token_budget:
            raise ValueError("output reserve must be smaller than token budget")
        return self


class AssembledContext(FrozenContract):
    """Complete model input plus its independently verifiable citation map."""

    messages: tuple[PromptMessage, ...]
    citations: tuple[CitationRef, ...]
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(gt=0)
    output_reserve_tokens: int = Field(default=0, ge=0)
    omitted_evidence_count: int = Field(ge=0)
    evidence_insufficient: bool = False

    @model_validator(mode="after")
    def validate_citations(self) -> AssembledContext:
        citation_ids = [citation.citation_id for citation in self.citations]
        if len(set(citation_ids)) != len(citation_ids):
            raise ValueError("citation_ids must be unique")
        chunk_ids = [citation.chunk_id for citation in self.citations]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("citation chunk_ids must be unique")
        if self.estimated_tokens + self.output_reserve_tokens > self.token_budget:
            raise ValueError("estimated tokens exceed token budget")
        return self
