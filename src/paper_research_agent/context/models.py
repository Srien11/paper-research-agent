"""Strict contracts for traceable, layered RAG context."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.figures.models import FigureRecord
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
    evidence_type: Literal["text", "figure_summary"] = "text"
    figure: FigureRecord | None = None
    storage_class: Literal["redistributable", "internal_research_only"] | None = None
    final_score: float
    final_rank: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> ContextEvidence:
        if self.page_end < self.page_start:
            raise ValueError("context evidence page range is reversed")
        actual_sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_sha256 != actual_sha256:
            raise ValueError("context evidence text_sha256 does not match text")
        if self.evidence_type == "figure_summary" and self.figure is None:
            raise ValueError("图片摘要证据必须携带完整图片记录")
        if self.evidence_type == "text" and self.figure is not None:
            raise ValueError("正文证据不能携带图片记录")
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
    evidence_type: Literal["text", "figure_summary"] = "text"
    figure: FigureRecord | None = None
    storage_class: Literal["redistributable", "internal_research_only"] | None = None

    @model_validator(mode="after")
    def validate_pages(self) -> CitationRef:
        if self.page_end < self.page_start:
            raise ValueError("citation page range is reversed")
        if self.evidence_type == "figure_summary" and self.figure is None:
            raise ValueError("图片摘要引用必须携带完整图片记录")
        if self.evidence_type == "text" and self.figure is not None:
            raise ValueError("正文引用不能携带图片记录")
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


class ContextMemoryTurn(FrozenContract):
    """Low-trust conversational continuity projected without evidence or old citations."""

    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    user_question: str = Field(min_length=1)
    status: Literal["answered", "insufficient_evidence", "compiler_failed"]
    assistant_claims: tuple[str, ...] = ()

    @field_validator("user_question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory question must not be blank")
        return normalized

    @field_validator("assistant_claims")
    @classmethod
    def reject_old_citation_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("memory claims must not be blank")
        if any(re.search(r"\[E[1-9]\d*\]", value) for value in normalized):
            raise ValueError("memory claims cannot contain old citation labels")
        return normalized

    @model_validator(mode="after")
    def validate_status(self) -> ContextMemoryTurn:
        if self.status == "answered" and not self.assistant_claims:
            raise ValueError("answered memory requires claims")
        if self.status != "answered" and self.assistant_claims:
            raise ValueError("non-answer memory cannot contain claims")
        return self


class ContextLongTermMemory(FrozenContract):
    """Selected low-trust durable memory that can guide, but never prove, an answer."""

    memory_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    kind: Literal["preference", "project_context", "confirmed_conclusion"]
    content: str = Field(min_length=1, max_length=3000)
    relevance: float = Field(ge=0, le=1)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("long-term memory content must not be blank")
        return normalized


class ContextRequest(FrozenContract):
    """Inputs to deterministic context assembly."""

    system_rules: str = Field(min_length=1)
    user_question: str = Field(min_length=1)
    standalone_question: str | None = None
    evidence: tuple[ContextEvidence, ...]
    task_state: str | None = None
    allow_partial_answer: bool = False
    conversation_history: tuple[PromptMessage, ...] = ()
    short_term_memory: tuple[ContextMemoryTurn, ...] = ()
    memory_token_budget: int = Field(default=0, ge=0)
    long_term_memory: tuple[ContextLongTermMemory, ...] = ()
    long_term_memory_token_budget: int = Field(default=0, ge=0)
    protected_evidence_count: int = Field(default=1, gt=0, le=10)
    token_budget: int = Field(gt=0)
    output_reserve_tokens: int = Field(default=0, ge=0)

    @field_validator("system_rules", "user_question", "standalone_question")
    @classmethod
    def reject_blank_required_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
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
        memory_ids = [turn.turn_id for turn in self.short_term_memory]
        if len(set(memory_ids)) != len(memory_ids):
            raise ValueError("short-term memory turn IDs must be unique")
        long_term_memory_ids = [item.memory_id for item in self.long_term_memory]
        if len(set(long_term_memory_ids)) != len(long_term_memory_ids):
            raise ValueError("long-term memory IDs must be unique")
        if self.output_reserve_tokens >= self.token_budget:
            raise ValueError("output reserve must be smaller than token budget")
        if self.memory_token_budget >= self.token_budget:
            raise ValueError("memory token budget must be smaller than total token budget")
        if self.long_term_memory_token_budget >= self.token_budget:
            raise ValueError(
                "long-term memory token budget must be smaller than total token budget"
            )
        return self


class AssembledContext(FrozenContract):
    """Complete model input plus its independently verifiable citation map."""

    messages: tuple[PromptMessage, ...]
    citations: tuple[CitationRef, ...]
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(gt=0)
    output_reserve_tokens: int = Field(default=0, ge=0)
    omitted_evidence_count: int = Field(ge=0)
    included_memory_turn_ids: tuple[str, ...] = ()
    omitted_memory_turn_count: int = Field(default=0, ge=0)
    included_long_term_memory_ids: tuple[str, ...] = ()
    omitted_long_term_memory_count: int = Field(default=0, ge=0)
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
