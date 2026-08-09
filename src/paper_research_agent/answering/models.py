"""Strict input, provider draft, and validated output contracts for RAG answers."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.context.models import AssembledContext
from paper_research_agent.ingestion.models import Sha256

CitationId = str
StorageClass = Literal["redistributable", "internal_research_only"]
_INLINE_CITATION = re.compile(r"\[E[1-9]\d*\]")


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnswerClaim(FrozenContract):
    """One factual claim whose citations are separate from model-written prose."""

    text: str = Field(min_length=1)
    citation_ids: tuple[CitationId, ...] = Field(min_length=1)
    fact_ids: tuple[str, ...] = ()

    @field_validator("text")
    @classmethod
    def normalize_text_and_reject_inline_citations(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("answer claim text must not be blank")
        if _INLINE_CITATION.search(normalized):
            raise ValueError("answer claim text must not contain inline citation markers")
        return normalized

    @field_validator("citation_ids")
    @classmethod
    def validate_citation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("answer claim citation_ids must be unique")
        if any(re.fullmatch(r"E[1-9]\d*", identifier) is None for identifier in value):
            raise ValueError("answer claim contains an invalid citation ID")
        return value

    @field_validator("fact_ids")
    @classmethod
    def validate_fact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("answer claim fact_ids must be unique")
        if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", item) is None for item in value):
            raise ValueError("answer claim contains an invalid fact ID")
        return value


class ProviderAnswer(FrozenContract):
    """Exact JSON schema accepted from the untrusted model response."""

    status: Literal["answered", "insufficient_evidence"]
    claims: tuple[AnswerClaim, ...]
    insufficient_reason: str | None = None

    @field_validator("insufficient_reason")
    @classmethod
    def normalize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("insufficient_reason must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_state(self) -> ProviderAnswer:
        if self.status == "answered":
            if not self.claims:
                raise ValueError("answered provider result must contain claims")
            if self.insufficient_reason is not None:
                raise ValueError("answered provider result cannot contain insufficient_reason")
        else:
            if self.claims:
                raise ValueError("insufficient provider result cannot contain claims")
            if self.insufficient_reason is None:
                raise ValueError("insufficient provider result must contain a reason")
        return self


class AnswerRequest(FrozenContract):
    """The only supported v1 generation boundary: selected evidence for private research."""

    schema_version: Literal["rag-answer-request-v1"] = "rag-answer-request-v1"
    context: AssembledContext
    output_language: Literal["zh-CN"] = "zh-CN"
    output_mode: Literal["private_research"] = "private_research"
    provider_data_policy: Literal["selected-evidence-private-research-v1"] = (
        "selected-evidence-private-research-v1"
    )

    @model_validator(mode="after")
    def validate_provider_boundary(self) -> AnswerRequest:
        messages = self.context.messages
        if not messages or messages[0].role != "system":
            raise ValueError("answer context must start with one trusted system message")
        if sum(message.role == "system" for message in messages) != 1:
            raise ValueError("answer context must contain exactly one system message")
        missing_rights = [
            citation.citation_id
            for citation in self.context.citations
            if citation.storage_class is None
        ]
        if missing_rights:
            raise ValueError(
                f"answer context has citations without loaded rights: {missing_rights}"
            )
        return self


class ComparisonTarget(FrozenContract):
    target_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    label: str = Field(min_length=1, max_length=200)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")


class ComparisonDimension(FrozenContract):
    dimension_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    label: str = Field(min_length=1, max_length=200)


class ComparisonFact(FrozenContract):
    fact_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    target_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    dimension_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    statement: str = Field(min_length=1, max_length=1200)
    fact_requirement_ids: tuple[str, ...] = Field(default=(), max_length=6)
    citation_ids: tuple[CitationId, ...] = Field(min_length=1, max_length=20)
    qualifiers: tuple[dict[str, str], ...] = ()


class ComparisonAnswerRequest(FrozenContract):
    """Trusted comparison-generation boundary containing compiled facts, never raw blocks."""

    schema_version: Literal["comparison-answer-request-v1"] = "comparison-answer-request-v1"
    question: str = Field(min_length=1, max_length=10_000)
    context: AssembledContext
    targets: tuple[ComparisonTarget, ...] = Field(min_length=2, max_length=4)
    dimensions: tuple[ComparisonDimension, ...] = Field(min_length=1, max_length=5)
    facts: tuple[ComparisonFact, ...] = Field(min_length=1, max_length=480)

    @model_validator(mode="after")
    def validate_grid_references(self) -> ComparisonAnswerRequest:
        target_ids = {item.target_id for item in self.targets}
        target_corpora = {item.target_id: item.corpus_id for item in self.targets}
        dimension_ids = {item.dimension_id for item in self.dimensions}
        context_citations = {
            item.citation_id: item.corpus_id for item in self.context.citations
        }
        fact_ids = [item.fact_id for item in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("comparison answer fact IDs must be unique")
        if any(item.target_id not in target_ids for item in self.facts):
            raise ValueError("comparison fact references an unknown target")
        if any(item.dimension_id not in dimension_ids for item in self.facts):
            raise ValueError("comparison fact references an unknown dimension")
        if any(not set(item.citation_ids) <= set(context_citations) for item in self.facts):
            raise ValueError("comparison fact references an unknown citation")
        if any(
            context_citations[citation_id] != target_corpora[item.target_id]
            for item in self.facts
            for citation_id in item.citation_ids
        ):
            raise ValueError("comparison fact citation belongs to a different target corpus")
        return self


class GenerationResult(FrozenContract):
    """Sanitized provider output and accounting metadata before validation."""

    content: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    actual_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0)
    attempts: int = Field(gt=0)


class AnswerCitation(FrozenContract):
    """Minimal citation metadata safe for local answer output."""

    citation_id: str = Field(pattern=r"^E[1-9]\d*$")
    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    asset_id: str = Field(min_length=1)
    section_id: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text_sha256: Sha256
    evidence_type: Literal["text", "figure_summary"] = "text"
    storage_class: StorageClass

    @model_validator(mode="after")
    def validate_pages(self) -> AnswerCitation:
        if self.page_end < self.page_start:
            raise ValueError("answer citation page range is reversed")
        return self


class RAGAnswer(FrozenContract):
    """Validated local answer output with no evidence body or provider payload."""

    schema_version: Literal["rag-answer-v1"] = "rag-answer-v1"
    status: Literal["answered", "insufficient_evidence"]
    answer_markdown: str = Field(min_length=1)
    claims: tuple[AnswerClaim, ...]
    citations: tuple[AnswerCitation, ...]
    output_mode: Literal["private_research"] = "private_research"
    requested_model: str = Field(min_length=1)
    actual_model: str | None = None
    prompt_version: str = Field(min_length=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0)
    attempts: int = Field(ge=0)
    audit_persisted: bool = False

    @model_validator(mode="after")
    def validate_state(self) -> RAGAnswer:
        if self.status == "answered":
            if not self.claims or not self.citations or self.actual_model is None:
                raise ValueError("answered result requires claims, citations, and actual_model")
        elif self.claims or self.citations:
            raise ValueError("insufficient result cannot contain claims or citations")
        used = {identifier for claim in self.claims for identifier in claim.citation_ids}
        returned = {citation.citation_id for citation in self.citations}
        if used != returned:
            raise ValueError("answer citations must equal the citation IDs used by claims")
        if len(returned) != len(self.citations):
            raise ValueError("answer citations must be unique")
        return self
