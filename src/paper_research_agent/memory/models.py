"""Strict contracts for local short-term memory without evidence bodies."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.ingestion.models import Sha256

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
StorageClass = Literal["redistributable", "internal_research_only"]
MAX_MEMORY_CLAIM_CHARS = 1000
MAX_MEMORY_CLAIMS_CHARS = 3000


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def normalize_session_id(value: str) -> str:
    normalized = value.strip()
    if SESSION_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError("session_id contains unsafe characters or has an invalid length")
    return normalized


class MemorySourceRef(FrozenContract):
    """Stable source lineage retained without source text or per-turn citation labels."""

    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    text_sha256: Sha256
    storage_class: StorageClass


class ShortTermMemoryTurn(FrozenContract):
    """One completed, locally persisted question-answer turn."""

    schema_version: Literal["short-term-memory-v1"] = "short-term-memory-v1"
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    session_id: str
    created_at: datetime
    expires_at: datetime
    user_question: str = Field(min_length=1)
    standalone_question: str = Field(min_length=1, max_length=2000)
    status: Literal["answered", "insufficient_evidence", "compiler_failed"]
    assistant_claims: tuple[str, ...] = ()
    source_refs: tuple[MemorySourceRef, ...] = ()

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return normalize_session_id(value)

    @field_validator("user_question", "standalone_question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("user_question must not be blank")
        return normalized

    @field_validator("assistant_claims")
    @classmethod
    def normalize_claims(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("assistant claims must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("assistant claims must be unique")
        if any(len(value) > MAX_MEMORY_CLAIM_CHARS for value in normalized):
            raise ValueError("a memory claim exceeds the local persistence limit")
        if sum(len(value) for value in normalized) > MAX_MEMORY_CLAIMS_CHARS:
            raise ValueError("memory claims exceed the local persistence limit")
        if any(re.search(r"\[E[1-9]\d*\]", value) for value in normalized):
            raise ValueError("short-term memory cannot retain per-turn citation labels")
        return normalized

    @model_validator(mode="after")
    def validate_turn(self) -> ShortTermMemoryTurn:
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("memory timestamps must be timezone-aware")
        if self.expires_at <= self.created_at:
            raise ValueError("memory expiration must be after creation")
        if self.status == "answered" and not self.assistant_claims:
            raise ValueError("answered memory turn requires validated claims")
        if self.status != "answered" and (self.assistant_claims or self.source_refs):
            raise ValueError("non-answer memory turn cannot retain claims or sources")
        chunk_ids = [source.chunk_id for source in self.source_refs]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("memory source chunk IDs must be unique")
        return self
