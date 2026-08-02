"""Versioned configuration for bounded local short-term memory."""

from __future__ import annotations

import json
from pathlib import Path, PurePath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class ShortTermMemoryConfig(BaseModel):
    """One reproducible policy for session memory and context projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["short-term-memory-config-v1"] = "short-term-memory-config-v1"
    ttl_hours: int = Field(default=24, gt=0, le=24 * 30)
    max_turns_per_session: int = Field(default=20, gt=0, le=100)
    context_turn_limit: int = Field(default=6, ge=0, le=20)
    context_token_budget: int = Field(default=1200, ge=0, le=4096)
    protected_evidence_count: int = Field(default=3, gt=0, le=10)
    follow_up_max_chars: int = Field(default=96, gt=0, le=500)
    store_path: Path = Path("data/runtime/short-term-memory-v1.sqlite3")

    @field_validator("store_path")
    @classmethod
    def require_safe_runtime_path(cls, value: Path) -> Path:
        path = PurePath(str(value))
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 3
            or path.parts[:2] != ("data", "runtime")
            or path.suffix != ".sqlite3"
        ):
            raise ValueError("memory store must be a project-relative SQLite file in data/runtime")
        return Path(*path.parts)

    @field_serializer("store_path")
    def serialize_store_path(self, value: Path) -> str:
        return value.as_posix()

    @model_validator(mode="after")
    def validate_turn_limits(self) -> ShortTermMemoryConfig:
        if self.context_turn_limit > self.max_turns_per_session:
            raise ValueError("context_turn_limit cannot exceed max_turns_per_session")
        return self


def load_memory_config(path: Path) -> ShortTermMemoryConfig:
    return ShortTermMemoryConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
