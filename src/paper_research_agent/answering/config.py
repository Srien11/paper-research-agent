"""Versioned configuration for grounded RAG answer generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnsweringConfig(FrozenConfig):
    """One reproducible factual-answering profile."""

    schema_version: Literal["rag-answering-v1"] = "rag-answering-v1"
    model: str = "qwen3.7-plus-2026-05-26"
    prompt_version: Literal["rag-answer-json-v1"] = "rag-answer-json-v1"
    temperature: float = Field(default=0.1, ge=0, le=2)
    top_p: float = Field(default=0.7, gt=0, le=1)
    max_output_tokens: int = Field(default=1200, gt=0, le=8192)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=5)
    enable_thinking: Literal[False] = False

    @field_validator("model")
    @classmethod
    def require_pinned_model_version(cls, value: str) -> str:
        normalized = value.strip()
        if not re.search(r"-\d{4}-\d{2}-\d{2}$", normalized):
            raise ValueError("answering model must use a dated immutable version")
        return normalized


def load_answering_config(path: Path) -> AnsweringConfig:
    return AnsweringConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
