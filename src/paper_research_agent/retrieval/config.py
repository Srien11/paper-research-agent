"""Versioned configuration contracts for the retrieval baseline."""

from __future__ import annotations

import json
from pathlib import Path, PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkingConfig(FrozenConfig):
    schema_version: Literal["chunking-v1"] = "chunking-v1"
    tokenizer: Literal["regex-token-v1"] = "regex-token-v1"
    max_tokens: int = Field(default=512, ge=32, le=4096)
    overlap_tokens: int = Field(default=64, ge=0)
    output_dir: Path = Path("data/processed/chunks")

    @field_validator("output_dir")
    @classmethod
    def require_safe_local_output(cls, value: Path) -> Path:
        return _safe_project_relative_path(value)

    @model_validator(mode="after")
    def require_overlap_smaller_than_chunk(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class RetrievalConfig(FrozenConfig):
    schema_version: Literal["retrieval-v1"] = "retrieval-v1"
    embedding_model: str
    embedding_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    reranker_model: str
    reranker_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    rrf_k: int = Field(default=60, gt=0)
    sparse_candidates: int = Field(default=50, gt=0)
    vector_candidates: int = Field(default=50, gt=0)
    rerank_candidates: int = Field(default=30, gt=0)
    top_k: int = Field(default=10, gt=0)
    index_dir: Path = Path("data/indexes/retrieval-v1")

    @field_validator("embedding_model", "reranker_model")
    @classmethod
    def require_model_name(cls, value: str) -> str:
        if "/" not in value or value.startswith("/") or value.endswith("/"):
            raise ValueError("model must be a versioned repository name")
        return value

    @field_validator("index_dir")
    @classmethod
    def require_safe_local_index(cls, value: Path) -> Path:
        return _safe_project_relative_path(value)

    @model_validator(mode="after")
    def require_valid_candidate_counts(self) -> RetrievalConfig:
        if self.top_k > self.rerank_candidates:
            raise ValueError("top_k cannot exceed rerank_candidates")
        if self.rerank_candidates > self.sparse_candidates + self.vector_candidates:
            raise ValueError("rerank_candidates exceeds available candidates")
        return self


def _safe_project_relative_path(value: Path) -> Path:
    raw = str(value)
    path = PurePath(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "data":
        raise ValueError("artifact path must be project-relative and inside data/")
    return Path(*path.parts)


def load_chunking_config(path: Path) -> ChunkingConfig:
    return ChunkingConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_retrieval_config(path: Path) -> RetrievalConfig:
    return RetrievalConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
