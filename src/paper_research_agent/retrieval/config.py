"""Versioned configuration contracts for the retrieval baseline."""

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

    @field_serializer("output_dir")
    def serialize_output_dir(self, value: Path) -> str:
        return value.as_posix()

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

    @field_serializer("index_dir")
    def serialize_index_dir(self, value: Path) -> str:
        return value.as_posix()

    @model_validator(mode="after")
    def require_valid_candidate_counts(self) -> RetrievalConfig:
        if self.top_k > self.rerank_candidates:
            raise ValueError("top_k cannot exceed rerank_candidates")
        if self.rerank_candidates > self.sparse_candidates + self.vector_candidates:
            raise ValueError("rerank_candidates exceeds available candidates")
        return self


class BilingualRetrievalConfig(FrozenConfig):
    """Online Chinese-to-English retrieval orchestration settings."""

    schema_version: Literal["bilingual-retrieval-v1"] = "bilingual-retrieval-v1"
    pipeline_id: str = Field(default="zh-en-two-level-rrf-v1", min_length=1)
    rewrite_model: str = Field(default="qwen3.7-plus-2026-05-26", min_length=1)
    rewrite_prompt_version: Literal["query-rewrite-v2"] = "query-rewrite-v2"
    rewrite_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    rewrite_cache_fresh_days: int = Field(default=90, gt=0)
    rewrite_cache_stale_days: int = Field(default=365, gt=0)
    audit_plaintext_days: int = Field(default=30, ge=0)
    route_rrf_k: int = Field(default=60, gt=0)
    cache_path: Path = Path("data/runtime/query-rewrite-v2.sqlite3")
    audit_path: Path = Path("data/runtime/query-audit-v1.sqlite3")

    @field_validator("cache_path", "audit_path")
    @classmethod
    def require_safe_runtime_path(cls, value: Path) -> Path:
        return _safe_project_relative_path(value)

    @field_serializer("cache_path", "audit_path")
    def serialize_runtime_path(self, value: Path) -> str:
        return value.as_posix()

    @model_validator(mode="after")
    def require_stale_window_after_fresh_window(self) -> BilingualRetrievalConfig:
        if self.rewrite_cache_stale_days < self.rewrite_cache_fresh_days:
            raise ValueError("rewrite_cache_stale_days cannot be shorter than fresh days")
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


def load_bilingual_retrieval_config(path: Path) -> BilingualRetrievalConfig:
    return BilingualRetrievalConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
