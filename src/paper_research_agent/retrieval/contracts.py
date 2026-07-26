"""Stable retrieval index and result contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research_agent.ingestion.models import Sha256


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IndexManifest(FrozenContract):
    schema_version: Literal["retrieval-index-v1"] = "retrieval-index-v1"
    index_id: str = Field(min_length=1)
    chunk_build_sha256: Sha256
    chunk_count: int = Field(ge=0)
    embedding_model: str = Field(min_length=1)
    embedding_revision: str = Field(min_length=1)
    vector_dimension: int = Field(gt=0)
    faiss_factory: Literal["IndexFlatIP"] = "IndexFlatIP"
    vector_normalized: Literal[True] = True
    files_sha256: dict[str, Sha256]
    cpu_fingerprint: str = Field(min_length=1)


class SearchHit(FrozenContract):
    chunk_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    asset_id: str = Field(min_length=1)
    section_id: str | None = None
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text_sha256: Sha256
    scores: dict[str, float] = {}
    ranks: dict[str, int] = {}
    final_score: float
    final_rank: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_pages(self) -> SearchHit:
        if self.page_end < self.page_start:
            raise ValueError("search hit page range is reversed")
        if any(rank <= 0 for rank in self.ranks.values()):
            raise ValueError("stage ranks must be positive")
        return self


class RetrievalRun(FrozenContract):
    schema_version: Literal["retrieval-run-v1"] = "retrieval-run-v1"
    query: str = Field(min_length=1)
    variant: Literal["A", "B", "C"]
    top_k: int = Field(gt=0)
    hits: tuple[SearchHit, ...]
    index_id: str = Field(min_length=1)
    config_sha256: Sha256

    @model_validator(mode="after")
    def validate_rank_order(self) -> RetrievalRun:
        expected = list(range(1, len(self.hits) + 1))
        if [hit.final_rank for hit in self.hits] != expected:
            raise ValueError("final ranks must be contiguous and ordered")
        if len(self.hits) > self.top_k:
            raise ValueError("hit count exceeds top_k")
        return self
