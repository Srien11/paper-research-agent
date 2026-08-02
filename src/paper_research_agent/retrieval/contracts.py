"""Stable retrieval index and result contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research_agent.figures.models import FigureRecord
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
    evidence_type: Literal["text", "figure_summary"] = "text"
    figure: FigureRecord | None = None
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
        if self.evidence_type == "figure_summary" and self.figure is None:
            raise ValueError("图片摘要命中必须携带完整图片记录")
        if self.evidence_type == "text" and self.figure is not None:
            raise ValueError("正文命中不能携带图片记录")
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


class QueryRewriteTrace(FrozenContract):
    status: Literal["success", "cache_hit", "stale_cache", "timeout", "error"]
    english_query: str | None = None
    requested_model: str = Field(min_length=1)
    actual_model: str | None = None
    prompt_version: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    error_class: str | None = None
    fallback_reason: Literal["timeout", "error"] | None = None
    cache_error_class: str | None = None

    @model_validator(mode="after")
    def require_query_for_success(self) -> QueryRewriteTrace:
        successful = self.status in {"success", "cache_hit", "stale_cache"}
        if successful != bool(self.english_query and self.english_query.strip()):
            raise ValueError("successful rewrite status must carry a non-empty English query")
        if self.status == "stale_cache" and self.fallback_reason is None:
            raise ValueError("stale cache use must record why the provider path failed")
        return self


class BilingualRetrievalRun(FrozenContract):
    schema_version: Literal["bilingual-retrieval-run-v1"] = "bilingual-retrieval-run-v1"
    pipeline_id: str = Field(min_length=1)
    original_query: str = Field(min_length=1)
    rewrite: QueryRewriteTrace
    degraded: bool
    degraded_reason: str | None = None
    top_k: int = Field(gt=0)
    hits: tuple[SearchHit, ...]
    index_id: str = Field(min_length=1)
    config_sha256: Sha256
    storage_classes: dict[str, Literal["redistributable", "internal_research_only"]] = Field(
        default_factory=dict
    )
    rights_status: Literal["loaded", "not_loaded"]
    audit_persisted: bool = False

    @model_validator(mode="after")
    def validate_bilingual_run(self) -> BilingualRetrievalRun:
        expected = list(range(1, len(self.hits) + 1))
        if [hit.final_rank for hit in self.hits] != expected:
            raise ValueError("final ranks must be contiguous and ordered")
        if len(self.hits) > self.top_k:
            raise ValueError("hit count exceeds top_k")
        if self.degraded != (self.rewrite.status in {"timeout", "error", "stale_cache"}):
            raise ValueError("degraded flag must match rewrite failure status")
        if self.degraded != bool(self.degraded_reason):
            raise ValueError("degraded_reason must be present exactly for degraded runs")
        hit_corpus_ids = {hit.corpus_id for hit in self.hits}
        if self.rights_status == "loaded" and set(self.storage_classes) != hit_corpus_ids:
            raise ValueError("loaded storage rights must cover every hit corpus exactly")
        if self.rights_status == "not_loaded" and self.storage_classes:
            raise ValueError("storage_classes must be empty when rights are not loaded")
        return self
