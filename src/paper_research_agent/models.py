"""Stable contracts for frozen corpus metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FrozenPaper(BaseModel):
    """Minimum trustworthy metadata required before document ingestion."""

    model_config = ConfigDict(extra="allow")

    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    corpus_version: str
    dataset_split: Literal["core", "challenge"]
    canonical_key: str
    title: str = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
    authors: list[str] = Field(min_length=1)
    official_url: str
    fulltext_url: str
    selection_status: Literal["frozen"]
    content_status: Literal["downloaded_and_parse_verified"]
    storage_class: Literal["redistributable", "internal_research_only"]
    local_pdf_path: Path
    download_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    download_bytes: int = Field(gt=0)
    pdf_pages: int = Field(gt=0)
    parse_quality_status: Literal["machine_parse_pass", "visual_review_pass"]

    @field_validator("official_url", "fulltext_url")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("must be an HTTP(S) URL")
        return value


class CorpusReport(BaseModel):
    """Deterministic summary emitted by the corpus validation gate."""

    corpus_version: str
    paper_count: int
    core_count: int
    challenge_count: int
    redistributable_count: int
    internal_research_only_count: int
    total_pages: int
    canonical_key_count: int
    local_pdf_count: int

