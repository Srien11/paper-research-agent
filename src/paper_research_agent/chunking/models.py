"""Contracts for traceable evidence chunks and generated paper cards."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research_agent.figures.models import FigureRecord
from paper_research_agent.ingestion.models import Sha256


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceChunk(FrozenContract):
    schema_version: Literal["evidence-chunk-v2"] = "evidence-chunk-v2"
    chunk_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    section_id: str | None = None
    element_ids: tuple[str, ...] = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    token_start: int = Field(ge=0)
    token_end: int = Field(gt=0)
    tokenizer: Literal["regex-token-v1"] = "regex-token-v1"
    text: str = Field(min_length=1)
    text_sha256: Sha256
    config_sha256: Sha256
    evidence_type: Literal["text", "figure_summary"] = "text"
    content_origin: Literal["source_text", "generated"] = "source_text"
    figure: FigureRecord | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> EvidenceChunk:
        if self.page_end < self.page_start:
            raise ValueError("chunk page range is reversed")
        if self.token_end <= self.token_start:
            raise ValueError("chunk token range is empty or reversed")
        if len(set(self.element_ids)) != len(self.element_ids):
            raise ValueError("element_ids must be unique and ordered")
        if self.evidence_type == "figure_summary":
            if self.figure is None or self.content_origin != "generated":
                raise ValueError("图片摘要块必须包含生成的图片记录")
            if self.asset_id != self.figure.asset_id:
                raise ValueError("图片摘要块与图片记录的 asset_id 不一致")
            if self.page_start != self.page_end or self.page_start != self.figure.page_number:
                raise ValueError("图片摘要块必须精确定位到图片所在页")
            if self.element_ids != (self.figure.figure_id,):
                raise ValueError("图片摘要块必须用 figure_id 作为来源记录")
        elif self.figure is not None or self.content_origin != "source_text":
            raise ValueError("正文块不能携带生成的图片记录")
        return self


class PaperCard(FrozenContract):
    schema_version: Literal["paper-card-v1"] = "paper-card-v1"
    card_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    title: str = Field(min_length=1)
    abstract: str | None = None
    evidence_chunk_ids: tuple[str, ...] = Field(min_length=1)
    content_origin: Literal["generated"] = "generated"
    generation_method: Literal["deterministic-card-v1"] = "deterministic-card-v1"
    source_element_ids: tuple[str, ...] = Field(min_length=1)
    config_sha256: Sha256
