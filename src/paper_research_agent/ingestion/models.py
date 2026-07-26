"""本地论文解析产物的数据契约。

解析正文属于本地产物，不得直接提交到源码仓库。
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
StorageClass = Literal["redistributable", "internal_research_only"]
PageStatus = Literal["parsed", "empty", "failed", "quarantined"]
ElementType = Literal[
    "title",
    "heading",
    "paragraph",
    "list_item",
    "table_caption",
    "figure_caption",
    "table",
    "figure",
    "formula",
    "footnote",
    "reference",
    "other",
]
ContentOrigin = Literal["source_text", "generated"]


class FrozenContract(BaseModel):
    """拒绝未版本化字段，并避免解析后被意外修改。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentAsset(FrozenContract):
    asset_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    corpus_version: str = Field(min_length=1)
    source_sha256: Sha256
    source_bytes: int = Field(gt=0)
    expected_page_count: int = Field(gt=0)
    media_type: Literal["application/pdf"] = "application/pdf"
    storage_class: StorageClass


class PageRecord(FrozenContract):
    page_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    page_number: int = Field(ge=1)
    display_label: str | None = None
    status: PageStatus
    raw_text: str | None = None
    normalized_text: str | None = None
    raw_text_sha256: Sha256 | None = None
    normalized_text_sha256: Sha256 | None = None
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    source_sha256: Sha256
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> PageRecord:
        text_values = (
            self.raw_text,
            self.normalized_text,
            self.raw_text_sha256,
            self.normalized_text_sha256,
        )
        if self.status == "parsed":
            if any(value is None for value in text_values):
                raise ValueError("parsed 页面必须包含原文、归一化文本及其哈希")
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("parsed 页面不能包含错误信息")
        elif self.status == "empty":
            if any(value is not None for value in text_values):
                raise ValueError("empty 页面不能保存正文")
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("empty 页面不能包含错误信息")
        else:
            if self.error_code is None or self.error_message is None:
                raise ValueError("失败或隔离页面必须包含错误代码和错误说明")
            if any(value is not None for value in text_values):
                raise ValueError("失败或隔离页面不能保存正文")
        return self


class SectionRecord(FrozenContract):
    section_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    parent_section_id: str | None = None
    level: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    title_raw: str = Field(min_length=1)
    title_normalized: str = Field(min_length=1)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    source_sha256: Sha256
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_page_range_and_parent(self) -> SectionRecord:
        if self.end_page < self.start_page:
            raise ValueError("章节结束页不能早于开始页")
        if self.parent_section_id == self.section_id:
            raise ValueError("章节不能把自身作为父章节")
        return self


class DocumentElement(FrozenContract):
    element_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)
    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    page_number: int = Field(ge=1)
    section_id: str | None = None
    element_type: ElementType
    reading_order: int = Field(ge=0)
    raw_text: str
    normalized_text: str
    raw_start: int | None = Field(default=None, ge=0)
    raw_end: int | None = Field(default=None, ge=0)
    bbox: tuple[float, float, float, float] | None = None
    normalized_text_sha256: Sha256
    content_origin: ContentOrigin = "source_text"
    generation_method: str | None = None
    generation_model: str | None = None
    generation_version: str | None = None
    source_sha256: Sha256
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ranges_and_origin(self) -> DocumentElement:
        if (self.raw_start is None) != (self.raw_end is None):
            raise ValueError("原文位置必须同时提供起点和终点")
        if (
            self.raw_start is not None
            and self.raw_end is not None
            and self.raw_end < self.raw_start
        ):
            raise ValueError("原文终点不能早于起点")
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            if x1 < x0 or y1 < y0:
                raise ValueError("坐标框终点不能早于起点")
        generation_values = (
            self.generation_method,
            self.generation_model,
            self.generation_version,
        )
        if self.content_origin == "generated":
            if any(value is None for value in generation_values):
                raise ValueError("生成内容必须记录完整生成血缘")
        elif any(value is not None for value in generation_values):
            raise ValueError("来源正文不能包含生成血缘")
        return self


class IngestionManifest(FrozenContract):
    schema_version: Literal["ingestion-v1"] = "ingestion-v1"
    build_id: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    parser_name: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    parser_config_sha256: Sha256
    asset_count: int = Field(ge=0)
    expected_page_count: int = Field(ge=0)
    parsed_page_count: int = Field(ge=0)
    empty_page_count: int = Field(ge=0)
    failed_page_count: int = Field(ge=0)
    quarantined_page_count: int = Field(ge=0)
    section_count: int = Field(ge=0)
    element_count: int = Field(ge=0)
    artifact_sha256: dict[str, Sha256]

    @model_validator(mode="after")
    def validate_page_counts(self) -> IngestionManifest:
        actual_total = (
            self.parsed_page_count
            + self.empty_page_count
            + self.failed_page_count
            + self.quarantined_page_count
        )
        if actual_total != self.expected_page_count:
            raise ValueError("页面状态计数之和必须等于预期页数")
        return self

