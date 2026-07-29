"""图片语义信息的数据契约。

图片裁剪和视觉模型生成结果属于本地产物，不得直接提交到源码仓库。
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FigureRecord(BaseModel):
    """一张论文图片及其视觉模型生成语义。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    figure_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    figure_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    caption: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    figure_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_findings: tuple[str, ...]
    recognition_confidence: float = Field(ge=0, le=1)
    content_origin: Literal["视觉模型生成"] = "视觉模型生成"
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bbox_path_and_findings(self) -> FigureRecord:
        x0, y0, x1, y1 = self.bbox
        if x1 <= x0 or y1 <= y0:
            raise ValueError("图片坐标框必须具有正面积")

        path = PurePosixPath(self.image_path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.image_path:
            raise ValueError("image_path 必须是使用正斜杠的安全相对路径")
        if path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("image_path 必须指向支持的图片格式")

        if any(not finding.strip() for finding in self.key_findings):
            raise ValueError("key_findings 不能包含空项")
        return self
