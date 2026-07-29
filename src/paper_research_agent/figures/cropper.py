"""根据图注和 PDF 图形对象定位并裁剪论文图片。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber

from paper_research_agent.figures.identity import make_figure_id

_FIGURE_NAME = re.compile(
    r"^\s*((?:figure|fig\.?)\s*[A-Za-z]?\d+(?:[.\-][A-Za-z0-9]+)?)",
    re.IGNORECASE,
)
_GRAPHIC_OBJECT_TYPES = ("image", "curve", "rect", "line")
_CROP_PADDING = 8.0

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class FigureCrop:
    figure_id: str
    asset_id: str
    corpus_id: str
    caption_element_id: str
    figure_name: str
    page_number: int
    bbox: BBox
    caption: str
    image_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "figure_id": self.figure_id,
            "asset_id": self.asset_id,
            "corpus_id": self.corpus_id,
            "caption_element_id": self.caption_element_id,
            "figure_name": self.figure_name,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "caption": self.caption,
            "image_path": self.image_path,
        }


def extract_figure_name(caption: str) -> str:
    """从图注开头读取论文中的图编号。"""

    match = _FIGURE_NAME.match(caption)
    if match is None:
        raise ValueError(f"无法从图注提取图片名称: {caption[:80]}")
    return match.group(1).strip()


def detect_figure_bbox(
    page: object,
    caption_bbox: BBox,
) -> BBox:
    """选择图注上方最近的图形对象组，并返回 PDF 点坐标。"""

    page_width = float(page.width)  # type: ignore[attr-defined]
    page_height = float(page.height)  # type: ignore[attr-defined]
    caption_x0, caption_top, caption_x1, _ = caption_bbox
    band_x0, band_x1 = _caption_column_band(
        caption_x0,
        caption_x1,
        page_width,
    )
    objects = _graphic_bboxes(page.objects)  # type: ignore[attr-defined]
    candidates = [
        bbox
        for bbox in objects
        if bbox[3] <= caption_top + 2
        and bbox[1] >= page_height * 0.035
        and _horizontal_overlap_ratio(bbox, band_x0, band_x1) >= 0.3
        and not _is_page_sized_decoration(bbox, page_width, page_height)
    ]
    groups = _group_vertically(
        candidates,
        max_gap=max(24.0, min(72.0, page_height * 0.085)),
    )
    usable = [
        group
        for group in groups
        if _bbox_width(group) >= 36 and _bbox_height(group) >= 18
    ]
    if usable:
        nearest = min(
            usable,
            key=lambda bbox: (
                max(0.0, caption_top - bbox[3]),
                -(_bbox_width(bbox) * _bbox_height(bbox)),
            ),
        )
        return _padded_bbox(
            nearest,
            page_width,
            page_height,
            band_x0,
            band_x1,
        )

    fallback_height = min(page_height * 0.42, 320.0)
    return (
        band_x0,
        max(page_height * 0.035, caption_top - fallback_height),
        band_x1,
        max(page_height * 0.035 + 1, caption_top - 4),
    )


def render_figure_crop(
    page: object,
    bbox: BBox,
    output_path: Path,
    *,
    dpi: int,
) -> None:
    """把 PDF 坐标区域渲染为 PNG。"""

    if dpi < 72:
        raise ValueError("图片渲染 DPI 不能低于 72")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cropped = page.crop(bbox, strict=True)  # type: ignore[attr-defined]
    image = cropped.to_image(resolution=dpi, antialias=True)
    image.save(output_path, format="PNG")


def crop_pdf_figures(
    pdf_path: Path,
    captions: Iterable[Mapping[str, Any]],
    output_dir: Path,
    *,
    dpi: int = 160,
) -> list[FigureCrop]:
    """按一篇论文的图注裁剪图片，返回可供视觉模型处理的任务记录。"""

    caption_list = sorted(
        captions,
        key=lambda item: (
            int(item["page_number"]),
            int(item["reading_order"]),
            str(item["element_id"]),
        ),
    )
    if not caption_list:
        return []

    crops: list[FigureCrop] = []
    with pdfplumber.open(pdf_path) as document:
        for caption in caption_list:
            page_number = int(caption["page_number"])
            if page_number > len(document.pages):
                raise ValueError(f"图注页码超出 PDF 范围: {page_number}")
            raw_bbox = caption.get("bbox")
            if raw_bbox is None:
                raise ValueError(f"图注缺少坐标: {caption['element_id']}")
            caption_bbox = tuple(float(value) for value in raw_bbox)
            if len(caption_bbox) != 4:
                raise ValueError(f"图注坐标格式错误: {caption['element_id']}")
            typed_bbox = (
                caption_bbox[0],
                caption_bbox[1],
                caption_bbox[2],
                caption_bbox[3],
            )
            page = document.pages[page_number - 1]
            bbox = detect_figure_bbox(page, typed_bbox)
            figure_name = extract_figure_name(str(caption["normalized_text"]))
            figure_id = make_figure_id(
                str(caption["asset_id"]),
                page_number,
                figure_name,
                bbox,
            )
            relative_path = (
                Path("figures")
                / str(caption["asset_id"])
                / f"p{page_number:04d}_{figure_id}.png"
            )
            render_figure_crop(
                page,
                bbox,
                output_dir / relative_path,
                dpi=dpi,
            )
            crops.append(
                FigureCrop(
                    figure_id=figure_id,
                    asset_id=str(caption["asset_id"]),
                    corpus_id=str(caption["corpus_id"]),
                    caption_element_id=str(caption["element_id"]),
                    figure_name=figure_name,
                    page_number=page_number,
                    bbox=bbox,
                    caption=str(caption["normalized_text"]),
                    image_path=relative_path.as_posix(),
                )
            )
    return crops


def _caption_column_band(
    caption_x0: float,
    caption_x1: float,
    page_width: float,
) -> tuple[float, float]:
    margin = page_width * 0.045
    midpoint = page_width / 2
    center = (caption_x0 + caption_x1) / 2
    central = abs(center - midpoint) <= page_width * 0.12
    wide = caption_x1 - caption_x0 >= page_width * 0.45
    if central or wide:
        return margin, page_width - margin
    gutter_overlap = page_width * 0.01
    if center < midpoint:
        return margin, midpoint + gutter_overlap
    return midpoint - gutter_overlap, page_width - margin


def _graphic_bboxes(
    objects: Mapping[str, Iterable[Mapping[str, Any]]],
) -> list[BBox]:
    result: list[BBox] = []
    for object_type in _GRAPHIC_OBJECT_TYPES:
        for item in objects.get(object_type, ()):
            try:
                bbox = (
                    float(item["x0"]),
                    float(item["top"]),
                    float(item["x1"]),
                    float(item["bottom"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if bbox[2] >= bbox[0] and bbox[3] >= bbox[1]:
                result.append(bbox)
    return result


def _group_vertically(
    bboxes: Iterable[BBox],
    *,
    max_gap: float,
) -> list[BBox]:
    ordered = sorted(bboxes, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))
    if not ordered:
        return []
    groups: list[BBox] = []
    current = ordered[0]
    for bbox in ordered[1:]:
        if bbox[1] <= current[3] + max_gap:
            current = _union_bbox(current, bbox)
        else:
            groups.append(current)
            current = bbox
    groups.append(current)
    return groups


def _padded_bbox(
    bbox: BBox,
    page_width: float,
    page_height: float,
    band_x0: float,
    band_x1: float,
) -> BBox:
    return (
        band_x0,
        max(page_height * 0.02, bbox[1] - _CROP_PADDING),
        band_x1,
        min(page_height * 0.98, bbox[3] + _CROP_PADDING),
    )


def _union_bbox(left: BBox, right: BBox) -> BBox:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _horizontal_overlap_ratio(
    bbox: BBox,
    band_x0: float,
    band_x1: float,
) -> float:
    overlap = max(0.0, min(bbox[2], band_x1) - max(bbox[0], band_x0))
    width = max(0.001, bbox[2] - bbox[0])
    return overlap / width


def _is_page_sized_decoration(
    bbox: BBox,
    page_width: float,
    page_height: float,
) -> bool:
    return (
        _bbox_width(bbox) >= page_width * 0.94
        and _bbox_height(bbox) >= page_height * 0.94
    )


def _bbox_width(bbox: BBox) -> float:
    return bbox[2] - bbox[0]


def _bbox_height(bbox: BBox) -> float:
    return bbox[3] - bbox[1]
