"""使用 pdfplumber 逐页提取论文文本和坐标。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal

import pdfplumber

from paper_research_agent.ingestion.identity import (
    make_element_id,
    make_page_id,
    sha256_text,
)
from paper_research_agent.ingestion.models import (
    DocumentAsset,
    DocumentElement,
    PageRecord,
    SectionRecord,
)
from paper_research_agent.ingestion.structure import infer_document_structure
from paper_research_agent.ingestion.text import normalize_text

PARSER_NAME = "pdfplumber"
PARSER_VERSION = pdfplumber.__version__
PARSER_CONFIG_SCHEMA_VERSION = "pdf-parser-config-v1"
X_TOLERANCE = 2.0
COLUMN_SPLIT_GAP_RATIO = 0.018
MARGIN_RATIO = 0.08
REPEAT_RATIO = 0.3
MIN_REPEAT_PAGES = 3
_PAGE_NUMBER = re.compile(
    r"^\s*(?:page\s*)?(?:\d+|[ivxlcdm]+)(?:\s*(?:/|of)\s*\d+)?\s*$",
    re.IGNORECASE,
)
_DIGITS = re.compile(r"\d+")


class PdfParseError(RuntimeError):
    """PDF 无法打开或不满足资产契约。"""


@dataclass(frozen=True)
class TextLine:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    column: Literal["left", "right"] | None = None


@dataclass(frozen=True)
class PageDraft:
    page_number: int
    width: float
    height: float
    lines: tuple[TextLine, ...]
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    pages: tuple[PageRecord, ...]
    sections: tuple[SectionRecord, ...]
    elements: tuple[DocumentElement, ...]


def parser_config() -> dict[str, object]:
    """返回会影响解析结果的确定性配置。"""

    return {
        "schema_version": PARSER_CONFIG_SCHEMA_VERSION,
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "extract_text_lines": {
            "strip": True,
            "return_chars": True,
            "layout": True,
            "x_tolerance": X_TOLERANCE,
        },
        "header_footer": {
            "margin_ratio": MARGIN_RATIO,
            "repeat_ratio": REPEAT_RATIO,
            "minimum_repeat_pages": MIN_REPEAT_PAGES,
        },
        "column_ordering": {
            "midpoint_margin_ratio": 0.04,
            "spanning_width_ratio": 0.65,
            "minimum_lines_per_column": 3,
            "split_merged_line_gap_ratio": COLUMN_SPLIT_GAP_RATIO,
        },
        "serialization": {
            "format": "canonical-jsonl-v1",
            "encoding": "utf-8",
            "sort_keys": True,
            "escape_unicode_line_separators": True,
        },
        "bbox_policy": {
            "discard_fully_outside_page": True,
            "clip_partially_visible_lines": True,
            "discard_rotated_side_margin_lines": True,
        },
    }


def extract_lines(page: object) -> tuple[TextLine, ...]:
    """读取 pdfplumber 行记录，并转换为稳定的内部结构。"""

    raw_lines = page.extract_text_lines(  # type: ignore[attr-defined]
        strip=True,
        return_chars=True,
        layout=True,
        x_tolerance=X_TOLERANCE,
    )
    page_width = float(page.width)  # type: ignore[attr-defined]
    page_height = float(page.height)  # type: ignore[attr-defined]
    lines: list[TextLine] = []
    for item in raw_lines:
        if _is_rotated_side_margin(item, page_width):
            continue
        for line in _split_cross_column_line(item, page_width):
            clipped = _clip_line_to_page(line, page_width, page_height)
            if clipped is not None:
                lines.append(clipped)
    return tuple(lines)


def order_page_lines(
    lines: tuple[TextLine, ...],
    page_width: float,
) -> tuple[TextLine, ...]:
    """按跨栏标题分区，再按左栏、右栏顺序组织文本行。"""

    if not lines:
        return ()
    midpoint = page_width / 2
    margin = page_width * 0.04
    asymmetric_right_starts = [
        line.x0
        for line in lines
        if line.column == "right" and line.x0 < midpoint - margin
    ]
    asymmetric_right_start = (
        min(asymmetric_right_starts) if asymmetric_right_starts else None
    )
    if asymmetric_right_start is not None:
        asymmetric_lines = [
            line
            for line in lines
            if asymmetric_right_start - 1 <= line.x0 < midpoint - margin
        ]
        asymmetric_end = max(
            (line.bottom for line in asymmetric_lines),
            default=0.0,
        )
        upper_region = tuple(line for line in lines if line.top <= asymmetric_end)
        lower_region = tuple(line for line in lines if line.top > asymmetric_end)
        if upper_region and lower_region:
            return (
                *_order_page_region(
                    upper_region,
                    page_width,
                    asymmetric_right_start,
                ),
                *_order_page_region(lower_region, page_width, None),
            )
    return _order_page_region(lines, page_width, asymmetric_right_start)


def _order_page_region(
    lines: tuple[TextLine, ...],
    page_width: float,
    asymmetric_right_start: float | None,
) -> tuple[TextLine, ...]:
    midpoint = page_width / 2
    margin = page_width * 0.04
    spanning = tuple(
        line
        for line in lines
        if _classify_line(
            line,
            midpoint,
            margin,
            asymmetric_right_start,
        )
        is None
        and (
            (line.x0 < midpoint - margin and line.x1 > midpoint + margin)
            or (line.x1 - line.x0) >= page_width * 0.65
        )
    )
    non_spanning = tuple(line for line in lines if line not in spanning)
    left = tuple(
        line
        for line in non_spanning
        if _classify_line(line, midpoint, margin, asymmetric_right_start) == "left"
    )
    right = tuple(
        line
        for line in non_spanning
        if _classify_line(line, midpoint, margin, asymmetric_right_start) == "right"
    )
    two_columns = len(left) >= 3 and len(right) >= 3
    if not two_columns:
        return tuple(sorted(lines, key=lambda line: (line.top, line.x0, line.bottom)))

    ordered: list[TextLine] = []
    remaining = list(non_spanning)
    for separator in sorted(spanning, key=lambda line: (line.top, line.x0)):
        band = [line for line in remaining if line.top < separator.top]
        ordered.extend(
            _order_column_band(
                band,
                midpoint,
                margin,
                asymmetric_right_start,
            )
        )
        remaining = [line for line in remaining if line not in band]
        ordered.append(separator)
    ordered.extend(
        _order_column_band(
            remaining,
            midpoint,
            margin,
            asymmetric_right_start,
        )
    )
    return tuple(ordered)


def find_repeated_margin_signatures(
    drafts: tuple[PageDraft, ...],
) -> frozenset[tuple[str, str]]:
    """发现跨页重复的页眉页脚文本。"""

    successful = [draft for draft in drafts if draft.error_code is None]
    threshold = max(MIN_REPEAT_PAGES, math.ceil(len(successful) * REPEAT_RATIO))
    counts: dict[tuple[str, str], set[int]] = {}
    for draft in successful:
        for line in draft.lines:
            zone = _margin_zone(line, draft.height)
            if zone is None:
                continue
            signature = _margin_signature(line.text)
            if not signature or _PAGE_NUMBER.fullmatch(signature):
                continue
            counts.setdefault((zone, signature), set()).add(draft.page_number)
    return frozenset(key for key, pages in counts.items() if len(pages) >= threshold)


def clean_page_lines(
    draft: PageDraft,
    repeated_signatures: frozenset[tuple[str, str]],
) -> tuple[TextLine, ...]:
    """移除纯页码和已确认的重复页眉页脚。"""

    kept: list[TextLine] = []
    for line in draft.lines:
        zone = _margin_zone(line, draft.height)
        signature = _margin_signature(line.text)
        if zone is not None and _PAGE_NUMBER.fullmatch(signature):
            continue
        if zone is not None and (zone, signature) in repeated_signatures:
            continue
        kept.append(line)
    return tuple(kept)


def parse_pdf_asset(pdf_path: Path, asset: DocumentAsset) -> ParsedDocument:
    """解析一个已通过冻结清单校验的 PDF 资产。"""

    if not pdf_path.is_file():
        raise PdfParseError(f"PDF 不存在: {pdf_path}")
    try:
        document = pdfplumber.open(pdf_path)
    except Exception as exc:
        raise PdfParseError(f"PDF 无法打开: {exc}") from exc

    with document:
        if len(document.pages) != asset.expected_page_count:
            raise PdfParseError(
                "PDF 页数与冻结清单不一致: "
                f"{len(document.pages)} != {asset.expected_page_count}"
            )
        drafts = tuple(
            _extract_page_draft(page, page_number)
            for page_number, page in enumerate(document.pages, start=1)
        )

    repeated = find_repeated_margin_signatures(drafts)
    pages: list[PageRecord] = []
    elements: list[DocumentElement] = []
    for draft in drafts:
        page, page_elements = _build_page_records(draft, asset, repeated)
        pages.append(page)
        elements.extend(page_elements)
    structured = infer_document_structure(tuple(elements), asset)
    return ParsedDocument(
        pages=tuple(pages),
        sections=structured.sections,
        elements=structured.elements,
    )


def _extract_page_draft(page: object, page_number: int) -> PageDraft:
    width = float(page.width)  # type: ignore[attr-defined]
    height = float(page.height)  # type: ignore[attr-defined]
    try:
        lines = order_page_lines(extract_lines(page), width)
        return PageDraft(page_number, width, height, lines)
    except Exception as exc:  # noqa: BLE001 - isolate one malformed PDF page
        return PageDraft(
            page_number,
            width,
            height,
            (),
            error_code="page_extraction_error",
            error_message=f"{type(exc).__name__}: {exc}",
        )


def _build_page_records(
    draft: PageDraft,
    asset: DocumentAsset,
    repeated: frozenset[tuple[str, str]],
) -> tuple[PageRecord, tuple[DocumentElement, ...]]:
    page_id = make_page_id(asset.source_sha256, draft.page_number)
    if draft.error_code is not None:
        return (
            PageRecord(
                page_id=page_id,
                asset_id=asset.asset_id,
                corpus_id=asset.corpus_id,
                page_number=draft.page_number,
                width_points=draft.width,
                height_points=draft.height,
                source_sha256=asset.source_sha256,
                parser_name=PARSER_NAME,
                parser_version=PARSER_VERSION,
                status="failed",
                error_code=draft.error_code,
                error_message=draft.error_message,
            ),
            (),
        )

    raw_text = "\n".join(line.text for line in draft.lines)
    cleaned_lines = clean_page_lines(draft, repeated)
    normalized_text = normalize_text("\n".join(line.text for line in cleaned_lines))
    if not normalized_text:
        return (
            PageRecord(
                page_id=page_id,
                asset_id=asset.asset_id,
                corpus_id=asset.corpus_id,
                page_number=draft.page_number,
                width_points=draft.width,
                height_points=draft.height,
                source_sha256=asset.source_sha256,
                parser_name=PARSER_NAME,
                parser_version=PARSER_VERSION,
                status="empty",
            ),
            (),
        )

    page_record = PageRecord(
        page_id=page_id,
        asset_id=asset.asset_id,
        corpus_id=asset.corpus_id,
        page_number=draft.page_number,
        width_points=draft.width,
        height_points=draft.height,
        source_sha256=asset.source_sha256,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
        status="parsed",
        raw_text=raw_text,
        normalized_text=normalized_text,
        raw_text_sha256=sha256_text(raw_text),
        normalized_text_sha256=sha256_text(normalized_text),
    )
    raw_offsets = _raw_offsets(draft.lines)
    page_elements = tuple(
        _build_element(
            line=line,
            reading_order=reading_order,
            raw_start=raw_offsets[id(line)][0],
            raw_end=raw_offsets[id(line)][1],
            asset=asset,
            page_id=page_id,
            page_number=draft.page_number,
        )
        for reading_order, line in enumerate(cleaned_lines)
    )
    return page_record, page_elements


def _build_element(
    *,
    line: TextLine,
    reading_order: int,
    raw_start: int,
    raw_end: int,
    asset: DocumentAsset,
    page_id: str,
    page_number: int,
) -> DocumentElement:
    normalized = normalize_text(line.text)
    normalized_hash = sha256_text(normalized)
    return DocumentElement(
        element_id=make_element_id(
            asset.source_sha256,
            page_number,
            "paragraph",
            reading_order,
            normalized_hash,
        ),
        asset_id=asset.asset_id,
        page_id=page_id,
        corpus_id=asset.corpus_id,
        page_number=page_number,
        element_type="paragraph",
        reading_order=reading_order,
        raw_text=line.text,
        normalized_text=normalized,
        raw_start=raw_start,
        raw_end=raw_end,
        bbox=(line.x0, line.top, line.x1, line.bottom),
        normalized_text_sha256=normalized_hash,
        source_sha256=asset.source_sha256,
        parser_name=PARSER_NAME,
        parser_version=PARSER_VERSION,
    )


def _raw_offsets(lines: tuple[TextLine, ...]) -> dict[int, tuple[int, int]]:
    offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for line in lines:
        start = cursor
        end = start + len(line.text)
        offsets[id(line)] = (start, end)
        cursor = end + 1
    return offsets


def _order_column_band(
    lines: list[TextLine],
    midpoint: float,
    margin: float,
    asymmetric_right_start: float | None,
) -> list[TextLine]:
    left = sorted(
        (
            line
            for line in lines
            if _classify_line(line, midpoint, margin, asymmetric_right_start) == "left"
        ),
        key=lambda line: (line.top, line.x0),
    )
    right = sorted(
        (
            line
            for line in lines
            if _classify_line(line, midpoint, margin, asymmetric_right_start) == "right"
        ),
        key=lambda line: (line.top, line.x0),
    )
    return [*left, *right]


def _classify_line(
    line: TextLine,
    midpoint: float,
    margin: float,
    asymmetric_right_start: float | None,
) -> Literal["left", "right"] | None:
    if line.column is not None:
        return line.column
    if (
        asymmetric_right_start is not None
        and line.x0 >= asymmetric_right_start - 1
    ):
        return "right"
    if line.x1 <= midpoint + margin:
        return "left"
    if line.x0 >= midpoint - margin:
        return "right"
    return None


def _margin_zone(line: TextLine, page_height: float) -> str | None:
    if line.top <= page_height * MARGIN_RATIO:
        return "top"
    if line.bottom >= page_height * (1 - MARGIN_RATIO):
        return "bottom"
    return None


def _margin_signature(text: str) -> str:
    return _DIGITS.sub("#", normalize_text(text).casefold())


def _clip_line_to_page(
    line: TextLine,
    page_width: float,
    page_height: float,
) -> TextLine | None:
    values = (line.x0, line.top, line.x1, line.bottom)
    if not all(math.isfinite(value) for value in values):
        return None
    if line.x1 <= 0 or line.bottom <= 0 or line.x0 >= page_width or line.top >= page_height:
        return None
    x0 = max(0.0, line.x0)
    top = max(0.0, line.top)
    x1 = min(page_width, line.x1)
    bottom = min(page_height, line.bottom)
    if x1 <= x0 or bottom <= top:
        return None
    return TextLine(
        text=line.text,
        x0=x0,
        top=top,
        x1=x1,
        bottom=bottom,
        column=line.column,
    )


def _is_rotated_side_margin(item: dict[str, object], page_width: float) -> bool:
    """过滤位于左右页边的整行旋转水印，同时保留正文内的竖排图表文字。"""

    chars = item.get("chars")
    if not isinstance(chars, list) or not chars:
        return False
    if any(bool(char.get("upright", True)) for char in chars if isinstance(char, dict)):
        return False
    x0 = _as_float(item["x0"])
    x1 = _as_float(item["x1"])
    return x1 <= page_width * MARGIN_RATIO or x0 >= page_width * (1 - MARGIN_RATIO)


def _split_cross_column_line(
    item: dict[str, object],
    page_width: float,
) -> tuple[TextLine, ...]:
    """按跨越页面中线的大间隙拆开被 PDF 引擎合并的左右栏同行文本。"""

    original = _item_to_text_line(item)
    chars = item.get("chars")
    if not isinstance(chars, list):
        return (original,) if original.text else ()
    horizontal_chars = sorted(
        (
            char
            for char in chars
            if isinstance(char, dict)
            and bool(char.get("upright", True))
            and "x0" in char
            and "x1" in char
        ),
        key=lambda char: _as_float(char["x0"]),
    )
    midpoint = page_width / 2
    candidates = [
        (_as_float(right["x0"]) - _as_float(left["x1"]), index)
        for index, (left, right) in enumerate(pairwise(horizontal_chars), start=1)
        if _looks_like_column_gutter(left, right, midpoint, page_width)
    ]
    if not candidates:
        return (original,) if original.text else ()
    gap, split_index = max(candidates)
    if gap < page_width * COLUMN_SPLIT_GAP_RATIO:
        return (original,) if original.text else ()
    left_chars = horizontal_chars[:split_index]
    right_chars = horizontal_chars[split_index:]
    left_text, right_text = _split_text_at_char_boundary(original.text, left_chars)
    left_line = _chars_to_text_line(left_chars, left_text, "left")
    right_line = _chars_to_text_line(right_chars, right_text, "right")
    return tuple(line for line in (left_line, right_line) if line.text)


def _looks_like_column_gutter(
    left: dict[str, object],
    right: dict[str, object],
    midpoint: float,
    page_width: float,
) -> bool:
    left_edge = _as_float(left["x1"])
    right_edge = _as_float(right["x0"])
    if left_edge <= midpoint <= right_edge:
        return True
    gap_center_ratio = ((left_edge + right_edge) / 2) / page_width
    return 0.2 <= gap_center_ratio <= 0.45


def _item_to_text_line(item: dict[str, object]) -> TextLine:
    return TextLine(
        text=str(item.get("text", "")).strip(),
        x0=_as_float(item["x0"]),
        top=_as_float(item["top"]),
        x1=_as_float(item["x1"]),
        bottom=_as_float(item["bottom"]),
    )


def _split_text_at_char_boundary(
    text: str,
    left_chars: list[dict[str, object]],
) -> tuple[str, str]:
    """按左栏实际字符数切开已含合成空格的原始行文本。"""

    target = sum(
        1
        for char in left_chars
        for character in str(char.get("text", ""))
        if not character.isspace()
    )
    consumed = 0
    boundary = 0
    for boundary, character in enumerate(text, start=1):
        if not character.isspace():
            consumed += 1
        if consumed >= target:
            break
    return text[:boundary].strip(), text[boundary:].strip()


def _chars_to_text_line(
    chars: list[dict[str, object]],
    text: str,
    column: Literal["left", "right"],
) -> TextLine:
    return TextLine(
        text=text,
        x0=min(_as_float(char["x0"]) for char in chars),
        top=min(_as_float(char["top"]) for char in chars),
        x1=max(_as_float(char["x1"]) for char in chars),
        bottom=max(_as_float(char["bottom"]) for char in chars),
        column=column,
    )


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError("PDF coordinate must be numeric")
