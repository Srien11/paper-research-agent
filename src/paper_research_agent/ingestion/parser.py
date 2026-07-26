"""使用 pdfplumber 逐页提取论文文本和坐标。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

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
)
from paper_research_agent.ingestion.text import normalize_text

PARSER_NAME = "pdfplumber"
PARSER_VERSION = pdfplumber.__version__
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
    elements: tuple[DocumentElement, ...]


def extract_lines(page: object) -> tuple[TextLine, ...]:
    """读取 pdfplumber 行记录，并转换为稳定的内部结构。"""

    raw_lines = page.extract_text_lines(  # type: ignore[attr-defined]
        strip=True,
        return_chars=False,
        layout=True,
    )
    lines = [
        TextLine(
            text=str(item["text"]).strip(),
            x0=float(item["x0"]),
            top=float(item["top"]),
            x1=float(item["x1"]),
            bottom=float(item["bottom"]),
        )
        for item in raw_lines
        if str(item.get("text", "")).strip()
    ]
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
    spanning = tuple(
        line
        for line in lines
        if (line.x0 < midpoint - margin and line.x1 > midpoint + margin)
        or (line.x1 - line.x0) >= page_width * 0.65
    )
    non_spanning = tuple(line for line in lines if line not in spanning)
    left = tuple(line for line in non_spanning if line.x1 <= midpoint + margin)
    right = tuple(line for line in non_spanning if line.x0 >= midpoint - margin)
    two_columns = len(left) >= 3 and len(right) >= 3
    if not two_columns:
        return tuple(sorted(lines, key=lambda line: (line.top, line.x0, line.bottom)))

    ordered: list[TextLine] = []
    remaining = list(non_spanning)
    for separator in sorted(spanning, key=lambda line: (line.top, line.x0)):
        band = [line for line in remaining if line.top < separator.top]
        ordered.extend(_order_column_band(band, midpoint))
        remaining = [line for line in remaining if line not in band]
        ordered.append(separator)
    ordered.extend(_order_column_band(remaining, midpoint))
    return tuple(ordered)


def find_repeated_margin_signatures(
    drafts: tuple[PageDraft, ...],
) -> frozenset[tuple[str, str]]:
    """发现跨页重复的页眉页脚文本。"""

    successful = [draft for draft in drafts if draft.error_code is None]
    threshold = max(3, math.ceil(len(successful) * 0.3))
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
    return ParsedDocument(pages=tuple(pages), elements=tuple(elements))


def _extract_page_draft(page: object, page_number: int) -> PageDraft:
    width = float(page.width)  # type: ignore[attr-defined]
    height = float(page.height)  # type: ignore[attr-defined]
    try:
        lines = order_page_lines(extract_lines(page), width)
        return PageDraft(page_number, width, height, lines)
    except Exception as exc:
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
    common = {
        "page_id": page_id,
        "asset_id": asset.asset_id,
        "corpus_id": asset.corpus_id,
        "page_number": draft.page_number,
        "width_points": draft.width,
        "height_points": draft.height,
        "source_sha256": asset.source_sha256,
        "parser_name": PARSER_NAME,
        "parser_version": PARSER_VERSION,
    }
    if draft.error_code is not None:
        return (
            PageRecord(
                **common,
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
        return PageRecord(**common, status="empty"), ()

    page_record = PageRecord(
        **common,
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


def _order_column_band(lines: list[TextLine], midpoint: float) -> list[TextLine]:
    left = sorted(
        (line for line in lines if line.x0 < midpoint),
        key=lambda line: (line.top, line.x0),
    )
    right = sorted(
        (line for line in lines if line.x0 >= midpoint),
        key=lambda line: (line.top, line.x0),
    )
    return [*left, *right]


def _margin_zone(line: TextLine, page_height: float) -> str | None:
    if line.top <= page_height * 0.08:
        return "top"
    if line.bottom >= page_height * 0.92:
        return "bottom"
    return None


def _margin_signature(text: str) -> str:
    return _DIGITS.sub("#", normalize_text(text).casefold())

