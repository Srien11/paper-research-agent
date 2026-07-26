"""从页面文本行推断章节，并将元素绑定到最近章节。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from paper_research_agent.ingestion.identity import make_section_id, sha256_text
from paper_research_agent.ingestion.models import (
    DocumentAsset,
    DocumentElement,
    SectionRecord,
)

_NUMBERED_HEADING = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*)(?P<marker>[.)])?\s+(?P<title>\S.*)",
    re.IGNORECASE,
)
_APPENDIX_HEADING = re.compile(r"^(?:appendix|附录)(?:\s+[a-z0-9]+)?(?:[.:])?\s*", re.IGNORECASE)
_COMMON_HEADINGS = {
    "abstract",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "background",
    "conclusion",
    "conclusions",
    "discussion",
    "evaluation",
    "experiment",
    "experiments",
    "introduction",
    "limitations",
    "method",
    "methods",
    "methodology",
    "references",
    "related work",
    "results",
}
_REFERENCE_TITLES = {"references", "bibliography"}
_GENERIC_COUNT_LABELS = {"dataset", "datasets", "language", "languages", "task", "tasks"}
_SENTENCE_PUNCTUATION = re.compile(r"[,;!?]")
_TABLE_CAPTION = re.compile(r"^(?:table|tab\.)\s+[a-z]?\d+[.:]\s*\S", re.IGNORECASE)
_FIGURE_CAPTION = re.compile(r"^(?:figure|fig\.)\s+[a-z]?\d+[.:]\s*\S", re.IGNORECASE)


@dataclass(frozen=True)
class StructuredElements:
    sections: tuple[SectionRecord, ...]
    elements: tuple[DocumentElement, ...]


@dataclass(frozen=True)
class HeadingCandidate:
    element: DocumentElement
    level: int


def infer_document_structure(
    elements: tuple[DocumentElement, ...],
    asset: DocumentAsset,
) -> StructuredElements:
    """识别标题和章节，保留无法可靠判断的元素为普通段落。"""

    ordered = tuple(sorted(elements, key=lambda item: (item.page_number, item.reading_order)))
    if not ordered:
        return StructuredElements(sections=(), elements=())

    title_id = _find_title_element_id(ordered)
    candidates = tuple(
        candidate
        for element in ordered
        if (candidate := detect_heading(element)) is not None
    )
    sections, heading_section_ids = _build_sections(candidates, asset)

    current_section_id: str | None = None
    current_section_title: str | None = None
    section_by_id = {section.section_id: section for section in sections}
    updated: list[DocumentElement] = []
    for element in ordered:
        if element.element_id == title_id:
            updated.append(element.model_copy(update={"element_type": "title"}))
            continue
        if element.element_id in heading_section_ids:
            current_section_id = heading_section_ids[element.element_id]
            current_section_title = section_by_id[current_section_id].title_normalized.casefold()
            updated.append(
                element.model_copy(
                    update={
                        "element_type": "heading",
                        "section_id": current_section_id,
                    }
                )
            )
            continue
        element_type = element.element_type
        caption_type = detect_caption_type(element)
        if caption_type is not None:
            element_type = caption_type
        elif current_section_title in _REFERENCE_TITLES:
            element_type = "reference"
        updated.append(
            element.model_copy(
                update={
                    "section_id": current_section_id,
                    "element_type": element_type,
                }
            )
        )
    return StructuredElements(sections=sections, elements=tuple(updated))


def detect_heading(element: DocumentElement) -> HeadingCandidate | None:
    """使用保守规则识别常见英文章节标题。"""

    text = element.normalized_text.strip()
    if not text or len(text) > 120:
        return None
    numbered = _NUMBERED_HEADING.match(text)
    if numbered:
        number = numbered.group("number")
        raw_title = numbered.group("title").casefold()
        title = raw_title.rstrip(".:")
        plain_number = int(number) if "." not in number else None
        base_title_shape = (
            len(title) <= 80
            and title[:1].isalpha()
            and title not in _GENERIC_COUNT_LABELS
            and _SENTENCE_PUNCTUATION.search(title) is None
            and not raw_title.endswith(".")
        )
        looks_like_hierarchical_title = "." in number and base_title_shape
        looks_like_short_title = (
            plain_number is not None and plain_number <= 12 and base_title_shape
        )
        if (
            looks_like_hierarchical_title
            or title in _COMMON_HEADINGS
            or looks_like_short_title
        ):
            level = number.count(".") + 1
            return HeadingCandidate(element=element, level=level)
    lowered = text.casefold().rstrip(".:")
    if lowered in _COMMON_HEADINGS:
        return HeadingCandidate(element=element, level=1)
    if _APPENDIX_HEADING.match(lowered):
        return HeadingCandidate(element=element, level=1)
    return None


def detect_caption_type(
    element: DocumentElement,
) -> str | None:
    """识别带编号的表格与图片标题，供后续切块保留语义边界。"""

    text = element.normalized_text.strip()
    if _TABLE_CAPTION.match(text):
        return "table_caption"
    if _FIGURE_CAPTION.match(text):
        return "figure_caption"
    return None


def _find_title_element_id(elements: tuple[DocumentElement, ...]) -> str | None:
    first_page = [element for element in elements if element.page_number == 1]
    for element in first_page[:5]:
        if detect_heading(element) is None and 10 <= len(element.normalized_text) <= 300:
            return element.element_id
    return None


def _build_sections(
    candidates: tuple[HeadingCandidate, ...],
    asset: DocumentAsset,
) -> tuple[tuple[SectionRecord, ...], dict[str, str]]:
    if not candidates:
        return (), {}
    sections: list[SectionRecord] = []
    heading_section_ids: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    for ordinal, candidate in enumerate(candidates):
        element = candidate.element
        section_id = make_section_id(
            asset.source_sha256,
            ordinal,
            sha256_text(element.normalized_text),
        )
        while stack and stack[-1][0] >= candidate.level:
            stack.pop()
        parent_section_id = stack[-1][1] if stack else None
        next_page = (
            candidates[ordinal + 1].element.page_number
            if ordinal + 1 < len(candidates)
            else asset.expected_page_count
        )
        end_page = (
            max(element.page_number, next_page - 1)
            if next_page > element.page_number
            else element.page_number
        )
        section = SectionRecord(
            section_id=section_id,
            asset_id=asset.asset_id,
            corpus_id=asset.corpus_id,
            parent_section_id=parent_section_id,
            level=candidate.level,
            ordinal=ordinal,
            title_raw=element.raw_text,
            title_normalized=element.normalized_text,
            start_page=element.page_number,
            end_page=end_page,
            source_sha256=asset.source_sha256,
            parser_name=element.parser_name,
            parser_version=element.parser_version,
        )
        sections.append(section)
        heading_section_ids[element.element_id] = section_id
        stack.append((candidate.level, section_id))
    return tuple(sections), heading_section_ids
