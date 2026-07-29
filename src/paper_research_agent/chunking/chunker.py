"""Deterministic section-bounded chunking using regex token intervals."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from paper_research_agent.chunking.models import EvidenceChunk, PaperCard
from paper_research_agent.figures.models import FigureRecord
from paper_research_agent.ingestion.models import DocumentElement
from paper_research_agent.retrieval.config import ChunkingConfig

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class _Token:
    text: str
    element_id: str
    page_number: int


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def build_chunks(
    elements: Iterable[DocumentElement],
    config: ChunkingConfig,
    *,
    abstract_section_ids: set[str] | None = None,
) -> list[EvidenceChunk]:
    """Build stable chunks without crossing papers or section boundaries."""
    abstract_section_ids = abstract_section_ids or set()
    ordered = sorted(elements, key=lambda item: (item.corpus_id, item.page_number, item.reading_order))
    groups: dict[tuple[str, str | None, str], list[DocumentElement]] = defaultdict(list)
    for element in ordered:
        if not element.normalized_text.strip():
            continue
        boundary = "abstract" if element.section_id in abstract_section_ids else "body"
        groups[(element.asset_id, element.section_id, boundary)].append(element)

    config_hash = canonical_sha256(config.model_dump(mode="json"))
    chunks: list[EvidenceChunk] = []
    for group_key in sorted(groups, key=lambda value: tuple("" if v is None else v for v in value)):
        group = groups[group_key]
        tokens = [
            _Token(token, element.element_id, element.page_number)
            for element in group
            for token in tokenize(element.normalized_text)
        ]
        if not tokens:
            continue
        step = config.max_tokens - config.overlap_tokens
        for start in range(0, len(tokens), step):
            window = tokens[start : start + config.max_tokens]
            if not window:
                break
            element_ids = tuple(dict.fromkeys(token.element_id for token in window))
            text = " ".join(token.text for token in window)
            identity = {
                "asset_id": group[0].asset_id,
                "section_id": group[0].section_id,
                "element_ids": element_ids,
                "token_start": start,
                "token_end": start + len(window),
                "text": text,
                "config_sha256": config_hash,
            }
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"chk_{canonical_sha256(identity)[:24]}",
                    asset_id=group[0].asset_id,
                    corpus_id=group[0].corpus_id,
                    section_id=group[0].section_id,
                    element_ids=element_ids,
                    page_start=min(token.page_number for token in window),
                    page_end=max(token.page_number for token in window),
                    token_start=start,
                    token_end=start + len(window),
                    text=text,
                    text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    config_sha256=config_hash,
                )
            )
            if start + len(window) >= len(tokens):
                break
    return sorted(
        chunks,
        key=lambda item: (
            item.corpus_id,
            item.page_start,
            item.section_id or "",
            item.token_start,
            item.chunk_id,
        ),
    )


def build_figure_chunks(
    figures: Iterable[FigureRecord],
    config: ChunkingConfig,
    *,
    corpus_by_asset: Mapping[str, str],
) -> list[EvidenceChunk]:
    """把每张图片的可检索语义转换为一个独立证据块。"""

    config_hash = canonical_sha256(config.model_dump(mode="json"))
    chunks: list[EvidenceChunk] = []
    seen_ids: set[str] = set()
    for figure in sorted(
        figures,
        key=lambda item: (
            item.asset_id,
            item.page_number,
            item.figure_name,
            item.figure_id,
        ),
    ):
        if figure.figure_id in seen_ids:
            raise ValueError(f"重复 figure_id: {figure.figure_id}")
        seen_ids.add(figure.figure_id)
        corpus_id = corpus_by_asset.get(figure.asset_id)
        if corpus_id is None:
            raise ValueError(f"图片没有对应语料编号: {figure.asset_id}")
        findings = "\n".join(f"- {finding}" for finding in figure.key_findings)
        parts = [
            f"图片名称：{figure.figure_name}",
            f"图片类型：{figure.figure_type}",
            f"原始图注：{figure.caption}",
            f"视觉摘要：{figure.summary}",
        ]
        if findings:
            parts.append(f"关键发现：\n{findings}")
        text = "\n".join(parts)
        tokens = tokenize(text)
        identity = {
            "figure": figure.model_dump(mode="json"),
            "text": text,
            "config_sha256": config_hash,
        }
        chunks.append(
            EvidenceChunk(
                chunk_id=f"chk_{canonical_sha256(identity)[:24]}",
                asset_id=figure.asset_id,
                corpus_id=corpus_id,
                element_ids=(figure.figure_id,),
                page_start=figure.page_number,
                page_end=figure.page_number,
                token_start=0,
                token_end=len(tokens),
                text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                config_sha256=config_hash,
                evidence_type="figure_summary",
                content_origin="generated",
                figure=figure,
            )
        )
    return chunks


def build_paper_cards(
    elements: Iterable[DocumentElement],
    chunks: Iterable[EvidenceChunk],
    config: ChunkingConfig,
    *,
    abstract_section_ids: set[str] | None = None,
) -> list[PaperCard]:
    abstract_section_ids = abstract_section_ids or set()
    element_list = list(elements)
    chunk_list = list(chunks)
    cards: list[PaperCard] = []
    for asset_id in sorted({element.asset_id for element in element_list}):
        source = [element for element in element_list if element.asset_id == asset_id]
        related_chunks = [chunk for chunk in chunk_list if chunk.asset_id == asset_id]
        title_element = next((item for item in source if item.element_type == "title"), source[0])
        abstract_parts = [
            item.normalized_text
            for item in source
            if item.section_id in abstract_section_ids and item.normalized_text.strip()
        ]
        config_hash = canonical_sha256(config.model_dump(mode="json"))
        card_identity = {
            "asset_id": asset_id,
            "chunks": [chunk.chunk_id for chunk in related_chunks],
            "config_sha256": config_hash,
        }
        cards.append(
            PaperCard(
                card_id=f"card_{canonical_sha256(card_identity)[:24]}",
                asset_id=asset_id,
                corpus_id=title_element.corpus_id,
                title=title_element.normalized_text,
                abstract=" ".join(abstract_parts) or None,
                evidence_chunk_ids=tuple(chunk.chunk_id for chunk in related_chunks),
                source_element_ids=tuple(item.element_id for item in source),
                config_sha256=config_hash,
            )
        )
    return cards
