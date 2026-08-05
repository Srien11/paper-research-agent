"""Deterministic local evidence tools over immutable corpus artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from paper_research_agent.agent.tooling.contracts import (
    AdjacentChunksInput,
    ChunkIdsInput,
    ComparePapersInput,
    CorpusInput,
    PaperMetadataInput,
    ToolExecutionResult,
)
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.ingestion.models import SectionRecord
from paper_research_agent.models import FrozenPaper


class LocalResearchTools:
    def __init__(
        self,
        *,
        chunks: Sequence[EvidenceChunk],
        storage_classes: Mapping[str, str],
        papers: Sequence[FrozenPaper] = (),
        sections: Sequence[SectionRecord] = (),
    ) -> None:
        self._chunks = tuple(chunks)
        self._chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        self._papers = {paper.corpus_id: paper for paper in papers}
        self._sections = tuple(sections)
        self._storage_classes = dict(storage_classes)

    def get_adjacent_chunks(self, request: AdjacentChunksInput) -> ToolExecutionResult:
        anchor = self._chunk_map.get(request.chunk_id)
        if anchor is None:
            return _result("get_adjacent_chunks", "not_found")
        ordered = sorted(
            (chunk for chunk in self._chunks if chunk.corpus_id == anchor.corpus_id),
            key=lambda chunk: (chunk.page_start, chunk.token_start, chunk.chunk_id),
        )
        index = next(
            index for index, chunk in enumerate(ordered) if chunk.chunk_id == anchor.chunk_id
        )
        selected = ordered[max(0, index - request.before) : index + request.after + 1]
        return _result(
            "get_adjacent_chunks",
            items=tuple(_chunk_item(chunk, self._storage_classes) for chunk in selected),
            summary={"anchor_index": selected.index(anchor), "count": len(selected)},
        )

    def get_paper_metadata(self, request: PaperMetadataInput) -> ToolExecutionResult:
        items = []
        for corpus_id in request.corpus_ids:
            paper = self._papers.get(corpus_id)
            if paper is None:
                continue
            items.append(
                {
                    "corpus_id": paper.corpus_id,
                    "canonical_key": paper.canonical_key,
                    "title": paper.title,
                    "year": paper.year,
                    "authors": tuple(paper.authors),
                    "official_url": paper.official_url,
                    "fulltext_url": paper.fulltext_url,
                    "storage_class": paper.storage_class,
                    "pdf_pages": paper.pdf_pages,
                }
            )
        return _result(
            "get_paper_metadata",
            "ok" if items else "not_found",
            tuple(items),
            {"requested_count": len(request.corpus_ids), "returned_count": len(items)},
        )

    def trace_evidence_source(self, request: ChunkIdsInput) -> ToolExecutionResult:
        items = []
        for chunk_id in request.chunk_ids:
            chunk = self._chunk_map.get(chunk_id)
            if chunk is None:
                continue
            paper = self._papers.get(chunk.corpus_id)
            items.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "corpus_id": chunk.corpus_id,
                    "asset_id": chunk.asset_id,
                    "section_id": chunk.section_id,
                    "element_ids": chunk.element_ids,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "text_sha256": chunk.text_sha256,
                    "evidence_type": chunk.evidence_type,
                    "storage_class": self._storage_classes.get(chunk.corpus_id),
                    "paper_title": paper.title if paper else None,
                    "official_url": paper.official_url if paper else None,
                }
            )
        return _result(
            "trace_evidence_source",
            "ok" if items else "not_found",
            tuple(items),
            {"requested_count": len(request.chunk_ids), "returned_count": len(items)},
        )

    def get_paper_outline(self, request: CorpusInput) -> ToolExecutionResult:
        sections = sorted(
            (section for section in self._sections if section.corpus_id == request.corpus_id),
            key=lambda section: (section.ordinal, section.level, section.section_id),
        )
        items = tuple(
            {
                "section_id": section.section_id,
                "parent_section_id": section.parent_section_id,
                "level": section.level,
                "ordinal": section.ordinal,
                "title": section.title_normalized,
                "start_page": section.start_page,
                "end_page": section.end_page,
            }
            for section in sections[:100]
        )
        return _result(
            "get_paper_outline",
            "ok" if items else "not_found",
            items,
            {"count": len(items), "truncated": len(sections) > len(items)},
        )

    def compare_papers(self, request: ComparePapersInput) -> ToolExecutionResult:
        items: list[dict[str, object]] = []
        for corpus_id in request.corpus_ids:
            corpus_chunks = [chunk for chunk in self._chunks if chunk.corpus_id == corpus_id]
            for dimension in request.dimensions:
                terms = tuple(term for term in dimension.casefold().split() if len(term) > 1)
                ranked = sorted(
                    corpus_chunks,
                    key=lambda chunk: (
                        -sum(term in chunk.text.casefold() for term in terms),
                        chunk.page_start,
                        chunk.chunk_id,
                    ),
                )
                evidence = ranked[: request.evidence_per_dimension]
                items.append(
                    {
                        "corpus_id": corpus_id,
                        "dimension": dimension,
                        "evidence": tuple(
                            {
                                "chunk_id": chunk.chunk_id,
                                "page_start": chunk.page_start,
                                "page_end": chunk.page_end,
                                "text": chunk.text[:1500],
                                "text_sha256": chunk.text_sha256,
                            }
                            for chunk in evidence
                        ),
                    }
                )
        return _result(
            "compare_papers",
            "ok" if items else "not_found",
            tuple(items),
            {"paper_count": len(request.corpus_ids), "dimension_count": len(request.dimensions)},
        )


def _chunk_item(chunk: EvidenceChunk, storage: Mapping[str, str]) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "corpus_id": chunk.corpus_id,
        "section_id": chunk.section_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "text": chunk.text,
        "text_sha256": chunk.text_sha256,
        "evidence_type": chunk.evidence_type,
        "storage_class": storage.get(chunk.corpus_id),
    }


def _result(
    name: str,
    status: str = "ok",
    items: tuple[dict[str, object], ...] = (),
    summary: dict[str, object] | None = None,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name=name,
        status=status,
        items=items,
        summary=summary or {},
    )
