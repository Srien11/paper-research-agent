"""Strict adapters from retrieval artifacts to context evidence."""

from __future__ import annotations

from collections.abc import Iterable

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.models import ContextEvidence
from paper_research_agent.retrieval.contracts import BilingualRetrievalRun, RetrievalRun


class EvidenceJoinError(ValueError):
    """A retrieval hit cannot be traced exactly to its source chunk."""


def join_retrieval_evidence(
    run: RetrievalRun | BilingualRetrievalRun,
    chunks: Iterable[EvidenceChunk],
) -> tuple[ContextEvidence, ...]:
    if isinstance(run, BilingualRetrievalRun) and run.rights_status != "loaded":
        raise EvidenceJoinError(
            "bilingual retrieval rights were not loaded; context assembly fails closed"
        )
    chunk_map: dict[str, EvidenceChunk] = {}
    for source_chunk in chunks:
        if source_chunk.chunk_id in chunk_map:
            raise EvidenceJoinError(f"duplicate source chunk_id: {source_chunk.chunk_id}")
        chunk_map[source_chunk.chunk_id] = source_chunk

    evidence: list[ContextEvidence] = []
    for hit in run.hits:
        matched_chunk = chunk_map.get(hit.chunk_id)
        if matched_chunk is None:
            raise EvidenceJoinError(f"retrieval hit has no source chunk: {hit.chunk_id}")
        expected = (
            hit.corpus_id,
            hit.asset_id,
            hit.section_id,
            hit.page_start,
            hit.page_end,
            hit.text_sha256,
            hit.evidence_type,
            hit.figure,
        )
        actual = (
            matched_chunk.corpus_id,
            matched_chunk.asset_id,
            matched_chunk.section_id,
            matched_chunk.page_start,
            matched_chunk.page_end,
            matched_chunk.text_sha256,
            matched_chunk.evidence_type,
            matched_chunk.figure,
        )
        if expected != actual:
            raise EvidenceJoinError(
                f"retrieval metadata does not match source chunk: {hit.chunk_id}"
            )
        evidence.append(
            ContextEvidence(
                chunk_id=matched_chunk.chunk_id,
                corpus_id=matched_chunk.corpus_id,
                asset_id=matched_chunk.asset_id,
                section_id=matched_chunk.section_id,
                page_start=matched_chunk.page_start,
                page_end=matched_chunk.page_end,
                text=matched_chunk.text,
                text_sha256=matched_chunk.text_sha256,
                evidence_type=matched_chunk.evidence_type,
                figure=matched_chunk.figure,
                storage_class=(
                    run.storage_classes[hit.corpus_id]
                    if isinstance(run, BilingualRetrievalRun)
                    else None
                ),
                final_score=hit.final_score,
                final_rank=hit.final_rank,
            )
        )
    return tuple(evidence)
