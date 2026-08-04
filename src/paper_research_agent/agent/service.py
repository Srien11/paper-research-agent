"""Framework-independent implementations of the read-only research tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from paper_research_agent.agent.models import (
    EvidenceRecord,
    GetEvidenceInput,
    GetEvidenceResult,
    SearchCorpusHit,
    SearchCorpusInput,
    SearchCorpusResult,
    StorageClass,
)
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.adapters import join_retrieval_evidence
from paper_research_agent.retrieval.contracts import BilingualRetrievalRun


class AsyncResearchRetriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        privacy_ttl_days: int | None = None,
    ) -> BilingualRetrievalRun: ...


class ResearchToolService:
    """Expose bounded search and evidence hydration over immutable local artifacts."""

    def __init__(
        self,
        *,
        retriever: AsyncResearchRetriever,
        chunks: Sequence[EvidenceChunk],
        storage_classes: Mapping[str, StorageClass],
    ) -> None:
        if not chunks:
            raise ValueError("research tools require at least one evidence chunk")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("research tool chunks contain duplicate chunk IDs")
        corpus_ids = {chunk.corpus_id for chunk in chunks}
        missing_rights = corpus_ids - set(storage_classes)
        if missing_rights:
            raise ValueError("storage rights do not cover every research tool chunk")

        self._retriever = retriever
        self._chunks = tuple(chunks)
        self._chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        self._storage_classes = {
            corpus_id: storage_classes[corpus_id] for corpus_id in corpus_ids
        }

    async def search_corpus(self, request: SearchCorpusInput) -> SearchCorpusResult:
        """Search the existing bilingual index and validate every hit against source chunks."""
        run = await self._retriever.search(request.query, top_k=request.top_k)
        evidence = join_retrieval_evidence(run, self._chunks)
        hits: list[SearchCorpusHit] = []
        for item in evidence:
            if item.storage_class is None:
                raise ValueError("retrieval result is missing loaded storage rights")
            hits.append(
                SearchCorpusHit(
                    chunk_id=item.chunk_id,
                    corpus_id=item.corpus_id,
                    section_id=item.section_id,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    text_sha256=item.text_sha256,
                    evidence_type=item.evidence_type,
                    storage_class=item.storage_class,
                    final_rank=item.final_rank,
                )
            )
        return SearchCorpusResult(
            query=request.query,
            index_id=run.index_id,
            degraded=run.degraded,
            degraded_reason=run.degraded_reason,
            hits=tuple(hits),
        )

    async def get_evidence(self, request: GetEvidenceInput) -> GetEvidenceResult:
        """Hydrate explicit chunk IDs from the immutable local evidence catalog."""
        records: list[EvidenceRecord] = []
        missing: list[str] = []
        for chunk_id in request.chunk_ids:
            chunk = self._chunk_map.get(chunk_id)
            if chunk is None:
                missing.append(chunk_id)
                continue
            records.append(
                EvidenceRecord(
                    chunk_id=chunk.chunk_id,
                    corpus_id=chunk.corpus_id,
                    section_id=chunk.section_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    text_sha256=chunk.text_sha256,
                    evidence_type=chunk.evidence_type,
                    storage_class=self._storage_classes[chunk.corpus_id],
                )
            )
        return GetEvidenceResult(records=tuple(records), missing_chunk_ids=tuple(missing))
