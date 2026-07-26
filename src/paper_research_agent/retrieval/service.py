"""One deterministic service exposing A/B/C retrieval variants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from paper_research_agent.chunking.chunker import canonical_sha256
from paper_research_agent.retrieval.bm25 import BM25Index
from paper_research_agent.retrieval.config import RetrievalConfig
from paper_research_agent.retrieval.contracts import RetrievalRun, SearchHit
from paper_research_agent.retrieval.hybrid import reciprocal_rank_fusion
from paper_research_agent.retrieval.rerank import Reranker, rerank
from paper_research_agent.retrieval.vector import SearchableVectorIndex


class RetrievalService:
    def __init__(
        self,
        sparse: BM25Index,
        vector: SearchableVectorIndex,
        reranker: Reranker,
        config: RetrievalConfig,
        *,
        index_id: str,
    ):
        self.sparse = sparse
        self.vector = vector
        self.reranker = reranker
        self.config = config
        self.index_id = index_id

    def search(
        self,
        query: str,
        variant: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, str] | None = None,
    ) -> RetrievalRun:
        if variant not in {"A", "B", "C"}:
            raise ValueError("variant must be A, B, or C")
        limit = self.config.top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be positive")
        vector = self.vector.search(query, self.config.vector_candidates, filters=filters)
        if variant == "A":
            candidates = [
                (chunk, score, {"vector": rank}, {"vector": score})
                for rank, (chunk, score) in enumerate(vector, start=1)
            ]
        else:
            sparse = self.sparse.search(query, self.config.sparse_candidates, filters=filters)
            candidates = reciprocal_rank_fusion(
                sparse, vector, rrf_k=self.config.rrf_k, top_k=self.config.rerank_candidates
            )
            if variant == "C":
                candidates = rerank(query, candidates, self.reranker)
        hits = tuple(
            SearchHit(
                chunk_id=chunk.chunk_id,
                corpus_id=chunk.corpus_id,
                asset_id=chunk.asset_id,
                section_id=chunk.section_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text_sha256=chunk.text_sha256,
                scores=scores,
                ranks={**ranks, "final": rank},
                final_score=score,
                final_rank=rank,
            )
            for rank, (chunk, score, ranks, scores) in enumerate(candidates[:limit], start=1)
        )
        return RetrievalRun(
            query=query,
            variant=cast(Literal["A", "B", "C"], variant),
            top_k=limit,
            hits=hits,
            index_id=self.index_id,
            config_sha256=canonical_sha256(self.config.model_dump(mode="json")),
        )
