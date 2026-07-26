"""Reciprocal-rank fusion for deterministic hybrid recall."""

from __future__ import annotations

from collections.abc import Sequence

from paper_research_agent.chunking.models import EvidenceChunk


def reciprocal_rank_fusion(
    sparse: Sequence[tuple[EvidenceChunk, float]],
    vector: Sequence[tuple[EvidenceChunk, float]],
    *,
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[tuple[EvidenceChunk, float, dict[str, int], dict[str, float]]]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    chunks: dict[str, EvidenceChunk] = {}
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float]] = {}
    for stage, results in (("bm25", sparse), ("vector", vector)):
        seen: set[str] = set()
        for rank, (chunk, score) in enumerate(results, start=1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks[chunk.chunk_id] = chunk
            ranks.setdefault(chunk.chunk_id, {})[stage] = rank
            scores.setdefault(chunk.chunk_id, {})[stage] = score
    fused = [
        (
            chunks[chunk_id],
            sum(1 / (rrf_k + rank) for rank in stage_ranks.values()),
            stage_ranks,
            scores[chunk_id],
        )
        for chunk_id, stage_ranks in ranks.items()
    ]
    fused.sort(key=lambda item: (-item[1], item[0].chunk_id))
    return fused[:top_k] if top_k is not None else fused
