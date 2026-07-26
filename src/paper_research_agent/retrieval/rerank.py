"""Stable reranking over a fixed candidate pool."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from paper_research_agent.chunking.models import EvidenceChunk


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


def rerank(
    query: str,
    candidates: Sequence[tuple[EvidenceChunk, float, dict[str, int], dict[str, float]]],
    model: Reranker,
) -> list[tuple[EvidenceChunk, float, dict[str, int], dict[str, float]]]:
    values = model.score(query, [candidate[0].text for candidate in candidates])
    if len(values) != len(candidates):
        raise ValueError("reranker result count does not match candidates")
    results = []
    for candidate, rerank_score in zip(candidates, values, strict=True):
        chunk, _, ranks, scores = candidate
        results.append((chunk, float(rerank_score), dict(ranks), {**scores, "rerank": float(rerank_score)}))
    results.sort(key=lambda item: (-item[1], item[0].chunk_id))
    return results
