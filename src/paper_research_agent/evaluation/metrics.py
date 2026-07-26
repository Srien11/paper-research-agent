"""Transparent retrieval metrics with no external dependencies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from collections.abc import Set as AbstractSet


def recall_at_k(retrieved: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: AbstractSet[str]) -> float:
    for rank, identifier in enumerate(retrieved, start=1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: AbstractSet[str], k: int) -> float:
    seen: set[str] = set()
    gain = 0.0
    for rank, identifier in enumerate(retrieved[:k], start=1):
        if identifier in relevant and identifier not in seen:
            gain += 1.0 / math.log2(rank + 1)
        seen.add(identifier)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return gain / ideal if ideal else 0.0


def evidence_hit_at_k(
    retrieved_chunks: Sequence[str], relevant_chunks: AbstractSet[str], k: int
) -> float | None:
    if not relevant_chunks:
        return None
    return float(bool(set(retrieved_chunks[:k]) & relevant_chunks))
