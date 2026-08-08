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


def candidate_paper_recall(
    candidate_corpus_ids: Sequence[str], relevant_corpus_ids: AbstractSet[str]
) -> float | None:
    """Recall of gold papers in the independently discovered candidate set."""
    if not relevant_corpus_ids:
        return None
    return len(set(candidate_corpus_ids) & relevant_corpus_ids) / len(relevant_corpus_ids)


def explicit_corpus_id_accuracy(
    predicted: Sequence[Sequence[str]], expected: Sequence[Sequence[str]]
) -> float | None:
    """Exact ordered parsing accuracy across explicit-ID diagnostic cases."""
    if len(predicted) != len(expected):
        raise ValueError("explicit-ID prediction and expectation counts differ")
    if not expected:
        return None
    correct = sum(tuple(left) == tuple(right) for left, right in zip(predicted, expected, strict=True))
    return correct / len(expected)
