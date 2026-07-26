"""Small deterministic BM25 implementation with explicit tokenization."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping

from paper_research_agent.chunking.chunker import tokenize
from paper_research_agent.chunking.models import EvidenceChunk


class BM25Index:
    def __init__(self, chunks: Iterable[EvidenceChunk], *, k1: float = 1.2, b: float = 0.75):
        self.chunks = sorted(chunks, key=lambda item: item.chunk_id)
        self.k1 = k1
        self.b = b
        self._terms = [Counter(token.casefold() for token in tokenize(item.text)) for item in self.chunks]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._document_frequency: Counter[str] = Counter()
        for terms in self._terms:
            self._document_frequency.update(terms.keys())

    def score(self, query: str, document_index: int) -> float:
        if not self.chunks or self._average_length == 0:
            return 0.0
        score = 0.0
        query_terms = {token.casefold() for token in tokenize(query)}
        terms = self._terms[document_index]
        length = self._lengths[document_index]
        for term in query_terms:
            frequency = terms[term]
            if frequency == 0:
                continue
            frequency_docs = self._document_frequency[term]
            inverse_document_frequency = math.log(
                1 + (len(self.chunks) - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / self._average_length
            )
            score += inverse_document_frequency * frequency * (self.k1 + 1) / denominator
        return score

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[EvidenceChunk, float]]:
        filters = filters or {}
        scored = [
            (chunk, self.score(query, position))
            for position, chunk in enumerate(self.chunks)
            if _matches(chunk, filters)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored[:top_k]


def _matches(chunk: EvidenceChunk, filters: Mapping[str, str]) -> bool:
    return all(getattr(chunk, key, None) == value for key, value in filters.items())
