"""Normalized vector retrieval with an injectable encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from paper_research_agent.chunking.models import EvidenceChunk


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SearchableVectorIndex(Protocol):
    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[EvidenceChunk, float]]: ...


def normalize(vector: Sequence[float]) -> tuple[float, ...]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise ValueError("encoder returned a zero vector")
    return tuple(value / magnitude for value in vector)


class VectorIndex:
    def __init__(self, chunks: Sequence[EvidenceChunk], encoder: Encoder):
        self.chunks = sorted(chunks, key=lambda item: item.chunk_id)
        self.encoder = encoder
        self.vectors = [normalize(vector) for vector in encoder.encode([item.text for item in self.chunks])]
        if len(self.vectors) != len(self.chunks):
            raise ValueError("encoder result count does not match chunks")
        dimensions = {len(vector) for vector in self.vectors}
        if len(dimensions) > 1 or (self.chunks and not dimensions):
            raise ValueError("encoder returned inconsistent dimensions")

    @property
    def dimension(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[EvidenceChunk, float]]:
        query_vectors = self.encoder.encode([query])
        if len(query_vectors) != 1:
            raise ValueError("encoder must return exactly one query vector")
        query_vector = normalize(query_vectors[0])
        if self.vectors and len(query_vector) != self.dimension:
            raise ValueError("query vector dimension does not match index")
        filters = filters or {}
        scored = [
            (chunk, sum(left * right for left, right in zip(vector, query_vector, strict=True)))
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
            if all(getattr(chunk, key, None) == value for key, value in filters.items())
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored[:top_k]


class FaissVectorIndex:
    """Query an on-disk IndexFlatIP while keeping source text outside SQLite."""

    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        encoder: Encoder,
        index_path: Path,
    ):
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("install the retrieval extra to query a FAISS index") from error
        self.chunks = sorted(chunks, key=lambda item: item.chunk_id)
        self.encoder = encoder
        self._index = faiss.read_index(str(index_path))
        if self._index.ntotal != len(self.chunks):
            raise ValueError("FAISS vector count does not match chunk artifact")
        if self.chunks and self._index.d <= 0:
            raise ValueError("FAISS index has an invalid vector dimension")

    def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[tuple[EvidenceChunk, float]]:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError("install the retrieval extra to query a FAISS index") from error
        query_vectors = self.encoder.encode([query])
        if len(query_vectors) != 1:
            raise ValueError("encoder must return exactly one query vector")
        query_vector = normalize(query_vectors[0])
        if len(query_vector) != self._index.d:
            raise ValueError("query vector dimension does not match FAISS index")
        scores, positions = self._index.search(
            np.asarray([query_vector], dtype="float32"), len(self.chunks)
        )
        filters = filters or {}
        results = [
            (self.chunks[int(position)], float(score))
            for position, score in zip(positions[0], scores[0], strict=True)
            if position >= 0
            and all(
                getattr(self.chunks[int(position)], key, None) == value
                for key, value in filters.items()
            )
        ]
        results.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return results[:top_k]
