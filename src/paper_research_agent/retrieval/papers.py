"""Deterministic paper-level hybrid retrieval over titles and abstracts."""

from __future__ import annotations

import asyncio
import math
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from paper_research_agent.chunking.chunker import tokenize
from paper_research_agent.chunking.models import EvidenceChunk, PaperCard
from paper_research_agent.retrieval.vector import Encoder, normalize


@dataclass(frozen=True)
class PaperCandidateDocument:
    corpus_id: str
    title: str
    abstract: str | None
    fallback_text: str | None

    @property
    def used_fallback(self) -> bool:
        return self.abstract is None

    @property
    def retrieval_text(self) -> str:
        body = self.abstract if self.abstract is not None else self.fallback_text
        return f"{self.title}\n{body or ''}".strip()


@dataclass(frozen=True)
class PaperCandidateHit:
    corpus_id: str
    title: str
    abstract: str | None
    used_fallback: bool
    final_score: float
    ranks: dict[str, int]


@dataclass(frozen=True)
class PaperCandidateQuery:
    """Keep the authoritative question while selecting an English retrieval view."""

    original_query: str
    english_query: str | None = None

    def __post_init__(self) -> None:
        if not self.original_query.strip():
            raise ValueError("paper candidate original query cannot be blank")
        if self.english_query is not None and not self.english_query.strip():
            raise ValueError("paper candidate English query cannot be blank")

    @property
    def retrieval_query(self) -> str:
        """Use English for English paper cards, falling back only on rewrite failure."""
        return self.english_query if self.english_query is not None else self.original_query


class AsyncPaperCandidateRetriever(Protocol):
    async def search(
        self, query: PaperCandidateQuery, *, top_k: int
    ) -> tuple[PaperCandidateHit, ...]: ...


def build_paper_candidate_documents(
    cards: Sequence[PaperCard],
    chunks: Sequence[EvidenceChunk],
    *,
    fallback_chunk_limit: int = 3,
    fallback_chars: int = 2400,
) -> tuple[PaperCandidateDocument, ...]:
    """Build one bounded retrieval document per paper without crossing corpora."""
    if fallback_chunk_limit <= 0:
        raise ValueError("fallback_chunk_limit must be positive")
    if fallback_chars <= 0:
        raise ValueError("fallback_chars must be positive")
    chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
    if len(chunk_map) != len(chunks):
        raise ValueError("paper candidate chunks contain duplicate chunk IDs")
    seen_corpora: set[str] = set()
    documents: list[PaperCandidateDocument] = []
    for card in sorted(cards, key=lambda item: item.corpus_id):
        if card.corpus_id in seen_corpora:
            raise ValueError("paper cards contain duplicate corpus IDs")
        seen_corpora.add(card.corpus_id)
        abstract = _normalize_optional(card.abstract)
        fallback_text: str | None = None
        if abstract is None:
            fallback_parts: list[str] = []
            for chunk_id in card.evidence_chunk_ids:
                chunk = chunk_map.get(chunk_id)
                if chunk is None:
                    raise ValueError("paper card references a missing fallback chunk")
                if chunk.corpus_id != card.corpus_id:
                    raise ValueError("paper card fallback crosses corpus boundary")
                if chunk.evidence_type != "text" or chunk.content_origin != "source_text":
                    continue
                fallback_parts.append(" ".join(chunk.text.split()))
                if len(fallback_parts) >= fallback_chunk_limit:
                    break
            fallback_text = " ".join(fallback_parts)[:fallback_chars].strip()
            if not fallback_text:
                raise ValueError("paper card without abstract has no usable fallback text")
        documents.append(
            PaperCandidateDocument(
                corpus_id=card.corpus_id,
                title=card.title,
                abstract=abstract,
                fallback_text=fallback_text,
            )
        )
    return tuple(documents)


class HybridPaperCandidateRetriever:
    """Fuse paper-level BM25 and dense rankings with deterministic RRF."""

    def __init__(self, documents: Sequence[PaperCandidateDocument], encoder: Encoder):
        if not documents:
            raise ValueError("paper candidate retriever requires documents")
        corpus_ids = [document.corpus_id for document in documents]
        if len(corpus_ids) != len(set(corpus_ids)):
            raise ValueError("paper candidate documents contain duplicate corpus IDs")
        self._documents = tuple(sorted(documents, key=lambda item: item.corpus_id))
        self._encoder = encoder
        self._terms = [
            Counter(token.casefold() for token in tokenize(document.retrieval_text))
            for document in self._documents
        ]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._document_frequency: Counter[str] = Counter()
        for terms in self._terms:
            self._document_frequency.update(terms.keys())
        self._vectors: tuple[tuple[float, ...], ...] | None = None
        self._vector_lock = threading.Lock()

    async def search(
        self, query: PaperCandidateQuery, *, top_k: int
    ) -> tuple[PaperCandidateHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return await asyncio.to_thread(self._search_sync, query, top_k)

    def _search_sync(
        self, query: PaperCandidateQuery, top_k: int
    ) -> tuple[PaperCandidateHit, ...]:
        return self._hybrid_route(query.retrieval_query)[:top_k]

    def _hybrid_route(self, query: str) -> tuple[PaperCandidateHit, ...]:
        sparse = [
            (document, score)
            for position, document in enumerate(self._documents)
            if (score := self._bm25_score(query, position)) > 0
        ]
        sparse.sort(key=lambda item: (-item[1], item[0].corpus_id))
        vectors = self._document_vectors()
        query_vector = normalize(self._encoder.encode_query(query))
        if vectors and len(query_vector) != len(vectors[0]):
            raise ValueError("paper query vector dimension does not match documents")
        dense = [
            (
                document,
                sum(
                    left * right
                    for left, right in zip(vector, query_vector, strict=True)
                ),
            )
            for document, vector in zip(self._documents, vectors, strict=True)
        ]
        dense.sort(key=lambda item: (-item[1], item[0].corpus_id))
        ranks: dict[str, dict[str, int]] = {}
        documents: dict[str, PaperCandidateDocument] = {}
        for route, results in (("bm25", sparse), ("vector", dense)):
            for rank, (document, _) in enumerate(results, start=1):
                documents[document.corpus_id] = document
                ranks.setdefault(document.corpus_id, {})[route] = rank
        fused = [
            PaperCandidateHit(
                corpus_id=corpus_id,
                title=documents[corpus_id].title,
                abstract=documents[corpus_id].abstract,
                used_fallback=documents[corpus_id].used_fallback,
                final_score=sum(1 / (60 + rank) for rank in route_ranks.values()),
                ranks=dict(route_ranks),
            )
            for corpus_id, route_ranks in ranks.items()
        ]
        fused.sort(key=lambda item: (-item.final_score, item.corpus_id))
        return tuple(fused)

    def _document_vectors(self) -> tuple[tuple[float, ...], ...]:
        if self._vectors is None:
            with self._vector_lock:
                if self._vectors is None:
                    encoded = self._encoder.encode_documents(
                        [document.retrieval_text for document in self._documents]
                    )
                    if len(encoded) != len(self._documents):
                        raise ValueError("paper encoder result count does not match documents")
                    vectors = tuple(normalize(vector) for vector in encoded)
                    dimensions = {len(vector) for vector in vectors}
                    if len(dimensions) != 1:
                        raise ValueError("paper encoder returned inconsistent dimensions")
                    self._vectors = vectors
        return self._vectors

    def _bm25_score(self, query: str, document_index: int) -> float:
        score = 0.0
        terms = self._terms[document_index]
        length = self._lengths[document_index]
        for term in {token.casefold() for token in tokenize(query)}:
            frequency = terms[term]
            if frequency == 0:
                continue
            frequency_docs = self._document_frequency[term]
            inverse_document_frequency = math.log(
                1
                + (len(self._documents) - frequency_docs + 0.5)
                / (frequency_docs + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * length / self._average_length
            )
            score += inverse_document_frequency * frequency * 2.2 / denominator
        return score


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None
