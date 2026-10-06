"""Lazy FastEmbed adapters; imports happen only for local model-backed runs."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from paper_research_agent.exact_cache import ExactCache, exact_key


class FastEmbedEncoder:
    def __init__(self, model_name: str, *, revision: str, cache_entries: int = 256):
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError("install the retrieval extra to use FastEmbed") from error
        self._model = TextEmbedding(model_name=model_name, revision=revision)
        self.query_cache = ExactCache[tuple[float, ...]](cache_entries)
        self._cache_namespace = (model_name, revision, "query-embed-v1")

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(vector) for vector in self._model.passage_embed(list(texts))]

    def encode_query(self, query: str) -> list[float]:
        key = exact_key([self._cache_namespace, query])
        cached = self.query_cache.get(key)
        if cached is not None:
            return list(cached)
        vectors = [list(vector) for vector in self._model.query_embed(query)]
        if len(vectors) != 1:
            raise ValueError("FastEmbed 必须为单条查询返回一个向量")
        if vectors[0] and all(isfinite(value) for value in vectors[0]):
            self.query_cache.put(key, tuple(vectors[0]))
        return vectors[0]


class FastEmbedReranker:
    def __init__(self, model_name: str, *, revision: str, cache_entries: int = 256):
        try:
            # TextCrossEncoder is not re-exported by fastembed 0.8.
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as error:
            raise RuntimeError("install the retrieval extra to use the cross encoder") from error
        self._model = TextCrossEncoder(model_name=model_name, revision=revision)
        self.score_cache = ExactCache[tuple[float, ...]](cache_entries)
        self._cache_namespace = (model_name, revision, "rerank-v1")

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        inputs = list(texts)
        key = exact_key([self._cache_namespace, query, inputs])
        cached = self.score_cache.get(key)
        if cached is not None:
            return list(cached)
        scores = [float(value) for value in self._model.rerank(query, inputs)]
        if len(scores) == len(inputs) and all(isfinite(value) for value in scores):
            self.score_cache.put(key, tuple(scores))
        return scores
