"""Lazy FastEmbed adapters; imports happen only for local model-backed runs."""

from __future__ import annotations

from collections.abc import Sequence


class FastEmbedEncoder:
    def __init__(self, model_name: str, *, revision: str):
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError("install the retrieval extra to use FastEmbed") from error
        self._model = TextEmbedding(model_name=model_name, revision=revision)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(vector) for vector in self._model.passage_embed(list(texts))]

    def encode_query(self, query: str) -> list[float]:
        vectors = [list(vector) for vector in self._model.query_embed(query)]
        if len(vectors) != 1:
            raise ValueError("FastEmbed 必须为单条查询返回一个向量")
        return vectors[0]


class FastEmbedReranker:
    def __init__(self, model_name: str, *, revision: str):
        try:
            # TextCrossEncoder is not re-exported by fastembed 0.8.
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as error:
            raise RuntimeError("install the retrieval extra to use the cross encoder") from error
        self._model = TextCrossEncoder(model_name=model_name, revision=revision)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(value) for value in self._model.rerank(query, list(texts))]
