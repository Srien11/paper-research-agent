from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.retrieval.model_adapters import (
    FastEmbedEncoder,
    FastEmbedReranker,
)


class FakeTextEmbedding:
    model_name: str | None = None
    revision: str | None = None

    def __init__(self, *, model_name: str, revision: str):
        type(self).model_name = model_name
        type(self).revision = revision

    document_call = None
    query_call = None

    def passage_embed(self, documents):
        type(self).document_call = documents
        return (vector for vector in ((1.0, 2.0), (3.0, 4.0)))

    def query_embed(self, query):
        type(self).query_call = query
        return (vector for vector in ((5.0, 6.0),))


class FakeTextCrossEncoder:
    model_name: str | None = None
    revision: str | None = None
    call: tuple[str, list[str]] | None = None

    def __init__(self, *, model_name: str, revision: str):
        type(self).model_name = model_name
        type(self).revision = revision

    def rerank(self, query, documents):
        type(self).call = (query, documents)
        return (value for value in (0.25, 0.75))


class FastEmbedAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fastembed = types.ModuleType("fastembed")
        self.fastembed.TextEmbedding = FakeTextEmbedding
        self.rerank = types.ModuleType("fastembed.rerank")
        self.cross_encoder = types.ModuleType("fastembed.rerank.cross_encoder")
        self.cross_encoder.TextCrossEncoder = FakeTextCrossEncoder

    def test_encoder_uses_public_api_and_consumes_embedding_generator(self) -> None:
        with patch.dict(sys.modules, {"fastembed": self.fastembed}):
            encoder = FastEmbedEncoder("org/embedding", revision="embed-sha")
            self.assertEqual(
                encoder.encode_documents(("first", "second")),
                [[1.0, 2.0], [3.0, 4.0]],
            )
            self.assertEqual(encoder.encode_query("query"), [5.0, 6.0])
        self.assertEqual(FakeTextEmbedding.model_name, "org/embedding")
        self.assertEqual(FakeTextEmbedding.revision, "embed-sha")
        self.assertEqual(FakeTextEmbedding.document_call, ["first", "second"])
        self.assertEqual(FakeTextEmbedding.query_call, "query")

    def test_reranker_uses_v08_api_and_consumes_score_generator(self) -> None:
        modules = {
            "fastembed": self.fastembed,
            "fastembed.rerank": self.rerank,
            "fastembed.rerank.cross_encoder": self.cross_encoder,
        }
        with patch.dict(sys.modules, modules):
            reranker = FastEmbedReranker("org/reranker", revision="rerank-sha")
            self.assertEqual(reranker.score("query", ("first", "second")), [0.25, 0.75])
        self.assertEqual(FakeTextCrossEncoder.model_name, "org/reranker")
        self.assertEqual(FakeTextCrossEncoder.revision, "rerank-sha")
        self.assertEqual(FakeTextCrossEncoder.call, ("query", ["first", "second"]))


if __name__ == "__main__":
    unittest.main()
