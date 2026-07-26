from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.retrieval.bm25 import BM25Index
from paper_research_agent.retrieval.config import RetrievalConfig
from paper_research_agent.retrieval.hybrid import reciprocal_rank_fusion
from paper_research_agent.retrieval.service import RetrievalService
from paper_research_agent.retrieval.vector import VectorIndex
from tests.retrieval.test_bm25 import chunk
from tests.retrieval.test_vector import FakeEncoder


class FakeReranker:
    def score(self, query, texts):
        return [float("beta" in text) for text in texts]


class HybridTests(unittest.TestCase):
    def test_rrf_deduplicates_and_stabilizes_ties(self) -> None:
        a, b = chunk("a", "alpha"), chunk("b", "beta")
        fused = reciprocal_rank_fusion([(a, 1), (a, 0.5)], [(b, 1)], rrf_k=60)
        self.assertEqual([item[0].chunk_id for item in fused], ["a", "b"])

    def test_abc_pipeline_is_repeatable_and_preserves_stage_scores(self) -> None:
        chunks = [chunk("a", "alpha"), chunk("b", "beta alpha")]
        config = RetrievalConfig(
            embedding_model="org/embed",
            embedding_revision="a" * 40,
            reranker_model="org/rerank",
            reranker_revision="b" * 40,
            sparse_candidates=2,
            vector_candidates=2,
            rerank_candidates=2,
            top_k=2,
        )
        service = RetrievalService(
            BM25Index(chunks), VectorIndex(chunks, FakeEncoder()), FakeReranker(), config, index_id="i"
        )
        first = service.search("alpha", "C")
        self.assertEqual(first, service.search("alpha", "C"))
        self.assertIn("rerank", first.hits[0].scores)
        self.assertEqual(first.hits[0].chunk_id, "b")

        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            service.search("alpha", "C", top_k=0)
