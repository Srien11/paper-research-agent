from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.retrieval.config import (
    RetrievalConfig,
    load_chunking_config,
    load_retrieval_config,
)


class RetrievalConfigTests(unittest.TestCase):
    def test_checked_in_configs_are_valid_and_models_are_pinned(self) -> None:
        chunking = load_chunking_config(PROJECT_ROOT / "configs/chunking/baseline-v1.json")
        retrieval = load_retrieval_config(
            PROJECT_ROOT / "configs/retrieval/hybrid-rerank-v1.json"
        )
        self.assertEqual(chunking.max_tokens, 512)
        self.assertEqual(retrieval.embedding_model, "BAAI/bge-small-en-v1.5")
        self.assertTrue(retrieval.embedding_revision)
        self.assertTrue(retrieval.reranker_revision)

    def test_missing_model_revision_is_rejected(self) -> None:
        payload = self._valid_payload()
        del payload["embedding_revision"]
        with self.assertRaises(ValidationError):
            RetrievalConfig.model_validate(payload)

    def test_invalid_candidate_counts_are_rejected(self) -> None:
        payload = self._valid_payload()
        payload["top_k"] = 31
        with self.assertRaises(ValidationError):
            RetrievalConfig.model_validate(payload)

    def test_unsafe_local_path_is_rejected(self) -> None:
        for path in ("../secrets", "C:/models", "/tmp/index"):
            payload = self._valid_payload()
            payload["index_dir"] = path
            with self.subTest(path=path), self.assertRaises(ValidationError):
                RetrievalConfig.model_validate(payload)

    @staticmethod
    def _valid_payload() -> dict[str, object]:
        return {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_revision": "a" * 40,
            "reranker_model": "Xenova/ms-marco-MiniLM-L-6-v2",
            "reranker_revision": "b" * 40,
            "rrf_k": 60,
            "sparse_candidates": 50,
            "vector_candidates": 50,
            "rerank_candidates": 30,
            "top_k": 10,
            "index_dir": "data/indexes/test",
        }
