from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.retrieval.vector import VectorIndex
from tests.retrieval.test_bm25 import chunk


class FakeEncoder:
    def encode(self, texts):
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]


class VectorTests(unittest.TestCase):
    def test_cosine_ranking_and_stable_ties(self) -> None:
        index = VectorIndex(
            [chunk("b", "alpha second"), chunk("a", "alpha first"), chunk("c", "beta")],
            FakeEncoder(),
        )
        self.assertEqual([item.chunk_id for item, _ in index.search("alpha", 3)], ["a", "b", "c"])
