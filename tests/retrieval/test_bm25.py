from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.retrieval.bm25 import BM25Index


def chunk(identifier: str, text: str, corpus_id: str = "C001") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=identifier,
        asset_id="a",
        corpus_id=corpus_id,
        element_ids=(f"e-{identifier}",),
        page_start=1,
        page_end=1,
        token_start=0,
        token_end=max(1, len(text.split())),
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        config_sha256="0" * 64,
    )


class BM25Tests(unittest.TestCase):
    def test_ranking_ties_and_filter_are_deterministic(self) -> None:
        index = BM25Index([chunk("b", "alpha"), chunk("a", "alpha"), chunk("c", "beta", "T001")])
        self.assertEqual([item.chunk_id for item, _ in index.search("alpha", 3)], ["a", "b", "c"])
        filtered = index.search("alpha", 3, filters={"corpus_id": "T001"})
        self.assertEqual([item.chunk_id for item, _ in filtered], ["c"])
