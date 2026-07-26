from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk


class EvidenceChunkContractTests(unittest.TestCase):
    def test_invalid_page_range_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EvidenceChunk(
                chunk_id="c1",
                asset_id="a1",
                corpus_id="C001",
                element_ids=("e1",),
                page_start=2,
                page_end=1,
                token_start=0,
                token_end=1,
                text="x",
                text_sha256="0" * 64,
                config_sha256="1" * 64,
            )

    def test_duplicate_element_references_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EvidenceChunk(
                chunk_id="c1",
                asset_id="a1",
                corpus_id="C001",
                element_ids=("e1", "e1"),
                page_start=1,
                page_end=1,
                token_start=0,
                token_end=1,
                text="x",
                text_sha256="0" * 64,
                config_sha256="1" * 64,
            )
