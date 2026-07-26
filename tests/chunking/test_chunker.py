from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.chunker import build_chunks
from paper_research_agent.ingestion.models import DocumentElement
from paper_research_agent.retrieval.config import ChunkingConfig


def element(element_id: str, text: str, section_id: str, order: int) -> DocumentElement:
    import hashlib

    return DocumentElement(
        element_id=element_id,
        asset_id="asset-1",
        page_id="page-1",
        corpus_id="C001",
        page_number=1,
        section_id=section_id,
        element_type="paragraph",
        reading_order=order,
        raw_text=text,
        normalized_text=text,
        normalized_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_sha256="a" * 64,
        parser_name="fixture",
        parser_version="1",
    )


class DeterministicChunkerTests(unittest.TestCase):
    def test_never_crosses_sections_and_is_stable(self) -> None:
        config = ChunkingConfig(max_tokens=32, overlap_tokens=4)
        elements = [element("e1", "one two", "s1", 0), element("e2", "three four", "s2", 1)]
        first = build_chunks(elements, config)
        second = build_chunks(reversed(elements), config)
        self.assertEqual(first, second)
        self.assertEqual({chunk.section_id for chunk in first}, {"s1", "s2"})
        self.assertTrue(all(len(chunk.element_ids) == 1 for chunk in first))

    def test_long_section_respects_limit_and_overlap(self) -> None:
        config = ChunkingConfig(max_tokens=32, overlap_tokens=4)
        chunks = build_chunks(
            [element("e1", " ".join(f"t{i}" for i in range(40)), "s1", 0)], config
        )
        self.assertEqual([(item.token_start, item.token_end) for item in chunks], [(0, 32), (28, 40)])
