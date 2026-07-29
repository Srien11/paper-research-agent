from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.chunker import build_chunks, build_figure_chunks
from paper_research_agent.figures.models import FigureRecord
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

    def test_figure_is_one_generated_chunk_with_full_metadata(self) -> None:
        figure = FigureRecord(
            figure_id="figure_001",
            asset_id="asset-1",
            figure_name="Figure 3",
            page_number=12,
            bbox=(72.0, 180.0, 520.0, 610.0),
            caption="Figure 3. Overall architecture.",
            image_path="figures/asset-1/p0012.png",
            figure_type="系统架构图",
            summary="该图展示检索、重排与生成模块。",
            key_findings=("检索结果进入重排模块", "生成模块保留证据引用"),
            recognition_confidence=0.9,
            model_id="fixture-vision-v1",
            prompt_version="figure-summary-v1",
        )
        chunks = build_figure_chunks(
            [figure],
            ChunkingConfig(max_tokens=64, overlap_tokens=8),
            corpus_by_asset={"asset-1": "C001"},
        )
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.evidence_type, "figure_summary")
        self.assertEqual(chunk.content_origin, "generated")
        self.assertEqual(chunk.figure, figure)
        self.assertEqual(chunk.element_ids, (figure.figure_id,))
        self.assertIn(figure.caption, chunk.text)
        self.assertIn(figure.summary, chunk.text)
        self.assertNotIn(figure.image_path, chunk.text)
        self.assertNotIn(figure.model_id, chunk.text)
