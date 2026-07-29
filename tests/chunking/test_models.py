from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.figures.models import FigureRecord


class EvidenceChunkContractTests(unittest.TestCase):
    def test_legacy_text_chunk_remains_readable(self) -> None:
        chunk = EvidenceChunk.model_validate(
            {
                "schema_version": "evidence-chunk-v1",
                "chunk_id": "c1",
                "asset_id": "a1",
                "corpus_id": "C001",
                "element_ids": ["e1"],
                "page_start": 1,
                "page_end": 1,
                "token_start": 0,
                "token_end": 1,
                "text": "x",
                "text_sha256": "0" * 64,
                "config_sha256": "1" * 64,
                "content_origin": "source_text",
            }
        )
        self.assertEqual(chunk.schema_version, "evidence-chunk-v1")
        self.assertEqual(chunk.evidence_type, "text")
        self.assertIsNone(chunk.figure)

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

    def test_figure_summary_requires_matching_generated_record(self) -> None:
        figure = FigureRecord(
            figure_id="figure_001",
            asset_id="asset-1",
            figure_name="Figure 3",
            page_number=12,
            bbox=(72.0, 180.0, 520.0, 610.0),
            caption="Figure 3. Overall architecture.",
            image_path="figures/asset-1/p0012.png",
            figure_type="系统架构图",
            summary="该图展示系统结构。",
            key_findings=(),
            recognition_confidence=0.8,
            model_id="fixture-vision-v1",
            prompt_version="figure-summary-v1",
        )
        with self.assertRaisesRegex(ValidationError, "必须包含生成"):
            EvidenceChunk(
                chunk_id="c1",
                asset_id="asset-1",
                corpus_id="C001",
                element_ids=(figure.figure_id,),
                page_start=12,
                page_end=12,
                token_start=0,
                token_end=1,
                text="x",
                text_sha256="0" * 64,
                config_sha256="1" * 64,
                evidence_type="figure_summary",
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
