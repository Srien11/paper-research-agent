from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.chunker import build_figure_chunks
from paper_research_agent.figures.models import FigureRecord
from paper_research_agent.retrieval.config import ChunkingConfig
from paper_research_agent.retrieval.indexer import write_chunk_metadata


class RetrievalIndexerTests(unittest.TestCase):
    def test_index_metadata_contains_exact_figure_record(self) -> None:
        figure = FigureRecord(
            figure_id="figure-1",
            asset_id="asset-1",
            figure_name="Figure 1",
            page_number=2,
            bbox=(10.0, 20.0, 100.0, 120.0),
            caption="Figure 1. Architecture.",
            image_path="figures/asset-1/p0002.png",
            figure_type="系统架构图",
            summary="展示模块关系。",
            key_findings=("模块 A 连接模块 B",),
            recognition_confidence=0.9,
            model_id="fixture-vision-v1",
            prompt_version="figure-summary-v1",
        )
        chunks = build_figure_chunks(
            [figure],
            ChunkingConfig(max_tokens=64, overlap_tokens=8),
            corpus_by_asset={"asset-1": "C001"},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metadata_path = output / "metadata.sqlite"
            write_chunk_metadata(chunks, metadata_path)
            with closing(sqlite3.connect(metadata_path)) as connection:
                row = connection.execute(
                    "SELECT evidence_type, figure_json FROM chunks"
                ).fetchone()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row[0], "figure_summary")
            payload = json.loads(row[1])
            self.assertEqual(payload, figure.model_dump(mode="json"))
            self.assertEqual(len(payload), 14)


if __name__ == "__main__":
    unittest.main()
