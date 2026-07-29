from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.chunking.runner import run_chunking
from paper_research_agent.figures.models import FigureRecord
from paper_research_agent.ingestion.models import DocumentElement


class ChunkingRunnerTests(unittest.TestCase):
    def test_optional_figures_are_merged_into_same_chunk_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = "source paragraph"
            element = DocumentElement(
                element_id="element-1",
                asset_id="asset-1",
                page_id="page-1",
                corpus_id="C001",
                page_number=1,
                element_type="paragraph",
                reading_order=0,
                raw_text=text,
                normalized_text=text,
                normalized_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                source_sha256="a" * 64,
                parser_name="fixture",
                parser_version="1",
            )
            figure = FigureRecord(
                figure_id="figure-1",
                asset_id="asset-1",
                figure_name="Figure 1",
                page_number=1,
                bbox=(10.0, 10.0, 100.0, 100.0),
                caption="Figure 1. Architecture.",
                image_path="figures/asset-1/p0001.png",
                figure_type="系统架构图",
                summary="展示两个模块之间的数据流。",
                key_findings=("模块 A 连接模块 B",),
                recognition_confidence=0.9,
                model_id="fixture-vision-v1",
                prompt_version="figure-summary-v1",
            )
            elements_path = root / "elements.jsonl"
            figures_path = root / "figures.jsonl"
            sections_path = root / "sections.jsonl"
            elements_path.write_text(element.model_dump_json() + "\n", encoding="utf-8")
            figures_path.write_text(figure.model_dump_json() + "\n", encoding="utf-8")
            sections_path.write_text("", encoding="utf-8")

            chunks_path, _ = run_chunking(
                elements_path,
                sections_path,
                PROJECT_ROOT / "configs" / "chunking" / "baseline-v1.json",
                root / "chunks",
                figures_path=figures_path,
            )

            chunks = [
                EvidenceChunk.model_validate_json(line)
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(chunks), 2)
            figure_chunk = next(
                chunk for chunk in chunks if chunk.evidence_type == "figure_summary"
            )
            self.assertEqual(figure_chunk.figure, figure)


if __name__ == "__main__":
    unittest.main()
