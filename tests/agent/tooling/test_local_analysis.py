from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from paper_research_agent.agent.tooling.analysis import AnalysisResearchTools
from paper_research_agent.agent.tooling.contracts import (
    AdjacentChunksInput,
    AnalyzeExperimentDataInput,
    CalculateInput,
    ChunkIdsInput,
    ComparePapersInput,
    CorpusInput,
    PaperMetadataInput,
)
from paper_research_agent.agent.tooling.local import LocalResearchTools
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.ingestion.models import SectionRecord
from paper_research_agent.models import FrozenPaper


def _chunk(index: int, text: str) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=f"chunk-{index}",
        asset_id="asset-1",
        corpus_id="C001",
        section_id="section-1",
        element_ids=(f"element-{index}",),
        page_start=index,
        page_end=index,
        token_start=index * 10,
        token_end=index * 10 + 5,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        config_sha256="a" * 64,
    )


def _paper() -> FrozenPaper:
    return FrozenPaper(
        corpus_id="C001",
        corpus_version="v1",
        dataset_split="core",
        canonical_key="doi:10.1/test",
        title="Test Paper",
        year=2026,
        authors=["Researcher"],
        official_url="https://example.org/paper",
        fulltext_url="https://example.org/paper.pdf",
        selection_status="frozen",
        content_status="downloaded_and_parse_verified",
        storage_class="internal_research_only",
        local_pdf_path=Path("private.pdf"),
        download_sha256="b" * 64,
        download_bytes=100,
        pdf_pages=3,
        parse_quality_status="machine_parse_pass",
    )


class LocalAndAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = (
            _chunk(1, "The method uses public source code."),
            _chunk(2, "Accuracy improves by ten percent."),
            _chunk(3, "Random seed and hyperparameters are reported."),
        )
        section = SectionRecord(
            section_id="section-1",
            asset_id="asset-1",
            corpus_id="C001",
            level=1,
            ordinal=0,
            title_raw="Methods",
            title_normalized="Methods",
            start_page=1,
            end_page=3,
            source_sha256="c" * 64,
            parser_name="test",
            parser_version="1",
        )
        self.local = LocalResearchTools(
            chunks=self.chunks,
            storage_classes={"C001": "internal_research_only"},
            papers=(_paper(),),
            sections=(section,),
        )

    def test_local_evidence_tools(self) -> None:
        adjacent = self.local.get_adjacent_chunks(
            AdjacentChunksInput(chunk_id="chunk-2", before=1, after=1)
        )
        self.assertEqual(
            [item["chunk_id"] for item in adjacent.items], ["chunk-1", "chunk-2", "chunk-3"]
        )
        metadata = self.local.get_paper_metadata(PaperMetadataInput(corpus_ids=("C001",)))
        self.assertEqual(metadata.items[0]["title"], "Test Paper")
        self.assertNotIn("local_pdf_path", metadata.items[0])
        trace = self.local.trace_evidence_source(ChunkIdsInput(chunk_ids=("chunk-1",)))
        self.assertEqual(trace.items[0]["page_start"], 1)
        outline = self.local.get_paper_outline(CorpusInput(corpus_id="C001"))
        self.assertEqual(outline.items[0]["title"], "Methods")
        compared = self.local.compare_papers(
            ComparePapersInput(corpus_ids=("C001", "T001"), dimensions=("accuracy",))
        )
        self.assertEqual(compared.summary["dimension_count"], 1)

    def test_safe_calculation_statistics_and_reproducibility(self) -> None:
        analysis = AnalysisResearchTools(chunks=self.chunks)
        self.assertEqual(
            analysis.calculate(CalculateInput(expression="(2 + 3) * 4")).items[0]["value"], 20
        )
        with self.assertRaisesRegex(ValueError, "forbidden"):
            analysis.calculate(CalculateInput(expression="__import__('os').system('dir')"))
        stats = analysis.analyze_experiment_data(
            AnalyzeExperimentDataInput(
                columns=("score",),
                rows=((1.0,), (3.0,)),
                operations=("mean", "stdev"),
            )
        )
        self.assertEqual(stats.items[0]["mean"], 2)
        reproducibility = analysis.check_reproducibility(CorpusInput(corpus_id="C001"))
        self.assertTrue(reproducibility.items[0]["code"])
        self.assertTrue(reproducibility.items[0]["random_seed"])


if __name__ == "__main__":
    unittest.main()
