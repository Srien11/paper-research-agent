from __future__ import annotations

import hashlib
import unittest

from paper_research_agent.agent.tooling.content import ContentResearchTools
from paper_research_agent.agent.tooling.contracts import ElementLookupInput
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.figures.models import FigureRecord
from paper_research_agent.ingestion.models import DocumentElement


def _element(element_type: str, text: str, index: int) -> DocumentElement:
    return DocumentElement(
        element_id=f"element-{index}",
        asset_id="asset-1",
        page_id="page-1",
        corpus_id="C001",
        page_number=1,
        section_id="results",
        element_type=element_type,
        reading_order=index,
        raw_text=text,
        normalized_text=text,
        normalized_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_sha256="a" * 64,
        parser_name="test",
        parser_version="1",
    )


class ContentTests(unittest.TestCase):
    def test_reads_tables_equations_and_stored_figures(self) -> None:
        figure = FigureRecord(
            figure_id="figure-1",
            asset_id="asset-1",
            figure_name="Figure 1",
            page_number=1,
            bbox=(0, 0, 10, 10),
            caption="Accuracy curve",
            image_path="figures/f1.png",
            figure_type="line_chart",
            summary="Accuracy increases.",
            key_findings=("Higher is better",),
            recognition_confidence=0.9,
            model_id="vlm-test",
            prompt_version="v1",
        )
        figure_text = "Figure 1 Accuracy curve"
        figure_chunk = EvidenceChunk(
            chunk_id="figure-chunk",
            asset_id="asset-1",
            corpus_id="C001",
            element_ids=("figure-1",),
            page_start=1,
            page_end=1,
            token_start=0,
            token_end=4,
            text=figure_text,
            text_sha256=hashlib.sha256(figure_text.encode()).hexdigest(),
            config_sha256="b" * 64,
            evidence_type="figure_summary",
            content_origin="generated",
            figure=figure,
        )
        tools = ContentResearchTools(
            elements=(_element("table", "Table 1 | A | 1", 1), _element("formula", "x = y + 1", 2)),
            chunks=(figure_chunk,),
        )
        self.assertEqual(
            tools.extract_table(ElementLookupInput(corpus_id="C001")).items[0]["element_type"],
            "table",
        )
        self.assertEqual(
            tools.extract_equation(ElementLookupInput(corpus_id="C001")).items[0]["element_type"],
            "formula",
        )
        inspected = tools.inspect_figure(ElementLookupInput(corpus_id="C001", label="Figure 1"))
        self.assertEqual(inspected.items[0]["summary"], "Accuracy increases.")
        self.assertNotIn("image_path", inspected.items[0])


if __name__ == "__main__":
    unittest.main()
