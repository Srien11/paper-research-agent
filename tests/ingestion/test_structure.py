from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.identity import sha256_text
from paper_research_agent.ingestion.models import DocumentAsset, DocumentElement
from paper_research_agent.ingestion.structure import (
    detect_caption_type,
    detect_heading,
    infer_document_structure,
)

SHA_A = "a" * 64


class DocumentStructureTests(unittest.TestCase):
    def test_table_and_figure_captions_are_classified(self) -> None:
        self.assertEqual(
            detect_caption_type(self._element(0, "Table 3: Retrieval results.")),
            "table_caption",
        )
        self.assertEqual(
            detect_caption_type(self._element(0, "Fig. 2. System overview.")),
            "figure_caption",
        )
        self.assertIsNone(
            detect_caption_type(self._element(0, "The table summarizes results."))
        )

        elements = (
            self._element(0, "A Reliable Evaluation Framework"),
            self._element(1, "1 Results"),
            self._element(2, "Table 3: Retrieval results."),
            self._element(3, "Figure 2: System overview."),
        )

        result = infer_document_structure(elements, self._asset())

        self.assertEqual(result.elements[2].element_type, "table_caption")
        self.assertEqual(result.elements[3].element_type, "figure_caption")

    def test_numbered_and_common_headings_are_detected(self) -> None:
        self.assertEqual(detect_heading(self._element(0, "1 Introduction")).level, 1)
        self.assertEqual(detect_heading(self._element(0, "1.2 Retrieval")).level, 2)
        self.assertEqual(detect_heading(self._element(0, "Abstract")).level, 1)
        self.assertIsNone(detect_heading(self._element(0, "9 Tasks")))
        self.assertIsNone(
            detect_heading(
                self._element(
                    0,
                    "2. Term-weighting fails, document expansion captures vocabulary.",
                )
            )
        )
        self.assertIsNone(
            detect_heading(
                self._element(
                    0,
                    "This long body sentence should remain a paragraph "
                    "rather than a heading.",
                )
            )
        )

    def test_elements_bind_to_nearest_section_and_nested_parent(self) -> None:
        elements = (
            self._element(0, "A Reliable Evaluation Framework"),
            self._element(1, "Abstract"),
            self._element(2, "Summary evidence."),
            self._element(3, "1 Introduction"),
            self._element(4, "Introductory evidence."),
            self._element(5, "1.1 Background"),
            self._element(6, "Background evidence."),
            self._element(7, "References", page_number=2),
            self._element(8, "[1] Source", page_number=2),
        )

        result = infer_document_structure(elements, self._asset(page_count=2))

        self.assertEqual([section.title_normalized for section in result.sections], [
            "Abstract",
            "1 Introduction",
            "1.1 Background",
            "References",
        ])
        introduction = result.sections[1]
        background = result.sections[2]
        self.assertEqual(background.parent_section_id, introduction.section_id)
        self.assertEqual(result.elements[0].element_type, "title")
        self.assertEqual(result.elements[2].section_id, result.sections[0].section_id)
        self.assertEqual(result.elements[-1].element_type, "reference")

    def test_elements_before_first_heading_have_no_section(self) -> None:
        elements = (
            self._element(0, "Authors and affiliations"),
            self._element(1, "Body before heading"),
            self._element(2, "1 Introduction"),
        )

        result = infer_document_structure(elements, self._asset())

        self.assertIsNone(result.elements[0].section_id)
        self.assertIsNone(result.elements[1].section_id)
        self.assertEqual(result.elements[2].element_type, "heading")

    @staticmethod
    def _asset(page_count: int = 1) -> DocumentAsset:
        return DocumentAsset(
            asset_id="asset-a",
            corpus_id="C001",
            corpus_version="corpus-v1",
            source_sha256=SHA_A,
            source_bytes=100,
            expected_page_count=page_count,
            storage_class="internal_research_only",
        )

    @staticmethod
    def _element(
        reading_order: int,
        text: str,
        *,
        page_number: int = 1,
    ) -> DocumentElement:
        normalized_hash = sha256_text(text)
        return DocumentElement(
            element_id=f"element-{page_number}-{reading_order}",
            asset_id="asset-a",
            page_id=f"page-{page_number}",
            corpus_id="C001",
            page_number=page_number,
            element_type="paragraph",
            reading_order=reading_order,
            raw_text=text,
            normalized_text=text,
            normalized_text_sha256=normalized_hash,
            source_sha256=SHA_A,
            parser_name="test",
            parser_version="1",
        )


if __name__ == "__main__":
    unittest.main()
