from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.models import DocumentAsset
from paper_research_agent.ingestion.parser import (
    TextLine,
    order_page_lines,
    parse_pdf_asset,
)

SHA_A = "a" * 64


class FakePage:
    width = 600.0
    height = 800.0

    def __init__(self, lines: list[dict[str, object]] | Exception) -> None:
        self._lines = lines
        self.last_extract_kwargs: dict[str, object] | None = None

    def extract_text_lines(self, **kwargs: object) -> list[dict[str, object]]:
        self.last_extract_kwargs = kwargs
        if isinstance(self._lines, Exception):
            raise self._lines
        return self._lines


class FakeDocument:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> FakeDocument:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class PdfParserTests(unittest.TestCase):
    def test_word_spacing_uses_stricter_horizontal_tolerance(self) -> None:
        page = FakePage([self._line("Body text", top=100)])

        self._parse([page])

        self.assertEqual(page.last_extract_kwargs["x_tolerance"], 2.0)

    def test_two_columns_are_read_left_then_right(self) -> None:
        lines = tuple(
            [
                TextLine("Title", 50, 20, 550, 40),
                TextLine("L1", 50, 100, 250, 110),
                TextLine("R1", 350, 100, 550, 110),
                TextLine("L2", 50, 120, 250, 130),
                TextLine("R2", 350, 120, 550, 130),
                TextLine("L3", 50, 140, 250, 150),
                TextLine("R3", 350, 140, 550, 150),
            ]
        )

        ordered = order_page_lines(lines, 600)

        self.assertEqual(
            [line.text for line in ordered],
            ["Title", "L1", "L2", "L3", "R1", "R2", "R3"],
        )

    def test_repeated_headers_and_page_numbers_are_removed(self) -> None:
        pages = [
            FakePage(
                [
                    self._line("Paper Header 2026", top=10),
                    self._line(f"Body {page_number}", top=100),
                    self._line(str(page_number), top=760),
                ]
            )
            for page_number in range(1, 4)
        ]

        result = self._parse(pages)

        self.assertEqual(len(result.pages), 3)
        self.assertTrue(all(page.status == "parsed" for page in result.pages))
        self.assertEqual(
            [page.normalized_text for page in result.pages],
            ["Body 1", "Body 2", "Body 3"],
        )
        self.assertEqual(
            [element.raw_text for element in result.elements],
            ["Body 1", "Body 2", "Body 3"],
        )

    def test_page_failure_is_isolated(self) -> None:
        pages = [
            FakePage([self._line("Body", top=100)]),
            FakePage(RuntimeError("broken page")),
        ]

        result = self._parse(pages)

        self.assertEqual(result.pages[0].status, "parsed")
        self.assertEqual(result.pages[1].status, "failed")
        self.assertEqual(result.pages[1].error_code, "page_extraction_error")
        self.assertEqual(len(result.elements), 1)

    def test_outside_lines_are_removed_and_partial_lines_are_clipped(self) -> None:
        pages = [
            FakePage(
                [
                    {
                        "text": "Partial",
                        "x0": -20.0,
                        "top": 100.0,
                        "x1": 100.0,
                        "bottom": 110.0,
                    },
                    {
                        "text": "Hidden",
                        "x0": 20.0,
                        "top": -100.0,
                        "x1": 100.0,
                        "bottom": -90.0,
                    },
                    self._line("Body", top=200),
                ]
            )
        ]

        result = self._parse(pages)

        self.assertEqual([element.raw_text for element in result.elements], ["Partial", "Body"])
        self.assertEqual(result.elements[0].bbox, (0.0, 100.0, 100.0, 110.0))

    def _parse(self, pages: list[FakePage]):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "fixture.pdf"
            pdf_path.touch()
            asset = DocumentAsset(
                asset_id="asset-a",
                corpus_id="C001",
                corpus_version="corpus-v1",
                source_sha256=SHA_A,
                source_bytes=1,
                expected_page_count=len(pages),
                storage_class="internal_research_only",
            )
            with patch(
                "paper_research_agent.ingestion.parser.pdfplumber.open",
                return_value=FakeDocument(pages),
            ):
                return parse_pdf_asset(pdf_path, asset)

    @staticmethod
    def _line(text: str, *, top: float) -> dict[str, object]:
        return {
            "text": text,
            "x0": 50.0,
            "top": top,
            "x1": 550.0,
            "bottom": top + 10.0,
        }


if __name__ == "__main__":
    unittest.main()
