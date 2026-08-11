from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Self
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class PdfParserTests(unittest.TestCase):
    def test_word_spacing_uses_stricter_horizontal_tolerance(self) -> None:
        page = FakePage([self._line("Body text", top=100)])

        self._parse([page])

        self.assertEqual(page.last_extract_kwargs["x_tolerance"], 2.0)
        self.assertIs(page.last_extract_kwargs["return_chars"], True)

    def test_rotated_side_margin_watermark_is_removed(self) -> None:
        rotated = self._line("arXiv watermark", top=100)
        rotated.update(
            {
                "x0": 10.0,
                "x1": 20.0,
                "chars": [{"text": "a", "upright": False}],
            }
        )
        body = self._line("Body", top=200)

        result = self._parse([FakePage([rotated, body])])

        self.assertEqual([element.raw_text for element in result.elements], ["Body"])

    def test_rotated_text_inside_page_is_preserved(self) -> None:
        chart_label = self._line("vertical chart label", top=100)
        chart_label.update(
            {
                "x0": 100.0,
                "x1": 110.0,
                "chars": [{"text": "v", "upright": False}],
            }
        )

        result = self._parse([FakePage([chart_label])])

        self.assertEqual(
            [element.raw_text for element in result.elements],
            ["vertical chart label"],
        )

    def test_two_columns_are_read_left_then_right(self) -> None:
        lines = (
            TextLine("Title", 50, 20, 550, 40),
            TextLine("L1", 50, 100, 250, 110),
            TextLine("R1", 350, 100, 550, 110),
            TextLine("L2", 50, 120, 250, 130),
            TextLine("R2", 350, 120, 550, 130),
            TextLine("L3", 50, 140, 250, 150),
            TextLine("R3", 350, 140, 550, 150),
        )

        ordered = order_page_lines(lines, 600)

        self.assertEqual(
            [line.text for line in ordered],
            ["Title", "L1", "L2", "L3", "R1", "R2", "R3"],
        )

    def test_merged_cross_column_lines_are_split_before_ordering(self) -> None:
        lines = [
            self._merged_line(f"L{number}", f"R{number}", top=100 + number * 20)
            for number in range(1, 4)
        ]

        result = self._parse([FakePage(lines)])

        self.assertEqual(
            [element.raw_text for element in result.elements],
            ["L1", "L2", "L3", "R1", "R2", "R3"],
        )

    def test_split_columns_preserve_spaces_from_original_line(self) -> None:
        line = self._merged_line("Left words", "Right words", top=100)
        line["text"] = "Left words Right words"
        line["chars"] = [
            *self._chars("Leftwords", start=50, top=100),
            *self._chars("Rightwords", start=350, top=100),
        ]
        lines = [line, {**line, "top": 120.0, "bottom": 130.0}, {**line, "top": 140.0, "bottom": 150.0}]

        result = self._parse([FakePage(lines)])

        self.assertEqual(result.elements[0].raw_text, "Left words")
        self.assertEqual(result.elements[3].raw_text, "Right words")

    def test_narrow_nature_style_column_gap_is_split(self) -> None:
        lines = [
            self._narrow_merged_line(f"L{number}", f"R{number}", top=100 + number * 20)
            for number in range(1, 4)
        ]

        result = self._parse([FakePage(lines)])

        self.assertEqual(
            [element.raw_text for element in result.elements],
            ["L1", "L2", "L3", "R1", "R2", "R3"],
        )

    def test_asymmetric_nature_sidebar_is_split_from_body(self) -> None:
        lines = [
            self._sidebar_merged_line(f"M{number}", f"B{number}", top=100 + number * 20)
            for number in range(1, 4)
        ]

        result = self._parse([FakePage(lines)])

        self.assertEqual(
            [element.raw_text for element in result.elements],
            ["M1", "M2", "M3", "B1", "B2", "B3"],
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

    @staticmethod
    def _merged_line(left: str, right: str, *, top: float) -> dict[str, object]:
        chars = [
            {
                "text": text,
                "x0": x0,
                "x1": x0 + 5,
                "top": top,
                "bottom": top + 10,
                "upright": True,
            }
            for text, x0 in [
                (left[0], 50),
                (left[1], 55),
                (right[0], 350),
                (right[1], 355),
            ]
        ]
        return {
            "text": f"{left} {right}",
            "x0": 50.0,
            "top": top,
            "x1": 360.0,
            "bottom": top + 10.0,
            "chars": chars,
        }

    @staticmethod
    def _chars(text: str, *, start: float, top: float) -> list[dict[str, object]]:
        return [
            {
                "text": character,
                "x0": start + index * 5,
                "x1": start + (index + 1) * 5,
                "top": top,
                "bottom": top + 10,
                "upright": True,
            }
            for index, character in enumerate(text)
        ]

    @classmethod
    def _narrow_merged_line(
        cls,
        left: str,
        right: str,
        *,
        top: float,
    ) -> dict[str, object]:
        left_chars = cls._chars(left, start=284.8, top=top)
        right_chars = cls._chars(right, start=306.1, top=top)
        return {
            "text": f"{left} {right}",
            "x0": 284.8,
            "top": top,
            "x1": 316.1,
            "bottom": top + 10,
            "chars": [*left_chars, *right_chars],
        }

    @classmethod
    def _sidebar_merged_line(
        cls,
        left: str,
        right: str,
        *,
        top: float,
    ) -> dict[str, object]:
        left_chars = cls._chars(left, start=110, top=top)
        right_chars = cls._chars(right, start=217.3, top=top)
        return {
            "text": f"{left} {right}",
            "x0": 110.0,
            "top": top,
            "x1": 227.3,
            "bottom": top + 10,
            "chars": [*left_chars, *right_chars],
        }


if __name__ == "__main__":
    unittest.main()
