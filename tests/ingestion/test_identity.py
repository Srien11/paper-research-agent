from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.ingestion.identity import (
    make_build_id,
    make_element_id,
    make_page_id,
    sha256_text,
)
from paper_research_agent.ingestion.text import normalize_text


class DeterministicIdentityTests(unittest.TestCase):
    def test_equivalent_whitespace_normalizes_to_same_text(self) -> None:
        first = normalize_text("Evidence\u00a0  from\npaper")
        second = normalize_text("Evidence from paper")

        self.assertEqual(first, second)
        self.assertEqual(sha256_text(first), sha256_text(second))

    def test_hyphenated_line_break_is_rejoined(self) -> None:
        self.assertEqual(
            normalize_text("retriev-\nal quality"),
            "retrieval quality",
        )

    def test_page_id_is_stable_and_page_sensitive(self) -> None:
        source_hash = "a" * 64

        self.assertEqual(
            make_page_id(source_hash, 1),
            make_page_id(source_hash, 1),
        )
        self.assertNotEqual(
            make_page_id(source_hash, 1),
            make_page_id(source_hash, 2),
        )
        with self.assertRaisesRegex(ValueError, "页码必须从 1 开始"):
            make_page_id(source_hash, 0)

    def test_duplicate_text_uses_reading_order_to_avoid_collision(self) -> None:
        source_hash = "a" * 64
        text_hash = sha256_text(normalize_text("Repeated"))

        first = make_element_id(source_hash, 1, "paragraph", 0, text_hash)
        second = make_element_id(source_hash, 1, "paragraph", 1, text_hash)

        self.assertNotEqual(first, second)

    def test_build_id_ignores_source_hash_input_order(self) -> None:
        values = ("corpus-v1", "pdfplumber", "0.11.10", "c" * 64)

        first = make_build_id(*values, ["a" * 64, "b" * 64])
        second = make_build_id(*values, ["b" * 64, "a" * 64])

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

