from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.figures.cropper import FigureCrop
from paper_research_agent.figures.runner import (
    merge_caption_text,
    prune_orphaned_crops,
)
from paper_research_agent.ingestion.models import DocumentElement, ElementType


def element(
    identifier: str,
    order: int,
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    element_type: ElementType = "paragraph",
) -> DocumentElement:
    return DocumentElement(
        element_id=identifier,
        asset_id="asset-1",
        page_id="page-1",
        corpus_id="C001",
        page_number=1,
        element_type=element_type,
        reading_order=order,
        raw_text=text,
        normalized_text=text,
        bbox=bbox,
        normalized_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_sha256="a" * 64,
        parser_name="fixture",
        parser_version="1",
    )


class FigureRunnerTests(unittest.TestCase):
    def test_merges_adjacent_caption_lines_and_stops_at_heading(self) -> None:
        caption = element(
            "caption",
            10,
            "Figure 3: Accuracy gain over",
            (70.0, 500.0, 290.0, 510.0),
            element_type="figure_caption",
        )
        continuation = element(
            "continuation",
            11,
            "the neutral baseline.",
            (70.0, 512.0, 290.0, 522.0),
        )
        heading = element(
            "heading",
            12,
            "4 Results",
            (70.0, 530.0, 130.0, 540.0),
            element_type="heading",
        )
        self.assertEqual(
            merge_caption_text(caption, [heading, continuation, caption]),
            "Figure 3: Accuracy gain over the neutral baseline.",
        )

    def test_does_not_merge_nearby_text_from_other_column(self) -> None:
        caption = element(
            "caption",
            10,
            "Figure 1: Results.",
            (70.0, 500.0, 290.0, 510.0),
            element_type="figure_caption",
        )
        other_column = element(
            "right",
            11,
            "Unrelated paragraph.",
            (310.0, 512.0, 530.0, 522.0),
        )
        self.assertEqual(
            merge_caption_text(caption, [caption, other_column]),
            caption.normalized_text,
        )

    def test_prunes_only_unreferenced_png_inside_figures_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kept = root / "figures" / "asset-1" / "kept.png"
            orphan = root / "figures" / "asset-1" / "orphan.png"
            unrelated = root / "outside.png"
            kept.parent.mkdir(parents=True)
            kept.write_bytes(b"kept")
            orphan.write_bytes(b"orphan")
            unrelated.write_bytes(b"outside")
            crop = FigureCrop(
                figure_id="figure-1",
                asset_id="asset-1",
                corpus_id="C001",
                caption_element_id="element-1",
                figure_name="Figure 1",
                page_number=1,
                bbox=(10.0, 10.0, 100.0, 100.0),
                caption="Figure 1. Results.",
                image_path="figures/asset-1/kept.png",
            )

            removed = prune_orphaned_crops(root, [crop])

            self.assertEqual(removed, 1)
            self.assertTrue(kept.exists())
            self.assertFalse(orphan.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
