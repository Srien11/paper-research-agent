from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import ClassVar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.figures.cropper import (
    detect_figure_bbox,
    extract_figure_name,
)


class FakePage:
    width = 600.0
    height = 800.0
    objects: ClassVar[dict[str, list[dict[str, float]]]] = {
        "image": [
            {"x0": 100.0, "top": 80.0, "x1": 500.0, "bottom": 170.0},
        ],
        "curve": [
            {"x0": 120.0, "top": 60.0, "x1": 480.0, "bottom": 100.0},
        ],
        "rect": [
            {"x0": 20.0, "top": 25.0, "x1": 580.0, "bottom": 26.0},
        ],
    }


class FigureCropperTests(unittest.TestCase):
    def test_extracts_common_figure_names(self) -> None:
        self.assertEqual(extract_figure_name("Figure 3. Architecture"), "Figure 3")
        self.assertEqual(extract_figure_name("Fig. 2: Results"), "Fig. 2")
        self.assertEqual(extract_figure_name("FIGURE A1 Supplement"), "FIGURE A1")

    def test_detects_nearest_graphic_group_above_caption(self) -> None:
        bbox = detect_figure_bbox(
            FakePage(),
            (150.0, 190.0, 450.0, 202.0),
        )
        self.assertEqual(bbox, (27.0, 52.0, 573.0, 178.0))

    def test_uses_bounded_fallback_without_graphic_objects(self) -> None:
        page = FakePage()
        page.objects = {}
        bbox = detect_figure_bbox(page, (40.0, 400.0, 250.0, 412.0))
        self.assertEqual(bbox, (27.0, 80.0, 306.0, 396.0))

    def test_merges_plot_and_legend_separated_by_large_gap(self) -> None:
        page = FakePage()
        page.objects = {
            "line": [
                {"x0": 80.0, "top": 100.0, "x1": 270.0, "bottom": 260.0},
                {"x0": 110.0, "top": 305.0, "x1": 240.0, "bottom": 320.0},
            ]
        }
        bbox = detect_figure_bbox(page, (40.0, 340.0, 250.0, 352.0))
        self.assertEqual(bbox, (27.0, 92.0, 306.0, 328.0))


if __name__ == "__main__":
    unittest.main()
