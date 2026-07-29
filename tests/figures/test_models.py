from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.figures.identity import make_figure_id
from paper_research_agent.figures.models import FigureRecord


def record(**overrides: object) -> FigureRecord:
    payload: dict[str, object] = {
        "figure_id": "figure_001",
        "asset_id": "asset_001",
        "figure_name": "Figure 3",
        "page_number": 12,
        "bbox": (72.0, 180.0, 520.0, 610.0),
        "caption": "Figure 3. Overall architecture.",
        "image_path": "figures/asset_001/p12_fig3.png",
        "figure_type": "系统架构图",
        "summary": "该图展示了系统的主要组件与数据流。",
        "key_findings": ("检索模块连接生成模块", "证据随答案返回"),
        "recognition_confidence": 0.86,
        "content_origin": "视觉模型生成",
        "model_id": "vision-model",
        "prompt_version": "figure-summary-v1",
    }
    payload.update(overrides)
    return FigureRecord.model_validate(payload)


class FigureRecordTests(unittest.TestCase):
    def test_accepts_exact_requested_fields(self) -> None:
        value = record()
        self.assertEqual(
            set(value.model_dump()),
            {
                "figure_id",
                "asset_id",
                "figure_name",
                "page_number",
                "bbox",
                "caption",
                "image_path",
                "figure_type",
                "summary",
                "key_findings",
                "recognition_confidence",
                "content_origin",
                "model_id",
                "prompt_version",
            },
        )

    def test_rejects_invalid_bbox_and_unsafe_image_path(self) -> None:
        with self.assertRaisesRegex(ValidationError, "正面积"):
            record(bbox=(72.0, 180.0, 72.0, 610.0))
        with self.assertRaisesRegex(ValidationError, "安全相对路径"):
            record(image_path="../outside.png")
        with self.assertRaisesRegex(ValidationError, "安全相对路径"):
            record(image_path=r"figures\paper.png")

    def test_requires_generated_content_lineage(self) -> None:
        with self.assertRaises(ValidationError):
            record(content_origin="source_text")
        with self.assertRaises(ValidationError):
            record(model_id="")

    def test_figure_id_is_stable_and_position_sensitive(self) -> None:
        bbox = (72.0, 180.0, 520.0, 610.0)
        first = make_figure_id("asset_001", 12, "Figure 3", bbox)
        second = make_figure_id("asset_001", 12, "Figure 3", bbox)
        moved = make_figure_id("asset_001", 12, "Figure 3", (72.0, 181.0, 520.0, 610.0))
        self.assertEqual(first, second)
        self.assertNotEqual(first, moved)


if __name__ == "__main__":
    unittest.main()
