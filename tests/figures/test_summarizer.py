from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.figures.semantics import run_figure_summarization
from paper_research_agent.figures.summarizer import (
    PROMPT_VERSION,
    VisionSummary,
    build_summary_prompt,
    parse_summary_response,
)


class FakeSummarizer:
    model_id = "fixture-vision-v1"
    prompt_version = PROMPT_VERSION

    def __init__(self) -> None:
        self.calls = 0

    def summarize(
        self,
        image_path: Path,
        *,
        figure_name: str,
        caption: str,
    ) -> VisionSummary:
        self.calls += 1
        self.last_request = (image_path, figure_name, caption)
        return VisionSummary(
            figure_type="系统架构图",
            summary="该图展示检索、重排与生成模块之间的数据流。",
            key_findings=("检索结果进入重排模块", "生成模块保留证据引用"),
            recognition_confidence=0.9,
        )


class VisionSummarizerTests(unittest.TestCase):
    def test_prompt_requires_visible_evidence_and_exact_json_fields(self) -> None:
        prompt = build_summary_prompt(
            figure_name="Figure 3",
            caption="Figure 3. Overall architecture.",
        )
        self.assertIn("只根据图片中可见内容", prompt)
        self.assertIn("只能包含", prompt)
        self.assertIn("Figure 3", prompt)

    def test_parses_fenced_json_and_rejects_extra_fields(self) -> None:
        summary = parse_summary_response(
            """```json
{"figure_type":"曲线图","summary":"展示性能趋势","key_findings":["A 高于 B"],"recognition_confidence":0.8}
```"""
        )
        self.assertEqual(summary.figure_type, "曲线图")
        with self.assertRaises(ValidationError):
            parse_summary_response(
                '{"figure_type":"图","summary":"摘要","key_findings":[],"recognition_confidence":0.8,"extra":1}'
            )

    def test_run_writes_exact_record_and_resumes_without_duplicate_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "figures" / "asset_001" / "p0012.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"PNG fixture")
            candidates_path = root / "figure_candidates.jsonl"
            candidate = {
                "figure_id": "figure_001",
                "asset_id": "asset_001",
                "corpus_id": "C001",
                "caption_element_id": "element_001",
                "figure_name": "Figure 3",
                "page_number": 12,
                "bbox": [72.0, 180.0, 520.0, 610.0],
                "caption": "Figure 3. Overall architecture.",
                "image_path": "figures/asset_001/p0012.png",
            }
            candidates_path.write_text(
                json.dumps(candidate, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output_path = root / "figures.jsonl"
            summarizer = FakeSummarizer()

            first = run_figure_summarization(
                candidates_path,
                output_path,
                summarizer,
            )
            second = run_figure_summarization(
                candidates_path,
                output_path,
                summarizer,
            )

            self.assertEqual(first, second)
            self.assertEqual(summarizer.calls, 1)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 14)
            self.assertEqual(payload["content_origin"], "视觉模型生成")
            self.assertEqual(payload["model_id"], "fixture-vision-v1")
            self.assertEqual(payload["prompt_version"], PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
