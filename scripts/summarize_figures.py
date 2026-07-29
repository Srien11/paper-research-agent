"""调用视觉模型生成结构化图片信息。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.figures.semantics import run_figure_summarization
from paper_research_agent.figures.summarizer import ZaiCliVisionSummarizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--executable", default="z-ai")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output_path = args.output or args.candidates.parent / "figures.jsonl"
    summarizer = ZaiCliVisionSummarizer(
        model_id=args.model_id,
        executable=args.executable,
        timeout_seconds=args.timeout_seconds,
    )
    records = run_figure_summarization(
        args.candidates,
        output_path,
        summarizer,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "figure_count": len(records),
                "model_id": summarizer.model_id,
                "prompt_version": summarizer.prompt_version,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
