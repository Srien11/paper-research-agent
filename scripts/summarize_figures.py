"""调用视觉模型生成结构化图片信息。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.figures.dashscope import DashScopeVisionSummarizer
from paper_research_agent.figures.semantics import run_figure_summarization
from paper_research_agent.figures.summarizer import VisionSummarizer, ZaiCliVisionSummarizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provider", choices=("zai-cli", "dashscope"), default="zai-cli")
    parser.add_argument("--model-id", action="append", required=True)
    parser.add_argument("--executable", default="z-ai")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    output_path = args.output or args.candidates.parent / "figures.jsonl"
    summarizer: VisionSummarizer
    if args.provider == "dashscope":
        summarizer = DashScopeVisionSummarizer(
            model_ids=args.model_id,
            api_key_env=args.api_key_env,
            base_url=args.base_url or os.getenv("DASHSCOPE_BASE_URL"),
            timeout_seconds=args.timeout_seconds,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
        )
    else:
        if len(args.model_id) != 1:
            parser.error("zai-cli provider 只允许一个 --model-id")
        summarizer = ZaiCliVisionSummarizer(
            model_id=args.model_id[0],
            executable=args.executable,
            timeout_seconds=args.timeout_seconds,
        )
    records = run_figure_summarization(
        args.candidates,
        output_path,
        summarizer,
        limit=args.limit,
        workers=args.workers,
    )
    model_counts: dict[str, int] = {}
    for record in records:
        model_counts[record.model_id] = model_counts.get(record.model_id, 0) + 1
    result: dict[str, object] = {
        "output_path": str(output_path),
        "figure_count": len(records),
        "model_counts": model_counts,
        "model_id": next(iter(model_counts)) if len(model_counts) == 1 else None,
        "prompt_version": summarizer.prompt_version,
    }
    if isinstance(summarizer, DashScopeVisionSummarizer):
        result["usage"] = summarizer.usage_report()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
