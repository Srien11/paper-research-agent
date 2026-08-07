"""Validate a private RAG gold dataset and replay its frozen evidence sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from paper_research_agent.evaluation.gold_dataset import (
    dataset_summary,
    load_gold_dataset,
    validate_source_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Private gold JSONL, normally below data/evaluations/gold/.",
    )
    parser.add_argument(
        "--elements",
        type=Path,
        required=True,
        help="Frozen elements.jsonl used to replay raw spans.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Current chunks.jsonl used to validate chunk projections.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        questions = load_gold_dataset(args.dataset)
        replay = validate_source_replay(questions, args.elements, args.chunks)
    except (OSError, ValueError) as error:
        # Gold validation can process copyrighted source spans.  Do not print
        # exception payloads, source text, or absolute paths on failure.
        print(f"gold validation failed: {type(error).__name__}", file=sys.stderr)
        return 1
    result = {
        **dataset_summary(questions),
        "source_replay": replay.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
