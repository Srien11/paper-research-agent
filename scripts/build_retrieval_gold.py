from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.gold_dataset import load_gold_dataset
from paper_research_agent.evaluation.retrieval_gold import build_retrieval_gold


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a private 30-query span retrieval view.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/rag-answer-candidates-v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/retrieval-gold-v1.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    rows = build_retrieval_gold(load_gold_dataset(args.dataset), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(item.model_dump_json() + "\n" for item in rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "query_count": len(rows),
                "category_counts": {
                    category: sum(item.category == category for item in rows)
                    for category in sorted({item.category for item in rows})
                },
                "annotation_status": sorted({item.annotation_status for item in rows}),
                "output": args.output.name,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
