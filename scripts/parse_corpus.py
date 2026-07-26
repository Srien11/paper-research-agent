"""将冻结论文语料解析为页面、章节和元素产物。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from paper_research_agent.ingestion.runner import (  # noqa: E402
    IngestionRunError,
    run_corpus_ingestion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="包含 core_frozen.jsonl 和 challenge_frozen.jsonl 的目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="本地解析产物根目录，默认位于被 Git 忽略的 data/processed。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_corpus_ingestion(args.corpus_dir, args.output_root)
    except IngestionRunError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "manifest": result.manifest.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

