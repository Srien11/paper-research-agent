"""审计解析产物，并生成不含论文正文的聚合质量报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from paper_research_agent.ingestion.quality import (  # noqa: E402
    QualityAuditError,
    assess_ingestion,
    write_quality_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manual-review-id",
        action="append",
        default=[],
        help="仍需人工复核的 corpus_id，可重复传入。",
    )
    parser.add_argument(
        "--known-warning",
        action="append",
        default=[],
        help="不包含正文的已知解析警告，可重复传入。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        assessment = assess_ingestion(
            args.build_dir,
            manual_review_ids=tuple(args.manual_review_id),
            known_warnings=tuple(args.known_warning),
        )
    except QualityAuditError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.output is not None:
        write_quality_report(args.output, assessment)
    print(
        json.dumps(
            assessment.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if assessment.status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())

