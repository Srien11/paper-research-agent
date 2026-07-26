"""Validate the frozen corpus before parsing or indexing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from paper_research_agent import CorpusValidationError, validate_corpus_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory containing core_frozen.jsonl and challenge_frozen.jsonl.",
    )
    parser.add_argument(
        "--skip-local-pdf-check",
        action="store_true",
        help="Validate metadata without requiring local PDF files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [
        args.corpus_dir / "core_frozen.jsonl",
        args.corpus_dir / "challenge_frozen.jsonl",
    ]
    try:
        report = validate_corpus_files(
            paths,
            require_local_pdfs=not args.skip_local_pdf_check,
        )
    except CorpusValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

