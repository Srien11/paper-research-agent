from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.comparison_end_to_end import (
    COMPARISON_E2E_RUN_SCHEMA_VERSION,
    ComparisonCaseDiagnostic,
    aggregate_compilation_audits,
)


def recompute_payload(
    payload: Mapping[str, object],
    *,
    source_run_sha256: str,
) -> dict[str, object]:
    """Recompute body-free Compiler summaries without invoking runtime services."""
    raw_cases = payload.get("cases", ())
    if not isinstance(raw_cases, (list, tuple)):
        raise TypeError("run cases must be a list")
    cases = tuple(ComparisonCaseDiagnostic.model_validate(item) for item in raw_cases)

    updated = dict(payload)
    updated["schema_version"] = COMPARISON_E2E_RUN_SCHEMA_VERSION
    updated["source_run_sha256"] = source_run_sha256
    updated["compilation"] = aggregate_compilation_audits(cases)

    raw_split_summaries = updated.get("split_summaries", {})
    if not isinstance(raw_split_summaries, Mapping):
        raise TypeError("run split_summaries must be an object")
    split_summaries = dict(raw_split_summaries)
    for split in sorted({case.split for case in cases}):
        raw_split_summary = split_summaries.get(split, {})
        if not isinstance(raw_split_summary, Mapping):
            raise TypeError(f"split summary for {split} must be an object")
        split_summary = dict(raw_split_summary)
        split_summary["compilation"] = aggregate_compilation_audits(
            case for case in cases if case.split == split
        )
        split_summaries[split] = split_summary
    updated["split_summaries"] = split_summaries
    return updated


def recompute_file(input_path: Path, output_path: Path) -> dict[str, object]:
    """Read one run and atomically save its recomputed summary to a new path."""
    resolved_input = input_path.resolve(strict=True)
    resolved_output = output_path.resolve(strict=False)
    if resolved_input == resolved_output:
        raise ValueError("input and output must be different paths")
    if resolved_output.exists():
        raise FileExistsError(f"output already exists: {resolved_output}")

    raw_input = resolved_input.read_bytes()
    parsed: Any = json.loads(raw_input.decode("utf-8"))
    if not isinstance(parsed, Mapping):
        raise TypeError("run JSON root must be an object")
    updated = recompute_payload(
        parsed,
        source_run_sha256=hashlib.sha256(raw_input).hexdigest(),
    )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{resolved_output.name}.",
            suffix=".tmp",
            dir=resolved_output.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(updated, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if resolved_output.exists():
            raise FileExistsError(f"output already exists: {resolved_output}")
        os.replace(temporary_path, resolved_output)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return updated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute comparison Compiler summaries without model calls."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _count(summary: Mapping[str, object], section: str, field: str) -> object:
    nested = summary.get(section)
    if not isinstance(nested, Mapping):
        raise TypeError(f"compilation summary is missing {section}")
    return nested.get(field, 0)


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    updated = recompute_file(args.input, args.output)
    compilation = updated.get("compilation")
    if not isinstance(compilation, Mapping):
        raise TypeError("recomputed compilation summary is invalid")
    cases = updated.get("cases", ())
    if not isinstance(cases, (list, tuple)):
        raise TypeError("recomputed cases are invalid")

    print(f"schema_version={updated['schema_version']}")
    print(f"question_count={len(cases)}")
    print(f"attempt_count={_count(compilation, 'attempts', 'attempt_count')}")
    print(f"retry_case_count={compilation.get('retry_case_count', 0)}")
    print(
        "attempt_requested_unit_count="
        f"{_count(compilation, 'attempts', 'requested_unit_count')}"
    )
    print(
        "attempt_accepted_unit_count="
        f"{_count(compilation, 'attempts', 'accepted_unit_count')}"
    )
    print(
        "attempt_failed_unit_count="
        f"{_count(compilation, 'attempts', 'failed_unit_count')}"
    )
    print(
        "final_requested_unit_count="
        f"{_count(compilation, 'final', 'requested_unit_count')}"
    )
    print(
        "final_accepted_unit_count="
        f"{_count(compilation, 'final', 'accepted_unit_count')}"
    )
    print(
        "final_failed_unit_count="
        f"{_count(compilation, 'final', 'failed_unit_count')}"
    )
    print(
        "final_unresolved_fact_requirement_count="
        f"{_count(compilation, 'final', 'unresolved_fact_requirement_count')}"
    )
    print(
        "final_retained_fact_count="
        f"{_count(compilation, 'final', 'retained_fact_count')}"
    )
    print(f"output={args.output.resolve(strict=False)}")


if __name__ == "__main__":
    main()
