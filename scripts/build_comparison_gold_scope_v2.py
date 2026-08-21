from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.candidate_gold import load_candidate_paper_gold
from paper_research_agent.evaluation.comparison_end_to_end import ComparisonEndToEndGold
from paper_research_agent.evaluation.gold_scope import (
    ComparisonGoldScopeManifest,
    apply_comparison_gold_scope,
)


def _load_gold(path: Path) -> tuple[ComparisonEndToEndGold, ...]:
    return tuple(
        ComparisonEndToEndGold.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    questions = load_candidate_paper_gold(args.questions)
    gold_rows = _load_gold(args.source_gold)
    manifest = ComparisonGoldScopeManifest.model_validate_json(
        args.scope.read_text(encoding="utf-8")
    )
    question_by_id = {item.question_id: item for item in questions}
    gold_by_id = {item.question_id: item for item in gold_rows}
    scope_by_id = {item.question_id: item for item in manifest.questions}
    expected_ids = set(question_by_id) & set(gold_by_id)
    if set(scope_by_id) != expected_ids:
        raise ValueError("scope question IDs must exactly match the source evaluation set")

    corrected = tuple(
        apply_comparison_gold_scope(
            question=question_by_id[question_id].question,
            gold=gold_by_id[question_id],
            scope=scope_by_id[question_id],
        )
        for question_id in sorted(expected_ids)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(item.model_dump_json() + "\n" for item in corrected),
        encoding="utf-8",
    )
    source_claim_count = sum(len(item.must_have_claims) for item in gold_rows)
    required_claim_count = sum(len(item.must_have_claims) for item in corrected)
    reason_counts: dict[str, int] = {}
    for scope in manifest.questions:
        for claim in scope.optional_claims:
            reason_counts[claim.reason] = reason_counts.get(claim.reason, 0) + 1
    metadata: dict[str, object] = {
        "schema_version": "comparison-end-to-end-gold-meta-v2",
        "question_count": len(corrected),
        "construction": "scope-projection-from-v1",
        "scope_adjudication": "agent-reviewed-question-to-claim-alignment",
        "human_adjudication_claimed": False,
        "source_gold_sha256": _sha256(args.source_gold),
        "scope_manifest_sha256": _sha256(args.scope),
        "source_claim_count": source_claim_count,
        "required_claim_count": required_claim_count,
        "optional_claim_count": source_claim_count - required_claim_count,
        "optional_reason_counts": dict(sorted(reason_counts.items())),
    }
    args.meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build scope-corrected comparison gold v2.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/candidate-paper-gold-v1.jsonl",
    )
    parser.add_argument(
        "--source-gold",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/comparison-end-to-end-gold-v1.jsonl",
    )
    parser.add_argument(
        "--scope",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/comparison-end-to-end-gold-scope-v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/comparison-end-to-end-gold-v2.jsonl",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/comparison-end-to-end-gold-v2.meta.json",
    )
    return parser


def main() -> None:
    metadata = run(_build_parser().parse_args())
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
