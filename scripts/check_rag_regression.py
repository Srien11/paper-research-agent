from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.regression_gate import (
    RegressionGateConfig,
    evaluate_regression_gate,
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("regression input must be a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare safe RAG evaluation metrics with a frozen baseline."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation/rag-gold-regression-gate-v1.json",
    )
    args = parser.parse_args()
    config = RegressionGateConfig.model_validate(_load_object(args.config))
    result = evaluate_regression_gate(
        _load_object(args.baseline),
        _load_object(args.candidate),
        config,
    )
    print(result.model_dump_json(indent=2))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
