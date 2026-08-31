from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.ragas_eval import (
    RagasEvaluationConfig,
    merge_ragas_results,
    write_ragas_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge privacy-safe Ragas batch results from isolated processes."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation/ragas-v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/runs/ragas-gold-v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/Ragas论文研究Agent评测-v1.md",
    )
    args = parser.parse_args()

    config = RagasEvaluationConfig.from_path(args.config)
    batches = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    context = dict(batches[0].get("evaluation_context", {}))
    context.update(
        {
            "process_isolation_batch_size": config.max_concurrency,
            "batch_count": len(batches),
        }
    )
    result = merge_ragas_results(
        batches,
        args.output,
        evaluation_context=context,
    )
    write_ragas_report(
        result,
        args.report,
        thresholds=config.diagnostic_thresholds,
        release_gate_enabled=config.release_gate_enabled,
    )
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
