from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.comparison_end_to_end import (
    ComparisonCaseDiagnostic,
    aggregate_answer_scores,
    percentile,
)
from paper_research_agent.evaluation.hydration_diagnostics import (
    FactRankingDiagnostic,
    summarize_hydration_cutoffs,
)


def summarize_run_payload(payload: dict[str, Any]) -> dict[str, object]:
    """Create a privacy-safe summary from one completed E2E run payload."""
    cases = tuple(
        ComparisonCaseDiagnostic.model_validate(item) for item in payload.get("cases", ())
    )
    facts = tuple(
        FactRankingDiagnostic(
            fact_id=f"{case.question_id}:{item.claim_id}",
            loss_stage=item.loss_stage,
            semantic_alternative=item.same_paper_alternative_chunk_compiled,
            best_final_rank=item.best_final_rank,
            stage_ranks=item.best_stage_ranks,
            search_occurrences=item.search_occurrences,
            same_page_top4=item.same_page_top4,
            same_section_top4=item.same_section_top4,
        )
        for case in cases
        for item in case.fact_lineage
    )
    hydration = summarize_hydration_cutoffs(facts, cutoffs=(4, 6, 8, 10))
    answer_scores = aggregate_answer_scores(cases)
    latencies = tuple(case.total_latency_ms for case in cases)
    return {
        "experiment_fingerprint": payload.get("experiment_fingerprint"),
        "code_revision": payload.get("code_revision"),
        "retrieval_config_sha256": payload.get("retrieval_config_sha256"),
        "checkpoint_id": payload.get("checkpoint_id"),
        "evidence_per_step": payload.get("evidence_per_step"),
        "rerank_mode": payload.get("rerank_mode"),
        "question_count": len(cases),
        "planning_contract_failure_count": sum(
            case.error_reason_code == "planner_contract_invalid" for case in cases
        ),
        "hydration": hydration.model_dump(),
        "answer_scores": answer_scores,
        "latency_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "mean_generation_input_tokens": (
            statistics.fmean(case.generation_input_tokens for case in cases)
            if cases
            else None
        ),
        "mean_tool_call_count": (
            statistics.fmean(case.tool_call_count for case in cases) if cases else None
        ),
    }


def render_markdown(summaries: tuple[dict[str, object], ...]) -> str:
    """Render only aggregate values and non-reversible experiment identifiers."""
    lines = [
        "# 检索事实块排名与水化消融汇总",
        "",
        "本报告只包含安全标识、名次、计数和聚合指标，不包含题目、答案或证据正文。",
        "",
        "| 证据数 | 重排 | 题数 | 真实水化缺失 | Top 4 剩余 | Top 6 剩余 | Top 8 剩余 | Top 10 剩余 | 必要事实覆盖率 | 引用正确率 | 禁用事实率 | 严格通过率 | P95 延迟(ms) | 平均输入 Token | 平均工具调用 | 规划失败 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        hydration = summary["hydration"]
        assert isinstance(hydration, dict)
        remaining = hydration["remaining_by_cutoff"]
        assert isinstance(remaining, dict)
        answer_scores = summary["answer_scores"]
        assert isinstance(answer_scores, dict)
        model_judge = answer_scores["model_judge"]
        assert isinstance(model_judge, dict)
        latency = summary["latency_ms"]
        assert isinstance(latency, dict)
        lines.append(
            "| {evidence} | {rerank} | {questions} | {losses} | {top4} | {top6} | "
            "{top8} | {top10} | {coverage} | {citation} | {forbidden} | {strict} | "
            "{p95} | {tokens} | {calls} | {planning} |".format(
                evidence=summary.get("evidence_per_step", "—"),
                rerank=summary.get("rerank_mode", "—"),
                questions=summary["question_count"],
                losses=hydration["true_hydration_loss_count"],
                top4=remaining.get(4, remaining.get("4", "—")),
                top6=remaining.get(6, remaining.get("6", "—")),
                top8=remaining.get(8, remaining.get("8", "—")),
                top10=remaining.get(10, remaining.get("10", "—")),
                coverage=_percent(model_judge.get("must_have_claim_recall")),
                citation=_percent(model_judge.get("citation_correctness")),
                forbidden=_percent(model_judge.get("forbidden_claim_rate")),
                strict=_percent(answer_scores.get("end_to_end_all_correct_rate")),
                p95=_number(latency.get("p95")),
                tokens=_number(summary.get("mean_generation_input_tokens")),
                calls=_number(summary.get("mean_tool_call_count")),
                planning=summary["planning_contract_failure_count"],
            )
        )
    lines.extend(["", "## 排名诊断", ""])
    for summary in summaries:
        hydration = summary["hydration"]
        assert isinstance(hydration, dict)
        lines.extend(
            [
                (
                    f"- `{str(summary.get('experiment_fingerprint', 'unknown'))[:12]}`："
                    f"最终排名中位数 {_number(hydration.get('final_rank_median'))}，"
                    f"重排升权 {hydration.get('reranker_promoted_count', 0)}、"
                    f"降权 {hydration.get('reranker_demoted_count', 0)}、"
                    f"不变 {hydration.get('reranker_unchanged_count', 0)}；"
                    f"同页 Top 4 竞争率 {_percent(hydration.get('same_page_top4_rate'))}。"
                )
            ]
        )
    return "\n".join(lines) + "\n"


def _percent(value: object) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _number(value: object) -> str:
    return "—" if value is None else f"{float(value):.1f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize hydration ablation runs.")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summaries = tuple(
        summarize_run_payload(json.loads(path.read_text(encoding="utf-8")))
        for path in args.runs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summaries), encoding="utf-8")


if __name__ == "__main__":
    main()
