from __future__ import annotations

import json
from pathlib import Path

from paper_research_agent.evaluation.regression_gate import (
    RegressionGateConfig,
    RegressionMetricRule,
    evaluate_regression_gate,
    failed_metric_names,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config() -> RegressionGateConfig:
    return RegressionGateConfig(
        rules=(
            RegressionMetricRule(
                metric="aggregates.citation_f1",
                direction="higher",
            ),
            RegressionMetricRule(
                metric="aggregates.latency_p95_ms",
                direction="lower",
                max_relative_regression=0.1,
            ),
        )
    )


def test_gate_accepts_equal_quality_and_latency_within_tolerance() -> None:
    result = evaluate_regression_gate(
        {"aggregates": {"citation_f1": 0.9, "latency_p95_ms": 1000}},
        {"aggregates": {"citation_f1": 0.9, "latency_p95_ms": 1099}},
        _config(),
    )

    assert result.passed
    assert result.checked_count == 2
    assert result.violations == ()


def test_gate_rejects_quality_drop_and_latency_regression() -> None:
    result = evaluate_regression_gate(
        {"aggregates": {"citation_f1": 0.9, "latency_p95_ms": 1000}},
        {"aggregates": {"citation_f1": 0.89, "latency_p95_ms": 1101}},
        _config(),
    )

    assert not result.passed
    assert failed_metric_names(result) == (
        "aggregates.citation_f1",
        "aggregates.latency_p95_ms",
    )


def test_gate_fails_closed_when_a_required_metric_is_missing() -> None:
    result = evaluate_regression_gate(
        {"aggregates": {"citation_f1": 0.9, "latency_p95_ms": 1000}},
        {"aggregates": {"citation_f1": 0.9}},
        _config(),
    )

    assert not result.passed
    assert result.violations[0].reason == "missing"
    assert result.violations[0].metric == "aggregates.latency_p95_ms"


def test_checked_in_rag_gate_covers_latency_and_quality() -> None:
    config = RegressionGateConfig.model_validate(
        json.loads(
            (PROJECT_ROOT / "configs/evaluation/rag-gold-regression-gate-v1.json").read_text(
                encoding="utf-8"
            )
        )
    )
    by_metric = {rule.metric: rule for rule in config.rules}

    assert by_metric["aggregates.citation_f1"].direction == "higher"
    assert by_metric["aggregates.required_group_recall"].direction == "higher"
    assert by_metric["aggregates.latency_p95_ms"].direction == "lower"
    assert by_metric["aggregates.latency_p95_ms"].max_relative_regression == 0.1
