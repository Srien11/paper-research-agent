"""Deterministic latency and quality regression checks over safe evaluation summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegressionMetricRule(FrozenModel):
    metric: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    direction: Literal["higher", "lower"]
    max_absolute_regression: float = Field(default=0, ge=0)
    max_relative_regression: float = Field(default=0, ge=0, le=1)


class RegressionGateConfig(FrozenModel):
    schema_version: Literal["interaction-regression-gate-v1"] = "interaction-regression-gate-v1"
    rules: tuple[RegressionMetricRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_metrics(self) -> RegressionGateConfig:
        metrics = tuple(rule.metric for rule in self.rules)
        if len(metrics) != len(set(metrics)):
            raise ValueError("regression gate metrics must be unique")
        return self


class RegressionViolation(FrozenModel):
    metric: str
    reason: Literal["missing", "regressed"]
    direction: Literal["higher", "lower"]
    baseline: float | None = None
    candidate: float | None = None
    boundary: float | None = None


class RegressionGateResult(FrozenModel):
    schema_version: Literal["interaction-regression-result-v1"] = "interaction-regression-result-v1"
    passed: bool
    checked_count: int = Field(ge=0)
    violations: tuple[RegressionViolation, ...] = ()


def evaluate_regression_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    config: RegressionGateConfig,
) -> RegressionGateResult:
    violations: list[RegressionViolation] = []
    for rule in config.rules:
        baseline_value = _number_at_path(baseline, rule.metric)
        candidate_value = _number_at_path(candidate, rule.metric)
        if baseline_value is None or candidate_value is None:
            violations.append(
                RegressionViolation(
                    metric=rule.metric,
                    reason="missing",
                    direction=rule.direction,
                    baseline=baseline_value,
                    candidate=candidate_value,
                )
            )
            continue
        tolerance = rule.max_absolute_regression + (
            abs(baseline_value) * rule.max_relative_regression
        )
        boundary = (
            baseline_value - tolerance if rule.direction == "higher" else baseline_value + tolerance
        )
        regressed = (
            candidate_value < boundary if rule.direction == "higher" else candidate_value > boundary
        )
        if regressed:
            violations.append(
                RegressionViolation(
                    metric=rule.metric,
                    reason="regressed",
                    direction=rule.direction,
                    baseline=baseline_value,
                    candidate=candidate_value,
                    boundary=boundary,
                )
            )
    return RegressionGateResult(
        passed=not violations,
        checked_count=len(config.rules),
        violations=tuple(violations),
    )


def _number_at_path(source: Mapping[str, Any], path: str) -> float | None:
    value: Any = source
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def failed_metric_names(result: RegressionGateResult) -> Sequence[str]:
    return tuple(item.metric for item in result.violations)
