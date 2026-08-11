"""Run real structured routing decisions without executing any research tool."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from paper_research_agent.agent.dynamic.memory import DynamicMemoryProposer, MemoryProposal
from paper_research_agent.agent.dynamic.models import ToolDecision
from paper_research_agent.agent.dynamic.router import DynamicToolRouter
from paper_research_agent.agent.tooling.catalog import (
    TOOL_SPEC_BY_NAME,
    effective_tool_spec,
)
from paper_research_agent.agent.tooling.contracts import TOOL_INPUT_SCHEMAS
from paper_research_agent.evaluation.tool_routing import ToolRoutingCase

_SAFE_CONTEXT_FIELDS = frozenset(
    {
        "model_id",
        "answer_config_sha256",
        "dataset_sha256",
        "temperature",
        "top_p",
        "timeout_seconds",
        "max_retries",
        "enable_thinking",
    }
)


async def evaluate_tool_routing(
    router: DynamicToolRouter,
    memory_proposer: DynamicMemoryProposer,
    cases: Sequence[ToolRoutingCase],
    output_path: Path,
    *,
    evaluation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate model decisions while persisting no prompt or free-form model text."""
    safe_context = _safe_context(evaluation_context or {})
    records: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        try:
            action, tool_name, risk, trust, approval, predicted_arguments = await _predict(
                router,
                memory_proposer,
                case,
            )
            exact_tool_match = tool_name == case.expected_tool
            allowed_tool_match = (
                tool_name in case.allowed_tools
                if case.allowed_tools
                else tool_name is None
            )
            arguments_match = predicted_arguments == _expected_arguments(case)
            explicit_intent_match = (
                (action != "none") == (case.expected_action != "none")
            )
            record = {
                "case_id": case.case_id,
                "evaluation_stage": case.evaluation_stage,
                "scoring_scope": list(case.scoring_scope),
                "expected_action": case.expected_action,
                "expected_tool": case.expected_tool,
                "structured_success": True,
                "error_type": None,
                "predicted_action": action,
                "predicted_tool": tool_name,
                "predicted_risk": risk,
                "predicted_trust": trust,
                "predicted_approval_required": approval,
                "action_match": action == case.expected_action,
                "tool_match": exact_tool_match,
                "allowed_tool_match": allowed_tool_match,
                "arguments_match": arguments_match,
                "risk_match": risk == case.expected_risk,
                "trust_match": trust == case.expected_trust,
                "approval_match": approval == case.approval_required,
                "explicit_intent_match": explicit_intent_match,
                "neighbor_disambiguation_match": exact_tool_match,
                "latency_ms": _elapsed_ms(started),
            }
        except Exception as error:  # noqa: BLE001 - persist the safe class name only
            record = {
                "case_id": case.case_id,
                "evaluation_stage": case.evaluation_stage,
                "scoring_scope": list(case.scoring_scope),
                "expected_action": case.expected_action,
                "expected_tool": case.expected_tool,
                "structured_success": False,
                "error_type": type(error).__name__,
                "predicted_action": None,
                "predicted_tool": None,
                "predicted_risk": None,
                "predicted_trust": None,
                "predicted_approval_required": None,
                "action_match": False,
                "tool_match": False,
                "allowed_tool_match": False,
                "arguments_match": False,
                "risk_match": False,
                "trust_match": False,
                "approval_match": False,
                "explicit_intent_match": False,
                "neighbor_disambiguation_match": False,
                "latency_ms": _elapsed_ms(started),
            }
        record["case_pass"] = _case_pass(record)
        records.append(record)

    result = {
        "schema_version": "tool-routing-evaluation-v1",
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(
                {
                    "context": safe_context,
                    "cases": [case.model_dump(mode="json") for case in cases],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "evaluation_context": safe_context,
        "case_count": len(cases),
        "stage_counts": dict(Counter(case.evaluation_stage for case in cases)),
        "aggregates": _aggregates(records),
        "stage_aggregates": {
            stage: _aggregates(
                [record for record in records if record["evaluation_stage"] == stage]
            )
            for stage in ("tool_router", "memory_proposer", "dynamic_pipeline")
            if any(record["evaluation_stage"] == stage for record in records)
        },
        "records": records,
        "limitations": [
            "只评估结构化决策，不执行任何本地、网络、计算或写入工具",
            "dynamic_pipeline 仅组合首个路由决策与记忆提议，不模拟工具观察后的后续轮次",
            "评测问题属于开发集，不是密封金标测试集",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


async def _predict(
    router: DynamicToolRouter,
    memory_proposer: DynamicMemoryProposer,
    case: ToolRoutingCase,
) -> tuple[str, str | None, str | None, str | None, bool, dict[str, Any]]:
    if case.evaluation_stage == "memory_proposer":
        proposal = await memory_proposer.propose(
            case.question,
            _memory_context(case),
            (),
        )
        if proposal.action == "none":
            return "none", None, None, None, False, {}
        arguments = proposal.tool_arguments(scope_id="evaluation")
        return _tool_prediction(
            "manage_long_term_memory",
            arguments,
            proposal.action,
            scored_arguments=_proposal_arguments(proposal),
        )

    decision = await router.decide(
        case.question,
        (),
        (),
        remaining_steps=1,
    )
    if case.evaluation_stage == "tool_router":
        return _router_prediction(decision)

    if decision.action == "call_tool":
        tool_name = _required_tool_name(decision)
        spec = effective_tool_spec(TOOL_SPEC_BY_NAME[tool_name], decision.arguments)
        action = "approval_required" if spec.approval_required else "call_tool"
        return (
            action,
            tool_name,
            spec.risk,
            spec.trust,
            spec.approval_required,
            _tool_arguments(tool_name, decision.arguments),
        )

    proposal = await memory_proposer.propose(
        case.question,
        _memory_context(case),
        (),
    )
    if proposal.action == "none":
        return "finish", None, None, None, False, {}
    arguments = proposal.tool_arguments(scope_id="evaluation")
    return _tool_prediction(
        "manage_long_term_memory",
        arguments,
        "approval_required",
        scored_arguments=_proposal_arguments(proposal),
    )


def _router_prediction(
    decision: ToolDecision,
) -> tuple[str, str | None, str | None, str | None, bool, dict[str, Any]]:
    if decision.action == "finish":
        return "finish", None, None, None, False, {}
    tool_name = _required_tool_name(decision)
    return _tool_prediction(tool_name, decision.arguments, "call_tool")


def _tool_prediction(
    tool_name: str,
    arguments: Mapping[str, Any],
    action: str,
    *,
    scored_arguments: dict[str, Any] | None = None,
) -> tuple[str, str, str, str, bool, dict[str, Any]]:
    spec = effective_tool_spec(TOOL_SPEC_BY_NAME[tool_name], arguments)
    return (
        action,
        tool_name,
        spec.risk,
        spec.trust,
        spec.approval_required,
        scored_arguments if scored_arguments is not None else _tool_arguments(tool_name, arguments),
    )


def _required_tool_name(decision: ToolDecision) -> str:
    if decision.tool_name is None:
        raise ValueError("tool decision is missing tool_name")
    return decision.tool_name


def _aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_ms"]) for record in records]
    return {
        "structured_success_rate": _rate(records, "structured_success"),
        "action_accuracy": _scoped_rate(records, "action_match", "action"),
        "exact_tool_accuracy": _scoped_rate(records, "tool_match", "tool"),
        "allowed_tool_accuracy": _scoped_rate(records, "allowed_tool_match", "tool"),
        "tool_macro_f1": _tool_macro_f1(records),
        "no_tool_f1": _no_tool_f1(records),
        "arguments_accuracy": _scoped_rate(records, "arguments_match", "arguments"),
        "risk_accuracy": _scoped_rate(records, "risk_match", "policy"),
        "trust_accuracy": _scoped_rate(records, "trust_match", "policy"),
        "approval_accuracy": _scoped_rate(records, "approval_match", "policy"),
        "explicit_intent_accuracy": _scoped_rate(
            records,
            "explicit_intent_match",
            "explicit_intent",
        ),
        "neighbor_disambiguation_accuracy": _scoped_rate(
            records,
            "neighbor_disambiguation_match",
            "neighbor_disambiguation",
        ),
        "case_pass_rate": _rate(records, "case_pass"),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "error_types": dict(
            Counter(
                str(record["error_type"])
                for record in records
                if record.get("error_type") is not None
            )
        ),
    }


def _rate(records: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(bool(record[field]) for record in records) if records else 0.0


def _scoped_rate(
    records: Sequence[Mapping[str, Any]],
    field: str,
    dimension: str,
) -> float | None:
    selected = [record for record in records if dimension in record["scoring_scope"]]
    return _rate(selected, field) if selected else None


def _case_pass(record: Mapping[str, Any]) -> bool:
    if not record["structured_success"]:
        return False
    checks = {
        "action": "action_match",
        "tool": "allowed_tool_match",
        "arguments": "arguments_match",
        "policy": "policy_match",
        "explicit_intent": "explicit_intent_match",
        "neighbor_disambiguation": "neighbor_disambiguation_match",
    }
    enriched = dict(record)
    enriched["policy_match"] = bool(
        record["risk_match"] and record["trust_match"] and record["approval_match"]
    )
    return all(bool(enriched[checks[dimension]]) for dimension in record["scoring_scope"])


def _tool_macro_f1(records: Sequence[Mapping[str, Any]]) -> float | None:
    selected = [record for record in records if "tool" in record["scoring_scope"]]
    labels = sorted(
        {
            str(record["expected_tool"])
            for record in selected
            if record.get("expected_tool") is not None
        }
    )
    if not labels:
        return None
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            record.get("expected_tool") == label and record.get("predicted_tool") == label
            for record in selected
        )
        false_positive = sum(
            record.get("expected_tool") != label and record.get("predicted_tool") == label
            for record in selected
        )
        false_negative = sum(
            record.get("expected_tool") == label and record.get("predicted_tool") != label
            for record in selected
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return statistics.fmean(scores)


def _no_tool_f1(records: Sequence[Mapping[str, Any]]) -> float | None:
    selected = [record for record in records if "action" in record["scoring_scope"]]
    if not selected:
        return None
    def expected_none(record: Mapping[str, Any]) -> bool:
        return record.get("expected_action") in {"finish", "none"}

    def predicted_none(record: Mapping[str, Any]) -> bool:
        return record.get("predicted_action") in {"finish", "none"}

    true_positive = sum(expected_none(record) and predicted_none(record) for record in selected)
    false_positive = sum(not expected_none(record) and predicted_none(record) for record in selected)
    false_negative = sum(expected_none(record) and not predicted_none(record) for record in selected)
    denominator = 2 * true_positive + false_positive + false_negative
    return (2 * true_positive / denominator) if denominator else None


def _expected_arguments(case: ToolRoutingCase) -> dict[str, Any]:
    if "arguments" not in case.scoring_scope:
        return {}
    if case.evaluation_stage == "memory_proposer":
        if case.expected_action == "none":
            return {}
        proposal = MemoryProposal.model_validate(
            {
                "action": case.expected_action,
                **case.expected_arguments,
                "rationale": "evaluation gold label",
            }
        )
        return _proposal_arguments(proposal)
    if case.expected_tool is None:
        return {}
    return _tool_arguments(case.expected_tool, case.expected_arguments)


def _tool_arguments(tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    request = TOOL_INPUT_SCHEMAS[tool_name].model_validate(arguments)
    payload = request.model_dump(mode="json", exclude_none=True)
    payload.pop("approval_token", None)
    return payload


def _proposal_arguments(proposal: MemoryProposal) -> dict[str, Any]:
    payload = proposal.model_dump(
        mode="json",
        exclude={"action", "rationale"},
        exclude_none=True,
    )
    if payload.get("source_chunk_ids") == []:
        payload.pop("source_chunk_ids")
    return payload


def _memory_context(case: ToolRoutingCase) -> tuple[dict[str, Any], ...]:
    memory_id = case.expected_arguments.get("memory_id")
    if not isinstance(memory_id, str):
        return ()
    return (
        {
            "memory_id": memory_id,
            "kind": "preference",
            "content": "evaluation recalled memory",
            "source_chunk_ids": (),
            "version": 1,
            "updated_at": "2026-08-06T00:00:00+00:00",
            "expires_at": None,
        },
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _safe_context(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key in _SAFE_CONTEXT_FIELDS
        and (item is None or isinstance(item, (str, int, float, bool)))
    }


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000)


def write_tool_routing_report(result: Mapping[str, Any], path: Path) -> None:
    values = result["aggregates"]

    def percentage(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def milliseconds(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f} ms"

    text = "\n".join(
        [
            "# 工具路由真实模型评测",
            "",
            f"- 实验指纹：`{result['fingerprint_sha256']}`",
            f"- 用例数：{result['case_count']}",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 结构化输出成功率 | {percentage(values['structured_success_rate'])} |",
            f"| 动作准确率 | {percentage(values['action_accuracy'])} |",
            f"| 工具准确率 | {percentage(values['exact_tool_accuracy'])} |",
            f"| 允许工具准确率 | {percentage(values['allowed_tool_accuracy'])} |",
            f"| 工具选择 Macro-F1 | {percentage(values['tool_macro_f1'])} |",
            f"| No-tool F1 | {percentage(values['no_tool_f1'])} |",
            f"| 参数准确率 | {percentage(values['arguments_accuracy'])} |",
            f"| 风险等级准确率 | {percentage(values['risk_accuracy'])} |",
            f"| 信任等级准确率 | {percentage(values['trust_accuracy'])} |",
            f"| 审批分类准确率 | {percentage(values['approval_accuracy'])} |",
            f"| 全字段通过率 | {percentage(values['case_pass_rate'])} |",
            f"| 决策延迟 P50 | {milliseconds(values['latency_p50_ms'])} |",
            f"| 决策延迟 P95 | {milliseconds(values['latency_p95_ms'])} |",
            "",
            "## 解释边界",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
