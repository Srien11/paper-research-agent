from __future__ import annotations

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from typing import Any

from paper_research_agent.agent.dynamic.memory import MemoryProposal
from paper_research_agent.agent.dynamic.models import ToolDecision
from paper_research_agent.evaluation.tool_routing import (
    ToolRoutingCase,
    load_tool_routing_dataset,
)
from paper_research_agent.evaluation.tool_routing_runner import (
    evaluate_tool_routing,
    write_tool_routing_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Router:
    def __init__(self, *values: ToolDecision | Exception):
        self.values = deque(values)
        self.calls: list[str] = []

    async def decide(
        self,
        question: str,
        observations: tuple[Any, ...],
        memory_context: tuple[dict[str, object], ...],
        *,
        remaining_steps: int,
    ) -> ToolDecision:
        del observations, memory_context, remaining_steps
        self.calls.append(question)
        value = self.values.popleft()
        if isinstance(value, Exception):
            raise value
        return value


class _MemoryProposer:
    def __init__(self, *values: MemoryProposal):
        self.values = deque(values)
        self.calls: list[str] = []

    async def propose(
        self,
        question: str,
        memories: tuple[dict[str, Any], ...],
        observations: tuple[Any, ...],
    ) -> MemoryProposal:
        del memories, observations
        self.calls.append(question)
        return self.values.popleft()


def _case(
    case_id: str,
    *,
    stage: str,
    action: str,
    tool: str | None,
    risk: str | None,
    trust: str | None,
    approval: bool,
    question: str,
    expected_arguments: dict[str, Any] | None = None,
    scoring_scope: list[str] | None = None,
) -> ToolRoutingCase:
    return ToolRoutingCase.model_validate(
        {
            "case_id": case_id,
            "evaluation_stage": stage,
            "question": question,
            "expected_action": action,
            "expected_tool": tool,
            "allowed_tools": [tool] if tool is not None else [],
            "expected_arguments": expected_arguments or {},
            "expected_risk": risk,
            "expected_trust": trust,
            "approval_required": approval,
            "scoring_scope": scoring_scope or ["action", "tool", "policy"],
            "test_reason": "runner unit test",
        }
    )


class ToolRoutingRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_v2_gold_decisions_score_without_executing_tools(self) -> None:
        cases = load_tool_routing_dataset(
            PROJECT_ROOT / "evaluation/datasets/tool-routing-v2.jsonl"
        )
        router_values: list[ToolDecision] = []
        proposal_values: list[MemoryProposal] = []
        for case in cases:
            if case.evaluation_stage == "tool_router":
                if case.expected_action == "finish":
                    router_values.append(
                        ToolDecision(
                            action="finish",
                            purpose="gold fixture",
                            final_summary="gold fixture",
                        )
                    )
                else:
                    router_values.append(
                        ToolDecision(
                            action="call_tool",
                            tool_name=case.expected_tool,
                            arguments=case.expected_arguments,
                            purpose="gold fixture",
                        )
                    )
            elif case.evaluation_stage == "memory_proposer":
                proposal_values.append(
                    MemoryProposal.model_validate(
                        {
                            "action": case.expected_action,
                            **case.expected_arguments,
                            "rationale": "gold fixture",
                        }
                    )
                )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = await evaluate_tool_routing(
                _Router(*router_values),
                _MemoryProposer(*proposal_values),
                cases,
                output,
            )

        self.assertEqual(result["case_count"], 119)
        self.assertEqual(result["aggregates"]["case_pass_rate"], 1.0)
        self.assertEqual(result["aggregates"]["arguments_accuracy"], 1.0)
        self.assertEqual(result["stage_aggregates"]["tool_router"]["tool_macro_f1"], 1.0)
        self.assertEqual(result["aggregates"]["no_tool_f1"], 1.0)

    async def test_evaluates_three_stages_without_persisting_sensitive_text(self) -> None:
        router = _Router(
            ToolDecision(
                action="call_tool",
                tool_name="calculate",
                arguments={"expression": "6 * 7"},
                purpose="provider purpose must not be saved",
            ),
            ToolDecision(
                action="finish",
                purpose="finish",
                final_summary="provider final summary must not be saved",
            ),
        )
        proposer = _MemoryProposer(
            MemoryProposal(
                action="add",
                kind="preference",
                content="private preference",
                rationale="private rationale",
            ),
            MemoryProposal(
                action="add",
                kind="preference",
                content="another private preference",
                rationale="another private rationale",
            ),
        )
        cases = (
            _case(
                "route-101",
                stage="tool_router",
                action="call_tool",
                tool="calculate",
                risk="restricted_compute",
                trust="computed_result",
                approval=False,
                question="sensitive calculate question",
                expected_arguments={"expression": "6 * 7"},
                scoring_scope=["action", "tool", "arguments", "policy"],
            ),
            _case(
                "route-102",
                stage="memory_proposer",
                action="add",
                tool="manage_long_term_memory",
                risk="write",
                trust="side_effect",
                approval=True,
                question="sensitive memory question",
            ),
            _case(
                "route-103",
                stage="dynamic_pipeline",
                action="approval_required",
                tool="manage_long_term_memory",
                risk="write",
                trust="side_effect",
                approval=True,
                question="sensitive pipeline question",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            report = Path(directory) / "report.md"
            result = await evaluate_tool_routing(
                router,
                proposer,
                cases,
                output,
                evaluation_context={
                    "model_id": "model-test",
                    "api_key": "secret-key-must-not-be-saved",
                    "provider_payload": "private payload",
                },
            )
            write_tool_routing_report(result, report)
            saved = output.read_text(encoding="utf-8")
            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(result["aggregates"]["case_pass_rate"], 1.0)
        self.assertEqual(result["aggregates"]["action_accuracy"], 1.0)
        self.assertEqual(result["aggregates"]["exact_tool_accuracy"], 1.0)
        self.assertEqual(result["aggregates"]["arguments_accuracy"], 1.0)
        self.assertEqual(result["aggregates"]["tool_macro_f1"], 1.0)
        self.assertIsNone(result["aggregates"]["no_tool_f1"])
        self.assertEqual(result["stage_counts"], {
            "tool_router": 1,
            "memory_proposer": 1,
            "dynamic_pipeline": 1,
        })
        self.assertEqual(result["evaluation_context"], {"model_id": "model-test"})
        for forbidden in (
            "sensitive calculate question",
            "sensitive memory question",
            "sensitive pipeline question",
            "provider purpose",
            "provider final summary",
            "private preference",
            "private rationale",
            "secret-key",
            "private payload",
            "6 * 7",
        ):
            self.assertNotIn(forbidden, saved)
        self.assertIn("工具路由真实模型评测", report_text)
        self.assertEqual(json.loads(saved)["records"][0]["predicted_tool"], "calculate")

    async def test_records_only_safe_error_type_and_continues(self) -> None:
        router = _Router(TimeoutError("provider response body must not be saved"))
        proposer = _MemoryProposer()
        case = _case(
            "route-104",
            stage="tool_router",
            action="call_tool",
            tool="calculate",
            risk="restricted_compute",
            trust="computed_result",
            approval=False,
            question="private failed question",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = await evaluate_tool_routing(router, proposer, (case,), output)
            saved = output.read_text(encoding="utf-8")

        self.assertEqual(result["aggregates"]["structured_success_rate"], 0.0)
        self.assertEqual(result["records"][0]["error_type"], "TimeoutError")
        self.assertNotIn("provider response body", saved)
        self.assertNotIn("private failed question", saved)


if __name__ == "__main__":
    unittest.main()
