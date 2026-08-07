from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from paper_research_agent.evaluation.tool_routing import (
    ToolRoutingCase,
    load_tool_routing_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ToolRoutingDatasetTests(unittest.TestCase):
    def test_legacy_case_defaults_to_tool_router_call(self) -> None:
        case = ToolRoutingCase(
            case_id="route-001",
            question="计算 2 + 2",
            expected_tool="calculate",
            expected_risk="restricted_compute",
            expected_trust="computed_result",
            approval_required=False,
        )

        self.assertEqual(case.evaluation_stage, "tool_router")
        self.assertEqual(case.expected_action, "call_tool")

    def test_stage_specific_contracts_are_validated(self) -> None:
        proposer = ToolRoutingCase(
            case_id="route-002",
            evaluation_stage="memory_proposer",
            question="请记住我偏好中文",
            expected_action="add",
            expected_tool="manage_long_term_memory",
            expected_risk="write",
            expected_trust="side_effect",
            approval_required=True,
        )
        pipeline = ToolRoutingCase(
            case_id="route-003",
            evaluation_stage="dynamic_pipeline",
            question="导出报告",
            expected_action="approval_required",
            expected_tool="export_research_report",
            expected_risk="write",
            expected_trust="side_effect",
            approval_required=True,
        )

        self.assertEqual(proposer.expected_action, "add")
        self.assertEqual(pipeline.expected_action, "approval_required")

        with self.assertRaises(ValidationError):
            ToolRoutingCase(
                case_id="route-004",
                evaluation_stage="memory_proposer",
                question="非法阶段动作",
                expected_action="call_tool",
                expected_tool="manage_long_term_memory",
                expected_risk="write",
                expected_trust="side_effect",
                approval_required=True,
            )

    def test_catalog_metadata_must_match_expected_tool(self) -> None:
        with self.assertRaises(ValidationError):
            ToolRoutingCase(
                case_id="route-005",
                question="计算",
                expected_tool="calculate",
                expected_risk="network_read",
                expected_trust="computed_result",
                approval_required=False,
            )

    def test_allowed_tools_arguments_and_scoring_scope_are_strict(self) -> None:
        base = {
            "case_id": "tr2-999",
            "evaluation_stage": "tool_router",
            "question": "计算 2 + 2",
            "expected_action": "call_tool",
            "expected_tool": "calculate",
            "allowed_tools": ["calculate"],
            "expected_arguments": {"expression": "2 + 2"},
            "expected_risk": "restricted_compute",
            "expected_trust": "computed_result",
            "approval_required": False,
            "scoring_scope": ["action", "tool", "arguments", "policy"],
            "test_reason": "严格字段测试",
        }
        valid = ToolRoutingCase.model_validate(base)
        self.assertEqual(valid.expected_arguments, {"expression": "2 + 2"})

        for update in (
            {"allowed_tools": ["run_shell"]},
            {"allowed_tools": ["get_paper_metadata"]},
            {"expected_arguments": {"expression": ""}},
            {"scoring_scope": ["action", "action"]},
            {"scoring_scope": ["explicit_intent"]},
        ):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                ToolRoutingCase.model_validate({**base, **update})

    def test_loader_rejects_duplicate_case_ids(self) -> None:
        row = (
            '{"case_id":"route-006","question":"计算 1+1",'
            '"expected_tool":"calculate","expected_risk":"restricted_compute",'
            '"expected_trust":"computed_result","approval_required":false}'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing.jsonl"
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "case_id"):
                load_tool_routing_dataset(path)

    def test_loads_full_v2_contract_without_dropping_high_value_fields(self) -> None:
        cases = load_tool_routing_dataset(
            PROJECT_ROOT / "evaluation/datasets/tool-routing-v2.jsonl"
        )

        self.assertEqual(len(cases), 124)
        self.assertEqual(cases[0].case_id, "tr2-001")
        self.assertEqual(cases[0].allowed_tools, ("get_adjacent_chunks",))
        self.assertEqual(cases[0].expected_arguments["before"], 1)
        self.assertIn("arguments", cases[0].scoring_scope)
        self.assertTrue(cases[0].test_reason)
        self.assertEqual(cases[-1].evaluation_stage, "memory_proposer")


if __name__ == "__main__":
    unittest.main()
