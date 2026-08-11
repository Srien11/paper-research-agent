from __future__ import annotations

import unittest
from pathlib import Path

from paper_research_agent.evaluation.main_agent import (
    MainAgentConversationCase,
    MainAgentInterfaceRecord,
    MainAgentInterfaceScoringResult,
    MainAgentScoringResult,
    MainAgentTurnRecord,
    load_main_agent_cases,
    load_main_agent_interface_cases,
    score_main_agent_case,
    score_main_agent_interface_case,
)

DATASET = (
    Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "main-agent-conversation-v1.jsonl"
)
INTERFACE_DATASET = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "datasets"
    / "main-agent-interface-v1.jsonl"
)


class MainAgentEvaluationTests(unittest.TestCase):
    def test_interface_dataset_covers_all_release_scenarios(self) -> None:
        cases = load_main_agent_interface_cases(INTERFACE_DATASET)
        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual(
            {case.scenario for case in cases},
            {
                "direct_chat",
                "local_rag",
                "paper_comparison",
                "hybrid_research",
                "attachment_qa",
                "file_edit",
                "approval_approved",
                "approval_rejected",
                "approval_expired",
                "duplicate_request",
                "rag_mode_required",
                "rag_mode_disabled",
                "commit_rejected",
            },
        )

    def test_interface_ideal_record_scores_every_release_axis(self) -> None:
        case = next(
            item
            for item in load_main_agent_interface_cases(INTERFACE_DATASET)
            if item.scenario == "local_rag"
        )
        record = MainAgentInterfaceRecord(
            status="completed",
            capabilities=("local_rag",),
            artifact_kinds=("local_rag",),
            input_source_ids=("chunk-1",),
            output_source_ids=("chunk-1",),
            goal_continuous=True,
            interpretation_count=1,
            planning_count=1,
            side_effect_count=0,
            commit_rejected=False,
            workspace_version_before=0,
            workspace_version_after=1,
            event_types=("run_started", "rag_result", "done"),
            event_ids=(1, 2, 3),
            done_count=1,
        )

        result = score_main_agent_interface_case(case, record)

        self.assertIsInstance(result, MainAgentInterfaceScoringResult)
        self.assertEqual(result.overall, 1.0)
        self.assertTrue(
            all(
                value == 1.0
                for value in (
                    result.route_correctness,
                    result.goal_continuity,
                    result.artifact_preservation,
                    result.citation_preservation,
                    result.approval_resume_correctness,
                    result.duplicate_side_effect,
                    result.commit_rejection,
                    result.event_contract_validity,
                )
            )
        )

    def test_interface_scoring_penalizes_replanning_duplicate_and_bad_events(self) -> None:
        case = next(
            item
            for item in load_main_agent_interface_cases(INTERFACE_DATASET)
            if item.scenario == "approval_approved"
        )
        record = MainAgentInterfaceRecord(
            status="completed",
            capabilities=("dynamic_tools",),
            artifact_kinds=("dynamic_tools",),
            goal_continuous=False,
            approval_resume_count=2,
            interpretation_count=2,
            planning_count=2,
            side_effect_count=2,
            commit_rejected=False,
            workspace_version_before=0,
            workspace_version_after=2,
            event_types=("run_started", "done", "done"),
            event_ids=(1, 3, 3),
            done_count=2,
        )

        result = score_main_agent_interface_case(case, record)

        self.assertEqual(result.approval_resume_correctness, 0.0)
        self.assertEqual(result.duplicate_side_effect, 0.0)
        self.assertEqual(result.event_contract_validity, 0.0)
        self.assertLess(result.overall, 1.0)

    def test_dataset_loads_valid_cases(self) -> None:
        cases = load_main_agent_cases(DATASET)
        self.assertEqual(len(cases), 3)
        self.assertTrue(all(case.turns for case in cases))
        self.assertTrue(all(case.expected_relations for case in cases))

    def test_ideal_records_score_full_marks(self) -> None:
        case = MainAgentConversationCase(
            case_id="goal-continuity",
            description="目标延续",
            turns=("比较", "继续"),
            expected_relations=("new_goal", "continue_goal"),
            goal_continuous=True,
            expected_capabilities=("local_rag",),
            expected_final_status="completed",
        )
        records = (
            MainAgentTurnRecord(
                turn_index=1,
                relation="new_goal",
                goal_id="a" * 32,
                plan_revision=1,
                capabilities=("local_rag",),
                status="completed",
                memory_recalled_before_route=True,
                side_effect_count=1,
            ),
            MainAgentTurnRecord(
                turn_index=2,
                relation="continue_goal",
                goal_id="a" * 32,
                plan_revision=1,
                capabilities=("local_rag",),
                status="completed",
                memory_recalled_before_route=True,
                side_effect_count=1,
            ),
        )
        result = score_main_agent_case(case, records)
        self.assertEqual(result.overall, 1.0)
        for axis in (
            result.goal_continuity,
            result.plan_continuity,
            result.route_correctness,
            result.memory_before_routing,
            result.duplicate_side_effect,
        ):
            self.assertEqual(axis, 1.0)

    def test_goal_switch_and_duplicate_side_effect_are_penalized(self) -> None:
        case = MainAgentConversationCase(
            case_id="goal-continuity",
            description="目标延续",
            turns=("比较", "继续"),
            expected_relations=("new_goal", "continue_goal"),
            goal_continuous=True,
            expected_capabilities=("local_rag",),
        )
        records = (
            MainAgentTurnRecord(
                turn_index=1,
                relation="new_goal",
                goal_id="a" * 32,
                plan_revision=1,
                capabilities=("local_rag",),
                status="completed",
                memory_recalled_before_route=True,
                side_effect_count=1,
            ),
            MainAgentTurnRecord(
                turn_index=2,
                relation="new_goal",
                goal_id="b" * 32,
                plan_revision=2,
                capabilities=("local_rag",),
                status="completed",
                memory_recalled_before_route=False,
                side_effect_count=2,
            ),
        )
        result = score_main_agent_case(case, records)
        self.assertEqual(result.goal_continuity, 0.0)
        self.assertEqual(result.duplicate_side_effect, 0.0)
        self.assertEqual(result.memory_before_routing, 0.0)
        self.assertLess(result.overall, 1.0)

    def test_route_correctness_penalizes_missing_capability(self) -> None:
        case = MainAgentConversationCase(
            case_id="hybrid-research",
            description="混合研究",
            turns=("比较论文并核验最新状态",),
            expected_relations=("new_goal",),
            goal_continuous=True,
            expected_capabilities=("local_rag", "dynamic_tools"),
        )
        records = (
            MainAgentTurnRecord(
                turn_index=1,
                relation="new_goal",
                goal_id="a" * 32,
                plan_revision=1,
                capabilities=("local_rag",),
                status="completed",
                memory_recalled_before_route=True,
            ),
        )
        result = score_main_agent_case(case, records)
        self.assertEqual(result.route_correctness, 0.0)

    def test_empty_records_score_zero(self) -> None:
        case = MainAgentConversationCase(
            case_id="empty",
            description="空记录",
            turns=("问题",),
            expected_relations=("new_goal",),
        )
        result: MainAgentScoringResult = score_main_agent_case(case, ())
        self.assertEqual(result.overall, 0.0)


if __name__ == "__main__":
    unittest.main()
