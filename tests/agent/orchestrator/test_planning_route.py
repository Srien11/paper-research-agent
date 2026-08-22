from __future__ import annotations

import unittest
from datetime import UTC, datetime

from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    AgentTask,
    ConversationWorkspace,
    GoalState,
    TaskPlan,
)
from paper_research_agent.agent.orchestrator.planning_route import (
    PlanningRouteDecision,
    classify_planning_route,
)


def _utc() -> datetime:
    return datetime(2026, 8, 22, tzinfo=UTC)


def _workspace(*, existing: bool = False) -> ConversationWorkspace:
    if not existing:
        return ConversationWorkspace(
            conversation_id="conversation-1",
            version=0,
            updated_at=_utc(),
        )
    goal = GoalState(
        goal_id="a" * 32,
        objective="已有目标",
        origin_turn_id="b" * 32,
        created_at=_utc(),
        updated_at=_utc(),
    )
    task = AgentTask(
        task_id="existing-task",
        goal_id=goal.goal_id,
        title="已有任务",
        objective="继续已有任务",
        success_criteria=("完成",),
        capability="local_rag",
    )
    plan = TaskPlan(
        plan_id="c" * 32,
        goal_id=goal.goal_id,
        tasks=(task,),
        created_at=_utc(),
        updated_at=_utc(),
    )
    return ConversationWorkspace(
        conversation_id="conversation-1",
        version=1,
        active_goal=goal,
        task_plan=plan,
        updated_at=_utc(),
    )


def _envelope(
    message: str,
    *,
    rag_mode: str = "preferred",
    attachments: tuple[str, ...] = (),
    existing: bool = False,
) -> AgentContextEnvelope:
    return AgentContextEnvelope(
        conversation_id="conversation-1",
        request_id="request-1",
        turn_id="d" * 32,
        current_message=message,
        rag_mode=rag_mode,  # type: ignore[arg-type]
        attachment_ids=attachments,
        workspace=_workspace(existing=existing),
        prepared_at=_utc(),
    )


class PlanningRouteTests(unittest.TestCase):
    def test_decision_contract_is_closed(self) -> None:
        decision = PlanningRouteDecision(
            route="fast_path",
            reason_code="clear_single_local_rag",
        )
        self.assertEqual(decision.route, "fast_path")
        with self.assertRaises(ValueError):
            PlanningRouteDecision(route="other", reason_code="unknown")  # type: ignore[arg-type]

    def test_clear_single_local_rag_requests_use_fast_path(self) -> None:
        cases = (
            _envelope("请总结本地论文 C001 的方法", rag_mode="required"),
            _envelope("比较 C001 与 T001 的实验方法"),
            _envelope(
                "在逻辑推理研究中，一篇认为没有外部反馈时无效，"
                "另一篇区分找错与改错。请找出论文。"
            ),
        )
        for envelope in cases:
            with self.subTest(message=envelope.current_message):
                decision = classify_planning_route(envelope, enabled=True)
                self.assertEqual(decision.route, "fast_path")
                self.assertEqual(
                    decision.reason_code,
                    "clear_single_local_rag",
                )

    def test_deny_rules_force_full_planner(self) -> None:
        cases = (
            (
                _envelope("比较 C001 与 T001", attachments=("attachment-1",)),
                "attachments_present",
            ),
            (_envelope("比较 C001 与 T001", existing=True), "existing_workspace"),
            (_envelope("比较 C001 与 T001", rag_mode="disabled"), "rag_disabled"),
            (
                _envelope("论文 " + "很长" * 500, rag_mode="required"),
                "contract_bounds_exceeded",
            ),
            (_envelope("修改报告文件并保存"), "complex_or_ambiguous"),
            (_envelope("查询这个项目今天的最新网页状态"), "complex_or_ambiguous"),
            (_envelope("比较论文，然后生成文件并发送审批"), "complex_or_ambiguous"),
            (_envelope("帮我研究一下", rag_mode="required"), "complex_or_ambiguous"),
            (_envelope("继续修改之前的目标"), "complex_or_ambiguous"),
        )
        for envelope, reason in cases:
            with self.subTest(message=envelope.current_message):
                decision = classify_planning_route(envelope, enabled=True)
                self.assertEqual(decision.route, "full_planner")
                self.assertEqual(decision.reason_code, reason)

    def test_feature_flag_is_first_deny_rule(self) -> None:
        decision = classify_planning_route(
            _envelope("比较 C001 与 T001", attachments=("attachment-1",)),
            enabled=False,
        )

        self.assertEqual(decision.route, "full_planner")
        self.assertEqual(decision.reason_code, "feature_disabled")


if __name__ == "__main__":
    unittest.main()
