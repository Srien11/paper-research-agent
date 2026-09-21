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
    SourcePolicyConflictError,
    classify_planning_route,
    infer_source_requirements,
    validate_source_policy,
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
                self.assertEqual(decision.capability, "local_rag")

    def test_simple_requests_use_single_direct_chat_task(self) -> None:
        cases = (
            _envelope("介绍一下 RAG"),
            _envelope("解释一下注意力机制", rag_mode="disabled"),
            _envelope("你好"),
        )

        for envelope in cases:
            with self.subTest(message=envelope.current_message):
                decision = classify_planning_route(envelope, enabled=True)
                self.assertEqual(decision.route, "fast_path")
                self.assertEqual(decision.reason_code, "simple_direct_chat")
                self.assertEqual(decision.capability, "direct_chat")

    def test_complex_comparison_keeps_full_planner(self) -> None:
        decision = classify_planning_route(
            _envelope("比较 RAG 与 GraphRAG 的适用边界、成本和实验差异"),
            enabled=True,
        )

        self.assertEqual(decision.route, "full_planner")
        self.assertEqual(decision.reason_code, "complex_or_ambiguous")

    def test_completed_workspace_allows_new_self_contained_request_only(self) -> None:
        existing = _envelope("介绍一下 RAG", existing=True)
        plan = existing.workspace.task_plan
        self.assertIsNotNone(plan)
        completed_plan = plan.model_copy(
            update={
                "tasks": tuple(
                    task.model_copy(update={"status": "completed"})
                    for task in plan.tasks
                )
            }
        )
        completed_workspace = existing.workspace.model_copy(
            update={"task_plan": completed_plan}
        )

        standalone = classify_planning_route(
            existing.model_copy(update={"workspace": completed_workspace}),
            enabled=True,
        )
        follow_up = classify_planning_route(
            existing.model_copy(
                update={
                    "current_message": "再详细说说",
                    "workspace": completed_workspace,
                }
            ),
            enabled=True,
        )

        self.assertEqual(standalone.route, "fast_path")
        self.assertEqual(standalone.capability, "direct_chat")
        self.assertEqual(follow_up.route, "full_planner")
        self.assertEqual(follow_up.reason_code, "existing_workspace")

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
            (_envelope("比较论文", rag_mode="required"), "complex_or_ambiguous"),
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

    def test_external_intent_never_uses_local_only_fast_path(self) -> None:
        messages = (
            "结合本地论文 C001 和联网搜索，分析 RAG",
            "比较 C001 与 C002，并用外部资料核验",
            "参考知识库，同时网上查最新版本",
            "根据 C001，再从网络来源确认项目状态",
        )
        for message in messages:
            with self.subTest(message=message):
                requirements = infer_source_requirements(message)
                decision = classify_planning_route(_envelope(message), enabled=True)
                self.assertTrue(requirements.external_required)
                self.assertEqual(decision.route, "full_planner")

    def test_source_negation_is_not_treated_as_positive_requirement(self) -> None:
        requirements = infer_source_requirements("不要使用知识库，只联网查询最新资料")

        self.assertTrue(requirements.local_forbidden)
        self.assertFalse(requirements.local_required)
        self.assertTrue(requirements.external_required)

    def test_hard_rag_modes_reject_conflicting_language(self) -> None:
        cases = (
            ("disabled", "请使用本地知识库回答", "rag_disabled_local_requested"),
            ("required", "请联网查询最新资料", "rag_required_external_requested"),
            ("required", "不要使用知识库", "rag_required_local_forbidden"),
        )

        for rag_mode, message, reason_code in cases:
            with self.subTest(rag_mode=rag_mode, message=message):
                with self.assertRaises(SourcePolicyConflictError) as captured:
                    validate_source_policy(message, rag_mode)  # type: ignore[arg-type]
                self.assertEqual(captured.exception.reason_code, reason_code)

    def test_preferred_mode_accepts_language_level_local_opt_out(self) -> None:
        validate_source_policy("不要使用知识库，只联网查询", "preferred")

        decision = classify_planning_route(
            _envelope("不要使用知识库，只联网查询", rag_mode="preferred"),
            enabled=True,
        )

        self.assertEqual(decision.route, "full_planner")

    def test_latest_local_paper_does_not_imply_external_source(self) -> None:
        requirements = validate_source_policy(
            "总结知识库中最新上传的论文",
            "required",
        )

        self.assertTrue(requirements.local_required)
        self.assertFalse(requirements.external_required)

    def test_source_terms_used_as_topics_do_not_trigger_source_policy(self) -> None:
        local_topic = validate_source_policy(
            "知识库和向量数据库有什么区别",
            "disabled",
        )
        network_topic = validate_source_policy(
            "解释神经网络架构",
            "required",
        )

        self.assertFalse(local_topic.local_required)
        self.assertFalse(network_topic.external_required)

    def test_explicit_external_scholarly_retrieval_is_detected(self) -> None:
        requirements = infer_source_requirements("结合本地论文和外部学术检索回答")

        self.assertTrue(requirements.local_required)
        self.assertTrue(requirements.external_required)


if __name__ == "__main__":
    unittest.main()
