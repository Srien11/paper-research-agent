from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from typing import Any

from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    AgentTask,
    ContextMessage,
    ConversationWorkspace,
    GoalDecision,
    GoalState,
    TaskPlan,
    TurnInterpretationV2,
)
from paper_research_agent.agent.orchestrator.planner import GoalReconciler, TaskPlanner


def _utc() -> datetime:
    return datetime(2026, 8, 7, tzinfo=UTC)


def _goal(**overrides: object) -> GoalState:
    values: dict[str, object] = {
        "goal_id": "a" * 32,
        "objective": "比较 RAG 与 GraphRAG",
        "status": "active",
        "origin_turn_id": "b" * 32,
        "created_at": _utc(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return GoalState(**values)


def _envelope(**overrides: object) -> AgentContextEnvelope:
    values: dict[str, object] = {
        "conversation_id": "conversation-1",
        "request_id": "request-1",
        "turn_id": "c" * 32,
        "current_message": "继续",
        "rag_mode": "preferred",
        "workspace": ConversationWorkspace(
            conversation_id="conversation-1",
            version=0,
            active_goal=_goal(),
            updated_at=_utc(),
        ),
        "recent_messages": (
            ContextMessage(turn_id="t1", sequence=1, role="user", content="之前的问题"),
        ),
        "recalled_context": (),
        "prepared_at": _utc(),
    }
    values.update(overrides)
    return AgentContextEnvelope(**values)


def _interpretation(relation: str, **overrides: object) -> TurnInterpretationV2:
    values: dict[str, Any] = {
        "relation": relation,
        "resolved_request": "继续比较 RAG 与 GraphRAG",
        "confidence": 0.9,
    }
    values.update(overrides)
    return TurnInterpretationV2(**values)


class _FakeModel:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def with_structured_output(self, schema: object, method: str = "function_calling") -> object:
        del schema, method
        return self

    async def ainvoke(self, messages: object) -> object:
        del messages
        self.calls += 1
        index = min(self.calls - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        return response


def _reconcile(
    reconciler: GoalReconciler,
    envelope: AgentContextEnvelope,
    interpretation: TurnInterpretationV2,
) -> object:
    return asyncio.run(reconciler.reconcile(envelope, interpretation))


class GoalReconcilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reconciler = GoalReconciler()

    def test_continue_reuses_goal_id(self) -> None:
        envelope = _envelope()
        decision = _reconcile(self.reconciler, envelope, _interpretation("continue_goal"))
        self.assertEqual(decision.action, "keep")
        self.assertEqual(decision.goal.goal_id, "a" * 32)

    def test_resume_after_approval_reuses_goal_id(self) -> None:
        decision = _reconcile(
            self.reconciler, _envelope(), _interpretation("resume_after_approval")
        )
        self.assertEqual(decision.action, "keep")
        self.assertEqual(decision.goal.goal_id, "a" * 32)

    def test_refine_reuses_goal_id_and_updates_objective(self) -> None:
        interpretation = _interpretation(
            "refine_goal", goal_change_summary="只比较推理成本和准确率"
        )
        decision = _reconcile(self.reconciler, _envelope(), interpretation)
        self.assertEqual(decision.action, "revise")
        self.assertEqual(decision.goal.goal_id, "a" * 32)
        self.assertEqual(decision.goal.objective, "只比较推理成本和准确率")

    def test_new_goal_creates_new_goal_id(self) -> None:
        decision = _reconcile(self.reconciler, _envelope(), _interpretation("new_goal"))
        self.assertEqual(decision.action, "create")
        self.assertNotEqual(decision.goal.goal_id, "a" * 32)
        self.assertEqual(decision.goal.objective, "继续比较 RAG 与 GraphRAG")

    def test_cancel_abandons_active_goal(self) -> None:
        decision = _reconcile(self.reconciler, _envelope(), _interpretation("cancel_goal"))
        self.assertEqual(decision.action, "abandon")
        self.assertEqual(decision.goal.goal_id, "a" * 32)
        self.assertEqual(decision.goal.status, "abandoned")

    def test_answer_within_goal_keeps_goal(self) -> None:
        decision = _reconcile(
            self.reconciler, _envelope(), _interpretation("answer_within_goal")
        )
        self.assertEqual(decision.action, "keep")
        self.assertEqual(decision.goal.goal_id, "a" * 32)

    def test_meta_conversation_does_not_create_goal(self) -> None:
        workspace = ConversationWorkspace(
            conversation_id="conversation-1", version=0, updated_at=_utc()
        )
        envelope = _envelope(workspace=workspace)
        decision = _reconcile(
            self.reconciler, envelope, _interpretation("meta_conversation")
        )
        self.assertEqual(decision.action, "keep")
        self.assertIsNone(decision.goal)

    def test_no_active_goal_creates_goal(self) -> None:
        workspace = ConversationWorkspace(
            conversation_id="conversation-1", version=0, updated_at=_utc()
        )
        envelope = _envelope(workspace=workspace)
        decision = _reconcile(
            self.reconciler, envelope, _interpretation("continue_goal")
        )
        self.assertEqual(decision.action, "create")
        self.assertEqual(decision.goal.origin_turn_id, "c" * 32)

    def test_model_completes_acceptance_criteria_for_create(self) -> None:
        fake = _FakeModel(
            [
                {
                    "objective": "比较 RAG 与 GraphRAG 并给出选型建议",
                    "acceptance_criteria": (
                        {
                            "criterion_id": "c1",
                            "description": "给出两项指标对比",
                        },
                    ),
                    "constraints": ("只依据本地论文",),
                }
            ]
        )
        reconciler = GoalReconciler(model=fake)
        decision = _reconcile(
            reconciler, _envelope(), _interpretation("new_goal")
        )
        self.assertEqual(decision.action, "create")
        self.assertEqual(decision.goal.objective, "比较 RAG 与 GraphRAG 并给出选型建议")
        self.assertEqual(len(decision.goal.acceptance_criteria), 1)

    def test_model_failure_falls_back_to_deterministic(self) -> None:
        fake = _FakeModel([RuntimeError("down")])
        reconciler = GoalReconciler(model=fake)
        decision = _reconcile(
            reconciler, _envelope(), _interpretation("new_goal")
        )
        self.assertEqual(decision.action, "create")
        self.assertEqual(decision.goal.objective, "继续比较 RAG 与 GraphRAG")
        self.assertIsNotNone(decision.goal.goal_id)


def _workspace(**overrides: object) -> ConversationWorkspace:
    values: dict[str, object] = {
        "conversation_id": "conversation-1",
        "version": 0,
        "active_goal": _goal(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return ConversationWorkspace(**values)


def _task(**overrides: object) -> AgentTask:
    values: dict[str, object] = {
        "task_id": "research-task",
        "goal_id": "a" * 32,
        "title": "检索证据",
        "objective": "比较 RAG 与 GraphRAG",
        "success_criteria": ("找到证据",),
        "capability": "local_rag",
        "status": "pending",
    }
    values.update(overrides)
    return AgentTask(**values)


def _plan(tasks: tuple[AgentTask, ...], **overrides: object) -> TaskPlan:
    values: dict[str, object] = {
        "plan_id": "d" * 32,
        "goal_id": "a" * 32,
        "revision": 1,
        "tasks": tasks,
        "created_at": _utc(),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return TaskPlan(**values)


def _goal_decision(action: str = "create", goal: GoalState | None = None) -> GoalDecision:
    return GoalDecision(
        action=action,
        goal=goal if goal is not None else _goal(),
        rationale="测试决策",
    )


class TaskPlannerTests(unittest.TestCase):
    def _plan_via(
        self,
        planner: TaskPlanner,
        envelope: AgentContextEnvelope,
        goal_decision: object,
    ) -> object:
        return asyncio.run(
            planner.plan(
                envelope,
                _interpretation("new_goal"),
                goal_decision,  # type: ignore[arg-type]
            )
        )

    def test_single_request_single_task(self) -> None:
        fake = _FakeModel(
            [
                {
                    "tasks": (
                        {
                            "task_id": "research",
                            "title": "检索证据",
                            "objective": "比较 RAG 与 GraphRAG",
                            "success_criteria": ("找到证据",),
                            "capability": "local_rag",
                        },
                    )
                }
            ]
        )
        decision = self._plan_via(TaskPlanner(model=fake), _envelope(), _goal_decision())
        self.assertEqual(decision.action, "create")
        self.assertEqual(len(decision.plan.tasks), 1)
        self.assertEqual(decision.plan.tasks[0].capability, "local_rag")

    def test_mixed_research_splits_into_two_tasks(self) -> None:
        fake = _FakeModel(
            [
                {
                    "tasks": (
                        {
                            "task_id": "local",
                            "title": "本地论文",
                            "objective": "比较论文中的方法",
                            "success_criteria": ("找到证据",),
                            "capability": "local_rag",
                        },
                        {
                            "task_id": "web",
                            "title": "核验最新状态",
                            "objective": "核验项目 2026 年维护状态",
                            "success_criteria": ("得到最新状态",),
                            "capability": "dynamic_tools",
                            "depends_on": ("local",),
                        },
                    )
                }
            ]
        )
        decision = self._plan_via(TaskPlanner(model=fake), _envelope(), _goal_decision())
        self.assertEqual(len(decision.plan.tasks), 2)
        self.assertEqual(
            {task.capability for task in decision.plan.tasks},
            {"local_rag", "dynamic_tools"},
        )

    def test_task_id_stable_and_completed_not_regressed_across_revision(self) -> None:
        current = _plan(
            tasks=(
                _task(task_id="stable-task", status="completed"),
                _task(task_id="pending-task", status="pending"),
            )
        )
        envelope = _envelope(workspace=_workspace(task_plan=current))
        fake = _FakeModel(
            [
                {
                    "tasks": (
                        {
                            "task_id": "pending-task",
                            "title": "更新后的任务",
                            "objective": "更新目标",
                            "success_criteria": ("完成",),
                            "capability": "local_rag",
                        },
                        {
                            "task_id": "new-task",
                            "title": "新增任务",
                            "objective": "新目标",
                            "success_criteria": ("完成",),
                            "capability": "direct_chat",
                        },
                    )
                }
            ]
        )
        decision = self._plan_via(
            TaskPlanner(model=fake), envelope, _goal_decision(action="revise")
        )
        self.assertEqual(decision.action, "revise")
        task_ids = [task.task_id for task in decision.plan.tasks]
        self.assertIn("stable-task", task_ids)
        stable = next(task for task in decision.plan.tasks if task.task_id == "stable-task")
        self.assertEqual(stable.status, "completed")
        self.assertEqual(decision.plan.revision, 2)

    def test_dependency_cycle_rejected_uses_single_task_fallback(self) -> None:
        fake = _FakeModel(
            [
                {
                    "tasks": (
                        {
                            "task_id": "a",
                            "title": "A",
                            "objective": "A",
                            "success_criteria": ("完成",),
                            "capability": "local_rag",
                            "depends_on": ("b",),
                        },
                        {
                            "task_id": "b",
                            "title": "B",
                            "objective": "B",
                            "success_criteria": ("完成",),
                            "capability": "local_rag",
                            "depends_on": ("a",),
                        },
                    )
                }
            ]
        )
        decision = self._plan_via(TaskPlanner(model=fake), _envelope(), _goal_decision())
        self.assertEqual(len(decision.plan.tasks), 1)

    def test_max_twelve_tasks_enforced(self) -> None:
        many = tuple(
            {
                "task_id": f"task-{index}",
                "title": f"任务{index}",
                "objective": f"目标{index}",
                "success_criteria": ("完成",),
                "capability": "direct_chat",
            }
            for index in range(13)
        )
        fake = _FakeModel([{"tasks": many}])
        decision = self._plan_via(TaskPlanner(model=fake), _envelope(), _goal_decision())
        self.assertEqual(len(decision.plan.tasks), 1)

    def test_revision_starts_at_one_and_increments(self) -> None:
        fake = _FakeModel(
            [
                {
                    "tasks": (
                        {
                            "task_id": "research",
                            "title": "检索证据",
                            "objective": "比较 RAG 与 GraphRAG",
                            "success_criteria": ("找到证据",),
                            "capability": "local_rag",
                        },
                    )
                }
            ]
        )
        decision = self._plan_via(TaskPlanner(model=fake), _envelope(), _goal_decision())
        self.assertEqual(decision.plan.revision, 1)
        current = _plan(tasks=(_task(),))
        envelope = _envelope(workspace=_workspace(task_plan=current))
        fake = _FakeModel(
            [
                {
                    "tasks": (
                        {
                            "task_id": "other",
                            "title": "其他",
                            "objective": "其他",
                            "success_criteria": ("完成",),
                            "capability": "direct_chat",
                        },
                    )
                }
            ]
        )
        revised = asyncio.run(
            TaskPlanner(model=fake).plan(
                envelope,
                _interpretation("refine_goal"),
                _goal_decision(action="revise"),  # type: ignore[arg-type]
            )
        )
        self.assertEqual(revised.plan.revision, 2)

    def test_model_failure_falls_back_to_single_task(self) -> None:
        fake = _FakeModel([RuntimeError("down")])
        decision = self._plan_via(TaskPlanner(model=fake), _envelope(), _goal_decision())
        self.assertEqual(len(decision.plan.tasks), 1)
        self.assertEqual(decision.plan.tasks[0].goal_id, "a" * 32)

    def test_resume_after_approval_preserves_existing_plan(self) -> None:
        current = _plan(tasks=(_task(),))
        envelope = _envelope(workspace=_workspace(task_plan=current))
        decision = asyncio.run(
            TaskPlanner().plan(
                envelope,
                _interpretation("resume_after_approval"),
                _goal_decision(action="keep"),  # type: ignore[arg-type]
            )
        )
        self.assertEqual(decision.action, "keep")
        self.assertIs(decision.plan, current)

    def test_new_turn_replans_completed_plan_when_goal_is_kept(self) -> None:
        current = _plan(tasks=(_task(task_id="answer", status="completed"),))
        envelope = _envelope(
            current_message="agent性能判断标准",
            workspace=_workspace(task_plan=current),
        )
        fake = _FakeModel(
            [
                {
                    "tasks": (
                        {
                            "task_id": "answer",
                            "title": "重新回答本轮问题",
                            "objective": "回答 agent 性能判断标准",
                            "success_criteria": ("给出本轮答案",),
                            "capability": "direct_chat",
                        },
                    )
                }
            ]
        )

        decision = asyncio.run(
            TaskPlanner(model=fake).plan(
                envelope,
                _interpretation(
                    "answer_within_goal",
                    resolved_request="回答 agent 性能判断标准",
                ),
                _goal_decision(action="keep"),  # type: ignore[arg-type]
            )
        )

        self.assertEqual(decision.action, "revise")
        self.assertEqual(fake.calls, 1)
        self.assertEqual(len(decision.plan.tasks), 1)
        self.assertEqual(decision.plan.tasks[0].status, "pending")
        self.assertEqual(decision.plan.revision, 2)

    def test_disabled_rag_fallback_uses_direct_chat(self) -> None:
        envelope = _envelope(
            rag_mode="disabled",
            workspace=_workspace(),
        )
        decision = self._plan_via(
            TaskPlanner(model=_FakeModel([RuntimeError("down")])), envelope, _goal_decision()
        )
        self.assertEqual(decision.plan.tasks[0].capability, "direct_chat")


if __name__ == "__main__":
    unittest.main()
