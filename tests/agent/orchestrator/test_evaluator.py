from __future__ import annotations

import unittest
from datetime import UTC, datetime

from paper_research_agent.agent.orchestrator.evaluator import (
    TaskEvaluation,
    evaluate_task,
    reduce_workspace,
)
from paper_research_agent.agent.orchestrator.models import (
    AgentTask,
    ChildTaskResult,
    ConversationWorkspace,
    GoalDecision,
    GoalState,
    TaskPlan,
    TaskPlanDecision,
)


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


def _task(**overrides: object) -> AgentTask:
    values: dict[str, object] = {
        "task_id": "task-1",
        "goal_id": "a" * 32,
        "title": "检索证据",
        "objective": "比较 RAG 与 GraphRAG",
        "success_criteria": ("找到至少两篇论文证据",),
        "capability": "local_rag",
        "status": "running",
        "attempt_count": 0,
    }
    values.update(overrides)
    return AgentTask(**values)


def _result(status: str = "completed", **overrides: object) -> ChildTaskResult:
    values: dict[str, object] = {
        "child_run_id": "r" * 32,
        "task_id": "task-1",
        "capability": "local_rag",
        "status": status,
        "summary": "已找到证据",
        "citation_kind": "local_paper" if status == "completed" else "none",
    }
    if status == "waiting_approval":
        values["pending_approval"] = {"approval_request_id": "a" * 32}
    if status == "failed":
        values["error_code"] = "test_failure"
    values.update(overrides)
    return ChildTaskResult(**values)


def _plan(tasks: tuple[AgentTask, ...]) -> TaskPlan:
    return TaskPlan(
        plan_id="c" * 32,
        goal_id="a" * 32,
        revision=1,
        tasks=tasks,
        created_at=_utc(),
        updated_at=_utc(),
    )


def _workspace(**overrides: object) -> ConversationWorkspace:
    values: dict[str, object] = {
        "conversation_id": "conversation-1",
        "version": 0,
        "active_goal": _goal(),
        "task_plan": _plan((_task(),)),
        "updated_at": _utc(),
    }
    values.update(overrides)
    return ConversationWorkspace(**values)


class EvaluateTaskTests(unittest.TestCase):
    def _evaluate(
        self,
        task: AgentTask,
        result: ChildTaskResult,
        *,
        child_calls_used: int = 0,
        replans_used: int = 0,
    ) -> TaskEvaluation:
        return evaluate_task(
            task,
            result,
            child_calls_used=child_calls_used,
            replans_used=replans_used,
        )

    def test_complete_local_with_citation(self) -> None:
        evaluation = self._evaluate(_task(), _result(status="completed"))
        self.assertEqual(evaluation.outcome, "complete")
        self.assertEqual(evaluation.satisfied_criteria, ("找到至少两篇论文证据",))

    def test_insufficient_evidence_retries_when_attempt_available(self) -> None:
        task = _task(attempt_count=0)
        result = _result(status="insufficient_evidence", citation_kind="none")
        evaluation = self._evaluate(task, result)
        self.assertEqual(evaluation.outcome, "retry")
        self.assertEqual(evaluation.missing_criteria, ("找到至少两篇论文证据",))

    def test_insufficient_evidence_replans_when_attempts_exhausted(self) -> None:
        task = _task(attempt_count=1)
        result = _result(status="insufficient_evidence", citation_kind="none")
        evaluation = self._evaluate(task, result, replans_used=0)
        self.assertEqual(evaluation.outcome, "replan")

    def test_insufficient_evidence_fails_without_budget(self) -> None:
        task = _task(attempt_count=1)
        result = _result(status="insufficient_evidence", citation_kind="none")
        evaluation = self._evaluate(task, result, replans_used=1)
        self.assertEqual(evaluation.outcome, "fail")

    def test_waiting_approval_waits_for_user(self) -> None:
        result = _result(status="waiting_approval", citation_kind="none")
        evaluation = self._evaluate(_task(), result)
        self.assertEqual(evaluation.outcome, "wait_user")

    def test_local_without_local_citation_fails(self) -> None:
        result = _result(status="completed", citation_kind="external")
        evaluation = self._evaluate(_task(capability="local_rag"), result)
        self.assertEqual(evaluation.outcome, "fail")
        self.assertIn("本地论文引用", evaluation.reason)

    def test_external_dynamic_result_completes_without_local_citation(self) -> None:
        result = _result(status="completed", citation_kind="external")
        evaluation = self._evaluate(
            _task(task_id="web-task", capability="dynamic_tools"), result
        )
        self.assertEqual(evaluation.outcome, "complete")

    def test_budget_exhausted_fails(self) -> None:
        result = _result(status="insufficient_evidence", citation_kind="none")
        evaluation = self._evaluate(_task(), result, child_calls_used=3)
        self.assertEqual(evaluation.outcome, "fail")

    def test_completed_result_not_failed_by_budget(self) -> None:
        result = _result(status="completed")
        evaluation = self._evaluate(_task(), result, child_calls_used=3)
        self.assertEqual(evaluation.outcome, "complete")

    def test_failed_task_retries_then_fails(self) -> None:
        result = _result(status="failed", citation_kind="none")
        first = self._evaluate(_task(attempt_count=0), result)
        self.assertEqual(first.outcome, "retry")
        second = self._evaluate(_task(attempt_count=1), result)
        self.assertEqual(second.outcome, "fail")


class ReduceWorkspaceTests(unittest.TestCase):
    def test_reducer_completes_task(self) -> None:
        workspace = _workspace()
        evaluation = TaskEvaluation(
            task_id="task-1",
            outcome="complete",
            satisfied_criteria=("找到至少两篇论文证据",),
            reason="成功",
        )
        reduced = reduce_workspace(
            workspace,
            task_id="task-1",
            evaluation=evaluation,
            result=_result(status="completed"),
        )
        task = reduced.task_plan.tasks[0]
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.result_ref, "r" * 32)

    def test_reducer_retries_task(self) -> None:
        workspace = _workspace()
        evaluation = TaskEvaluation(
            task_id="task-1",
            outcome="retry",
            missing_criteria=("找到至少两篇论文证据",),
            reason="证据不足",
        )
        reduced = reduce_workspace(
            workspace, task_id="task-1", evaluation=evaluation, result=None
        )
        task = reduced.task_plan.tasks[0]
        self.assertEqual(task.status, "pending")
        self.assertEqual(task.attempt_count, 1)

    def test_reducer_fails_task_with_blocked_reason(self) -> None:
        workspace = _workspace()
        evaluation = TaskEvaluation(
            task_id="task-1", outcome="fail", reason="无可用证据"
        )
        reduced = reduce_workspace(
            workspace, task_id="task-1", evaluation=evaluation, result=None
        )
        task = reduced.task_plan.tasks[0]
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.blocked_reason, "无可用证据")

    def test_reducer_waits_for_approval(self) -> None:
        workspace = _workspace()
        evaluation = TaskEvaluation(task_id="task-1", outcome="wait_user", reason="等待审批")
        reduced = reduce_workspace(
            workspace,
            task_id="task-1",
            evaluation=evaluation,
            result=_result(status="waiting_approval", citation_kind="none"),
        )
        task = reduced.task_plan.tasks[0]
        self.assertEqual(task.status, "waiting_approval")

    def test_reducer_applies_goal_create(self) -> None:
        workspace = _workspace()
        new_goal = _goal(goal_id="d" * 32, objective="新的目标")
        decision = GoalDecision(action="create", goal=new_goal, rationale="新目标")
        reduced = reduce_workspace(workspace, goal_decision=decision)
        self.assertEqual(reduced.active_goal.goal_id, "d" * 32)

    def test_reducer_applies_goal_abandon(self) -> None:
        workspace = _workspace()
        abandoned = _goal(status="abandoned")
        decision = GoalDecision(action="abandon", goal=abandoned, rationale="取消")
        reduced = reduce_workspace(workspace, goal_decision=decision)
        self.assertEqual(reduced.active_goal.status, "abandoned")

    def test_reducer_applies_plan_revise(self) -> None:
        workspace = _workspace()
        new_plan = _plan(
            (_task(task_id="other-task", capability="direct_chat", status="pending"),)
        )
        decision = TaskPlanDecision(action="revise", plan=new_plan, rationale="修订")
        reduced = reduce_workspace(workspace, plan_decision=decision)
        self.assertEqual(reduced.task_plan.tasks[0].task_id, "other-task")


if __name__ == "__main__":
    unittest.main()
