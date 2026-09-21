from __future__ import annotations

import unittest
from datetime import UTC, datetime

from paper_research_agent.agent.orchestrator.evaluator import (
    TaskEvaluation,
    degradation_codes_for_results,
    evaluate_task,
    reduce_workspace,
    reduce_workspace_batch,
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
    def test_degradation_matrix_uses_latest_branch_attempts(self) -> None:
        local_ok = _result(task_id="local", status="completed")
        local_failed = _result(task_id="local", status="failed")
        external_ok = _result(
            task_id="external",
            capability="dynamic_tools",
            status="completed",
            citation_kind="external",
        )
        external_failed = _result(
            task_id="external",
            capability="dynamic_tools",
            status="failed",
            error_code="provider_offline",
        )

        self.assertEqual(
            degradation_codes_for_results((local_ok, external_failed)),
            ("external_research_unavailable",),
        )
        self.assertEqual(
            degradation_codes_for_results((local_failed, external_ok)),
            ("local_evidence_insufficient",),
        )
        self.assertEqual(
            degradation_codes_for_results((local_failed, external_failed)),
            (
                "local_evidence_insufficient",
                "external_research_unavailable",
                "model_background_only",
            ),
        )
        self.assertEqual(
            degradation_codes_for_results(
                (local_failed, external_failed, local_ok, external_ok)
            ),
            (),
        )

    def test_external_timeout_and_write_denial_have_stable_codes(self) -> None:
        timeout = _result(
            task_id="external",
            capability="dynamic_tools",
            status="failed",
            error_code="parallel_child_timeout",
        )
        write = timeout.model_copy(update={"error_code": "parallel_write_not_allowed"})

        self.assertEqual(
            degradation_codes_for_results((timeout,)),
            ("external_research_timeout",),
        )
        self.assertEqual(
            degradation_codes_for_results((write,)),
            ("parallel_write_not_allowed",),
        )

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

    def test_insufficient_evidence_replans_without_blind_retry(self) -> None:
        task = _task(attempt_count=0)
        result = _result(status="insufficient_evidence", citation_kind="none")
        evaluation = self._evaluate(task, result)
        self.assertEqual(evaluation.outcome, "replan")
        self.assertEqual(evaluation.missing_criteria, ("找到至少两篇论文证据",))

    def test_insufficient_evidence_fails_without_replan_budget(self) -> None:
        task = _task(attempt_count=0)
        result = _result(status="insufficient_evidence", citation_kind="none")
        evaluation = self._evaluate(task, result, replans_used=1)
        self.assertEqual(evaluation.outcome, "fail")

    def test_fast_path_insufficient_evidence_does_not_replan(self) -> None:
        task = _task(attempt_count=0)
        result = _result(status="insufficient_evidence", citation_kind="none")

        evaluation = evaluate_task(
            task,
            result,
            child_calls_used=0,
            replans_used=0,
            allow_replan=False,
        )

        self.assertEqual(evaluation.outcome, "fail")
        self.assertIn("不重复执行", evaluation.reason)

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
        result = _result(
            status="completed",
            capability="dynamic_tools",
            citation_kind="external",
        )
        evaluation = self._evaluate(
            _task(task_id="web-task", capability="dynamic_tools"), result
        )
        self.assertEqual(evaluation.outcome, "complete")

    def test_routed_direct_result_is_not_held_to_original_local_policy(self) -> None:
        result = _result(
            status="completed",
            capability="direct_chat",
            citation_kind="none",
        )

        evaluation = self._evaluate(_task(capability="local_rag"), result)

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

    def test_batch_reducer_updates_both_tasks_atomically(self) -> None:
        local = _task(
            task_id="local-research",
            capability="local_rag",
            status="running",
            parallel_group_id="hybrid-research",
        )
        dynamic = _task(
            task_id="external-research",
            capability="dynamic_tools",
            status="running",
            parallel_group_id="hybrid-research",
            allowed_tool_risks=("network_read",),
            allowed_tool_names=(
                "search_scholarly_sources",
                "resolve_paper_identifier",
                "get_citation_graph",
                "check_paper_status",
            ),
        )
        workspace = _workspace(task_plan=_plan((local, dynamic)))
        evaluations = (
            TaskEvaluation(task_id=local.task_id, outcome="complete", reason="成功"),
            TaskEvaluation(task_id=dynamic.task_id, outcome="complete", reason="成功"),
        )
        results = (
            _result(task_id=local.task_id),
            _result(
                task_id=dynamic.task_id,
                capability="dynamic_tools",
                citation_kind="external",
            ),
        )

        reduced = reduce_workspace_batch(workspace, evaluations=evaluations, results=results)

        self.assertEqual(
            tuple(task.status for task in reduced.task_plan.tasks),
            ("completed", "completed"),
        )
        self.assertEqual(tuple(task.status for task in workspace.task_plan.tasks), ("running", "running"))

    def test_batch_reducer_rejects_unknown_task_without_partial_update(self) -> None:
        workspace = _workspace()
        evaluations = (
            TaskEvaluation(task_id="task-1", outcome="complete", reason="成功"),
            TaskEvaluation(task_id="missing", outcome="complete", reason="成功"),
        )
        results = (_result(), _result(task_id="missing"))

        with self.assertRaisesRegex(ValueError, "unknown task"):
            reduce_workspace_batch(workspace, evaluations=evaluations, results=results)
        self.assertEqual(workspace.task_plan.tasks[0].status, "running")


if __name__ == "__main__":
    unittest.main()
