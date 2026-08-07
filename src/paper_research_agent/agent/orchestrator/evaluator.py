"""Task result evaluation and pure workspace reduction for the main Agent."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from paper_research_agent.agent.orchestrator.models import (
    AgentTask,
    ChildTaskResult,
    ConversationWorkspace,
    FrozenModel,
    GoalDecision,
    TaskPlanDecision,
)

MAX_CHILD_CALLS_PER_RUN = 3
MAX_REPLANS_PER_RUN = 1
MAX_TASK_ATTEMPTS = 2


class TaskEvaluation(FrozenModel):
    task_id: str = Field(min_length=1, max_length=64)
    outcome: Literal["complete", "retry", "replan", "wait_user", "fail"]
    satisfied_criteria: tuple[str, ...] = Field(default=(), max_length=8)
    missing_criteria: tuple[str, ...] = Field(default=(), max_length=8)
    summary: str = Field(default="", max_length=5000)
    reason: str = Field(min_length=1, max_length=200)


def evaluate_task(
    task: AgentTask,
    result: ChildTaskResult,
    *,
    child_calls_used: int,
    replans_used: int,
    max_child_calls: int = MAX_CHILD_CALLS_PER_RUN,
    max_replans: int = MAX_REPLANS_PER_RUN,
    max_attempts: int = MAX_TASK_ATTEMPTS,
) -> TaskEvaluation:
    """Judge one child result against task criteria without relaxing citation safety."""
    if child_calls_used >= max_child_calls:
        return TaskEvaluation(
            task_id=task.task_id,
            outcome="fail",
            missing_criteria=task.success_criteria,
            summary=result.summary,
            reason="本轮子图调用预算耗尽",
        )
    if result.status == "waiting_approval":
        return TaskEvaluation(
            task_id=task.task_id,
            outcome="wait_user",
            summary=result.summary,
            reason="等待敏感工具审批",
        )
    if result.status == "failed":
        return _retry_or_fail(
            task, result, max_attempts, "子图执行失败"
        )
    if result.status == "insufficient_evidence":
        if task.attempt_count + 1 < max_attempts:
            return TaskEvaluation(
                task_id=task.task_id,
                outcome="retry",
                missing_criteria=task.success_criteria,
                summary=result.summary,
                reason="证据不足，重试任务",
            )
        if replans_used < max_replans:
            return TaskEvaluation(
                task_id=task.task_id,
                outcome="replan",
                missing_criteria=task.success_criteria,
                summary=result.summary,
                reason="证据不足，重新规划任务",
            )
        return TaskEvaluation(
            task_id=task.task_id,
            outcome="fail",
            missing_criteria=task.success_criteria,
            summary=result.summary,
            reason="证据不足且无重试与重规划预算",
        )
    if task.capability == "local_rag" and result.citation_kind != "local_paper":
        return TaskEvaluation(
            task_id=task.task_id,
            outcome="fail",
            missing_criteria=task.success_criteria,
            summary=result.summary,
            reason="本地研究未返回本地论文引用",
        )
    return TaskEvaluation(
        task_id=task.task_id,
        outcome="complete",
        satisfied_criteria=task.success_criteria,
        summary=result.summary,
        reason="任务成功完成",
    )


def _retry_or_fail(
    task: AgentTask,
    result: ChildTaskResult,
    max_attempts: int,
    reason_prefix: str,
) -> TaskEvaluation:
    if task.attempt_count + 1 < max_attempts:
        return TaskEvaluation(
            task_id=task.task_id,
            outcome="retry",
            missing_criteria=task.success_criteria,
            summary=result.summary,
            reason=f"{reason_prefix}，重试任务",
        )
    return TaskEvaluation(
        task_id=task.task_id,
        outcome="fail",
        missing_criteria=task.success_criteria,
        summary=result.summary,
        reason=f"{reason_prefix}且重试预算耗尽",
    )


def reduce_workspace(
    workspace: ConversationWorkspace,
    *,
    goal_decision: GoalDecision | None = None,
    plan_decision: TaskPlanDecision | None = None,
    task_id: str | None = None,
    evaluation: TaskEvaluation | None = None,
    result: ChildTaskResult | None = None,
) -> ConversationWorkspace:
    """Pure reduction: old workspace + decisions/results -> new workspace."""
    updated = workspace
    if goal_decision is not None:
        updated = _apply_goal_decision(updated, goal_decision)
    if plan_decision is not None:
        updated = _apply_plan_decision(updated, plan_decision)
    if task_id is not None and evaluation is not None:
        updated = _apply_task_result(updated, task_id, evaluation, result)
    return updated


def _apply_goal_decision(
    workspace: ConversationWorkspace, decision: GoalDecision
) -> ConversationWorkspace:
    if decision.action == "keep" or decision.goal is None:
        return workspace
    return workspace.model_copy(update={"active_goal": decision.goal})


def _apply_plan_decision(
    workspace: ConversationWorkspace, decision: TaskPlanDecision
) -> ConversationWorkspace:
    if decision.action == "keep":
        return workspace
    if decision.action == "clear":
        return workspace.model_copy(update={"task_plan": None})
    return workspace.model_copy(update={"task_plan": decision.plan})


def _apply_task_result(
    workspace: ConversationWorkspace,
    task_id: str,
    evaluation: TaskEvaluation,
    result: ChildTaskResult | None,
) -> ConversationWorkspace:
    plan = workspace.task_plan
    if plan is None:
        return workspace
    tasks = list(plan.tasks)
    index = next(
        (i for i, task in enumerate(tasks) if task.task_id == task_id), None
    )
    if index is None:
        return workspace
    task = tasks[index]
    updated_task = _updated_task(task, evaluation, result)
    tasks[index] = updated_task
    now = datetime.now(UTC)
    new_plan = plan.model_copy(update={"tasks": tuple(tasks), "updated_at": now})
    return workspace.model_copy(
        update={"task_plan": new_plan, "updated_at": now}
    )


def _updated_task(
    task: AgentTask, evaluation: TaskEvaluation, result: ChildTaskResult | None
) -> AgentTask:
    if evaluation.outcome == "complete":
        return task.model_copy(
            update={
                "status": "completed",
                "result_ref": result.child_run_id if result is not None else task.result_ref,
                "attempt_count": task.attempt_count + 1,
            }
        )
    if evaluation.outcome == "wait_user":
        status = (
            "waiting_approval"
            if result is not None and result.status == "waiting_approval"
            else "waiting_user"
        )
        return task.model_copy(update={"status": status})
    if evaluation.outcome == "retry":
        return task.model_copy(
            update={"status": "pending", "attempt_count": task.attempt_count + 1}
        )
    if evaluation.outcome == "replan":
        return task.model_copy(
            update={"status": "ready", "attempt_count": task.attempt_count + 1}
        )
    return task.model_copy(
        update={
            "status": "failed",
            "blocked_reason": evaluation.reason,
            "attempt_count": task.attempt_count + 1,
        }
    )
