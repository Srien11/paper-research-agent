from __future__ import annotations

from datetime import UTC, datetime

import pytest

from paper_research_agent.agent.orchestrator.control import (
    AgentRunControl,
    PlanEdit,
    PlanEditConflict,
    RunControlCommand,
    RunControlConflict,
    TaskEdit,
    apply_plan_edit,
    explain_task,
    task_budget_exhausted,
    transition_run_control,
)
from paper_research_agent.agent.orchestrator.models import (
    AcceptanceCriterion,
    AgentTask,
    ConversationWorkspace,
    GoalState,
    TaskBudget,
    TaskPlan,
    TaskUsage,
)

NOW = datetime(2026, 8, 11, tzinfo=UTC)


def _workspace() -> ConversationWorkspace:
    goal_id = "a" * 32
    tasks = (
        AgentTask(
            task_id="done",
            goal_id=goal_id,
            title="已完成",
            objective="保留结果",
            success_criteria=("已有产物",),
            capability="local_rag",
            status="completed",
            result_ref="artifact:done",
            execution_reason="先收集本地证据",
        ),
        AgentTask(
            task_id="failed",
            goal_id=goal_id,
            title="失败步骤",
            objective="只重试我",
            success_criteria=("成功",),
            capability="dynamic_tools",
            status="failed",
            depends_on=("done",),
            attempt_count=1,
            blocked_reason="temporary",
        ),
        AgentTask(
            task_id="pending",
            goal_id=goal_id,
            title="待执行",
            objective="可以跳过",
            success_criteria=("完成",),
            capability="direct_chat",
            depends_on=("failed",),
        ),
    )
    return ConversationWorkspace(
        conversation_id="conversation-123456",
        active_goal=GoalState(
            goal_id=goal_id,
            objective="旧目标",
            acceptance_criteria=(
                AcceptanceCriterion(criterion_id="old", description="旧标准"),
            ),
            origin_turn_id="b" * 32,
            created_at=NOW,
            updated_at=NOW,
        ),
        task_plan=TaskPlan(
            plan_id="c" * 32,
            goal_id=goal_id,
            revision=3,
            tasks=tasks,
            created_at=NOW,
            updated_at=NOW,
        ),
        updated_at=NOW,
    )


def test_edit_preserves_completed_results_and_only_retries_failed_step() -> None:
    edited = apply_plan_edit(
        _workspace(),
        PlanEdit(
            expected_revision=3,
            objective="新目标",
            acceptance_criteria=(
                AcceptanceCriterion(criterion_id="new", description="新标准"),
            ),
            task_edits=(
                TaskEdit(
                    task_id="failed",
                    success_criteria=("新成功标准",),
                    budget=TaskBudget(max_seconds=30, max_calls=2, max_cost_usd=0.5),
                ),
            ),
            retry_task_ids=("failed",),
            skip_task_ids=("pending",),
        ),
        now=NOW,
    )

    assert edited.active_goal is not None
    assert edited.active_goal.objective == "新目标"
    assert edited.task_plan is not None
    assert edited.task_plan.revision == 4
    done, failed, pending = edited.task_plan.tasks
    assert (done.status, done.result_ref, done.attempt_count) == (
        "completed",
        "artifact:done",
        0,
    )
    assert failed.status == "pending"
    assert failed.result_ref is None
    assert failed.blocked_reason is None
    assert failed.attempt_count == 1
    assert failed.success_criteria == ("新成功标准",)
    assert failed.budget.max_calls == 2
    assert pending.status == "skipped"


def test_completed_step_is_immutable() -> None:
    with pytest.raises(PlanEditConflict, match="completed tasks are immutable"):
        apply_plan_edit(
            _workspace(),
            PlanEdit(
                expected_revision=3,
                task_edits=(TaskEdit(task_id="done", objective="重跑它"),),
            ),
        )


def test_reorder_rejects_dependency_inversion() -> None:
    with pytest.raises(PlanEditConflict, match="dependencies before dependants"):
        apply_plan_edit(
            _workspace(),
            PlanEdit(
                expected_revision=3,
                ordered_task_ids=("done", "pending", "failed"),
            ),
        )


def test_only_failed_steps_can_be_retried() -> None:
    with pytest.raises(PlanEditConflict, match="only failed tasks"):
        apply_plan_edit(
            _workspace(),
            PlanEdit(expected_revision=3, retry_task_ids=("pending",)),
        )


@pytest.mark.parametrize(
    ("budget", "usage", "reason"),
    [
        (TaskBudget(max_calls=2), TaskUsage(call_count=2), "call_budget_exhausted"),
        (
            TaskBudget(max_seconds=3),
            TaskUsage(elapsed_seconds=3),
            "time_budget_exhausted",
        ),
        (
            TaskBudget(max_cost_usd=0.2),
            TaskUsage(cost_usd=0.2),
            "cost_budget_exhausted",
        ),
    ],
)
def test_budget_gate_is_durable(
    budget: TaskBudget, usage: TaskUsage, reason: str
) -> None:
    task = _workspace().task_plan.tasks[1].model_copy(  # type: ignore[union-attr]
        update={"budget": budget, "usage": usage}
    )
    assert task_budget_exhausted(task) == reason


def test_explanation_contains_reason_dependencies_and_success_criteria() -> None:
    explanation = explain_task(_workspace(), "failed")
    assert "完成目标所需的计划步骤" in explanation
    assert "done" in explanation
    assert "成功标准" in explanation


def test_pause_resume_and_cancel_are_optimistic_state_transitions() -> None:
    running = AgentRunControl(
        request_id="request-12345678",
        run_id="d" * 32,
        conversation_id="conversation-123456",
        updated_at=NOW,
    )
    pausing = transition_run_control(
        running, RunControlCommand(action="pause", expected_revision=0), now=NOW
    )
    assert (pausing.status, pausing.revision) == ("pause_requested", 1)
    paused = pausing.model_copy(update={"status": "paused"})
    resumed = transition_run_control(
        paused, RunControlCommand(action="resume", expected_revision=1), now=NOW
    )
    assert (resumed.status, resumed.revision) == ("resuming", 2)
    cancelled = transition_run_control(
        resumed, RunControlCommand(action="cancel", expected_revision=2), now=NOW
    )
    assert cancelled.status == "cancel_requested"


def test_control_rejects_stale_revision_and_terminal_transition() -> None:
    completed = AgentRunControl(
        request_id="request-12345678",
        run_id="d" * 32,
        conversation_id="conversation-123456",
        status="completed",
        revision=4,
        updated_at=NOW,
    )
    with pytest.raises(RunControlConflict, match="revision conflict"):
        transition_run_control(
            completed, RunControlCommand(action="cancel", expected_revision=3)
        )
    with pytest.raises(RunControlConflict, match="cannot cancel"):
        transition_run_control(
            completed, RunControlCommand(action="cancel", expected_revision=4)
        )
