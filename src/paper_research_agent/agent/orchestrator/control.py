"""Pure plan-control rules for interruptible, editable Agent runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from paper_research_agent.agent.orchestrator.models import (
    AcceptanceCriterion,
    AgentTask,
    ConversationWorkspace,
    FrozenModel,
    TaskBudget,
)

RunControlAction = Literal["pause", "resume", "cancel"]
RunControlStatus = Literal[
    "running",
    "pause_requested",
    "paused",
    "resuming",
    "cancel_requested",
    "cancelled",
    "completed",
    "failed",
    "waiting_approval",
]


class AgentRunControl(FrozenModel):
    request_id: str = Field(min_length=1, max_length=256)
    run_id: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(min_length=1, max_length=256)
    status: RunControlStatus = "running"
    revision: int = Field(default=0, ge=0)
    updated_at: datetime


class RunControlCommand(FrozenModel):
    action: RunControlAction
    expected_revision: int = Field(ge=0)


class RunControlConflict(ValueError):
    """The requested transition is stale or illegal for the current state."""


def transition_run_control(
    control: AgentRunControl,
    command: RunControlCommand,
    *,
    now: datetime | None = None,
) -> AgentRunControl:
    """Apply one optimistic and idempotency-safe run control transition."""

    if command.expected_revision != control.revision:
        raise RunControlConflict("run control revision conflict")
    targets: dict[tuple[RunControlStatus, RunControlAction], RunControlStatus] = {
        ("running", "pause"): "pause_requested",
        ("resuming", "pause"): "pause_requested",
        ("paused", "resume"): "resuming",
        ("running", "cancel"): "cancel_requested",
        ("pause_requested", "cancel"): "cancel_requested",
        ("paused", "cancel"): "cancel_requested",
        ("resuming", "cancel"): "cancel_requested",
        ("waiting_approval", "cancel"): "cancel_requested",
    }
    target = targets.get((control.status, command.action))
    if target is None:
        raise RunControlConflict(
            f"cannot {command.action} a run in {control.status} state"
        )
    return control.model_copy(
        update={
            "status": target,
            "revision": control.revision + 1,
            "updated_at": now or datetime.now(UTC),
        }
    )


class TaskEdit(FrozenModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    title: str | None = Field(default=None, min_length=1, max_length=200)
    objective: str | None = Field(default=None, min_length=1, max_length=1000)
    success_criteria: tuple[str, ...] | None = Field(default=None, min_length=1, max_length=8)
    execution_reason: str | None = Field(default=None, min_length=1, max_length=500)
    budget: TaskBudget | None = None

    @field_validator("title", "objective", "execution_reason")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("task edit text must not be blank")
        return normalized


class PlanEdit(FrozenModel):
    expected_revision: int = Field(ge=1)
    objective: str | None = Field(default=None, min_length=1, max_length=2000)
    acceptance_criteria: tuple[AcceptanceCriterion, ...] | None = Field(
        default=None, max_length=12
    )
    ordered_task_ids: tuple[str, ...] | None = Field(default=None, min_length=1, max_length=12)
    task_edits: tuple[TaskEdit, ...] = Field(default=(), max_length=12)
    skip_task_ids: tuple[str, ...] = Field(default=(), max_length=12)
    retry_task_ids: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def validate_unique_targets(self) -> PlanEdit:
        edited = [item.task_id for item in self.task_edits]
        for label, values in (
            ("ordered task IDs", self.ordered_task_ids or ()),
            ("edited task IDs", tuple(edited)),
            ("skipped task IDs", self.skip_task_ids),
            ("retried task IDs", self.retry_task_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        if set(self.skip_task_ids) & set(self.retry_task_ids):
            raise ValueError("a task cannot be skipped and retried in the same edit")
        return self


class PlanEditConflict(ValueError):
    """The requested edit cannot preserve established execution facts."""


def apply_plan_edit(
    workspace: ConversationWorkspace,
    edit: PlanEdit,
    *,
    now: datetime | None = None,
) -> ConversationWorkspace:
    """Apply one optimistic edit while preserving every completed task result."""

    plan = workspace.task_plan
    goal = workspace.active_goal
    if plan is None or goal is None:
        raise PlanEditConflict("the run has no editable active plan")
    if plan.revision != edit.expected_revision:
        raise PlanEditConflict("plan revision conflict")
    task_by_id = {task.task_id: task for task in plan.tasks}
    known = set(task_by_id)
    referenced = {
        *(edit.ordered_task_ids or ()),
        *(item.task_id for item in edit.task_edits),
        *edit.skip_task_ids,
        *edit.retry_task_ids,
    }
    unknown = referenced - known
    if unknown:
        raise PlanEditConflict(f"unknown task IDs: {', '.join(sorted(unknown))}")
    if edit.ordered_task_ids is not None and set(edit.ordered_task_ids) != known:
        raise PlanEditConflict("ordered task IDs must contain every task exactly once")

    edited_tasks = dict(task_by_id)
    for task_edit in edit.task_edits:
        current = edited_tasks[task_edit.task_id]
        if current.status == "completed":
            raise PlanEditConflict("completed tasks are immutable")
        updates = {
            key: value
            for key in (
                "title",
                "objective",
                "success_criteria",
                "execution_reason",
                "budget",
            )
            if (value := getattr(task_edit, key)) is not None
        }
        edited_tasks[current.task_id] = AgentTask.model_validate(
            {**current.model_dump(), **updates}
        )
    for task_id in edit.skip_task_ids:
        current = edited_tasks[task_id]
        if current.status == "completed":
            raise PlanEditConflict("completed tasks cannot be skipped")
        if current.status == "running":
            raise PlanEditConflict("the running task must be paused before it can be skipped")
        edited_tasks[task_id] = current.model_copy(
            update={"status": "skipped", "blocked_reason": "用户跳过该步骤"}
        )
    for task_id in edit.retry_task_ids:
        current = edited_tasks[task_id]
        if current.status != "failed":
            raise PlanEditConflict("only failed tasks can be retried")
        edited_tasks[task_id] = current.model_copy(
            update={"status": "pending", "blocked_reason": None, "result_ref": None}
        )

    order = edit.ordered_task_ids or tuple(task.task_id for task in plan.tasks)
    reordered = tuple(edited_tasks[task_id] for task_id in order)
    position = {task_id: index for index, task_id in enumerate(order)}
    for task in reordered:
        if any(position[dependency] > position[task.task_id] for dependency in task.depends_on):
            raise PlanEditConflict("task order must keep dependencies before dependants")

    changed_at = now or datetime.now(UTC)
    revised_plan = plan.model_copy(
        update={"revision": plan.revision + 1, "tasks": reordered, "updated_at": changed_at}
    )
    revised_goal = type(goal).model_validate(
        {
            **goal.model_dump(),
            "objective": edit.objective or goal.objective,
            "acceptance_criteria": (
                edit.acceptance_criteria
                if edit.acceptance_criteria is not None
                else goal.acceptance_criteria
            ),
            "updated_at": changed_at,
        }
    )
    return workspace.model_copy(update={"active_goal": revised_goal, "task_plan": revised_plan})


def task_budget_exhausted(task: AgentTask) -> str | None:
    """Return a stable machine-readable reason before another task attempt."""

    budget = task.budget
    usage = task.usage
    if budget.max_calls is not None and usage.call_count >= budget.max_calls:
        return "call_budget_exhausted"
    if budget.max_seconds is not None and usage.elapsed_seconds >= budget.max_seconds:
        return "time_budget_exhausted"
    if budget.max_cost_usd is not None and usage.cost_usd >= budget.max_cost_usd:
        return "cost_budget_exhausted"
    return None


def explain_task(workspace: ConversationWorkspace, task_id: str) -> str:
    """Build a user-safe explanation from plan facts only."""

    plan = workspace.task_plan
    if plan is None:
        raise PlanEditConflict("the run has no task plan")
    task = next((item for item in plan.tasks if item.task_id == task_id), None)
    if task is None:
        raise PlanEditConflict("unknown task ID")
    dependencies = ", ".join(task.depends_on) if task.depends_on else "无前置步骤"
    criteria = "；".join(task.success_criteria)
    return (
        f"{task.execution_reason}。能力：{task.capability}；前置：{dependencies}；"
        f"成功标准：{criteria}。"
    )
