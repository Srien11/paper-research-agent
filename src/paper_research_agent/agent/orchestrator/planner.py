"""Cross-turn goal reconciliation and session-level task planning for the main Agent."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field, ValidationError

from paper_research_agent.agent.orchestrator.models import (
    AcceptanceCriterion,
    AgentContextEnvelope,
    AgentTask,
    Capability,
    ConversationWorkspace,
    FrozenModel,
    GoalDecision,
    GoalState,
    TaskPlan,
    TaskPlanDecision,
    TurnInterpretationV2,
)
from paper_research_agent.agent.orchestrator.prompts import (
    GOAL_RECONCILER_PROMPT_VERSION,
    GOAL_RECONCILER_SYSTEM,
    TASK_PLANNER_PROMPT_VERSION,
    TASK_PLANNER_SYSTEM,
)


class _GoalDraft(FrozenModel):
    objective: str = Field(min_length=1, max_length=2000)
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = Field(default=(), max_length=12)
    constraints: tuple[str, ...] = Field(default=(), max_length=20)


class GoalReconciler:
    """Applies deterministic goal rules first, then optional model completion."""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        *,
        version: str = GOAL_RECONCILER_PROMPT_VERSION,
    ) -> None:
        self.version = version
        self._model = (
            model.with_structured_output(_GoalDraft, method="function_calling")
            if model is not None
            else None
        )

    async def reconcile(
        self, envelope: AgentContextEnvelope, interpretation: TurnInterpretationV2
    ) -> GoalDecision:
        decision = _deterministic_goal_decision(
            envelope.workspace, interpretation, envelope.turn_id
        )
        if decision.action in {"create", "revise"} and self._model is not None:
            decision = await self._complete_with_model(decision, envelope, interpretation)
        return decision

    async def _complete_with_model(
        self,
        decision: GoalDecision,
        envelope: AgentContextEnvelope,
        interpretation: TurnInterpretationV2,
    ) -> GoalDecision:
        model = self._model
        if model is None:
            return decision
        system = SystemMessage(
            content=f"{GOAL_RECONCILER_SYSTEM}\nPROMPT_VERSION={self.version}"
        )
        goal = decision.goal
        user = HumanMessage(
            content=(
                f"CURRENT_MESSAGE\n{envelope.current_message}\n\n"
                f"RESOLVED_REQUEST\n{interpretation.resolved_request}\n\n"
                f"EXISTING_GOAL\n{goal.objective if goal is not None else '（无）'}\n\n"
                f"RELATION\n{interpretation.relation}\n"
            )
        )
        try:
            raw = await model.ainvoke([system, user])
            draft = _GoalDraft.model_validate(raw) if not isinstance(raw, _GoalDraft) else raw
        except Exception:  # noqa: BLE001 - deterministic decision is the fallback
            return decision
        if goal is None:
            return decision
        updated = goal.model_copy(
            update={
                "objective": draft.objective,
                "acceptance_criteria": draft.acceptance_criteria,
                "constraints": _merge_constraints(goal.constraints, draft.constraints),
                "updated_at": datetime.now(UTC),
            }
        )
        return GoalDecision(
            action=decision.action, goal=updated, rationale=decision.rationale
        )


def _deterministic_goal_decision(
    workspace: ConversationWorkspace,
    interpretation: TurnInterpretationV2,
    turn_id: str,
) -> GoalDecision:
    existing = workspace.active_goal
    relation = interpretation.relation
    if relation == "cancel_goal":
        if existing is None:
            return GoalDecision(action="keep", goal=None, rationale="没有可取消的活动目标")
        abandoned = existing.model_copy(
            update={"status": "abandoned", "updated_at": datetime.now(UTC)}
        )
        return GoalDecision(action="abandon", goal=abandoned, rationale="用户取消当前目标")
    if relation == "meta_conversation":
        return GoalDecision(action="keep", goal=existing, rationale="元对话不改变活动目标")
    if relation in {"continue_goal", "resume_after_approval", "answer_within_goal"}:
        if existing is None:
            return _create_decision(interpretation, turn_id)
        return GoalDecision(action="keep", goal=existing, rationale="继续当前目标")
    if relation == "refine_goal":
        if existing is None:
            return _create_decision(interpretation, turn_id)
        objective = interpretation.goal_change_summary or interpretation.resolved_request
        revised = existing.model_copy(
            update={
                "objective": objective,
                "constraints": _merge_constraints(
                    existing.constraints, interpretation.new_constraints
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        return GoalDecision(action="revise", goal=revised, rationale="目标被修订")
    if relation == "new_goal":
        return _create_decision(interpretation, turn_id)
    if existing is None:
        return _create_decision(interpretation, turn_id)
    return GoalDecision(action="keep", goal=existing, rationale="保持当前目标")


def _create_decision(interpretation: TurnInterpretationV2, turn_id: str) -> GoalDecision:
    now = datetime.now(UTC)
    goal = GoalState(
        goal_id=uuid.uuid4().hex,
        objective=interpretation.resolved_request,
        status="active",
        constraints=interpretation.new_constraints,
        origin_turn_id=turn_id,
        created_at=now,
        updated_at=now,
    )
    return GoalDecision(action="create", goal=goal, rationale="建立新目标")


def _merge_constraints(
    existing: tuple[str, ...], new: tuple[str, ...]
) -> tuple[str, ...]:
    merged = tuple(dict.fromkeys((*existing, *new)))
    return merged[:20]


class _TaskDraft(FrozenModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=8)
    capability: Capability
    depends_on: tuple[str, ...] = Field(default=(), max_length=8)
    execution_reason: str = Field(
        default="完成目标所需的计划步骤", min_length=1, max_length=500
    )


class _TaskPlanDraft(FrozenModel):
    tasks: tuple[_TaskDraft, ...] = Field(min_length=1, max_length=12)


class TaskPlanner:
    """Builds and revises session-level task plans for the active goal."""

    def __init__(
        self,
        model: BaseChatModel | None = None,
        *,
        version: str = TASK_PLANNER_PROMPT_VERSION,
    ) -> None:
        self.version = version
        self._model = (
            model.with_structured_output(_TaskPlanDraft, method="function_calling")
            if model is not None
            else None
        )

    async def plan(
        self,
        envelope: AgentContextEnvelope,
        interpretation: TurnInterpretationV2,
        goal_decision: GoalDecision,
    ) -> TaskPlanDecision:
        current = envelope.workspace.task_plan
        if goal_decision.action == "keep":
            return TaskPlanDecision(
                action="keep", plan=current, rationale="目标未变，计划保持不变"
            )
        if goal_decision.action in {"abandon", "satisfy", "block"}:
            return TaskPlanDecision(
                action="keep", plan=current, rationale="目标结束，任务状态由结果评估更新"
            )
        model = self._model
        if model is None:
            return self._fallback_decision(goal_decision, current, envelope, interpretation)
        system = SystemMessage(
            content=f"{TASK_PLANNER_SYSTEM}\nPROMPT_VERSION={self.version}"
        )
        user = HumanMessage(content=_task_planner_user_content(envelope, interpretation))
        try:
            raw = await model.ainvoke([system, user])
            draft = _TaskPlanDraft.model_validate(raw) if not isinstance(raw, _TaskPlanDraft) else raw
        except Exception:  # noqa: BLE001 - single-task fallback on planner failure
            return self._fallback_decision(goal_decision, current, envelope, interpretation)
        try:
            return self._build_decision(
                goal_decision, current, envelope, draft
            )
        except ValidationError:
            return self._fallback_decision(goal_decision, current, envelope, interpretation)

    def _build_decision(
        self,
        goal_decision: GoalDecision,
        current: TaskPlan | None,
        envelope: AgentContextEnvelope,
        draft: _TaskPlanDraft,
    ) -> TaskPlanDecision:
        goal_id = _goal_id(goal_decision, current, envelope)
        completed = (
            tuple(task for task in current.tasks if task.status == "completed")
            if current is not None
            else ()
        )
        kept_ids = {task.task_id for task in completed}
        new_tasks = tuple(
            AgentTask(
                task_id=item.task_id,
                goal_id=goal_id,
                title=item.title,
                objective=item.objective,
                success_criteria=item.success_criteria,
                capability=item.capability,
                status="pending",
                depends_on=item.depends_on,
                execution_reason=item.execution_reason,
            )
            for item in draft.tasks
            if item.task_id not in kept_ids
        )
        tasks = (*completed, *new_tasks)
        revision = (
            1
            if goal_decision.action == "create" or current is None
            else current.revision + 1
        )
        now = datetime.now(UTC)
        plan = TaskPlan(
            plan_id=uuid.uuid4().hex,
            goal_id=goal_id,
            revision=revision,
            tasks=tasks,
            created_at=now,
            updated_at=now,
        )
        action: Literal["create", "revise"] = (
            "create" if current is None else "revise"
        )
        return TaskPlanDecision(
            action=action,
            plan=plan,
            rationale="根据本轮目标修订会话任务计划",
        )

    def _fallback_decision(
        self,
        goal_decision: GoalDecision,
        current: TaskPlan | None,
        envelope: AgentContextEnvelope,
        interpretation: TurnInterpretationV2,
    ) -> TaskPlanDecision:
        goal_id = _goal_id(goal_decision, current, envelope)
        capability: Capability = (
            "direct_chat" if envelope.rag_mode == "disabled" else "local_rag"
        )
        task = AgentTask(
            task_id="single-research-task",
            goal_id=goal_id,
            title="完成当前请求",
            objective=interpretation.resolved_request,
            success_criteria=("完成当前请求",),
            capability=capability,
            status="pending",
            execution_reason="直接完成当前请求并据此判断目标是否达成",
        )
        now = datetime.now(UTC)
        revision = (
            1
            if goal_decision.action == "create" or current is None
            else current.revision + 1
        )
        plan = TaskPlan(
            plan_id=uuid.uuid4().hex,
            goal_id=goal_id,
            revision=revision,
            tasks=(task,),
            created_at=now,
            updated_at=now,
        )
        action: Literal["create", "revise"] = (
            "create" if current is None else "revise"
        )
        return TaskPlanDecision(
            action=action,
            plan=plan,
            rationale="模型规划不可用，使用单任务降级",
        )


def _goal_id(
    goal_decision: GoalDecision,
    current: TaskPlan | None,
    envelope: AgentContextEnvelope,
) -> str:
    if goal_decision.goal is not None:
        return goal_decision.goal.goal_id
    if current is not None:
        return current.goal_id
    if envelope.workspace.active_goal is not None:
        return envelope.workspace.active_goal.goal_id
    raise ValueError("no active goal to plan tasks for")


def _task_planner_user_content(
    envelope: AgentContextEnvelope, interpretation: TurnInterpretationV2
) -> str:
    goal = envelope.workspace.active_goal
    goal_text = f"{goal.objective}" if goal is not None else "（无）"
    task_plan = envelope.workspace.task_plan
    existing_tasks = (
        "; ".join(f"{task.status}: {task.task_id}" for task in task_plan.tasks)
        if task_plan is not None
        else "（无）"
    )
    return (
        f"CURRENT_MESSAGE\n{envelope.current_message}\n\n"
        f"RESOLVED_REQUEST\n{interpretation.resolved_request}\n\n"
        f"ACTIVE_GOAL\n{goal_text}\n\n"
        f"EXISTING_TASKS\n{existing_tasks}\n\n"
        f"RAG_MODE\n{envelope.rag_mode}\n"
    )
