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
from paper_research_agent.agent.orchestrator.planning_route import (
    validate_source_policy,
)
from paper_research_agent.agent.orchestrator.prompts import (
    GOAL_RECONCILER_PROMPT_VERSION,
    GOAL_RECONCILER_SYSTEM,
    TASK_PLANNER_PROMPT_VERSION,
    TASK_PLANNER_SYSTEM,
)
from paper_research_agent.agent.tooling.catalog import SCHOLARLY_NETWORK_TOOL_NAMES


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


def _build_single_task_plan(
    *,
    goal_id: str,
    request: str,
    capability: Capability,
    success_criteria: tuple[str, ...],
    title: str,
    execution_reason: str,
    revision: int,
) -> TaskPlan:
    now = datetime.now(UTC)
    task = AgentTask(
        task_id=uuid.uuid4().hex,
        goal_id=goal_id,
        title=title,
        objective=request,
        success_criteria=success_criteria,
        capability=capability,
        status="pending",
        depends_on=(),
        execution_reason=execution_reason,
    )
    return TaskPlan(
        plan_id=uuid.uuid4().hex,
        goal_id=goal_id,
        revision=revision,
        tasks=(task,),
        created_at=now,
        updated_at=now,
    )


def build_single_local_rag_decisions(
    envelope: AgentContextEnvelope,
) -> tuple[TurnInterpretationV2, GoalDecision, TaskPlanDecision]:
    """Materialize a strict new-goal local-RAG plan without model calls."""
    request = " ".join(envelope.current_message.split())
    if not request:
        raise ValueError("single local RAG request must not be blank")
    if len(request) > 1000:
        raise ValueError("single local RAG request exceeds the 1000 character limit")
    now = datetime.now(UTC)
    interpretation = TurnInterpretationV2(
        relation="new_goal",
        resolved_request=request,
        needs_clarification=False,
        confidence=1.0,
    )
    goal = GoalState(
        goal_id=uuid.uuid4().hex,
        objective=request,
        status="active",
        origin_turn_id=envelope.turn_id,
        created_at=now,
        updated_at=now,
    )
    goal_decision = GoalDecision(
        action="create",
        goal=goal,
        rationale="确定性单一私有论文请求",
    )
    plan = _build_single_task_plan(
        goal_id=goal.goal_id,
        request=request,
        capability="local_rag",
        success_criteria=("使用本地论文证据完成当前请求",),
        title="完成当前本地论文请求",
        execution_reason="使用本地论文证据完成单一研究请求",
        revision=1,
    )
    plan_decision = TaskPlanDecision(
        action="create",
        plan=plan,
        rationale="确定性构造单一 local_rag 任务",
    )
    return interpretation, goal_decision, plan_decision


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
        if goal_decision.action == "keep" and (
            interpretation.relation == "resume_after_approval"
            or (goal_decision.goal is None and current is None)
        ):
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
                goal_decision, current, envelope, interpretation, draft
            )
        except ValidationError:
            return self._fallback_decision(goal_decision, current, envelope, interpretation)

    def _build_decision(
        self,
        goal_decision: GoalDecision,
        current: TaskPlan | None,
        envelope: AgentContextEnvelope,
        interpretation: TurnInterpretationV2,
        draft: _TaskPlanDraft,
    ) -> TaskPlanDecision:
        goal_id = _goal_id(goal_decision, current, envelope)
        completed = (
            tuple(
                task.model_copy(update={"parallel_group_id": None})
                for task in current.tasks
                if task.status == "completed"
            )
            if current is not None and goal_decision.action == "revise"
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
        tasks = enforce_capability_plan(
            (*completed, *new_tasks),
            envelope=envelope,
            resolved_request=interpretation.resolved_request,
            goal_id=goal_id,
        )
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
        revision = (
            1
            if goal_decision.action == "create" or current is None
            else current.revision + 1
        )
        plan = _build_single_task_plan(
            goal_id=goal_id,
            request=interpretation.resolved_request,
            capability=capability,
            success_criteria=("完成当前请求",),
            title="完成当前请求",
            execution_reason="直接完成当前请求并据此判断目标是否达成",
            revision=revision,
        )
        tasks = enforce_capability_plan(
            plan.tasks,
            envelope=envelope,
            resolved_request=interpretation.resolved_request,
            goal_id=goal_id,
        )
        if tasks != plan.tasks:
            plan = plan.model_copy(update={"tasks": tasks})
        action: Literal["create", "revise"] = (
            "create" if current is None else "revise"
        )
        return TaskPlanDecision(
            action=action,
            plan=plan,
            rationale="模型规划不可用，使用单任务降级",
        )


def enforce_capability_plan(
    tasks: tuple[AgentTask, ...],
    *,
    envelope: AgentContextEnvelope,
    resolved_request: str,
    goal_id: str,
) -> tuple[AgentTask, ...]:
    """Deterministically enforce local/external evidence policy after model planning."""
    goal = envelope.workspace.active_goal
    source_text = "\n".join(
        part
        for part in (
            envelope.current_message,
            resolved_request,
            goal.objective if goal is not None else "",
        )
        if part
    )
    requirements = validate_source_policy(source_text, envelope.rag_mode)
    if not requirements.external_required:
        return tasks
    if envelope.rag_mode == "preferred" and requirements.local_forbidden:
        return tasks

    active = tuple(task for task in tasks if task.status != "completed")
    completed = tuple(task for task in tasks if task.status == "completed")
    if envelope.rag_mode == "required":
        local = next((task for task in active if task.capability == "local_rag"), None)
        if local is None:
            local = _evidence_task(
                tasks,
                goal_id=goal_id,
                capability="local_rag",
                objective=resolved_request,
            )
        return (*completed, local.model_copy(update={"parallel_group_id": None}))

    if envelope.rag_mode == "disabled":
        dynamic = next((task for task in active if task.capability == "dynamic_tools"), None)
        if dynamic is None:
            dynamic = _evidence_task(
                tasks,
                goal_id=goal_id,
                capability="dynamic_tools",
                objective=resolved_request,
            )
        return (
            *completed,
            dynamic.model_copy(
                update={
                    "parallel_group_id": None,
                    "allowed_tool_risks": ("network_read",),
                    "allowed_tool_names": SCHOLARLY_NETWORK_TOOL_NAMES,
                }
            ),
        )

    local = next((task for task in active if task.capability == "local_rag"), None)
    dynamic = next((task for task in active if task.capability == "dynamic_tools"), None)
    if local is None:
        local = _evidence_task(
            tasks,
            goal_id=goal_id,
            capability="local_rag",
            objective=resolved_request,
        )
    if dynamic is None:
        dynamic = _evidence_task(
            (*tasks, local),
            goal_id=goal_id,
            capability="dynamic_tools",
            objective=resolved_request,
        )
    member_ids = {local.task_id, dynamic.task_id}
    common_dependencies = tuple(
        dict.fromkeys(
            dependency
            for task in (local, dynamic)
            for dependency in task.depends_on
            if dependency not in member_ids
        )
    )
    group_id = _parallel_group_id(completed)
    local = local.model_copy(
        update={
            "parallel_group_id": group_id,
            "depends_on": common_dependencies,
            "allowed_tool_risks": (),
            "allowed_tool_names": (),
        }
    )
    dynamic = dynamic.model_copy(
        update={
            "parallel_group_id": group_id,
            "depends_on": common_dependencies,
            "allowed_tool_risks": ("network_read",),
            "allowed_tool_names": SCHOLARLY_NETWORK_TOOL_NAMES,
        }
    )
    selected = {local.task_id, dynamic.task_id}
    other_active = tuple(
        task
        for task in active
        if task.task_id not in selected and task.capability not in {"direct_chat", "local_rag", "dynamic_tools"}
    )
    return (*completed, local, dynamic, *other_active)


def _evidence_task(
    existing: tuple[AgentTask, ...],
    *,
    goal_id: str,
    capability: Literal["local_rag", "dynamic_tools"],
    objective: str,
) -> AgentTask:
    base_id = "local-research" if capability == "local_rag" else "external-research"
    task_id = _unique_task_id(base_id, existing)
    if capability == "local_rag":
        title = "检索本地论文证据"
        success = ("获得可追溯的本地论文证据或明确证据不足",)
        reason = "为最终回答提供本地论文证据"
    else:
        title = "检索外部学术信息"
        success = ("通过学术 Provider 获得外部信息或明确不可用原因",)
        reason = "为最终回答提供最新外部学术核验"
    return AgentTask(
        task_id=task_id,
        goal_id=goal_id,
        title=title,
        objective=objective,
        success_criteria=success,
        capability=capability,
        status="pending",
        execution_reason=reason,
    )


def _unique_task_id(base: str, tasks: tuple[AgentTask, ...]) -> str:
    used = {task.task_id for task in tasks}
    if base not in used:
        return base
    index = 2
    while f"{base}-{index}" in used:
        index += 1
    return f"{base}-{index}"


def _parallel_group_id(completed: tuple[AgentTask, ...]) -> str:
    used = {task.parallel_group_id for task in completed if task.parallel_group_id is not None}
    if "hybrid-research" not in used:
        return "hybrid-research"
    index = 2
    while f"hybrid-research-{index}" in used:
        index += 1
    return f"hybrid-research-{index}"


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
