"""Cross-turn goal reconciliation and session-level task planning for the main Agent."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from paper_research_agent.agent.orchestrator.models import (
    AcceptanceCriterion,
    AgentContextEnvelope,
    ConversationWorkspace,
    FrozenModel,
    GoalDecision,
    GoalState,
    TurnInterpretationV2,
)
from paper_research_agent.agent.orchestrator.prompts import (
    GOAL_RECONCILER_PROMPT_VERSION,
    GOAL_RECONCILER_SYSTEM,
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
