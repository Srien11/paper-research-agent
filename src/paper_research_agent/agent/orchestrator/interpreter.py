"""Turn-relation interpreter for the main Agent."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    TurnInterpretationV2,
)
from paper_research_agent.agent.orchestrator.prompts import (
    TURN_INTERPRETER_PROMPT_VERSION,
    TURN_INTERPRETER_SYSTEM,
)

_GOAL_CHANGING_RELATIONS = frozenset({"new_goal", "refine_goal", "cancel_goal"})


class TurnInterpreter:
    """Maps the current message to its relation with the active goal only."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        version: str = TURN_INTERPRETER_PROMPT_VERSION,
    ) -> None:
        self._model = model.with_structured_output(
            TurnInterpretationV2, method="function_calling"
        )
        self.version = version

    async def interpret(self, envelope: AgentContextEnvelope) -> TurnInterpretationV2:
        allowed_ids = _allowed_context_ids(envelope)
        system = SystemMessage(
            content=f"{TURN_INTERPRETER_SYSTEM}\nPROMPT_VERSION={self.version}"
        )
        user = HumanMessage(content=_interpreter_user_content(envelope, allowed_ids))
        for attempt in range(2):
            try:
                raw = await self._model.ainvoke([system, user])
                if isinstance(raw, TurnInterpretationV2):
                    return self._validate(raw, allowed_ids)
                interpretation = TurnInterpretationV2.model_validate(raw)
                return self._validate(interpretation, allowed_ids)
            except Exception:  # noqa: BLE001 - one retry, then deterministic fallback
                if attempt == 1:
                    return self._fallback(envelope)
        raise AssertionError("unreachable")

    def _validate(
        self, interpretation: TurnInterpretationV2, allowed_ids: frozenset[str]
    ) -> TurnInterpretationV2:
        unknown = tuple(
            context_id
            for context_id in interpretation.selected_context_ids
            if context_id not in allowed_ids
        )
        if unknown:
            raise ValueError("interpretation selected an unknown context ID")
        if (
            interpretation.relation in _GOAL_CHANGING_RELATIONS
            and interpretation.confidence < 0.55
            and not interpretation.needs_clarification
        ):
            return interpretation.model_copy(
                update={
                    "needs_clarification": True,
                    "clarification_question": "你希望我按这个新的方向继续吗？请确认后再继续。",
                }
            )
        return interpretation

    def _fallback(self, envelope: AgentContextEnvelope) -> TurnInterpretationV2:
        relation: Literal["continue_goal", "new_goal"] = (
            "continue_goal" if envelope.workspace.active_goal is not None else "new_goal"
        )
        return TurnInterpretationV2(
            relation=relation,
            resolved_request=envelope.current_message,
            confidence=0.0,
        )


def _allowed_context_ids(envelope: AgentContextEnvelope) -> frozenset[str]:
    ids = {message.turn_id for message in envelope.recent_messages}
    ids.update(item.source_id for item in envelope.recalled_context)
    return frozenset(ids)


def _interpreter_user_content(
    envelope: AgentContextEnvelope, allowed_ids: frozenset[str]
) -> str:
    goal = envelope.workspace.active_goal
    goal_text = f"{goal.status}: {goal.objective}" if goal is not None else "（无活动目标）"
    task_plan = envelope.workspace.task_plan
    tasks_text = (
        "; ".join(f"{task.status}: {task.title}" for task in task_plan.tasks)
        if task_plan is not None
        else "（无任务计划）"
    )
    return (
        f"CURRENT_MESSAGE\n{envelope.current_message}\n\n"
        f"ACTIVE_GOAL\n{goal_text}\n\n"
        f"TASK_PLAN\n{tasks_text}\n\n"
        f"UNRESOLVED_QUESTIONS\n{envelope.workspace.unresolved_questions}\n\n"
        f"ALLOWED_CONTEXT_IDS\n{sorted(allowed_ids)}\n"
    )
