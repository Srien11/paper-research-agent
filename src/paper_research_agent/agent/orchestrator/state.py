"""LangGraph state container for the main Agent graph."""

from __future__ import annotations

from typing import TypedDict

from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    ChildTaskResult,
    ConversationWorkspace,
    GoalDecision,
    MainAgentRequest,
    TaskPlanDecision,
    TurnInterpretationV2,
)


class MainAgentGraphState(TypedDict, total=False):
    """Temporary graph container; every node boundary uses strict Pydantic models."""

    run_id: str
    request: MainAgentRequest
    base_workspace_version: int
    context: AgentContextEnvelope
    interpretation: TurnInterpretationV2
    goal_decision: GoalDecision
    plan_decision: TaskPlanDecision
    workspace_draft: ConversationWorkspace
    active_task_id: str
    route: str
    child_results: list[ChildTaskResult]
    direct_answer: str
    final_answer: str
    pending_approval: dict[str, object]
    remaining_child_calls: int
    remaining_replans: int
    termination_reason: str
    next_action: str
