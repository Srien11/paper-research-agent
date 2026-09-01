"""LangGraph state container for the main Agent graph."""

from __future__ import annotations

from typing import Literal, TypedDict

from paper_research_agent.agent.orchestrator.evaluator import TaskEvaluation
from paper_research_agent.agent.orchestrator.models import (
    AgentContextEnvelope,
    AgentRunStart,
    Capability,
    ChildTaskResult,
    CommitOutcome,
    ConversationWorkspace,
    DegradationCode,
    GoalDecision,
    MainAgentRequest,
    TaskPlanDecision,
    TurnInterpretationV2,
)
from paper_research_agent.agent.orchestrator.parallel import TaskBatch


class MainAgentGraphState(TypedDict, total=False):
    """Temporary graph container; every node boundary uses strict Pydantic models."""

    run_id: str
    run_start: AgentRunStart
    turn_id: str
    request: MainAgentRequest
    base_workspace_version: int
    context: AgentContextEnvelope
    planning_route: Literal["fast_path", "full_planner"]
    planning_route_reason: str
    interpretation: TurnInterpretationV2
    goal_decision: GoalDecision
    plan_decision: TaskPlanDecision
    workspace_draft: ConversationWorkspace
    active_task_id: str
    active_batch: TaskBatch
    batch_routes: dict[str, Capability]
    batch_results: tuple[ChildTaskResult, ...]
    batch_evaluations: tuple[TaskEvaluation, ...]
    batch_elapsed_seconds: float
    route: Capability
    child_results: list[ChildTaskResult]
    child_result: ChildTaskResult
    evaluation: TaskEvaluation
    direct_answer: str
    final_answer: str
    degraded: bool
    degradation_codes: tuple[DegradationCode, ...]
    pending_approval: dict[str, object]
    remaining_child_calls: int
    remaining_replans: int
    termination_reason: str
    next_action: str
    validation_errors: tuple[str, ...]
    commit_outcome: CommitOutcome
    route_trace: list[str]
    resuming: bool
