"""Strict domain contracts for the main Agent orchestrator layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.agent.orchestrator.artifacts import ChildArtifact

GoalStatus = Literal["active", "satisfied", "blocked", "abandoned"]
TaskStatus = Literal[
    "pending",
    "ready",
    "running",
    "waiting_approval",
    "waiting_user",
    "completed",
    "failed",
    "skipped",
    "cancelled",
]
Capability = Literal[
    "direct_chat",
    "local_rag",
    "dynamic_tools",
    "attachment_qa",
    "file_edit",
]
RequestRelation = Literal[
    "new_goal",
    "continue_goal",
    "refine_goal",
    "answer_within_goal",
    "cancel_goal",
    "resume_after_approval",
    "meta_conversation",
]
RunStatus = Literal[
    "running",
    "waiting_user",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
    "conflict",
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcceptanceCriterion(FrozenModel):
    criterion_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str = Field(min_length=1, max_length=500)
    satisfied: bool = False

    @field_validator("criterion_id", "description")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("acceptance criterion text must not be blank")
        return normalized


class GoalState(FrozenModel):
    goal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    objective: str = Field(min_length=1, max_length=2000)
    status: GoalStatus = "active"
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = Field(default=(), max_length=12)
    constraints: tuple[str, ...] = Field(default=(), max_length=20)
    origin_turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: datetime
    updated_at: datetime

    @field_validator("objective")
    @classmethod
    def normalize_objective(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("goal objective must not be blank")
        return normalized

    @field_validator("constraints")
    @classmethod
    def validate_constraints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("goal constraints must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("goal constraints must be unique")
        return normalized


class AgentTask(FrozenModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    goal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    title: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=8)
    capability: Capability
    status: TaskStatus = "pending"
    depends_on: tuple[str, ...] = Field(default=(), max_length=8)
    attempt_count: int = Field(default=0, ge=0, le=5)
    result_ref: str | None = Field(default=None, max_length=128)
    blocked_reason: str | None = Field(default=None, max_length=500)

    @field_validator("title", "objective")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("task text must not be blank")
        return normalized

    @field_validator("success_criteria")
    @classmethod
    def validate_success_criteria(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("task success criteria must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("task success criteria must be unique")
        return normalized

    @field_validator("depends_on")
    @classmethod
    def validate_dependencies(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("task dependency IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("task dependency IDs must be unique")
        return normalized


class TaskPlan(FrozenModel):
    plan_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    goal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    revision: int = Field(default=1, ge=1)
    tasks: tuple[AgentTask, ...] = Field(default=(), max_length=12)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_plan(self) -> TaskPlan:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task plan task IDs must be unique")
        if any(task.goal_id != self.goal_id for task in self.tasks):
            raise ValueError("task plan goal ID must match every task goal ID")
        known = set(task_ids)
        for task in self.tasks:
            if any(dependency == task.task_id for dependency in task.depends_on):
                raise ValueError("task cannot depend on itself")
            if any(dependency not in known for dependency in task.depends_on):
                raise ValueError("task depends on an unknown task ID")
        children: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
        indegree = {task_id: 0 for task_id in task_ids}
        for task in self.tasks:
            for dependency in task.depends_on:
                children[dependency].append(task.task_id)
                indegree[task.task_id] += 1
        ready = [task_id for task_id in task_ids if indegree[task_id] == 0]
        visited = 0
        while ready:
            current = ready.pop()
            visited += 1
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(task_ids):
            raise ValueError("task plan dependencies must not contain a cycle")
        return self


class ConversationWorkspace(FrozenModel):
    schema_version: Literal["conversation-workspace-v1"] = "conversation-workspace-v1"
    conversation_id: str = Field(min_length=1, max_length=256)
    version: int = Field(default=0, ge=0)
    summary: str = Field(default="", max_length=3000)
    active_goal: GoalState | None = None
    task_plan: TaskPlan | None = None
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=10)
    stable_constraints: tuple[str, ...] = Field(default=(), max_length=20)
    updated_at: datetime


class ContextMessage(FrozenModel):
    turn_id: str
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)
    trust: Literal["non_evidence"] = "non_evidence"

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("context message content must not be blank")
        return normalized


class RecalledContext(FrozenModel):
    source_id: str
    kind: Literal["conversation_turn", "episode", "long_term_memory"]
    content: str = Field(min_length=1, max_length=3000)
    relevance: float = Field(ge=0, le=1)
    trust: Literal["non_evidence", "research_context"]

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("recalled context content must not be blank")
        return normalized


class AgentContextEnvelope(FrozenModel):
    conversation_id: str
    request_id: str
    turn_id: str
    current_message: str
    rag_mode: Literal["disabled", "preferred", "required"]
    attachment_ids: tuple[str, ...] = ()
    workspace: ConversationWorkspace
    recent_messages: tuple[ContextMessage, ...] = ()
    recalled_context: tuple[RecalledContext, ...] = ()
    prepared_at: datetime


class TurnInterpretationV2(FrozenModel):
    relation: RequestRelation
    resolved_request: str = Field(min_length=1, max_length=2000)
    selected_context_ids: tuple[str, ...] = Field(default=(), max_length=10)
    goal_change_summary: str = Field(default="", max_length=1000)
    new_constraints: tuple[str, ...] = Field(default=(), max_length=10)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=500)
    confidence: float = Field(ge=0, le=1)

    @field_validator("resolved_request")
    @classmethod
    def normalize_resolved_request(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("resolved request must not be blank")
        return normalized

    @field_validator("goal_change_summary")
    @classmethod
    def normalize_optional_summary(cls, value: str) -> str:
        return value.strip()

    @field_validator("selected_context_ids", "new_constraints")
    @classmethod
    def validate_reference_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("interpretation reference IDs must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("interpretation reference IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_clarification(self) -> TurnInterpretationV2:
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("clarification question is required")
        if not self.needs_clarification and self.clarification_question is not None:
            raise ValueError("clarification question must be null when clarification is not needed")
        return self


class GoalDecision(FrozenModel):
    action: Literal["keep", "create", "revise", "satisfy", "block", "abandon"]
    goal: GoalState | None
    rationale: str = Field(min_length=1, max_length=500)

    @field_validator("rationale")
    @classmethod
    def normalize_rationale(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("goal decision rationale must not be blank")
        return normalized


class TaskPlanDecision(FrozenModel):
    action: Literal["keep", "create", "revise", "clear"]
    plan: TaskPlan | None
    rationale: str = Field(min_length=1, max_length=500)

    @field_validator("rationale")
    @classmethod
    def normalize_rationale(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("plan decision rationale must not be blank")
        return normalized


class ChildTaskRequest(FrozenModel):
    run_id: str
    conversation_id: str
    goal_id: str
    task_id: str
    objective: str = Field(min_length=1, max_length=1000)
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=8)
    capability: Capability
    current_message: str = Field(min_length=1, max_length=10_000)
    conversation_summary: str = Field(default="", max_length=3000)
    constraints: tuple[str, ...] = Field(default=(), max_length=20)
    selected_context: tuple[RecalledContext, ...] = Field(default=(), max_length=10)
    rag_mode: Literal["disabled", "preferred", "required"]
    attachment_ids: tuple[str, ...] = Field(default=(), max_length=20)


class ChildTaskResult(FrozenModel):
    child_run_id: str
    task_id: str
    capability: Capability
    status: Literal[
        "completed",
        "insufficient_evidence",
        "waiting_approval",
        "failed",
    ]
    summary: str = Field(default="", max_length=5000)
    source_ids: tuple[str, ...] = Field(default=(), max_length=100)
    citation_kind: Literal["none", "local_paper", "external"] = "none"
    pending_approval: dict[str, object] | None = None
    error_code: str | None = None
    artifact: ChildArtifact | None = None

    @model_validator(mode="after")
    def validate_result_consistency(self) -> ChildTaskResult:
        if self.status == "waiting_approval" and self.pending_approval is None:
            raise ValueError("waiting approval result requires a pending approval payload")
        if self.status != "waiting_approval" and self.pending_approval is not None:
            raise ValueError("pending approval payload requires waiting approval status")
        if self.status == "failed" and not self.error_code:
            raise ValueError("failed result requires an error code")
        return self


class MainAgentRequest(FrozenModel):
    request_id: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=10_000)
    rag_mode: Literal["disabled", "preferred", "required"]
    attachment_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("main agent request message must not be blank")
        return normalized


class MainAgentResult(FrozenModel):
    run_id: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(min_length=1, max_length=256)
    status: RunStatus
    answer: str = Field(default="", max_length=20_000)
    route_trace: tuple[str, ...] = Field(default=(), max_length=64)
    child_results: tuple[ChildTaskResult, ...] = Field(default=(), max_length=12)
    pending_approval: dict[str, object] | None = None
    workspace_version: int = Field(default=0, ge=0)

    @field_validator("route_trace")
    @classmethod
    def validate_route_trace(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("route trace entries must not be blank")
        return normalized


class AgentRunStart(FrozenModel):
    """Idempotent start of one main Agent run returned by begin_agent_run."""

    run_id: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=256)
    conversation_id: str = Field(min_length=1, max_length=256)
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    workspace: ConversationWorkspace
    outcome: Literal["created", "running_reused", "completed_cached"]
    result: MainAgentResult | None = None


class CommitOutcome(FrozenModel):
    """Result of one atomic workspace + turn + run commit."""

    committed: bool
    reason: Literal[
        "committed",
        "already_completed",
        "version_conflict",
        "run_not_found",
        "turn_conflict",
    ] = "committed"
    workspace_version: int = Field(default=0, ge=0)
