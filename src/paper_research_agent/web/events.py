"""Safe, versioned NDJSON events for the unified main-Agent Web API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from paper_research_agent.agent.orchestrator.models import (
    Capability,
    ChildTaskResult,
    MainAgentResult,
)
from paper_research_agent.web.models import (
    SafeEvidenceSource,
    SafePendingToolApproval,
    WebModel,
)

LegacyAgentStreamEventType = Literal[
    "run_started",
    "run_reused",
    "context_ready",
    "goal_updated",
    "plan_updated",
    "task_started",
    "route_selected",
    "rag_result",
    "tool_result",
    "attachment_result",
    "file_result",
    "task_completed",
    "approval_required",
    "delta",
    "error",
    "done",
]
AgentStreamEventType = Literal[
    "run_started",
    "run_reused",
    "run_resumed",
    "run_paused",
    "run_waiting_user",
    "run_waiting_approval",
    "run_completed",
    "run_failed",
    "run_cancelled",
    "run_conflict",
    "reasoning_started",
    "reasoning_summary",
    "reasoning_completed",
    "goal_updated",
    "plan_updated",
    "task_started",
    "task_completed",
    "task_failed",
    "tool_started",
    "tool_completed",
    "tool_failed",
    "retrieval_completed",
    "file_created",
    "answer_started",
    "answer_delta",
    "answer_completed",
    "interaction_required",
    "interaction_resolved",
]
AgentPublicStatus = Literal[
    "running",
    "paused",
    "waiting_user",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
    "conflict",
]
RunNodeStatus = Literal[
    "pending",
    "running",
    "paused",
    "waiting_user",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
    "conflict",
]
DeliveryMode = Literal["provider_live", "validated_replay", "event_only"]


class SafeRunEventDetail(WebModel):
    """Strict public projection; arbitrary provider and tool payloads are forbidden."""

    tool_name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.:-]{1,128}$")
    purpose: str | None = Field(default=None, max_length=500)
    arguments_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expires_at_epoch: float | None = Field(default=None, gt=0)
    capability: Capability | None = None
    delivery_mode: DeliveryMode | None = None
    route: str | None = Field(default=None, max_length=64)
    reason_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{1,63}$",
    )
    goal_action: str | None = Field(default=None, max_length=32)
    plan_action: str | None = Field(default=None, max_length=32)
    query_kind: str | None = Field(default=None, max_length=64)
    source_count: int | None = Field(default=None, ge=0, le=100_000)
    returned_count: int | None = Field(default=None, ge=0, le=100_000)
    attachment_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    output_attachment_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    degraded: bool | None = None
    call_count: int | None = Field(default=None, ge=0, le=100_000)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    first_token_ms: int | None = Field(default=None, ge=0)
    browser_first_answer_byte_ms: int | None = Field(default=None, ge=0)
    citations: tuple[SafeEvidenceSource, ...] = Field(default=(), max_length=100)


class AgentStreamEventDraft(WebModel):
    """Validated event payload before the durable store assigns its event ID."""

    schema_version: Literal["main-agent-stream-v2"] = "main-agent-stream-v2"
    type: AgentStreamEventType
    occurred_at: datetime
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{16,128}$")
    run_id: str = Field(min_length=1, max_length=256)
    turn_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    node_id: str = Field(min_length=1, max_length=256)
    parent_node_id: str | None = Field(default=None, max_length=256)
    task_id: str | None = Field(default=None, max_length=64)
    status: RunNodeStatus | None = None
    title: str | None = Field(default=None, max_length=200)
    summary: str | None = Field(default=None, max_length=1_000)
    duration_ms: int | None = Field(default=None, ge=0)
    detail: SafeRunEventDetail = Field(default_factory=SafeRunEventDetail)
    delta: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def validate_payload(self) -> AgentStreamEventDraft:
        if self.type == "answer_delta" and self.delta is None:
            raise ValueError("answer_delta event requires delta")
        if self.type != "answer_delta" and self.delta is not None:
            raise ValueError("only answer_delta events may contain delta")
        required_status = {
            "run_paused": "paused",
            "run_waiting_user": "waiting_user",
            "run_waiting_approval": "waiting_approval",
            "run_completed": "completed",
            "run_failed": "failed",
            "run_cancelled": "cancelled",
            "run_conflict": "conflict",
        }.get(self.type)
        if required_status is not None and self.status != required_status:
            raise ValueError(f"{self.type} requires status={required_status}")
        return self

    @property
    def is_terminal(self) -> bool:
        return self.type in {
            "run_completed",
            "run_failed",
            "run_cancelled",
            "run_conflict",
        }

    @property
    def closes_delivery_segment(self) -> bool:
        return self.is_terminal or self.type in {
            "run_paused",
            "run_waiting_user",
            "run_waiting_approval",
        }

    def with_event_id(self, event_id: int) -> AgentStreamEvent:
        return AgentStreamEvent(
            event_id=event_id,
            **self.model_dump(),
        )


class AgentStreamEvent(AgentStreamEventDraft):
    """Versioned, durable product event used by live and historical transcripts."""

    event_id: int = Field(ge=1)

    def to_ndjson(self) -> bytes:
        return (self.model_dump_json(exclude_none=True) + "\n").encode("utf-8")


class LegacyAgentStreamEvent(WebModel):
    schema_version: Literal["main-agent-stream-v1"] = "main-agent-stream-v1"
    event_id: int = Field(ge=1)
    type: LegacyAgentStreamEventType
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{16,128}$")
    run_id: str = Field(min_length=1, max_length=256)
    status: AgentPublicStatus | None = None
    workspace_version: int | None = Field(default=None, ge=0)
    text: str | None = Field(default=None, max_length=20_000)
    reason_code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{1,63}$",
    )
    counts: dict[str, int] = Field(default_factory=dict, max_length=20)
    task_id: str | None = Field(default=None, max_length=64)
    capability: Capability | None = None
    source_ids: tuple[str, ...] = Field(default=(), max_length=100)
    output_attachment_ids: tuple[str, ...] = Field(default=(), max_length=5)
    tool_names: tuple[str, ...] = Field(default=(), max_length=20)
    pending_approval: SafePendingToolApproval | None = None

    @field_validator("counts")
    @classmethod
    def validate_counts(cls, values: dict[str, int]) -> dict[str, int]:
        if any(not key or len(key) > 64 for key in values):
            raise ValueError("event count names must contain between 1 and 64 characters")
        if any(value < 0 for value in values.values()):
            raise ValueError("event counts must be non-negative")
        return values

    @model_validator(mode="after")
    def validate_payload(self) -> LegacyAgentStreamEvent:
        if self.type == "delta" and self.text is None:
            raise ValueError("delta event requires text")
        if self.type != "delta" and self.text is not None:
            raise ValueError("only delta events may contain text")
        if self.type == "done":
            if self.status is None or self.workspace_version is None:
                raise ValueError("done event requires status and workspace version")
        elif self.status is not None or self.workspace_version is not None:
            raise ValueError("status and workspace version are reserved for done events")
        task_events = {
            "task_started",
            "route_selected",
            "rag_result",
            "tool_result",
            "attachment_result",
            "file_result",
            "task_completed",
        }
        if self.type in task_events and (self.task_id is None or self.capability is None):
            raise ValueError("task events require task_id and capability")
        if self.type not in task_events and (
            self.task_id is not None or self.capability is not None
        ):
            raise ValueError("task fields are reserved for task events")
        if self.output_attachment_ids and self.type != "file_result":
            raise ValueError("output attachments are reserved for file_result")
        if self.tool_names and self.type != "tool_result":
            raise ValueError("tool names are reserved for tool_result")
        if self.type == "approval_required" and self.pending_approval is None:
            raise ValueError("approval_required requires a safe approval projection")
        if self.type != "approval_required" and self.pending_approval is not None:
            raise ValueError("pending approval is reserved for approval_required")
        return self

    def to_ndjson(self) -> bytes:
        return (self.model_dump_json(exclude_none=True) + "\n").encode("utf-8")


class AgentEventProjector:
    """Allocate monotonic event IDs and close a stream exactly once."""

    def __init__(self, *, request_id: str, run_id: str) -> None:
        self._request_id = request_id
        self._run_id = run_id
        self._next_event_id = 1
        self._done = False

    def event(
        self,
        event_type: LegacyAgentStreamEventType,
        *,
        text: str | None = None,
        reason_code: str | None = None,
        counts: dict[str, int] | None = None,
        task_id: str | None = None,
        capability: Capability | None = None,
        source_ids: tuple[str, ...] = (),
        output_attachment_ids: tuple[str, ...] = (),
        tool_names: tuple[str, ...] = (),
        pending_approval: SafePendingToolApproval | None = None,
    ) -> LegacyAgentStreamEvent:
        if self._done:
            raise RuntimeError("event stream is already complete")
        if event_type == "done":
            raise ValueError("use done() to emit the terminal event")
        event = LegacyAgentStreamEvent(
            event_id=self._next_event_id,
            type=event_type,
            request_id=self._request_id,
            run_id=self._run_id,
            text=text,
            reason_code=reason_code,
            counts=counts or {},
            task_id=task_id,
            capability=capability,
            source_ids=source_ids,
            output_attachment_ids=output_attachment_ids,
            tool_names=tool_names,
            pending_approval=pending_approval,
        )
        self._next_event_id += 1
        return event

    def done(
        self,
        *,
        status: AgentPublicStatus,
        workspace_version: int,
    ) -> LegacyAgentStreamEvent:
        if self._done:
            raise RuntimeError("done event already emitted")
        event = LegacyAgentStreamEvent(
            event_id=self._next_event_id,
            type="done",
            request_id=self._request_id,
            run_id=self._run_id,
            status=status,
            workspace_version=workspace_version,
        )
        self._next_event_id += 1
        self._done = True
        return event

    def project_result(
        self,
        result: MainAgentResult,
        *,
        reused: bool = False,
    ) -> tuple[LegacyAgentStreamEvent, ...]:
        """Project a runtime result without exposing graph state, prompts, or summaries."""
        events = [self.event("run_reused" if reused else "run_started")]
        events.append(self.event("context_ready"))
        for child in result.child_results:
            events.extend(self._project_child(child))
        pending = _safe_pending_approval(result.pending_approval)
        if pending is not None:
            events.append(self.event("approval_required", pending_approval=pending))
        if result.answer:
            events.append(self.event("delta", text=result.answer))
        if result.status == "failed":
            events.append(self.event("error", reason_code="run_failed"))
        events.append(
            self.done(
                status=result.status,
                workspace_version=result.workspace_version,
            )
        )
        return tuple(events)

    def _project_child(
        self, child: ChildTaskResult
    ) -> list[LegacyAgentStreamEvent]:
        events = [
            self.event(
                "task_started", task_id=child.task_id, capability=child.capability
            ),
            self.event(
                "route_selected", task_id=child.task_id, capability=child.capability
            ),
        ]
        event_types: dict[Capability, LegacyAgentStreamEventType] = {
            "local_rag": "rag_result",
            "dynamic_tools": "tool_result",
            "attachment_qa": "attachment_result",
            "file_edit": "file_result",
            "direct_chat": "task_completed",
        }
        event_type = event_types[child.capability]
        artifact = child.artifact
        artifact_metrics = getattr(artifact, "metrics", None)
        metric_counts = (
            {
                key: int(value)
                for key, value in artifact_metrics.model_dump().items()
                if isinstance(value, int) and value >= 0
            }
            if artifact_metrics is not None
            else {}
        )
        output_ids = tuple(getattr(artifact, "output_attachment_ids", ()))
        tool_names = tuple(getattr(artifact, "tool_names", ()))
        source_ids = tuple(child.source_ids)
        if event_type != "task_completed":
            events.append(
                self.event(
                    event_type,
                    source_ids=source_ids,
                    output_attachment_ids=output_ids,
                    tool_names=tool_names,
                    counts={"source_count": len(source_ids), **metric_counts},
                    task_id=child.task_id,
                    capability=child.capability,
                )
            )
        if child.status != "waiting_approval":
            events.append(
                self.event(
                    "task_completed",
                    counts={"source_count": len(source_ids), **metric_counts},
                    task_id=child.task_id,
                    capability=child.capability,
                )
            )
        return events


def _safe_pending_approval(
    payload: dict[str, object] | None,
) -> SafePendingToolApproval | None:
    if payload is None:
        return None
    return SafePendingToolApproval.model_validate(
        {
            "tool_name": payload.get("tool_name"),
            "purpose": payload.get("purpose"),
            "arguments_sha256": payload.get("arguments_sha256"),
            "expires_at_epoch": payload.get("expires_at_epoch"),
        }
    )
