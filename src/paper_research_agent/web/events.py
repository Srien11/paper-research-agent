"""Safe, versioned NDJSON events for the unified main-Agent Web API."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from paper_research_agent.web.models import WebModel

AgentStreamEventType = Literal[
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
AgentPublicStatus = Literal[
    "running",
    "waiting_user",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
    "conflict",
]


class AgentStreamEvent(WebModel):
    schema_version: Literal["main-agent-stream-v1"] = "main-agent-stream-v1"
    event_id: int = Field(ge=1)
    type: AgentStreamEventType
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

    @field_validator("counts")
    @classmethod
    def validate_counts(cls, values: dict[str, int]) -> dict[str, int]:
        if any(not key or len(key) > 64 for key in values):
            raise ValueError("event count names must contain between 1 and 64 characters")
        if any(value < 0 for value in values.values()):
            raise ValueError("event counts must be non-negative")
        return values

    @model_validator(mode="after")
    def validate_payload(self) -> AgentStreamEvent:
        if self.type == "delta" and self.text is None:
            raise ValueError("delta event requires text")
        if self.type != "delta" and self.text is not None:
            raise ValueError("only delta events may contain text")
        if self.type == "done":
            if self.status is None or self.workspace_version is None:
                raise ValueError("done event requires status and workspace version")
        elif self.status is not None or self.workspace_version is not None:
            raise ValueError("status and workspace version are reserved for done events")
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
        event_type: AgentStreamEventType,
        *,
        text: str | None = None,
        reason_code: str | None = None,
        counts: dict[str, int] | None = None,
    ) -> AgentStreamEvent:
        if self._done:
            raise RuntimeError("event stream is already complete")
        if event_type == "done":
            raise ValueError("use done() to emit the terminal event")
        event = AgentStreamEvent(
            event_id=self._next_event_id,
            type=event_type,
            request_id=self._request_id,
            run_id=self._run_id,
            text=text,
            reason_code=reason_code,
            counts=counts or {},
        )
        self._next_event_id += 1
        return event

    def done(
        self,
        *,
        status: AgentPublicStatus,
        workspace_version: int,
    ) -> AgentStreamEvent:
        if self._done:
            raise RuntimeError("done event already emitted")
        event = AgentStreamEvent(
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

