"""Privacy-safe structured events for the local research Agent."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AgentEventType = Literal[
    "run_started",
    "run_completed",
    "run_failed",
    "node_started",
    "node_completed",
    "node_failed",
    "tool_started",
    "tool_completed",
    "tool_failed",
    "runtime_intercepted",
    "output_rejected",
    "main_runtime_built",
    "main_run_started",
    "capability_routed",
    "child_completed",
    "main_run_paused",
    "main_commit_rejected",
    "main_run_completed",
    "deprecated_endpoint_used",
]
AgentEventStatus = Literal["started", "succeeded", "failed", "intercepted"]
AgentEventComponent = Literal["runtime", "node", "tool"]
MainCapability = Literal[
    "direct_chat",
    "local_rag",
    "dynamic_tools",
    "attachment_qa",
    "file_edit",
]
ChildStatus = Literal[
    "completed", "insufficient_evidence", "failed", "waiting_approval"
]
DeprecatedEndpoint = Literal["ask", "chat_stream", "tools_run", "tools_approval"]


class AgentEvent(BaseModel):
    """One body-free event containing only fixed fields and safe summaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["agent-event-v1"] = "agent-event-v1"
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    occurred_at: datetime
    event_type: AgentEventType
    status: AgentEventStatus
    component: AgentEventComponent
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    duration_ms: float | None = Field(default=None, ge=0)
    question_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    thread_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    step_id_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    query_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error_type: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
    reason_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    termination_reason: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )
    degraded: bool | None = None
    evidence_sufficient: bool | None = None
    hit_count: int | None = Field(default=None, ge=0)
    requested_count: int | None = Field(default=None, ge=0)
    returned_count: int | None = Field(default=None, ge=0)
    evidence_count: int | None = Field(default=None, ge=0)
    tool_call_count: int | None = Field(default=None, ge=0)
    replan_count: int | None = Field(default=None, ge=0)
    max_steps: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    capability: MainCapability | None = None
    child_status: ChildStatus | None = None
    endpoint: DeprecatedEndpoint | None = None
    workspace_version: int | None = Field(default=None, ge=0)
    validation_error_count: int | None = Field(default=None, ge=0)

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("agent event timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_status(self) -> AgentEvent:
        expected: AgentEventStatus
        if self.event_type.endswith("_started"):
            expected = "started"
        elif self.event_type == "runtime_intercepted":
            expected = "intercepted"
        elif self.event_type.endswith("_completed") or self.event_type in {
            "main_runtime_built",
            "main_run_paused",
            "deprecated_endpoint_used",
            "capability_routed",
        }:
            expected = "succeeded"
        else:
            expected = "failed"
        if self.status != expected:
            raise ValueError("agent event status does not match its type")
        if expected == "started" and self.duration_ms is not None:
            raise ValueError("started events cannot contain a duration")
        if expected in {"failed", "intercepted"} and self.reason_code is None:
            raise ValueError("failed and intercepted events require a reason code")
        if (
            self.event_type in {"capability_routed", "child_completed"}
            and self.capability is None
        ):
            raise ValueError("capability event requires a fixed capability")
        if self.event_type == "child_completed" and self.child_status is None:
            raise ValueError("child completion requires a fixed child status")
        if self.event_type == "deprecated_endpoint_used" and self.endpoint is None:
            raise ValueError("deprecated endpoint event requires an endpoint")
        return self


class AgentEventSink(Protocol):
    def write(self, event: AgentEvent) -> bool: ...


def safe_fingerprint(value: str) -> str:
    """Return a stable SHA-256 over normalized text without retaining the text."""
    normalized = value.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def emit_agent_event(sink: AgentEventSink | None, event: AgentEvent) -> bool:
    """Write one event without allowing telemetry failure to break research."""
    if sink is None:
        return False
    try:
        return bool(sink.write(event))
    except (OSError, RuntimeError, sqlite3.Error):
        return False


class SQLiteAgentEventLogger:
    """Concurrency-safe local event storage with no raw prompt or evidence columns."""

    _COLUMNS = (
        "schema_version",
        "run_id",
        "occurred_at",
        "event_type",
        "status",
        "component",
        "name",
        "duration_ms",
        "question_sha256",
        "thread_sha256",
        "step_id_sha256",
        "query_sha256",
        "error_type",
        "reason_code",
        "termination_reason",
        "degraded",
        "evidence_sufficient",
        "hit_count",
        "requested_count",
        "returned_count",
        "evidence_count",
        "tool_call_count",
        "replan_count",
        "max_steps",
        "max_tool_calls",
        "timeout_seconds",
        "capability",
        "child_status",
        "endpoint",
        "workspace_version",
        "validation_error_count",
    )

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def write(self, event: AgentEvent) -> bool:
        payload = event.model_dump(mode="python")
        payload["occurred_at"] = event.occurred_at.astimezone(UTC).isoformat()
        payload["degraded"] = _sqlite_bool(event.degraded)
        payload["evidence_sufficient"] = _sqlite_bool(event.evidence_sufficient)
        placeholders = ",".join("?" for _ in self._COLUMNS)
        columns = ",".join(self._COLUMNS)
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                connection.execute(
                    f"INSERT INTO agent_events ({columns}) VALUES ({placeholders})",
                    tuple(payload[column] for column in self._COLUMNS),
                )
        except (OSError, sqlite3.Error):
            return False
        return True

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    schema_version TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    component TEXT NOT NULL,
                    name TEXT NOT NULL,
                    duration_ms REAL,
                    question_sha256 TEXT,
                    thread_sha256 TEXT,
                    step_id_sha256 TEXT,
                    query_sha256 TEXT,
                    error_type TEXT,
                    reason_code TEXT,
                    termination_reason TEXT,
                    degraded INTEGER,
                    evidence_sufficient INTEGER,
                    hit_count INTEGER,
                    requested_count INTEGER,
                    returned_count INTEGER,
                    evidence_count INTEGER,
                    tool_call_count INTEGER,
                    replan_count INTEGER,
                    max_steps INTEGER,
                    max_tool_calls INTEGER,
                    timeout_seconds REAL,
                    capability TEXT,
                    child_status TEXT,
                    endpoint TEXT,
                    workspace_version INTEGER,
                    validation_error_count INTEGER
                );
                CREATE INDEX IF NOT EXISTS agent_events_run
                    ON agent_events(run_id, event_id);
                CREATE INDEX IF NOT EXISTS agent_events_type_time
                    ON agent_events(event_type, occurred_at DESC);
                """
            )
            existing = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(agent_events)")
            }
            additions = {
                "capability": "TEXT",
                "child_status": "TEXT",
                "endpoint": "TEXT",
                "workspace_version": "INTEGER",
                "validation_error_count": "INTEGER",
            }
            for column, sql_type in additions.items():
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE agent_events ADD COLUMN {column} {sql_type}"
                    )
            connection.execute("PRAGMA user_version = 2")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection


def _sqlite_bool(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(value)
