"""Strict contracts for dynamic tool selection and trust-separated observations."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research_agent.agent.tooling.catalog import EXTENDED_TOOL_NAMES
from paper_research_agent.agent.tooling.contracts import TOOL_INPUT_SCHEMAS, ToolExecutionResult


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolDecision(FrozenModel):
    action: Literal["call_tool", "finish"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str = Field(min_length=1, max_length=500)
    final_summary: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_action(self) -> ToolDecision:
        if self.action == "finish":
            if self.tool_name is not None or self.arguments or self.final_summary is None:
                raise ValueError("finish decision requires only final_summary")
            return self
        if self.tool_name not in EXTENDED_TOOL_NAMES or self.final_summary is not None:
            raise ValueError("call_tool decision contains an unavailable tool")
        if "approval_token" in self.arguments:
            raise ValueError("router decisions cannot supply approval tokens")
        schema = TOOL_INPUT_SCHEMAS[self.tool_name]
        schema.model_validate(self.arguments)
        return self

    @property
    def fingerprint(self) -> str:
        payload = {"tool_name": self.tool_name, "arguments": self.arguments}
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class ToolObservation(FrozenModel):
    sequence: int = Field(ge=1)
    decision_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_name: str
    purpose: str = Field(min_length=1, max_length=500)
    result: ToolExecutionResult

    @model_validator(mode="after")
    def validate_tool(self) -> ToolObservation:
        if self.tool_name != self.result.tool_name:
            raise ValueError("tool observation name does not match its result")
        return self


class PendingApproval(FrozenModel):
    tool_name: str
    arguments: dict[str, Any]
    purpose: str
    decision_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at_epoch: float = Field(gt=0)


class ApprovalDecision(FrozenModel):
    approved: bool


class DynamicResearchResult(FrozenModel):
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    thread_id: str = Field(min_length=1, max_length=256)
    status: Literal["completed", "approval_required"]
    observations: tuple[ToolObservation, ...] = ()
    final_summary: str | None = None
    termination_reason: (
        Literal[
            "router_finished",
            "max_steps",
            "repeated_tool_call",
            "approval_denied",
            "approval_expired",
        ]
        | None
    ) = None
    pending_approval: PendingApproval | None = None

    @model_validator(mode="after")
    def validate_status(self) -> DynamicResearchResult:
        if self.status == "approval_required" and self.pending_approval is None:
            raise ValueError("approval-required result needs pending approval")
        if self.status == "completed" and self.pending_approval is not None:
            raise ValueError("completed result cannot retain pending approval")
        return self
