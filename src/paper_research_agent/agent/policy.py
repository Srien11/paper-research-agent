"""Fail-closed runtime limits for the read-only research workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ResearchToolName = Literal["search_corpus", "get_evidence"]
READ_ONLY_RESEARCH_TOOLS: frozenset[ResearchToolName] = frozenset(
    {"search_corpus", "get_evidence"}
)


class ResearchRuntimePolicy(BaseModel):
    """One immutable capability and resource budget for an Agent invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_tools: frozenset[ResearchToolName] = READ_ONLY_RESEARCH_TOOLS
    # This bounds the initial plan only. Evidence-driven replans may continue
    # until the tool, timeout, repetition, or stagnation guards stop the run.
    max_steps: int = Field(default=4, ge=1, le=6)
    evidence_per_step: int = Field(default=4, ge=1, le=20)
    max_tool_calls: int = Field(default=12, ge=1, le=12)
    timeout_seconds: float = Field(default=90, gt=0, le=300)

    @field_validator("allowed_tools")
    @classmethod
    def require_non_empty_allowlist(
        cls,
        value: frozenset[ResearchToolName],
    ) -> frozenset[ResearchToolName]:
        if not value:
            raise ValueError("research tool allowlist cannot be empty")
        return value

    def consume(self, tool_name: ResearchToolName, current_calls: int) -> int:
        """Authorize one exact tool call and return the updated invocation count."""
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"research tool is not allowed: {tool_name}")
        next_calls = current_calls + 1
        if next_calls > self.max_tool_calls:
            raise RuntimeError("research tool call budget exceeded")
        return next_calls
