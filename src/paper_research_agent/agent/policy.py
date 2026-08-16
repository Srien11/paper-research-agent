"""Fail-closed runtime limits for the read-only research workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ResearchToolName = Literal["search_corpus", "get_evidence"]
MAX_INITIAL_PLAN_STEPS = 20
ABSOLUTE_MAX_RESEARCH_STEPS = 24
ABSOLUTE_MAX_TOOL_CALLS = 48
MAX_FOLLOWUP_STEPS = 4
READ_ONLY_RESEARCH_TOOLS: frozenset[ResearchToolName] = frozenset(
    {"search_corpus", "get_evidence"}
)


class ResearchRuntimePolicy(BaseModel):
    """One immutable capability and resource budget for an Agent invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_tools: frozenset[ResearchToolName] = READ_ONLY_RESEARCH_TOOLS
    # Absolute invocation ceiling. Each run freezes a smaller question-specific
    # budget after planning and can only consume, never expand, that budget.
    max_steps: int = Field(
        default=ABSOLUTE_MAX_RESEARCH_STEPS,
        ge=1,
        le=ABSOLUTE_MAX_RESEARCH_STEPS,
    )
    max_followup_steps: int = Field(default=MAX_FOLLOWUP_STEPS, ge=0, le=MAX_FOLLOWUP_STEPS)
    max_dynamic_tool_steps: int = Field(default=6, ge=1, le=12)
    comparison_search_concurrency: int = Field(default=2, ge=1, le=4)
    evidence_per_step: int = Field(default=4, ge=1, le=20)
    max_tool_calls: int = Field(
        default=ABSOLUTE_MAX_TOOL_CALLS,
        ge=1,
        le=ABSOLUTE_MAX_TOOL_CALLS,
    )
    timeout_seconds: float = Field(default=180, gt=0, le=300)

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

    def freeze_invocation_budget(self, initial_steps: int) -> tuple[int, int]:
        """Freeze a smaller per-question budget that can never grow during execution."""
        if initial_steps <= 0 or initial_steps > MAX_INITIAL_PLAN_STEPS:
            raise ValueError(
                f"initial_steps must be between 1 and {MAX_INITIAL_PLAN_STEPS}"
            )
        proportional_margin = max(2, (initial_steps + 3) // 4)
        followup_steps = min(self.max_followup_steps, proportional_margin)
        step_budget = min(self.max_steps, initial_steps + followup_steps)
        tool_call_budget = min(self.max_tool_calls, step_budget * 2)
        if step_budget < initial_steps or tool_call_budget < initial_steps:
            raise ValueError("initial plan cannot fit the frozen invocation budget")
        return step_budget, tool_call_budget
