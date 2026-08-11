"""Validated contracts for privacy-safe research-tool routing evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.agent.dynamic.memory import MemoryProposal
from paper_research_agent.agent.tooling.catalog import (
    EXTENDED_TOOL_NAMES,
    TOOL_SPEC_BY_NAME,
    ToolRisk,
    ToolTrust,
    effective_tool_spec,
)
from paper_research_agent.agent.tooling.contracts import TOOL_INPUT_SCHEMAS

EvaluationStage = Literal["tool_router", "memory_proposer", "dynamic_pipeline"]
ExpectedAction = Literal[
    "call_tool",
    "finish",
    "none",
    "add",
    "update",
    "delete",
    "approval_required",
]
ScoringDimension = Literal[
    "action",
    "tool",
    "arguments",
    "policy",
    "explicit_intent",
    "neighbor_disambiguation",
]

_ACTIONS_BY_STAGE: dict[EvaluationStage, frozenset[str]] = {
    "tool_router": frozenset({"call_tool", "finish"}),
    "memory_proposer": frozenset({"none", "add", "update", "delete"}),
    "dynamic_pipeline": frozenset({"call_tool", "finish", "approval_required"}),
}
_NO_TOOL_ACTIONS = frozenset({"finish", "none"})
_MEMORY_MUTATIONS = frozenset({"add", "update", "delete"})


class ToolRoutingCase(BaseModel):
    """One stage-specific routing gold label.

    ``evaluation_stage`` and ``expected_action`` default to the original
    one-step router contract so the existing v1 JSONL remains loadable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^(?:route|tr2)-\d{3}$")
    evaluation_stage: EvaluationStage = "tool_router"
    question: str = Field(min_length=1, max_length=10_000)
    expected_action: ExpectedAction = "call_tool"
    expected_tool: str | None = None
    allowed_tools: tuple[str, ...] = ()
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    expected_risk: ToolRisk | None = None
    expected_trust: ToolTrust | None = None
    approval_required: bool = False
    scoring_scope: tuple[ScoringDimension, ...] = ("action", "tool", "policy")
    test_reason: str = Field(default="legacy v1 routing case", min_length=1, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def add_legacy_defaults(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        tool = payload.get("expected_tool")
        payload.setdefault("allowed_tools", [tool] if isinstance(tool, str) else [])
        payload.setdefault("expected_arguments", {})
        payload.setdefault("scoring_scope", ["action", "tool", "policy"])
        payload.setdefault("test_reason", "legacy v1 routing case")
        return payload

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("routing question cannot be blank")
        return normalized

    @field_validator("test_reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("test_reason cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_stage_and_catalog(self) -> ToolRoutingCase:
        if self.expected_action not in _ACTIONS_BY_STAGE[self.evaluation_stage]:
            raise ValueError("expected_action is invalid for evaluation_stage")
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError("allowed_tools must be unique")
        if any(tool not in EXTENDED_TOOL_NAMES for tool in self.allowed_tools):
            raise ValueError("allowed_tools contains an unavailable tool")
        if len(self.scoring_scope) != len(set(self.scoring_scope)):
            raise ValueError("scoring_scope must be unique")
        if not self.scoring_scope:
            raise ValueError("scoring_scope cannot be empty")
        if (
            "explicit_intent" in self.scoring_scope
            and self.evaluation_stage != "memory_proposer"
        ):
            raise ValueError("explicit_intent scoring requires memory_proposer")
        if (
            "neighbor_disambiguation" in self.scoring_scope
            and self.evaluation_stage != "tool_router"
        ):
            raise ValueError("neighbor_disambiguation scoring requires tool_router")

        if self.expected_action in _NO_TOOL_ACTIONS:
            if any(
                value is not None
                for value in (self.expected_tool, self.expected_risk, self.expected_trust)
            ) or self.approval_required or self.allowed_tools or self.expected_arguments:
                raise ValueError("no-tool actions cannot contain tool metadata")
            return self

        if self.expected_tool not in EXTENDED_TOOL_NAMES:
            raise ValueError("expected_tool is not in the extended tool catalog")
        if self.expected_tool is None:
            raise ValueError("tool actions require expected_tool")
        if self.expected_tool not in self.allowed_tools:
            raise ValueError("expected_tool must be included in allowed_tools")
        if (
            self.evaluation_stage == "memory_proposer"
            and self.expected_action in _MEMORY_MUTATIONS
            and self.expected_tool != "manage_long_term_memory"
        ):
            raise ValueError("memory mutations must resolve to manage_long_term_memory")

        arguments_for_policy = dict(self.expected_arguments)
        if self.evaluation_stage == "memory_proposer":
            arguments_for_policy["action"] = self.expected_action
        spec = effective_tool_spec(
            TOOL_SPEC_BY_NAME[self.expected_tool],
            arguments_for_policy,
        )
        expected_metadata = (spec.risk, spec.trust, spec.approval_required)
        actual_metadata = (
            self.expected_risk,
            self.expected_trust,
            self.approval_required,
        )
        if actual_metadata != expected_metadata:
            raise ValueError("expected risk, trust, or approval does not match the tool catalog")
        if (
            self.evaluation_stage == "dynamic_pipeline"
            and spec.approval_required != (self.expected_action == "approval_required")
        ):
            raise ValueError("dynamic pipeline action does not match the approval policy")
        if "arguments" in self.scoring_scope:
            self._validate_expected_arguments()
        return self

    def _validate_expected_arguments(self) -> None:
        if self.evaluation_stage == "memory_proposer":
            MemoryProposal.model_validate(
                {
                    "action": self.expected_action,
                    **self.expected_arguments,
                    "rationale": "evaluation gold label",
                }
            )
            return
        if self.expected_tool is None:
            raise ValueError("tool argument scoring requires expected_tool")
        schema = TOOL_INPUT_SCHEMAS[self.expected_tool]
        schema.model_validate(self.expected_arguments)


def load_tool_routing_dataset(path: Path) -> list[ToolRoutingCase]:
    cases = [
        ToolRoutingCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("tool-routing case_id values must be unique")
    return cases
