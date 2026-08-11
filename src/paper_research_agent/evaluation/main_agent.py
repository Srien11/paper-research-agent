"""Multi-turn gold evaluation for the main Agent."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import Field

from paper_research_agent.agent.orchestrator.models import Capability, FrozenModel, RunStatus

InterfaceScenario = Literal[
    "direct_chat",
    "local_rag",
    "paper_comparison",
    "hybrid_research",
    "attachment_qa",
    "file_edit",
    "approval_approved",
    "approval_rejected",
    "approval_expired",
    "duplicate_request",
    "rag_mode_required",
    "rag_mode_disabled",
    "commit_rejected",
]
RAGMode = Literal["disabled", "preferred", "required"]
ApprovalOutcome = Literal["none", "approved", "rejected", "expired"]
ArtifactKind = Literal["chat", "local_rag", "dynamic_tools", "attachment_qa", "file_edit"]


class MainAgentInterfaceCase(FrozenModel):
    """One release-gate scenario for the unified main-Agent interface."""

    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,96}$")
    scenario: InterfaceScenario
    description: str = Field(min_length=1, max_length=500)
    message: str = Field(min_length=1, max_length=20_000)
    rag_mode: RAGMode = "disabled"
    expected_capabilities: tuple[Capability, ...] = Field(min_length=1, max_length=5)
    expected_artifact_kinds: tuple[ArtifactKind, ...] = Field(min_length=1, max_length=5)
    expected_status: RunStatus
    requires_citation_preservation: bool = False
    approval_outcome: ApprovalOutcome = "none"
    duplicate_request: bool = False
    expect_commit_rejected: bool = False
    expected_event_types: tuple[str, ...] = Field(min_length=1, max_length=16)


class MainAgentInterfaceRecord(FrozenModel):
    """Observed facts from one interface scenario execution."""

    status: RunStatus
    capabilities: tuple[Capability, ...] = Field(default=(), max_length=5)
    artifact_kinds: tuple[ArtifactKind, ...] = Field(default=(), max_length=5)
    input_source_ids: tuple[str, ...] = Field(default=(), max_length=256)
    output_source_ids: tuple[str, ...] = Field(default=(), max_length=256)
    goal_continuous: bool = False
    approval_resume_count: int = Field(default=0, ge=0)
    interpretation_count: int = Field(default=0, ge=0)
    planning_count: int = Field(default=0, ge=0)
    side_effect_count: int = Field(default=0, ge=0)
    commit_rejected: bool = False
    workspace_version_before: int = Field(default=0, ge=0)
    workspace_version_after: int = Field(default=0, ge=0)
    event_types: tuple[str, ...] = Field(default=(), max_length=64)
    event_ids: tuple[int, ...] = Field(default=(), max_length=64)
    done_count: int = Field(default=0, ge=0)


class MainAgentInterfaceScoringResult(FrozenModel):
    route_correctness: float = Field(ge=0, le=1)
    goal_continuity: float = Field(ge=0, le=1)
    artifact_preservation: float = Field(ge=0, le=1)
    citation_preservation: float = Field(ge=0, le=1)
    approval_resume_correctness: float = Field(ge=0, le=1)
    duplicate_side_effect: float = Field(ge=0, le=1)
    commit_rejection: float = Field(ge=0, le=1)
    event_contract_validity: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)


class MainAgentTurnRecord(FrozenModel):
    turn_index: int = Field(ge=1)
    relation: str
    goal_id: str | None = None
    plan_revision: int | None = Field(default=None, ge=1)
    capabilities: tuple[str, ...] = Field(default=(), max_length=8)
    status: str
    memory_recalled_before_route: bool = False
    side_effect_count: int = Field(default=0, ge=0)


class MainAgentConversationCase(FrozenModel):
    case_id: str
    description: str
    turns: tuple[str, ...] = Field(min_length=1, max_length=20)
    expected_relations: tuple[str, ...] = Field(min_length=1)
    goal_continuous: bool = False
    expected_capabilities: tuple[str, ...] = Field(default=(), max_length=8)
    expected_final_status: str = "completed"


class MainAgentScoringResult(FrozenModel):
    goal_continuity: float = Field(ge=0, le=1)
    plan_continuity: float = Field(ge=0, le=1)
    route_correctness: float = Field(ge=0, le=1)
    memory_before_routing: float = Field(ge=0, le=1)
    duplicate_side_effect: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)


def load_main_agent_cases(path: Path) -> tuple[MainAgentConversationCase, ...]:
    cases: list[MainAgentConversationCase] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            cases.append(MainAgentConversationCase.model_validate(json.loads(raw)))
    return tuple(cases)


def load_main_agent_interface_cases(path: Path) -> tuple[MainAgentInterfaceCase, ...]:
    """Load strict JSONL release cases and reject ambiguous duplicate IDs."""
    cases: list[MainAgentInterfaceCase] = []
    case_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            case = MainAgentInterfaceCase.model_validate(json.loads(raw))
            if case.case_id in case_ids:
                raise ValueError(f"duplicate interface case_id at line {line_number}: {case.case_id}")
            case_ids.add(case.case_id)
            cases.append(case)
    return tuple(cases)


def score_main_agent_interface_case(
    case: MainAgentInterfaceCase,
    record: MainAgentInterfaceRecord,
) -> MainAgentInterfaceScoringResult:
    """Score the eight observable release guarantees of the unified interface."""
    route_correctness = float(record.capabilities == case.expected_capabilities)
    goal_continuity = float(record.goal_continuous)
    artifact_preservation = float(record.artifact_kinds == case.expected_artifact_kinds)
    citation_preservation = _score_interface_citations(case, record)
    approval_resume_correctness = _score_interface_approval(case, record)
    duplicate_side_effect = _score_interface_side_effects(case, record)
    commit_rejection = _score_interface_commit(case, record)
    event_contract_validity = _score_interface_events(case, record)
    axes = (
        route_correctness,
        goal_continuity,
        artifact_preservation,
        citation_preservation,
        approval_resume_correctness,
        duplicate_side_effect,
        commit_rejection,
        event_contract_validity,
    )
    return MainAgentInterfaceScoringResult(
        route_correctness=route_correctness,
        goal_continuity=goal_continuity,
        artifact_preservation=artifact_preservation,
        citation_preservation=citation_preservation,
        approval_resume_correctness=approval_resume_correctness,
        duplicate_side_effect=duplicate_side_effect,
        commit_rejection=commit_rejection,
        event_contract_validity=event_contract_validity,
        overall=round(sum(axes) / len(axes), 4),
    )


def _score_interface_citations(
    case: MainAgentInterfaceCase,
    record: MainAgentInterfaceRecord,
) -> float:
    if not case.requires_citation_preservation:
        return 1.0
    return float(
        bool(record.input_source_ids)
        and record.output_source_ids == record.input_source_ids
    )


def _score_interface_approval(
    case: MainAgentInterfaceCase,
    record: MainAgentInterfaceRecord,
) -> float:
    if case.approval_outcome == "none":
        return 1.0
    interpretation_and_plan_preserved = (
        record.interpretation_count == 1 and record.planning_count == 1
    )
    if case.approval_outcome == "expired":
        return float(interpretation_and_plan_preserved and record.approval_resume_count == 0)
    return float(interpretation_and_plan_preserved and record.approval_resume_count == 1)


def _score_interface_side_effects(
    case: MainAgentInterfaceCase,
    record: MainAgentInterfaceRecord,
) -> float:
    if case.approval_outcome in {"rejected", "expired"}:
        return float(record.side_effect_count == 0)
    if case.approval_outcome == "approved":
        return float(record.side_effect_count == 1)
    if case.duplicate_request:
        return float(record.side_effect_count <= 1)
    return float(record.side_effect_count <= 1)


def _score_interface_commit(
    case: MainAgentInterfaceCase,
    record: MainAgentInterfaceRecord,
) -> float:
    if case.expect_commit_rejected:
        return float(
            record.commit_rejected
            and record.workspace_version_after == record.workspace_version_before
        )
    return float(not record.commit_rejected)


def _score_interface_events(
    case: MainAgentInterfaceCase,
    record: MainAgentInterfaceRecord,
) -> float:
    expected_events_present = all(
        event_type in record.event_types for event_type in case.expected_event_types
    )
    event_ids_contiguous = record.event_ids == tuple(range(1, len(record.event_ids) + 1))
    return float(
        record.status == case.expected_status
        and expected_events_present
        and event_ids_contiguous
        and record.done_count == 1
    )


def score_main_agent_case(
    case: MainAgentConversationCase,
    records: Iterable[MainAgentTurnRecord],
) -> MainAgentScoringResult:
    """Score one conversation against the gold expectations; all axes in [0, 1]."""
    turn_records = tuple(records)
    if not turn_records:
        return MainAgentScoringResult(
            goal_continuity=0.0,
            plan_continuity=0.0,
            route_correctness=0.0,
            memory_before_routing=0.0,
            duplicate_side_effect=0.0,
            overall=0.0,
        )
    goal_continuity = _score_goal_continuity(case, turn_records)
    plan_continuity = _score_plan_continuity(turn_records)
    route_correctness = _score_route_correctness(case, turn_records)
    memory_before_routing = _score_memory_before_routing(turn_records)
    duplicate_side_effect = _score_duplicate_side_effect(turn_records)
    axes = (
        goal_continuity,
        plan_continuity,
        route_correctness,
        memory_before_routing,
        duplicate_side_effect,
    )
    overall = sum(axes) / len(axes)
    return MainAgentScoringResult(
        goal_continuity=goal_continuity,
        plan_continuity=plan_continuity,
        route_correctness=route_correctness,
        memory_before_routing=memory_before_routing,
        duplicate_side_effect=duplicate_side_effect,
        overall=round(overall, 4),
    )


def _score_goal_continuity(
    case: MainAgentConversationCase, records: tuple[MainAgentTurnRecord, ...]
) -> float:
    if not case.goal_continuous:
        return 1.0
    goal_ids = [record.goal_id for record in records if record.goal_id is not None]
    if not goal_ids:
        return 0.0
    return 1.0 if len(set(goal_ids)) == 1 else 0.0


def _score_plan_continuity(records: tuple[MainAgentTurnRecord, ...]) -> float:
    revisions = [record.plan_revision for record in records if record.plan_revision is not None]
    if not revisions:
        return 1.0
    if revisions != sorted(revisions):
        return 0.0
    return 1.0


def _score_route_correctness(
    case: MainAgentConversationCase, records: tuple[MainAgentTurnRecord, ...]
) -> float:
    expected = set(case.expected_capabilities)
    actual = {capability for record in records for capability in record.capabilities}
    if not expected:
        return 1.0
    return 1.0 if actual == expected else 0.0


def _score_memory_before_routing(records: tuple[MainAgentTurnRecord, ...]) -> float:
    routed = [record for record in records if record.capabilities]
    if not routed:
        return 1.0
    return 1.0 if all(record.memory_recalled_before_route for record in routed) else 0.0


def _score_duplicate_side_effect(records: tuple[MainAgentTurnRecord, ...]) -> float:
    return 1.0 if all(record.side_effect_count <= 1 for record in records) else 0.0
