"""Multi-turn gold evaluation for the main Agent."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import Field

from paper_research_agent.agent.orchestrator.models import FrozenModel


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
