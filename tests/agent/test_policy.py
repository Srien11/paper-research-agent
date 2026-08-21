from __future__ import annotations

import pytest
from pydantic import ValidationError

from paper_research_agent.agent.policy import (
    ResearchRuntimePolicy,
    evidence_cutoff_after_assessments,
)


def test_default_policy_leaves_room_for_maximum_grid_and_followups() -> None:
    policy = ResearchRuntimePolicy()

    assert policy.max_steps == 24
    assert policy.max_followup_steps == 4
    assert policy.comparison_search_concurrency == 2
    assert policy.adaptive_evidence_hydration_enabled is False
    assert policy.evidence_per_step == 4
    assert policy.initial_evidence_per_step == 4
    assert policy.first_followup_evidence_per_step == 6
    assert policy.later_followup_evidence_per_step == 10
    assert policy.max_tool_calls == 48
    assert policy.timeout_seconds == 180
    assert policy.freeze_invocation_budget(2) == (4, 8)
    assert policy.freeze_invocation_budget(10) == (13, 26)
    assert policy.freeze_invocation_budget(20) == (24, 48)


def test_adaptive_hydration_cutoff_is_deterministic_and_bounded() -> None:
    policy = ResearchRuntimePolicy(adaptive_evidence_hydration_enabled=True)

    assert evidence_cutoff_after_assessments(
        policy, assessment_count=0, is_comparison=True
    ) == 4
    assert evidence_cutoff_after_assessments(
        policy, assessment_count=1, is_comparison=True
    ) == 6
    assert evidence_cutoff_after_assessments(
        policy, assessment_count=2, is_comparison=True
    ) == 10
    assert evidence_cutoff_after_assessments(
        policy, assessment_count=8, is_comparison=True
    ) == 10
    assert evidence_cutoff_after_assessments(
        policy, assessment_count=8, is_comparison=False
    ) == 4


def test_adaptive_hydration_is_disabled_by_default() -> None:
    policy = ResearchRuntimePolicy()

    assert evidence_cutoff_after_assessments(
        policy, assessment_count=1, is_comparison=True
    ) == 4
    assert evidence_cutoff_after_assessments(
        policy, assessment_count=8, is_comparison=True
    ) == 4


def test_adaptive_hydration_cutoffs_must_be_non_decreasing() -> None:
    with pytest.raises(ValidationError, match="non-decreasing"):
        ResearchRuntimePolicy(
            initial_evidence_per_step=6,
            first_followup_evidence_per_step=4,
            later_followup_evidence_per_step=10,
        )


def test_legacy_evidence_per_step_input_maps_to_initial_cutoff() -> None:
    policy = ResearchRuntimePolicy(evidence_per_step=3)

    assert policy.initial_evidence_per_step == 3
    assert policy.evidence_per_step == 3
