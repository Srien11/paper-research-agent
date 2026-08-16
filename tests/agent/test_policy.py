from __future__ import annotations

from paper_research_agent.agent.policy import ResearchRuntimePolicy


def test_default_policy_leaves_room_for_maximum_grid_and_followups() -> None:
    policy = ResearchRuntimePolicy()

    assert policy.max_steps == 24
    assert policy.max_followup_steps == 4
    assert policy.comparison_search_concurrency == 2
    assert policy.evidence_per_step == 4
    assert policy.max_tool_calls == 48
    assert policy.timeout_seconds == 180
    assert policy.freeze_invocation_budget(2) == (4, 8)
    assert policy.freeze_invocation_budget(10) == (13, 26)
    assert policy.freeze_invocation_budget(20) == (24, 48)
