"""Fail-closed assessment boundaries shared by the graph and runtime."""

from __future__ import annotations

from collections.abc import Mapping


def direct_assessment_limits(
    state: Mapping[str, object], *, observation_count: int, assessment_count: int
) -> tuple[int, ...]:
    """Read exact assessed prefixes, with one-to-one legacy compatibility."""
    raw_counts = state.get("assessment_observation_counts")
    if raw_counts is None or raw_counts == []:
        if assessment_count != observation_count:
            raise ValueError("research assessments do not match observations")
        return tuple(range(1, observation_count + 1))
    if not isinstance(raw_counts, list) or len(raw_counts) != assessment_count:
        raise ValueError("research assessments do not match observations")
    maximum_batch = 2 if state.get("direct_batching_allowed") is True else 1
    limits: list[int] = []
    previous = 0
    for count in raw_counts:
        if (
            type(count) is not int
            or not previous < count <= observation_count
            or count - previous > maximum_batch
        ):
            raise ValueError("research assessments do not match observations")
        limits.append(count)
        previous = count
    if previous != observation_count:
        raise ValueError("research assessments do not match observations")
    return tuple(limits)
