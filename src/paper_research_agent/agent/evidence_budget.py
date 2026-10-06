"""Shared excerpt limits and conservative capacity checks for direct research."""

from __future__ import annotations

from collections.abc import Sequence

from paper_research_agent.agent.models import ResearchObservation

MAX_DIRECT_EVIDENCE_CHARS = 24_000
MAX_RECORD_EXCERPT_CHARS = 2_000


def direct_batch_fits_evidence_budget(
    observations: Sequence[ResearchObservation], *, additional_records: int
) -> bool:
    """Keep every existing excerpt and reserve full excerpts for new records."""
    seen: set[str] = set()
    used = 0
    for observation in observations:
        for record in observation.evidence.records:
            if record.chunk_id not in seen:
                seen.add(record.chunk_id)
                used += min(len(record.text), MAX_RECORD_EXCERPT_CHARS)
    return used + additional_records * MAX_RECORD_EXCERPT_CHARS <= MAX_DIRECT_EVIDENCE_CHARS
