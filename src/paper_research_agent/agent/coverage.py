"""Deterministic evidence-coverage validation for structured research plans."""

from __future__ import annotations

from paper_research_agent.agent.models import (
    EvidenceAssessment,
    EvidenceCoverage,
    ResearchObservation,
    ResearchPlan,
)


def validate_evidence_assessment(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    assessment: EvidenceAssessment,
) -> EvidenceAssessment:
    """Reject subjective sufficiency claims that violate the auditable plan state."""
    if plan.task_type == "direct":
        if assessment.coverage or assessment.next_requirement_ids:
            raise ValueError("direct assessment cannot declare comparison coverage")
        return assessment

    expected_ids = {item.requirement_id for item in plan.requirements}
    actual_ids = {item.requirement_id for item in assessment.coverage}
    if actual_ids != expected_ids or len(assessment.coverage) != len(expected_ids):
        raise ValueError("comparison coverage IDs must exactly match plan requirements")

    available_chunk_ids = {
        record.chunk_id
        for observation in observations
        for record in observation.evidence.records
    }
    for item in assessment.coverage:
        unknown = set(item.chunk_ids) - available_chunk_ids
        if unknown:
            raise ValueError("comparison coverage references an unknown evidence chunk")

    missing_ids = {item.requirement_id for item in assessment.coverage if not item.covered}
    if assessment.evidence_sufficient and missing_ids:
        raise ValueError("comparison cannot be sufficient with missing coverage")
    if assessment.status == "missing_coverage" and not missing_ids:
        raise ValueError("missing-coverage status requires an uncovered requirement")
    if not set(assessment.next_requirement_ids) <= missing_ids:
        raise ValueError("next requirement IDs must reference uncovered requirements")
    if assessment.next_query is not None and not assessment.next_requirement_ids:
        raise ValueError("comparison follow-up query requires missing requirement IDs")
    return assessment


def repair_evidence_assessment(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    assessment: EvidenceAssessment | None,
) -> EvidenceAssessment:
    """Conservatively repair model-only structural drift without inventing evidence."""
    if assessment is None:
        return _empty_assessment(plan, observations)

    if plan.task_type == "direct":
        repaired = assessment.model_copy(
            update={"coverage": (), "next_requirement_ids": ()}
        )
        return validate_evidence_assessment(plan, observations, repaired)

    available_chunk_ids = {
        record.chunk_id
        for observation in observations
        for record in observation.evidence.records
    }
    supplied = {item.requirement_id: item for item in assessment.coverage}
    coverage: list[EvidenceCoverage] = []
    for requirement in plan.requirements:
        item = supplied.get(requirement.requirement_id)
        chunk_ids = (
            tuple(chunk_id for chunk_id in item.chunk_ids if chunk_id in available_chunk_ids)
            if item is not None and item.covered
            else ()
        )
        coverage.append(
            EvidenceCoverage(
                requirement_id=requirement.requirement_id,
                covered=bool(chunk_ids),
                chunk_ids=chunk_ids,
            )
        )

    missing_ids = tuple(item.requirement_id for item in coverage if not item.covered)
    if not missing_ids:
        repaired = EvidenceAssessment(
            evidence_sufficient=True,
            status="sufficient",
            coverage=tuple(coverage),
        )
        return validate_evidence_assessment(plan, observations, repaired)

    requested_ids = tuple(
        requirement_id
        for requirement_id in assessment.next_requirement_ids
        if requirement_id in missing_ids
    )
    if assessment.next_query is not None and not requested_ids:
        requested_ids = (missing_ids[0],)
    has_follow_up = assessment.next_query is not None and bool(requested_ids)
    status = (
        assessment.status
        if assessment.status != "sufficient"
        else "missing_coverage"
    )
    repaired = EvidenceAssessment(
        evidence_sufficient=False,
        status=status,
        coverage=tuple(coverage),
        next_query=assessment.next_query if has_follow_up else None,
        next_objective=assessment.next_objective if has_follow_up else None,
        next_requirement_ids=requested_ids if has_follow_up else (),
    )
    return validate_evidence_assessment(plan, observations, repaired)


def _empty_assessment(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
) -> EvidenceAssessment:
    has_evidence = any(item.evidence.records for item in observations)
    coverage = (
        tuple(
            EvidenceCoverage(
                requirement_id=requirement.requirement_id,
                covered=False,
            )
            for requirement in plan.requirements
        )
        if plan.task_type == "comparison"
        else ()
    )
    return EvidenceAssessment(
        evidence_sufficient=False,
        status="missing_coverage" if has_evidence else "no_hits",
        coverage=coverage,
    )
