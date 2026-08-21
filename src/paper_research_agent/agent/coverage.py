"""Deterministic evidence-coverage validation for structured research plans."""

from __future__ import annotations

from collections.abc import Mapping

from paper_research_agent.agent.fact_queries import compose_fact_search_query
from paper_research_agent.agent.models import (
    AssessmentStatus,
    CompiledEvidenceFact,
    EvidenceAssessment,
    EvidenceCellCompilation,
    EvidenceCompilationRepairAudit,
    EvidenceCoverage,
    EvidenceFollowup,
    EvidenceLedgerCell,
    EvidenceRequirement,
    ResearchObservation,
    ResearchPlan,
)


def validate_evidence_compilation_cell(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    cell: EvidenceCellCompilation,
) -> EvidenceCellCompilation:
    """Validate one minimal model-authored cell against trusted plan scope."""
    if plan.task_type != "comparison":
        raise ValueError("evidence cell compilation requires a comparison plan")
    requirement_by_id = {item.requirement_id: item for item in plan.requirements}
    requirement = requirement_by_id.get(cell.requirement_id)
    if requirement is None:
        raise ValueError("compiled evidence cell references an unknown requirement")
    allowed_chunks = _available_chunks_by_requirement(plan, observations)[
        cell.requirement_id
    ]
    expected_fact_ids = {
        item.fact_requirement_id for item in requirement.fact_requirements
    }
    for fact in cell.facts:
        if not set(fact.chunk_ids) <= allowed_chunks:
            raise ValueError(
                "compiled evidence fact references evidence outside its comparison cell"
            )
        if not set(fact.fact_requirement_ids) <= expected_fact_ids:
            raise ValueError(
                "compiled evidence fact references an unknown fact requirement"
            )
        required_qualifiers = {
            kind
            for item in requirement.fact_requirements
            if item.fact_requirement_id in fact.fact_requirement_ids
            for kind in item.required_qualifier_kinds
        }
        if not required_qualifiers <= {item.kind for item in fact.qualifiers}:
            raise ValueError("compiled evidence fact omits a required qualifier")
    return cell


def project_evidence_compilation(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    *,
    committed_cells: Mapping[str, EvidenceCellCompilation],
    compiler_failed_requirement_ids: tuple[str, ...] = (),
) -> EvidenceAssessment:
    """Deterministically project committed minimal cells into a strict assessment."""
    if plan.task_type != "comparison":
        raise ValueError("evidence compilation projection requires a comparison plan")
    expected_ids = {item.requirement_id for item in plan.requirements}
    if set(committed_cells) - expected_ids:
        raise ValueError("committed evidence contains an unknown requirement")
    failed_ids = set(compiler_failed_requirement_ids)
    if failed_ids - expected_ids:
        raise ValueError("compiler failure references an unknown requirement")
    if failed_ids & set(committed_cells):
        raise ValueError("a compilation unit cannot be both committed and failed")

    coverage: list[EvidenceCoverage] = []
    ledger: list[EvidenceLedgerCell] = []
    for requirement in plan.requirements:
        cell = committed_cells.get(requirement.requirement_id)
        if cell is not None:
            validate_evidence_compilation_cell(plan, observations, cell)
        facts = tuple(
            CompiledEvidenceFact(
                fact_id=f"{requirement.requirement_id}-f{index}",
                statement=fact.statement,
                chunk_ids=fact.chunk_ids,
                fact_requirement_ids=fact.fact_requirement_ids,
                qualifiers=fact.qualifiers,
            )
            for index, fact in enumerate(() if cell is None else cell.facts, 1)
        )
        satisfied_fact_ids = {
            fact_requirement_id
            for fact in facts
            for fact_requirement_id in fact.fact_requirement_ids
        }
        missing_fact_ids = tuple(
            item.fact_requirement_id
            for item in requirement.fact_requirements
            if item.fact_requirement_id not in satisfied_fact_ids
        )
        chunk_ids = tuple(
            dict.fromkeys(chunk_id for fact in facts for chunk_id in fact.chunk_ids)
        )
        coverage.append(
            EvidenceCoverage(
                requirement_id=requirement.requirement_id,
                covered=bool(facts),
                chunk_ids=chunk_ids,
            )
        )
        ledger.append(
            EvidenceLedgerCell(
                requirement_id=requirement.requirement_id,
                status=(
                    "missing"
                    if not facts
                    else "partial" if missing_fact_ids else "sufficient"
                ),
                facts=facts,
                missing_fact_requirement_ids=missing_fact_ids,
            )
        )

    evidence_sufficient = all(item.status == "sufficient" for item in ledger)
    status: AssessmentStatus
    if failed_ids:
        status = "compiler_failed"
        evidence_sufficient = False
    elif evidence_sufficient:
        status = "sufficient"
    else:
        has_evidence = any(item.evidence.records for item in observations)
        status = "missing_coverage" if has_evidence else "no_hits"
    return validate_evidence_assessment(
        plan,
        observations,
        EvidenceAssessment(
            evidence_sufficient=evidence_sufficient,
            status=status,
            coverage=tuple(coverage),
            ledger=tuple(ledger),
        ),
    )


def validate_evidence_assessment(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    assessment: EvidenceAssessment,
) -> EvidenceAssessment:
    """Reject subjective sufficiency claims that violate the auditable plan state."""
    if plan.task_type == "direct":
        if (
            assessment.coverage
            or assessment.ledger
            or assessment.next_requirement_ids
            or assessment.followups
        ):
            raise ValueError("direct assessment cannot declare comparison coverage")
        return assessment

    if assessment.evidence_sufficient:
        required_structured_fact_ids = {
            fact.fact_requirement_id
            for requirement in plan.requirements
            for fact in requirement.fact_requirements
            if fact.origin == "planned" and fact.protected_anchors
        }
        step_by_id = {step.step_id: step for step in plan.steps}
        executed_fact_ids = {
            step.fact_requirement_id
            for observation in observations
            if (step := step_by_id.get(observation.step_id)) is not None
            and step.fact_requirement_id is not None
        }
        if not required_structured_fact_ids <= executed_fact_ids:
            raise ValueError(
                "structured fact query has not been executed before sufficiency"
            )

    expected_ids = {item.requirement_id for item in plan.requirements}
    actual_ids = {item.requirement_id for item in assessment.coverage}
    if actual_ids != expected_ids or len(assessment.coverage) != len(expected_ids):
        raise ValueError("comparison coverage IDs must exactly match plan requirements")

    available_by_requirement = _available_chunks_by_requirement(plan, observations)
    for item in assessment.coverage:
        unknown = set(item.chunk_ids) - available_by_requirement[item.requirement_id]
        if unknown:
            raise ValueError("comparison coverage references evidence outside its comparison cell")

    if assessment.ledger:
        ledger_ids = {item.requirement_id for item in assessment.ledger}
        if ledger_ids != expected_ids or len(assessment.ledger) != len(expected_ids):
            raise ValueError("comparison ledger IDs must exactly match plan requirements")
        coverage_by_id = {item.requirement_id: item for item in assessment.coverage}
        requirement_by_id = {item.requirement_id: item for item in plan.requirements}
        for cell in assessment.ledger:
            allowed = available_by_requirement[cell.requirement_id]
            requirement = requirement_by_id[cell.requirement_id]
            expected_fact_ids = {
                item.fact_requirement_id for item in requirement.fact_requirements
            }
            satisfied_fact_ids: set[str] = set()
            for fact in cell.facts:
                mapped_ids = _fact_requirement_ids(fact, requirement)
                if not mapped_ids:
                    raise ValueError(
                        "compiled evidence fact requires a fact requirement mapping"
                    )
                if not set(mapped_ids) <= expected_fact_ids:
                    raise ValueError(
                        "compiled evidence fact references an unknown fact requirement"
                    )
                required_qualifiers = {
                    kind
                    for item in requirement.fact_requirements
                    if item.fact_requirement_id in mapped_ids
                    for kind in item.required_qualifier_kinds
                }
                if not required_qualifiers <= {item.kind for item in fact.qualifiers}:
                    raise ValueError(
                        "compiled evidence fact omits a required qualifier"
                    )
                satisfied_fact_ids.update(mapped_ids)
            missing_fact_ids = expected_fact_ids - satisfied_fact_ids
            legacy_missing_omitted = (
                not cell.missing_fact_requirement_ids
                and all(item.origin == "derived" for item in requirement.fact_requirements)
            )
            if (
                set(cell.missing_fact_requirement_ids) != missing_fact_ids
                and not legacy_missing_omitted
            ):
                raise ValueError(
                    "ledger missing fact requirement IDs do not match compiled facts"
                )
            fact_chunks = tuple(
                dict.fromkeys(chunk_id for fact in cell.facts for chunk_id in fact.chunk_ids)
            )
            if set(fact_chunks) - allowed:
                raise ValueError(
                    "comparison ledger fact references evidence outside its comparison cell"
                )
            coverage = coverage_by_id[cell.requirement_id]
            if tuple(coverage.chunk_ids) != fact_chunks:
                raise ValueError("comparison ledger facts must exactly project to coverage chunks")
            if coverage.covered != bool(cell.facts):
                raise ValueError("comparison ledger fact state must match coverage")
            expected_status = (
                "missing"
                if not cell.facts
                else "partial" if missing_fact_ids else "sufficient"
            )
            if cell.status != "conflicting" and cell.status != expected_status:
                raise ValueError("ledger status does not match fact intent coverage")

    missing_ids = (
        {
            item.requirement_id
            for item in assessment.ledger
            if item.status != "sufficient"
        }
        if assessment.ledger
        else {item.requirement_id for item in assessment.coverage if not item.covered}
    )
    if assessment.evidence_sufficient and missing_ids:
        raise ValueError("comparison cannot be sufficient with missing coverage")
    if assessment.status == "missing_coverage" and not missing_ids:
        raise ValueError("missing-coverage status requires an uncovered requirement")
    if not set(assessment.next_requirement_ids) <= missing_ids:
        raise ValueError("next requirement IDs must reference uncovered requirements")
    if assessment.next_query is not None and not assessment.next_requirement_ids:
        raise ValueError("comparison follow-up query requires missing requirement IDs")
    if len(assessment.next_requirement_ids) > 1:
        raise ValueError("comparison follow-up must target one requirement cell")
    followup_requirement_ids = {item.requirement_id for item in assessment.followups}
    if not followup_requirement_ids <= missing_ids:
        raise ValueError("follow-ups must reference uncovered requirements")
    requirement_by_id = {item.requirement_id: item for item in plan.requirements}
    ledger_by_id = {item.requirement_id: item for item in assessment.ledger}
    for followup in assessment.followups:
        if followup.fact_requirement_id is None:
            continue
        requirement = requirement_by_id[followup.requirement_id]
        expected_fact_ids = {
            item.fact_requirement_id for item in requirement.fact_requirements
        }
        if followup.fact_requirement_id not in expected_fact_ids:
            raise ValueError(
                "follow-up fact requirement does not belong to its requirement cell"
            )
        followup_cell = ledger_by_id.get(followup.requirement_id)
        if followup_cell is None:
            raise ValueError("fact-bound follow-up requires a comparison ledger cell")
        missing_fact_ids = set(followup_cell.missing_fact_requirement_ids)
        if not missing_fact_ids and all(
            item.origin == "derived" for item in requirement.fact_requirements
        ):
            satisfied_fact_ids = {
                fact_requirement_id
                for fact in followup_cell.facts
                for fact_requirement_id in _fact_requirement_ids(fact, requirement)
            }
            missing_fact_ids = expected_fact_ids - satisfied_fact_ids
        if followup.fact_requirement_id not in missing_fact_ids:
            raise ValueError("follow-up fact requirement is not missing")
    return assessment


def repair_evidence_assessment(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    assessment: EvidenceAssessment | None,
) -> EvidenceAssessment:
    """Conservatively repair model-only structural drift without inventing evidence."""
    repaired, _ = repair_evidence_assessment_with_audit(plan, observations, assessment)
    return repaired


def repair_evidence_assessment_with_audit(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    assessment: EvidenceAssessment | None,
) -> tuple[EvidenceAssessment, EvidenceCompilationRepairAudit]:
    """Repair an assessment and report body-free fact retention counts."""
    if assessment is None:
        empty = _empty_assessment(plan, observations)
        return empty, EvidenceCompilationRepairAudit(
            applied=True,
            source_assessment_available=False,
            missing_ledger_cell_count=len(plan.requirements),
            fallback_empty_used=True,
        )

    if plan.task_type == "direct":
        input_fact_count = sum(len(cell.facts) for cell in assessment.ledger)
        repaired = assessment.model_copy(
            update={
                "coverage": (),
                "ledger": (),
                "followups": (),
                "next_requirement_ids": (),
            }
        )
        validated = validate_evidence_assessment(plan, observations, repaired)
        return validated, EvidenceCompilationRepairAudit(
            applied=True,
            source_assessment_available=True,
            input_fact_count=input_fact_count,
            dropped_fact_mapping_count=input_fact_count,
        )

    available_by_requirement = _available_chunks_by_requirement(plan, observations)
    supplied = {item.requirement_id: item for item in assessment.coverage}
    supplied_ledger = {item.requirement_id: item for item in assessment.ledger}
    expected_ids = {item.requirement_id for item in plan.requirements}
    input_fact_count = sum(len(cell.facts) for cell in assessment.ledger)
    dropped_chunk_scope_count = 0
    dropped_fact_mapping_count = sum(
        len(cell.facts)
        for cell in assessment.ledger
        if cell.requirement_id not in expected_ids
    )
    retained_fact_count = 0
    missing_ledger_cell_count = sum(
        item.requirement_id not in supplied_ledger for item in plan.requirements
    )
    coverage: list[EvidenceCoverage] = []
    ledger: list[EvidenceLedgerCell] = []
    for requirement in plan.requirements:
        item = supplied.get(requirement.requirement_id)
        ledger_item = supplied_ledger.get(requirement.requirement_id)
        repaired_facts: list[CompiledEvidenceFact] = []
        for fact in (() if ledger_item is None else ledger_item.facts):
            if not all(
                chunk_id in available_by_requirement[requirement.requirement_id]
                for chunk_id in fact.chunk_ids
            ):
                dropped_chunk_scope_count += 1
                continue
            mapped_ids = _repair_fact_requirement_ids(fact, requirement)
            if not mapped_ids:
                dropped_fact_mapping_count += 1
                continue
            repaired_facts.append(
                CompiledEvidenceFact(
                    fact_id=fact.fact_id,
                    statement=fact.statement,
                    chunk_ids=fact.chunk_ids,
                    fact_requirement_ids=mapped_ids,
                    qualifiers=fact.qualifiers,
                )
            )
            retained_fact_count += 1
        facts = tuple(repaired_facts)
        satisfied_fact_ids = {
            fact_requirement_id
            for fact in facts
            for fact_requirement_id in fact.fact_requirement_ids
        }
        missing_fact_ids = tuple(
            fact_requirement.fact_requirement_id
            for fact_requirement in requirement.fact_requirements
            if fact_requirement.fact_requirement_id not in satisfied_fact_ids
        )
        chunk_ids = tuple(
            dict.fromkeys(chunk_id for fact in facts for chunk_id in fact.chunk_ids)
        )
        if not facts and item is not None and item.covered:
            chunk_ids = ()
        coverage.append(
            EvidenceCoverage(
                requirement_id=requirement.requirement_id,
                covered=bool(chunk_ids),
                chunk_ids=chunk_ids,
            )
        )
        ledger.append(
            EvidenceLedgerCell(
                requirement_id=requirement.requirement_id,
                status=(
                    "missing"
                    if not facts
                    else "partial" if missing_fact_ids else "sufficient"
                ),
                facts=facts,
                missing_fact_requirement_ids=missing_fact_ids,
            )
        )

    missing_ids = tuple(
        item.requirement_id for item in ledger if item.status != "sufficient"
    )
    if not missing_ids:
        repaired = EvidenceAssessment(
            evidence_sufficient=True,
            status="sufficient",
            coverage=tuple(coverage),
            ledger=tuple(ledger),
        )
        validated = validate_evidence_assessment(plan, observations, repaired)
        return validated, EvidenceCompilationRepairAudit(
            applied=True,
            source_assessment_available=True,
            input_fact_count=input_fact_count,
            retained_fact_count=retained_fact_count,
            dropped_chunk_scope_count=dropped_chunk_scope_count,
            dropped_fact_mapping_count=dropped_fact_mapping_count,
            missing_ledger_cell_count=missing_ledger_cell_count,
        )

    requested_ids = tuple(
        requirement_id
        for requirement_id in assessment.next_requirement_ids
        if requirement_id in missing_ids
    )[:1]
    repaired_followups = tuple(
        item
        for item in assessment.followups
        if item.requirement_id in missing_ids
    )
    if repaired_followups:
        requested_ids = ()
    if assessment.next_query is not None and not requested_ids:
        requested_ids = () if repaired_followups else (missing_ids[0],)
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
        ledger=tuple(ledger),
        followups=repaired_followups,
        next_query=assessment.next_query if has_follow_up else None,
        next_objective=assessment.next_objective if has_follow_up else None,
        next_requirement_ids=requested_ids if has_follow_up else (),
    )
    validated = validate_evidence_assessment(plan, observations, repaired)
    return validated, EvidenceCompilationRepairAudit(
        applied=True,
        source_assessment_available=True,
        input_fact_count=input_fact_count,
        retained_fact_count=retained_fact_count,
        dropped_chunk_scope_count=dropped_chunk_scope_count,
        dropped_fact_mapping_count=dropped_fact_mapping_count,
        missing_ledger_cell_count=missing_ledger_cell_count,
    )


def ensure_incomplete_followups(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
    assessment: EvidenceAssessment,
    *,
    remaining_steps: int,
) -> EvidenceAssessment:
    """Deterministically turn uncovered fact intents into bounded local retries."""
    if (
        plan.task_type != "comparison"
        or assessment.evidence_sufficient
        or remaining_steps <= 0
        or assessment.followups
        or assessment.next_query is not None
        or not assessment.ledger
    ):
        return assessment
    requirement_by_id = {item.requirement_id: item for item in plan.requirements}
    target_by_id = {item.target_id: item for item in plan.targets}
    dimension_by_id = {item.dimension_id: item for item in plan.dimensions}
    followups: list[EvidenceFollowup] = []
    for cell in assessment.ledger:
        if cell.status == "sufficient":
            continue
        requirement = requirement_by_id[cell.requirement_id]
        missing_ids = set(cell.missing_fact_requirement_ids)
        if not missing_ids and all(
            item.origin == "derived" for item in requirement.fact_requirements
        ):
            satisfied_ids = {
                fact_requirement_id
                for fact in cell.facts
                for fact_requirement_id in _fact_requirement_ids(fact, requirement)
            }
            missing_ids = {
                item.fact_requirement_id for item in requirement.fact_requirements
            } - satisfied_ids
        missing_fact_requirements = [
            item
            for item in requirement.fact_requirements
            if item.fact_requirement_id in missing_ids
        ]
        target = target_by_id[requirement.target_id]
        dimension = dimension_by_id[requirement.dimension_id]
        missing_facts: tuple[tuple[str | None, str, str], ...]
        if not missing_fact_requirements:
            focus = requirement.description
            missing_facts = (
                (None, f"{target.label} {dimension.label}: {focus}"[:2000], focus),
            )
        else:
            missing_facts = tuple(
                (
                    item.fact_requirement_id,
                    (
                        compose_fact_search_query(target.label, item)
                        if item.protected_anchors
                        else f"{target.label} {dimension.label}: "
                        f"{item.search_query or item.description}"[:2000]
                    ),
                    item.search_query or item.description,
                )
                for item in missing_fact_requirements
            )
        for fact_requirement_id, query, focus in missing_facts:
            followups.append(
                EvidenceFollowup(
                    requirement_id=requirement.requirement_id,
                    fact_requirement_id=fact_requirement_id,
                    query=query,
                    objective=(
                        f"Find missing fact for {target.label} {dimension.label}: {focus}"
                    )[:500],
                )
            )
            if len(followups) >= min(remaining_steps, 4):
                break
        if len(followups) >= min(remaining_steps, 4):
            break
    if not followups:
        return assessment
    return validate_evidence_assessment(
        plan,
        observations,
        assessment.model_copy(update={"followups": tuple(followups)}),
    )


def _available_chunks_by_requirement(
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
) -> dict[str, set[str]]:
    steps = {step.step_id: step for step in plan.steps}
    targets = {target.target_id: target for target in plan.targets}
    observations_by_step = {observation.step_id: observation for observation in observations}
    result: dict[str, set[str]] = {}
    for requirement in plan.requirements:
        corpus_id = targets[requirement.target_id].corpus_id
        eligible_step_ids = {
            step.step_id
            for step in steps.values()
            if requirement.target_id in step.target_ids
        }
        result[requirement.requirement_id] = {
            record.chunk_id
            for step_id in eligible_step_ids
            if (observation := observations_by_step.get(step_id)) is not None
            for record in observation.evidence.records
            if corpus_id is None or record.corpus_id == corpus_id
        }
    return result


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
    ledger = (
        tuple(
            EvidenceLedgerCell(
                requirement_id=requirement.requirement_id,
                status="missing",
                missing_fact_requirement_ids=tuple(
                    item.fact_requirement_id
                    for item in requirement.fact_requirements
                ),
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
        ledger=ledger,
    )


def _fact_requirement_ids(
    fact: CompiledEvidenceFact,
    requirement: EvidenceRequirement,
) -> tuple[str, ...]:
    if fact.fact_requirement_ids:
        return fact.fact_requirement_ids
    fact_requirements = requirement.fact_requirements
    if len(fact_requirements) == 1:
        return (fact_requirements[0].fact_requirement_id,)
    return ()


def _repair_fact_requirement_ids(
    fact: CompiledEvidenceFact,
    requirement: EvidenceRequirement,
) -> tuple[str, ...]:
    expected = {
        item.fact_requirement_id for item in requirement.fact_requirements
    }
    mapped = _fact_requirement_ids(fact, requirement)
    return tuple(item for item in mapped if item in expected)
