from __future__ import annotations

import unittest

from paper_research_agent.agent.coverage import (
    repair_evidence_assessment,
    repair_evidence_assessment_with_audit,
    validate_evidence_assessment,
)
from paper_research_agent.agent.models import (
    CompiledEvidenceFact,
    EvidenceAssessment,
    EvidenceCoverage,
    EvidenceFactRequirement,
    EvidenceFollowup,
    EvidenceLedgerCell,
    EvidenceRecord,
    EvidenceRequirement,
    GetEvidenceResult,
    ResearchDimension,
    ResearchObservation,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
    SearchCorpusResult,
)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        task_type="comparison",
        targets=(
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        ),
        dimensions=(ResearchDimension(dimension_id="method", label="Method"),),
        requirements=(
            EvidenceRequirement(
                requirement_id="a-method",
                target_id="a",
                dimension_id="method",
                description="Paper A method",
            ),
            EvidenceRequirement(
                requirement_id="b-method",
                target_id="b",
                dimension_id="method",
                description="Paper B method",
            ),
        ),
        steps=(
            ResearchStep(
                step_id="a",
                objective="Find A",
                query="Paper A method",
                corpus_id="C001",
                target_ids=("a",),
                dimension_ids=("method",),
            ),
            ResearchStep(
                step_id="b",
                objective="Find B",
                query="Paper B method",
                corpus_id="T001",
                target_ids=("b",),
                dimension_ids=("method",),
            ),
        ),
    )


def _observation() -> ResearchObservation:
    return ResearchObservation(
        step_id="a",
        objective="Find A",
        search=SearchCorpusResult(
            query="Paper A method",
            corpus_id="C001",
            index_id="idx-test",
            degraded=False,
            hits=(),
        ),
        evidence=GetEvidenceResult(
            records=(
                EvidenceRecord(
                    chunk_id="chunk-a",
                    corpus_id="C001",
                    page_start=1,
                    page_end=1,
                    text="Paper A uses method X.",
                    text_sha256="a" * 64,
                    storage_class="internal_research_only",
                ),
            )
        ),
    )


def _two_dimension_plan() -> ResearchPlan:
    base = _plan()
    return ResearchPlan.model_validate(
        {
            **base.model_dump(mode="json"),
            "dimensions": [
                *[item.model_dump(mode="json") for item in base.dimensions],
                {"dimension_id": "metric", "label": "Metric"},
            ],
            "requirements": [
                *[item.model_dump(mode="json") for item in base.requirements],
                {
                    "requirement_id": "a-metric",
                    "target_id": "a",
                    "dimension_id": "metric",
                    "description": "Paper A metric",
                },
                {
                    "requirement_id": "b-metric",
                    "target_id": "b",
                    "dimension_id": "metric",
                    "description": "Paper B metric",
                },
            ],
            "steps": [
                *[item.model_dump(mode="json") for item in base.steps],
                {
                    "step_id": "a-metric",
                    "objective": "Find A metric",
                    "query": "Paper A metric",
                    "corpus_id": "C001",
                    "target_ids": ["a"],
                    "dimension_ids": ["metric"],
                },
                {
                    "step_id": "b-metric",
                    "objective": "Find B metric",
                    "query": "Paper B metric",
                    "corpus_id": "T001",
                    "target_ids": ["b"],
                    "dimension_ids": ["metric"],
                },
            ],
        }
    )


class EvidenceCoverageValidationTests(unittest.TestCase):
    def test_partial_cell_tracks_unsatisfied_fact_intent_and_allows_followup(self) -> None:
        base = _plan()
        requirements = list(base.requirements)
        requirements[0] = requirements[0].model_copy(
            update={
                "fact_requirements": (
                    EvidenceFactRequirement(
                        fact_requirement_id="a-method-mechanism",
                        description="Core mechanism",
                    ),
                    EvidenceFactRequirement(
                        fact_requirement_id="a-method-input",
                        description="Input dependency",
                    ),
                )
            }
        )
        plan = base.model_copy(update={"requirements": tuple(requirements)})
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(
                EvidenceCoverage(
                    requirement_id="a-method",
                    covered=True,
                    chunk_ids=("chunk-a",),
                ),
                EvidenceCoverage(requirement_id="b-method", covered=False),
            ),
            ledger=(
                EvidenceLedgerCell(
                    requirement_id="a-method",
                    status="partial",
                    facts=(
                        CompiledEvidenceFact(
                            fact_id="a-method-f1",
                            statement="Paper A uses method X.",
                            chunk_ids=("chunk-a",),
                            fact_requirement_ids=("a-method-mechanism",),
                        ),
                    ),
                    missing_fact_requirement_ids=("a-method-input",),
                ),
                EvidenceLedgerCell(
                    requirement_id="b-method",
                    status="missing",
                    missing_fact_requirement_ids=("b-method-primary",),
                ),
            ),
            followups=(
                EvidenceFollowup(
                    requirement_id="a-method",
                    query="Paper A input dependency",
                    objective="Find the missing Paper A input dependency",
                ),
            ),
        )

        self.assertIs(
            validate_evidence_assessment(plan, (_observation(),), assessment),
            assessment,
        )
        with self.assertRaisesRegex(ValueError, "missing coverage"):
            validate_evidence_assessment(
                plan,
                (_observation(),),
                assessment.model_copy(
                    update={"evidence_sufficient": True, "status": "sufficient"}
                ),
            )

    def test_validates_compiled_fact_against_its_requirement_cell(self) -> None:
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(
                EvidenceCoverage(
                    requirement_id="a-method",
                    covered=True,
                    chunk_ids=("chunk-a",),
                ),
                EvidenceCoverage(requirement_id="b-method", covered=False),
            ),
            ledger=(
                EvidenceLedgerCell(
                    requirement_id="a-method",
                    status="sufficient",
                    facts=(
                        CompiledEvidenceFact(
                            fact_id="a-method-f1",
                            statement="Paper A uses method X.",
                            chunk_ids=("chunk-a",),
                        ),
                    ),
                ),
                EvidenceLedgerCell(requirement_id="b-method", status="missing"),
            ),
        )

        self.assertIs(
            validate_evidence_assessment(_plan(), (_observation(),), assessment),
            assessment,
        )
        tampered = assessment.model_copy(
            update={
                "ledger": (
                    assessment.ledger[0].model_copy(
                        update={
                            "facts": (
                                assessment.ledger[0].facts[0].model_copy(
                                    update={"chunk_ids": ("outside",)}
                                ),
                            )
                        }
                    ),
                    assessment.ledger[1],
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "outside its comparison cell"):
            validate_evidence_assessment(_plan(), (_observation(),), tampered)

    def test_accepts_queue_of_distinct_atomic_followups(self) -> None:
        plan = _two_dimension_plan()
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=tuple(
                EvidenceCoverage(requirement_id=item.requirement_id, covered=False)
                for item in plan.requirements
            ),
            followups=(
                EvidenceFollowup(
                    requirement_id="a-method",
                    query="Paper A method retry",
                    objective="Retry Paper A method",
                ),
                EvidenceFollowup(
                    requirement_id="a-metric",
                    query="Paper A metric retry",
                    objective="Retry Paper A metric",
                ),
            ),
        )

        self.assertIs(
            validate_evidence_assessment(plan, (_observation(),), assessment),
            assessment,
        )

    def test_accepts_complete_and_traceable_comparison_coverage(self) -> None:
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(
                EvidenceCoverage(
                    requirement_id="a-method", covered=True, chunk_ids=("chunk-a",)
                ),
                EvidenceCoverage(requirement_id="b-method", covered=False),
            ),
            next_query="Paper B method",
            next_objective="Find Paper B method",
            next_requirement_ids=("b-method",),
        )

        self.assertIs(validate_evidence_assessment(_plan(), (_observation(),), assessment), assessment)

    def test_rejects_missing_requirement_or_unknown_chunk(self) -> None:
        missing_requirement = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(EvidenceCoverage(requirement_id="a-method", covered=False),),
        )
        unknown_chunk = EvidenceAssessment(
            evidence_sufficient=True,
            status="sufficient",
            coverage=(
                EvidenceCoverage(
                    requirement_id="a-method", covered=True, chunk_ids=("chunk-a",)
                ),
                EvidenceCoverage(
                    requirement_id="b-method", covered=True, chunk_ids=("chunk-unknown",)
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_evidence_assessment(_plan(), (_observation(),), missing_requirement)
        with self.assertRaisesRegex(ValueError, "comparison cell"):
            validate_evidence_assessment(_plan(), (_observation(),), unknown_chunk)

    def test_rejects_sufficient_decision_with_missing_coverage(self) -> None:
        assessment = EvidenceAssessment(
            evidence_sufficient=True,
            status="sufficient",
            coverage=(
                EvidenceCoverage(
                    requirement_id="a-method", covered=True, chunk_ids=("chunk-a",)
                ),
                EvidenceCoverage(requirement_id="b-method", covered=False),
            ),
        )

        with self.assertRaisesRegex(ValueError, "cannot be sufficient"):
            validate_evidence_assessment(_plan(), (_observation(),), assessment)

    def test_rejects_chunk_from_the_wrong_target_cell(self) -> None:
        assessment = EvidenceAssessment(
            evidence_sufficient=True,
            status="sufficient",
            coverage=(
                EvidenceCoverage(
                    requirement_id="a-method", covered=True, chunk_ids=("chunk-a",)
                ),
                EvidenceCoverage(
                    requirement_id="b-method", covered=True, chunk_ids=("chunk-a",)
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "comparison cell"):
            validate_evidence_assessment(_plan(), (_observation(),), assessment)

    def test_accepts_same_paper_chunk_from_another_dimension_cell(self) -> None:
        plan = _two_dimension_plan()
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(
                EvidenceCoverage(requirement_id="a-method", covered=False),
                EvidenceCoverage(requirement_id="b-method", covered=False),
                EvidenceCoverage(
                    requirement_id="a-metric", covered=True, chunk_ids=("chunk-a",)
                ),
                EvidenceCoverage(requirement_id="b-metric", covered=False),
            ),
        )

        self.assertIs(
            validate_evidence_assessment(plan, (_observation(),), assessment),
            assessment,
        )

    def test_rejects_follow_up_that_spans_multiple_requirement_cells(self) -> None:
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(
                EvidenceCoverage(requirement_id="a-method", covered=False),
                EvidenceCoverage(requirement_id="b-method", covered=False),
            ),
            next_query="methods in both papers",
            next_objective="Compare methods in both papers",
            next_requirement_ids=("a-method", "b-method"),
        )

        with self.assertRaisesRegex(ValueError, "one requirement cell"):
            validate_evidence_assessment(_plan(), (_observation(),), assessment)

    def test_direct_plan_rejects_comparison_only_fields(self) -> None:
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(EvidenceCoverage(requirement_id="unexpected", covered=False),),
        )
        direct = ResearchPlan(
            steps=(ResearchStep(step_id="one", objective="Find", query="query"),)
        )

        with self.assertRaisesRegex(ValueError, "direct assessment"):
            validate_evidence_assessment(direct, (_observation(),), assessment)

    def test_repair_fills_missing_cells_and_removes_unknown_chunks(self) -> None:
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(
                EvidenceCoverage(
                    requirement_id="a-method",
                    covered=True,
                    chunk_ids=("chunk-unknown",),
                ),
            ),
            next_query="Paper B method",
            next_objective="Find Paper B method",
        )

        repaired = repair_evidence_assessment(
            _plan(),
            (_observation(),),
            assessment,
        )

        self.assertEqual(
            [item.requirement_id for item in repaired.coverage],
            ["a-method", "b-method"],
        )
        self.assertTrue(all(not item.covered for item in repaired.coverage))
        self.assertEqual(repaired.next_requirement_ids, ("a-method",))
        validate_evidence_assessment(_plan(), (_observation(),), repaired)

    def test_repair_audit_distinguishes_empty_fallback_from_fact_drops(self) -> None:
        fallback, fallback_audit = repair_evidence_assessment_with_audit(
            _plan(), (_observation(),), None
        )
        self.assertFalse(fallback.ledger[0].facts)
        self.assertTrue(fallback_audit.fallback_empty_used)
        self.assertFalse(fallback_audit.source_assessment_available)
        self.assertEqual(fallback_audit.missing_ledger_cell_count, 2)

        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            ledger=(
                EvidenceLedgerCell(
                    requirement_id="a-method",
                    status="sufficient",
                    facts=(
                        CompiledEvidenceFact(
                            fact_id="valid",
                            statement="Paper A uses method X.",
                            chunk_ids=("chunk-a",),
                        ),
                        CompiledEvidenceFact(
                            fact_id="outside",
                            statement="Outside evidence.",
                            chunk_ids=("chunk-outside",),
                        ),
                    ),
                ),
            ),
        )
        repaired, audit = repair_evidence_assessment_with_audit(
            _plan(), (_observation(),), assessment
        )

        self.assertEqual([fact.fact_id for fact in repaired.ledger[0].facts], ["valid"])
        self.assertEqual(audit.input_fact_count, 2)
        self.assertEqual(audit.retained_fact_count, 1)
        self.assertEqual(audit.dropped_chunk_scope_count, 1)
        self.assertEqual(audit.dropped_fact_mapping_count, 0)
        self.assertEqual(audit.missing_ledger_cell_count, 1)

    def test_repair_limits_follow_up_to_the_first_requirement_cell(self) -> None:
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(
                EvidenceCoverage(requirement_id="a-method", covered=False),
                EvidenceCoverage(requirement_id="b-method", covered=False),
            ),
            next_query="methods in both papers",
            next_objective="Compare methods in both papers",
            next_requirement_ids=("a-method", "b-method"),
        )

        repaired = repair_evidence_assessment(_plan(), (_observation(),), assessment)

        self.assertEqual(repaired.next_requirement_ids, ("a-method",))
        validate_evidence_assessment(_plan(), (_observation(),), repaired)

    def test_rejects_and_repairs_follow_up_with_multiple_dimensions_for_one_target(self) -> None:
        plan = _two_dimension_plan()
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=tuple(
                EvidenceCoverage(requirement_id=item.requirement_id, covered=False)
                for item in plan.requirements
            ),
            next_query="Paper A method and metric",
            next_objective="Fill Paper A missing evidence",
            next_requirement_ids=("a-method", "a-metric"),
        )

        with self.assertRaisesRegex(ValueError, "one requirement cell"):
            validate_evidence_assessment(plan, (_observation(),), assessment)

        repaired = repair_evidence_assessment(plan, (_observation(),), assessment)

        self.assertEqual(repaired.next_requirement_ids, ("a-method",))
        validate_evidence_assessment(plan, (_observation(),), repaired)


if __name__ == "__main__":
    unittest.main()
