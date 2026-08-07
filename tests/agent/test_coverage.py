from __future__ import annotations

import unittest

from paper_research_agent.agent.coverage import (
    repair_evidence_assessment,
    validate_evidence_assessment,
)
from paper_research_agent.agent.models import (
    EvidenceAssessment,
    EvidenceCoverage,
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
            ResearchTarget(target_id="a", label="Paper A"),
            ResearchTarget(target_id="b", label="Paper B"),
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
                target_ids=("a",),
                dimension_ids=("method",),
            ),
            ResearchStep(
                step_id="b",
                objective="Find B",
                query="Paper B method",
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


class EvidenceCoverageValidationTests(unittest.TestCase):
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
        with self.assertRaisesRegex(ValueError, "unknown evidence chunk"):
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


if __name__ == "__main__":
    unittest.main()
