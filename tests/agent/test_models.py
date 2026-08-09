from __future__ import annotations

import unittest

from pydantic import ValidationError

from paper_research_agent.agent.models import (
    CompiledEvidenceFact,
    EvidenceAssessment,
    EvidenceCompilationAttemptAudit,
    EvidenceCompilationAudit,
    EvidenceCompilationRepairAudit,
    EvidenceCompilationVisibility,
    EvidenceCoverage,
    EvidenceFactRequirement,
    EvidenceLedgerCell,
    EvidenceQualifier,
    EvidenceRecord,
    EvidenceRequirement,
    GetEvidenceInput,
    GetEvidenceResult,
    ResearchActionRecord,
    ResearchDimension,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
    SearchCorpusHit,
    SearchCorpusInput,
    SearchCorpusResult,
)


def _hit(*, rank: int = 1) -> SearchCorpusHit:
    return SearchCorpusHit(
        chunk_id=f"chunk-{rank}",
        corpus_id="C001",
        section_id="results",
        page_start=2,
        page_end=3,
        text_sha256=f"{rank}" * 64,
        evidence_type="text",
        storage_class="internal_research_only",
        final_rank=rank,
    )


class ResearchToolModelTests(unittest.TestCase):
    def test_compilation_audit_is_persisted_but_excluded_from_provider_schema(self) -> None:
        audit = EvidenceCompilationAudit(
            attempts=(
                EvidenceCompilationAttemptAudit(attempt=1, outcome="validated"),
            ),
            repair=EvidenceCompilationRepairAudit(
                applied=False,
                source_assessment_available=True,
            ),
        )
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            compilation_audit=audit,
        )

        self.assertEqual(
            assessment.model_dump()["compilation_audit"]["attempts"][0]["outcome"],
            "validated",
        )
        self.assertNotIn(
            "compilation_audit", str(EvidenceAssessment.model_json_schema())
        )

    def test_fact_requirements_support_partial_ledger_coverage(self) -> None:
        requirement = EvidenceRequirement(
            requirement_id="a-method",
            target_id="a",
            dimension_id="method",
            description="Paper A method and input dependency",
            fact_requirements=(
                EvidenceFactRequirement(
                    fact_requirement_id="a-method-mechanism",
                    description="Core mechanism",
                ),
                EvidenceFactRequirement(
                    fact_requirement_id="a-method-input",
                    description="Input dependency",
                    required_qualifier_kinds=("method",),
                ),
            ),
        )
        fact = CompiledEvidenceFact(
            fact_id="a-method-f1",
            statement="Paper A uses retrieval augmentation.",
            chunk_ids=("chunk-1",),
            fact_requirement_ids=("a-method-mechanism",),
        )
        cell = EvidenceLedgerCell(
            requirement_id=requirement.requirement_id,
            status="partial",
            facts=(fact,),
            missing_fact_requirement_ids=("a-method-input",),
        )
        visibility = EvidenceCompilationVisibility(
            requirement_id=requirement.requirement_id,
            available_chunk_ids=("chunk-1", "chunk-2"),
            visible_chunk_ids=("chunk-1",),
            truncated_chunk_ids=("chunk-1",),
        )

        self.assertEqual(cell.status, "partial")
        self.assertEqual(cell.missing_fact_requirement_ids, ("a-method-input",))
        self.assertEqual(visibility.visible_chunk_ids, ("chunk-1",))
        with self.assertRaisesRegex(ValidationError, "partial.*missing"):
            EvidenceLedgerCell(
                requirement_id=requirement.requirement_id,
                status="partial",
                facts=(fact,),
            )
        with self.assertRaisesRegex(ValidationError, "sufficient.*missing"):
            EvidenceLedgerCell(
                requirement_id=requirement.requirement_id,
                status="sufficient",
                facts=(fact,),
                missing_fact_requirement_ids=("a-method-input",),
            )

    def test_compiled_evidence_ledger_requires_atomic_traceable_facts(self) -> None:
        fact = CompiledEvidenceFact(
            fact_id="a-method-f1",
            statement="论文 A 使用检索增强生成方法。",
            chunk_ids=("chunk-1",),
            qualifiers=(EvidenceQualifier(kind="method", value="RAG"),),
        )
        cell = EvidenceLedgerCell(
            requirement_id="a-method",
            status="sufficient",
            facts=(fact,),
        )

        self.assertEqual(cell.facts[0].qualifiers[0].value, "RAG")
        with self.assertRaisesRegex(ValidationError, "requires facts"):
            EvidenceLedgerCell(requirement_id="a-method", status="sufficient")
        with self.assertRaisesRegex(ValidationError, "cannot contain facts"):
            EvidenceLedgerCell(
                requirement_id="a-method",
                status="missing",
                facts=(fact,),
            )
        with self.assertRaisesRegex(ValidationError, "globally unique"):
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                ledger=(
                    cell,
                    EvidenceLedgerCell(
                        requirement_id="b-method",
                        status="sufficient",
                        facts=(fact,),
                    ),
                ),
            )

    def test_comparison_plan_accepts_maximum_twenty_cell_initial_grid(self) -> None:
        targets = tuple(
            ResearchTarget(
                target_id=f"paper{index}",
                label=f"Paper {index}",
                corpus_id=f"C{index:03d}",
            )
            for index in range(1, 5)
        )
        dimensions = tuple(
            ResearchDimension(dimension_id=f"dim{index}", label=f"Dimension {index}")
            for index in range(1, 6)
        )
        requirements = tuple(
            EvidenceRequirement(
                requirement_id=f"{target.target_id}-{dimension.dimension_id}",
                target_id=target.target_id,
                dimension_id=dimension.dimension_id,
                description=f"{target.label} {dimension.label}",
            )
            for target in targets
            for dimension in dimensions
        )

        plan = ResearchPlan(
            task_type="comparison",
            targets=targets,
            dimensions=dimensions,
            requirements=requirements,
            steps=tuple(
                ResearchStep(
                    step_id=item.requirement_id,
                    objective=item.description,
                    query=f"query {item.requirement_id}",
                    corpus_id=next(
                        target.corpus_id
                        for target in targets
                        if target.target_id == item.target_id
                    ),
                    target_ids=(item.target_id,),
                    dimension_ids=(item.dimension_id,),
                )
                for item in requirements
            ),
        )

        self.assertEqual(len(plan.steps), 20)

    def test_scoped_search_result_rejects_cross_corpus_hits(self) -> None:
        hit = SearchCorpusHit(
            chunk_id="chunk-1",
            corpus_id="T001",
            page_start=1,
            page_end=1,
            text_sha256="a" * 64,
            storage_class="redistributable",
            final_rank=1,
        )

        with self.assertRaisesRegex(ValueError, "corpus scope"):
            SearchCorpusResult(
                query="method",
                corpus_id="C001",
                index_id="idx",
                degraded=False,
                hits=(hit,),
            )

    def test_search_input_normalizes_query_and_bounds_top_k(self) -> None:
        request = SearchCorpusInput(query="  grounded RAG  ", top_k=3)

        self.assertEqual(request.query, "grounded RAG")
        self.assertEqual(request.top_k, 3)
        with self.assertRaises(ValidationError):
            SearchCorpusInput(query=" ")
        with self.assertRaises(ValidationError):
            SearchCorpusInput(query="q", top_k=21)
        with self.assertRaises(ValidationError):
            SearchCorpusInput(query="q", unexpected=True)

    def test_search_result_requires_contiguous_ranks_and_degradation_reason(self) -> None:
        result = SearchCorpusResult(
            query="grounded RAG",
            index_id="idx-test",
            degraded=False,
            hits=(_hit(),),
        )

        self.assertEqual(result.hits[0].storage_class, "internal_research_only")
        with self.assertRaises(ValidationError):
            SearchCorpusResult(
                query="grounded RAG",
                index_id="idx-test",
                degraded=False,
                hits=(_hit(rank=2),),
            )
        with self.assertRaises(ValidationError):
            SearchCorpusResult(
                query="grounded RAG",
                index_id="idx-test",
                degraded=True,
                hits=(),
            )

    def test_get_evidence_rejects_duplicate_ids_and_result_overlap(self) -> None:
        with self.assertRaises(ValidationError):
            GetEvidenceInput(chunk_ids=("chunk-1", "chunk-1"))

        record = EvidenceRecord(
            chunk_id="chunk-1",
            corpus_id="C001",
            section_id=None,
            page_start=1,
            page_end=1,
            text="Evidence text.",
            text_sha256="a" * 64,
            evidence_type="text",
            storage_class="redistributable",
        )
        result = GetEvidenceResult(records=(record,), missing_chunk_ids=("missing",))
        self.assertEqual(result.records[0].text, "Evidence text.")
        with self.assertRaises(ValidationError):
            GetEvidenceResult(records=(record,), missing_chunk_ids=("chunk-1",))

    def test_evidence_assessment_requires_consistent_next_search(self) -> None:
        sufficient = EvidenceAssessment(
            evidence_sufficient=True,
            status="sufficient",
        )

        self.assertIsNone(sufficient.next_query)
        with self.assertRaises(ValidationError):
            EvidenceAssessment(
                evidence_sufficient=True,
                status="sufficient",
                next_query="another query",
                next_objective="find more evidence",
            )
        with self.assertRaises(ValidationError):
            EvidenceAssessment(
                evidence_sufficient=False,
                status="missing_coverage",
                next_query="missing comparison",
            )

        retry = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            next_query="  missing comparison  ",
            next_objective="  cover the missing dimension  ",
        )
        self.assertEqual(retry.next_query, "missing comparison")

    def test_comparison_plan_requires_complete_target_dimension_grid(self) -> None:
        targets = (
            ResearchTarget(target_id="ragas", label="RAGAS", corpus_id="C001"),
            ResearchTarget(target_id="ares", label="ARES", corpus_id="T001"),
        )
        dimensions = (
            ResearchDimension(dimension_id="method", label="评测方法"),
            ResearchDimension(dimension_id="metrics", label="评测指标"),
        )
        requirements = tuple(
            EvidenceRequirement(
                requirement_id=f"{target.target_id}-{dimension.dimension_id}",
                target_id=target.target_id,
                dimension_id=dimension.dimension_id,
                description=f"查明 {target.label} 的{dimension.label}",
            )
            for target in targets
            for dimension in dimensions
        )

        plan = ResearchPlan(
            task_type="comparison",
            targets=targets,
            dimensions=dimensions,
            requirements=requirements,
            steps=(
                ResearchStep(
                    step_id="ragas-method",
                    objective="检索 RAGAS 方法",
                    query="RAGAS evaluation method",
                    corpus_id="C001",
                    target_ids=("ragas",),
                    dimension_ids=("method",),
                ),
                ResearchStep(
                    step_id="ragas-metrics",
                    objective="检索 RAGAS 指标",
                    query="RAGAS evaluation metrics",
                    corpus_id="C001",
                    target_ids=("ragas",),
                    dimension_ids=("metrics",),
                ),
                ResearchStep(
                    step_id="ares-method",
                    objective="检索 ARES 方法",
                    query="ARES evaluation method",
                    corpus_id="T001",
                    target_ids=("ares",),
                    dimension_ids=("method",),
                ),
                ResearchStep(
                    step_id="ares-metrics",
                    objective="检索 ARES 指标",
                    query="ARES evaluation metrics",
                    corpus_id="T001",
                    target_ids=("ares",),
                    dimension_ids=("metrics",),
                ),
            ),
        )

        self.assertEqual(len(plan.requirements), 4)
        with self.assertRaisesRegex(ValidationError, "complete target-dimension grid"):
            ResearchPlan(
                task_type="comparison",
                targets=targets,
                dimensions=dimensions,
                requirements=requirements[:-1],
                steps=plan.steps,
            )

    def test_comparison_plan_rejects_one_query_for_multiple_dimensions(self) -> None:
        with self.assertRaisesRegex(ValidationError, "one target and one dimension"):
            ResearchPlan(
                task_type="comparison",
                targets=(
                    ResearchTarget(target_id="a", label="A", corpus_id="C001"),
                    ResearchTarget(target_id="b", label="B", corpus_id="T001"),
                ),
                dimensions=(
                    ResearchDimension(dimension_id="method", label="方法"),
                    ResearchDimension(dimension_id="metric", label="指标"),
                ),
                requirements=tuple(
                    EvidenceRequirement(
                        requirement_id=f"{target}-{dimension}",
                        target_id=target,
                        dimension_id=dimension,
                        description=f"{target} {dimension}",
                    )
                    for target in ("a", "b")
                    for dimension in ("method", "metric")
                ),
                steps=(
                    ResearchStep(
                        step_id="a-all",
                        objective="A all",
                        query="A method metric",
                        corpus_id="C001",
                        target_ids=("a",),
                        dimension_ids=("method", "metric"),
                    ),
                    ResearchStep(
                        step_id="b-all",
                        objective="B all",
                        query="B method metric",
                        corpus_id="T001",
                        target_ids=("b",),
                        dimension_ids=("method", "metric"),
                    ),
                ),
            )

    def test_comparison_plan_rejects_unplanned_requirement_cells(self) -> None:
        with self.assertRaisesRegex(ValidationError, "research steps do not cover"):
            ResearchPlan(
                task_type="comparison",
                targets=(
                    ResearchTarget(target_id="a", label="A", corpus_id="C001"),
                    ResearchTarget(target_id="b", label="B", corpus_id="T001"),
                ),
                dimensions=(ResearchDimension(dimension_id="method", label="方法"),),
                requirements=(
                    EvidenceRequirement(
                        requirement_id="a-method",
                        target_id="a",
                        dimension_id="method",
                        description="A 的方法",
                    ),
                    EvidenceRequirement(
                        requirement_id="b-method",
                        target_id="b",
                        dimension_id="method",
                        description="B 的方法",
                    ),
                ),
                steps=(
                    ResearchStep(
                        step_id="a",
                        objective="检索 A",
                        query="A method",
                        corpus_id="C001",
                        target_ids=("a",),
                        dimension_ids=("method",),
                    ),
                ),
            )

    def test_evidence_coverage_binds_covered_cells_to_chunks(self) -> None:
        covered = EvidenceCoverage(
            requirement_id="a-method",
            covered=True,
            chunk_ids=("chunk-1",),
        )
        missing = EvidenceCoverage(requirement_id="b-method", covered=False)
        assessment = EvidenceAssessment(
            evidence_sufficient=False,
            status="missing_coverage",
            coverage=(covered, missing),
            next_query="B method",
            next_objective="补齐 B 的方法",
            next_requirement_ids=("b-method",),
        )

        self.assertEqual(assessment.next_requirement_ids, ("b-method",))
        with self.assertRaisesRegex(ValidationError, "covered evidence requires chunk IDs"):
            EvidenceCoverage(requirement_id="a-method", covered=True)

    def test_action_record_enforces_fields_for_each_action(self) -> None:
        search = ResearchActionRecord(
            sequence=1,
            action="search_corpus",
            step_id="methods",
            query="RAG evaluation",
        )
        evidence = ResearchActionRecord(
            sequence=2,
            action="get_evidence",
            step_id="methods",
            chunk_ids=("chunk-1",),
        )
        assessment = ResearchActionRecord(
            sequence=3,
            action="assess_evidence",
            step_id="methods",
            outcome="missing_coverage",
        )
        finish = ResearchActionRecord(
            sequence=4,
            action="finish",
            outcome="plan_exhausted",
        )

        self.assertEqual(
            [search.action, evidence.action, assessment.action, finish.action],
            ["search_corpus", "get_evidence", "assess_evidence", "finish"],
        )
        with self.assertRaises(ValidationError):
            ResearchActionRecord(
                sequence=1,
                action="search_corpus",
                step_id="methods",
            )
        with self.assertRaises(ValidationError):
            ResearchActionRecord(
                sequence=1,
                action="get_evidence",
                step_id="methods",
                chunk_ids=(),
            )
        with self.assertRaises(ValidationError):
            ResearchActionRecord(
                sequence=1,
                action="finish",
                outcome="sufficient",
            )


if __name__ == "__main__":
    unittest.main()
