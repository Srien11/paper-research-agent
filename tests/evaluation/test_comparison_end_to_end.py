from __future__ import annotations

import pytest

from paper_research_agent.evaluation.comparison_end_to_end import (
    CitationDiagnostic,
    ComparisonCaseDiagnostic,
    ComparisonEndToEndGold,
    ComparisonGoldCitationRelation,
    ComparisonGoldClaim,
    CompilationAttemptDiagnostic,
    CompilationAuditDiagnostic,
    CompilationRepairDiagnostic,
    PlannerAttemptDiagnostic,
    RetrievalDiagnostic,
    aggregate_compilation_audits,
    aggregate_fact_lineage,
    aggregate_planner_attempts,
    classify_fact_lineage,
    deterministic_rewrite_retention,
    question_sha256,
    score_deterministic_case,
    validate_structural_guarantees,
)


def test_fact_lineage_assigns_each_fact_to_its_earliest_loss_stage() -> None:
    complete = classify_fact_lineage(
        "a",
        gold_chunk_ids=("c1",),
        retrieved_chunk_ids=("c1",),
        hydrated_chunk_ids=("c1",),
        visible_chunk_ids=("c1",),
        compiled_chunk_ids=("c1",),
        in_generation_input=True,
        expressed=True,
        citation_correct=True,
    )
    lost_during_compilation = classify_fact_lineage(
        "b",
        gold_chunk_ids=("t1",),
        retrieved_chunk_ids=("t1",),
        hydrated_chunk_ids=("t1",),
        visible_chunk_ids=("t1",),
        compiled_chunk_ids=(),
        in_generation_input=True,
        expressed=True,
        citation_correct=True,
    )

    summary = aggregate_fact_lineage((complete, lost_during_compilation))

    assert complete.loss_stage == "complete"
    assert lost_during_compilation.loss_stage == "not_compiled"
    assert summary["earliest_loss_counts"]["complete"] == 1
    assert summary["earliest_loss_counts"]["not_compiled"] == 1
    assert summary["stage_rates"]["compiled"] == 0.5


def test_fact_lineage_distinguishes_compiler_visibility_from_compilation() -> None:
    hidden = classify_fact_lineage(
        "hidden",
        gold_chunk_ids=("c1",),
        retrieved_chunk_ids=("c1",),
        hydrated_chunk_ids=("c1",),
        visible_chunk_ids=(),
        compiled_chunk_ids=(),
        in_generation_input=False,
        expressed=False,
        citation_correct=False,
    )

    assert hidden.hydrated is True
    assert hidden.visible_to_compiler is False
    assert hidden.loss_stage == "not_visible_to_compiler"


def test_fact_lineage_splits_exact_chunk_from_semantic_alternative() -> None:
    alternative = classify_fact_lineage(
        "alternative",
        gold_chunk_ids=("gold-c1",),
        retrieved_chunk_ids=("gold-c1", "alt-c2"),
        hydrated_chunk_ids=("gold-c1", "alt-c2"),
        visible_chunk_ids=("gold-c1", "alt-c2"),
        compiled_chunk_ids=("alt-c2",),
        semantic_compiled_chunk_ids=("alt-c2",),
        same_paper_alternative_chunk_ids=("alt-c2",),
        in_generation_input=True,
        expressed=True,
        citation_correct=True,
    )

    summary = aggregate_fact_lineage((alternative,))

    assert alternative.exact_gold_chunk_compiled is False
    assert alternative.same_paper_alternative_chunk_compiled is True
    assert alternative.semantic_fact_compiled is True
    assert alternative.compiled is True
    assert alternative.loss_stage == "complete"
    assert summary["exact_gold_chunk_recall"] == 0
    assert summary["same_paper_alternative_chunk_recall"] == 1
    assert summary["semantic_fact_recall"] == 1


def test_fact_lineage_preserves_one_search_path_ranking_without_bodies() -> None:
    diagnostic = classify_fact_lineage(
        "ranked",
        gold_chunk_ids=("gold-c1",),
        retrieved_chunk_ids=("gold-c1",),
        hydrated_chunk_ids=(),
        compiled_chunk_ids=(),
        in_generation_input=False,
        expressed=False,
        citation_correct=False,
        best_final_rank=6,
        best_stage_ranks={
            "zh.bm25": 9,
            "zh.vector": 4,
            "cross_route_rrf": 5,
            "final": 6,
        },
        search_occurrences=3,
        same_page_top4=True,
        same_section_top4=True,
    )

    assert diagnostic.loss_stage == "not_hydrated"
    assert diagnostic.best_final_rank == 6
    assert diagnostic.search_occurrences == 3
    assert diagnostic.best_stage_ranks["zh.bm25"] == 9
    assert diagnostic.same_page_top4 is True
    assert "text" not in diagnostic.model_dump()


def test_legacy_fact_lineage_remains_readable_without_ranking_fields() -> None:
    current = classify_fact_lineage(
        "legacy",
        gold_chunk_ids=("c1",),
        retrieved_chunk_ids=(),
        hydrated_chunk_ids=(),
        compiled_chunk_ids=(),
        in_generation_input=False,
        expressed=False,
        citation_correct=False,
    )
    legacy = current.model_dump()
    for field in (
        "best_final_rank",
        "best_stage_ranks",
        "search_occurrences",
        "same_page_top4",
        "same_section_top4",
    ):
        legacy.pop(field)

    restored = type(current).model_validate(legacy)

    assert restored.best_final_rank is None
    assert restored.best_stage_ranks == {}
    assert restored.search_occurrences == 0


def test_retrieval_diagnostic_tracks_fact_query_and_reads_legacy_rows() -> None:
    current = RetrievalDiagnostic(
        step_id_hash=question_sha256("a-method"),
        target_ids=("a",),
        dimension_ids=("method",),
        fact_requirement_id="a-reasoning-failure",
        corpus_id_filter="C001",
        search_hit_chunk_ids=("c1",),
        evidence_chunk_ids=("c1",),
    )
    legacy = current.model_dump()
    legacy.pop("fact_requirement_id")

    assert current.fact_requirement_id == "a-reasoning-failure"
    assert RetrievalDiagnostic.model_validate(legacy).fact_requirement_id is None


def _case(**updates: object) -> ComparisonCaseDiagnostic:
    values: dict[str, object] = {
        "question_id": "CPG001",
        "split": "dev",
        "original_question_sha256": question_sha256("raw question"),
        "raw_question_preserved": True,
        "rewrite_status": "success",
        "candidate_paper_ids_top8": ("C001", "T001"),
        "final_paper_ids": ("C001", "T001"),
        "planned_dimensions": ("method",),
        "retrievals": (
            RetrievalDiagnostic(
                step_id_hash=question_sha256("a-method"),
                target_ids=("a",),
                dimension_ids=("method",),
                corpus_id_filter="C001",
                search_hit_chunk_ids=("c1",),
                evidence_chunk_ids=("c1",),
            ),
            RetrievalDiagnostic(
                step_id_hash=question_sha256("b-method"),
                target_ids=("b",),
                dimension_ids=("method",),
                corpus_id_filter="T001",
                search_hit_chunk_ids=("t1",),
                evidence_chunk_ids=("t1",),
            ),
        ),
        "citations": (
            CitationDiagnostic(
                claim_index=0,
                citation_id="E1",
                chunk_id="c1",
                corpus_id="C001",
            ),
        ),
        "total_latency_ms": 1,
    }
    values.update(updates)
    return ComparisonCaseDiagnostic.model_validate(values)


def test_case_diagnostic_persists_body_free_compilation_audit() -> None:
    audit = CompilationAuditDiagnostic(
        attempts=(
            CompilationAttemptDiagnostic(
                attempt=1,
                outcome="contract_invalid",
                failure_code="fact_chunk_scope_invalid",
                raw_ledger_cell_count=2,
                raw_fact_count=2,
                accepted_fact_count=1,
                rejected_fact_count=1,
                unresolved_fact_requirement_count=1,
                requested_requirement_ids=("a-method", "b-method"),
                accepted_requirement_ids=("a-method",),
                failed_requirement_ids=("b-method",),
            ),
            CompilationAttemptDiagnostic(
                attempt=2,
                outcome="validated",
                raw_ledger_cell_count=1,
                raw_fact_count=1,
                accepted_fact_count=1,
                rejected_fact_count=0,
                unresolved_fact_requirement_count=0,
                requested_requirement_ids=("b-method",),
                accepted_requirement_ids=("b-method",),
                failed_requirement_ids=(),
            ),
        ),
        repair=CompilationRepairDiagnostic(
            applied=False,
            source_assessment_available=True,
            input_fact_count=2,
            retained_fact_count=2,
            dropped_chunk_scope_count=0,
            dropped_fact_mapping_count=0,
            missing_ledger_cell_count=0,
            fallback_empty_used=False,
        ),
    )

    case = _case(compilation_audit=audit)

    assert case.compilation_audit == audit
    assert "secret" not in str(case.compilation_audit.model_dump())
    summary = aggregate_compilation_audits((case,))
    assert summary["retry_case_count"] == 1
    assert summary["attempts"] == {
        "attempt_count": 2,
        "requested_unit_count": 3,
        "accepted_unit_count": 2,
        "failed_unit_count": 1,
        "schema_failed_unit_count": 0,
        "contract_failed_unit_count": 1,
        "accepted_fact_count": 2,
        "rejected_fact_count": 1,
        "unresolved_fact_requirement_count": 1,
    }
    assert summary["final"] == {
        "requested_unit_count": 2,
        "accepted_unit_count": 2,
        "failed_unit_count": 0,
        "schema_failed_unit_count": 0,
        "contract_failed_unit_count": 0,
        "unresolved_fact_requirement_count": 0,
        "retained_fact_count": 2,
    }


def test_compilation_audit_keeps_failure_when_retry_still_fails() -> None:
    audit = CompilationAuditDiagnostic(
        attempts=(
            CompilationAttemptDiagnostic(
                attempt=1,
                outcome="contract_invalid",
                failure_code="fact_chunk_scope_invalid",
                requested_requirement_ids=("a-method", "b-method"),
                accepted_requirement_ids=("a-method",),
                failed_requirement_ids=("b-method",),
                unresolved_fact_requirement_count=1,
            ),
            CompilationAttemptDiagnostic(
                attempt=2,
                outcome="contract_invalid",
                failure_code="fact_chunk_scope_invalid",
                requested_requirement_ids=("b-method",),
                accepted_requirement_ids=(),
                failed_requirement_ids=("b-method",),
                unresolved_fact_requirement_count=1,
            ),
        ),
        repair=CompilationRepairDiagnostic(
            applied=False,
            source_assessment_available=True,
            input_fact_count=2,
            retained_fact_count=1,
            dropped_chunk_scope_count=0,
            dropped_fact_mapping_count=0,
            missing_ledger_cell_count=0,
            fallback_empty_used=False,
        ),
    )

    summary = aggregate_compilation_audits((_case(compilation_audit=audit),))

    assert summary["final"]["failed_unit_count"] == 1
    assert summary["final"]["contract_failed_unit_count"] == 1
    assert summary["final"]["unresolved_fact_requirement_count"] == 1


def test_compilation_attempt_requires_disjoint_accepted_and_failed_ids() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        CompilationAttemptDiagnostic(
            attempt=1,
            outcome="validated",
            requested_requirement_ids=("a",),
            accepted_requirement_ids=("a",),
            failed_requirement_ids=("a",),
        )


def test_compilation_attempt_requires_a_complete_requested_partition() -> None:
    with pytest.raises(ValueError, match="partition"):
        CompilationAttemptDiagnostic(
            attempt=1,
            outcome="validated",
            requested_requirement_ids=("a", "b"),
            accepted_requirement_ids=("a",),
            failed_requirement_ids=(),
        )


@pytest.mark.parametrize(
    ("accepted_ids", "failed_ids"),
    [
        (("a", "outside"), ()),
        (("a",), ("outside",)),
    ],
)
def test_compilation_attempt_rejects_ids_outside_requested_partition(
    accepted_ids: tuple[str, ...],
    failed_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="partition"):
        CompilationAttemptDiagnostic(
            attempt=1,
            outcome="contract_invalid",
            failure_code="fact_chunk_scope_invalid",
            requested_requirement_ids=("a",),
            accepted_requirement_ids=accepted_ids,
            failed_requirement_ids=failed_ids,
        )


def test_validated_compilation_attempt_cannot_contain_a_failure() -> None:
    with pytest.raises(ValueError, match="validated"):
        CompilationAttemptDiagnostic(
            attempt=1,
            outcome="validated",
            failure_code="fact_chunk_scope_invalid",
            requested_requirement_ids=("a",),
            accepted_requirement_ids=("a",),
        )


@pytest.mark.parametrize(
    ("failure_code", "failed_ids"),
    [
        ("fact_chunk_scope_invalid", ()),
        (None, ("a",)),
    ],
)
def test_invalid_compilation_attempt_requires_failure_details(
    failure_code: str | None,
    failed_ids: tuple[str, ...],
) -> None:
    accepted_ids = ("a",) if not failed_ids else ()
    with pytest.raises(ValueError, match="invalid"):
        CompilationAttemptDiagnostic(
            attempt=1,
            outcome="contract_invalid",
            failure_code=failure_code,
            requested_requirement_ids=("a",),
            accepted_requirement_ids=accepted_ids,
            failed_requirement_ids=failed_ids,
        )


def test_case_diagnostic_persists_and_aggregates_body_free_planner_attempts() -> None:
    attempts = (
        PlannerAttemptDiagnostic(
            attempt=1,
            outcome="contract_invalid",
            failure_code="planner_grid_incomplete",
        ),
        PlannerAttemptDiagnostic(attempt=2, outcome="validated"),
    )

    case = _case(
        planner_attempts=attempts,
        planner_fallback_reason="fact_proposal_repair_exhausted",
        error_type="ComparisonPlanningError",
        error_reason_code="planner_fact_proposal_invalid",
    )
    target_failure = _case(
        planner_attempts=(),
        error_type="ComparisonTargetResolutionError",
        error_reason_code="insufficient_retrieval_candidates",
    )
    serialized = case.model_dump(mode="json")

    assert serialized["planner_attempts"] == [
        {
            "attempt": 1,
            "outcome": "contract_invalid",
            "failure_code": "planner_grid_incomplete",
        },
        {"attempt": 2, "outcome": "validated", "failure_code": None},
    ]
    for forbidden in ("question", "query", "description", "statement", "evidence"):
        assert forbidden not in str(serialized["planner_attempts"])
    summary = aggregate_planner_attempts((case,))
    assert summary["attempt_count"] == 2
    assert summary["failure_code_counts"] == {"planner_grid_incomplete": 1}
    assert summary["attempt_count_distribution"] == {"2": 1}
    assert summary["comparison_planning_failure_count"] == 1
    assert summary["target_resolution_failure_count"] == 0
    assert summary["fallback_count"] == 1
    assert summary["fallback_reason_counts"] == {
        "fact_proposal_repair_exhausted": 1
    }

    combined = aggregate_planner_attempts((case, target_failure))
    assert combined["attempt_count_distribution"] == {"0": 1, "2": 1}
    assert combined["comparison_planning_failure_count"] == 1
    assert combined["target_resolution_failure_count"] == 1


def test_rewrite_retention_preserves_identifiers_numbers_and_metrics() -> None:
    retained, required, matched = deterministic_rewrite_retention(
        "比较 C001 与 T001 在 MMLU、2024 年和 91.5% 指标上的结果",
        "Compare C001 and T001 on MMLU in 2024 using the 91.5% result.",
    )

    assert retained is True
    assert required == matched == 4


def test_same_semantic_query_in_different_corpora_is_not_duplicate() -> None:
    failures = validate_structural_guarantees(
        _case(), chunk_corpus_ids={"c1": "C001", "t1": "T001"}
    )

    assert failures == ()


def test_rejects_final_target_outside_candidates_and_cross_paper_citation() -> None:
    case = _case(
        final_paper_ids=("C001", "C999"),
        citations=(
            CitationDiagnostic(
                claim_index=0,
                citation_id="E1",
                chunk_id="t1",
                corpus_id="C001",
            ),
        ),
    )

    failures = validate_structural_guarantees(
        case, chunk_corpus_ids={"c1": "C001", "t1": "T001"}
    )

    assert "final_target_outside_candidates" in failures
    assert "citation_corpus_mismatch" in failures


def test_deterministic_answer_score_uses_gold_chunks_and_corpus_boundaries() -> None:
    gold = ComparisonEndToEndGold(
        question_id="CPG001",
        split="dev",
        relevant_paper_ids=("C001", "T001"),
        expected_dimensions=("method",),
        must_have_claims=(
            ComparisonGoldClaim(
                claim_id="a", corpus_id="C001", normalized_fact="fact a"
            ),
            ComparisonGoldClaim(
                claim_id="b", corpus_id="T001", normalized_fact="fact b"
            ),
        ),
        evidence_chunk_ids=("c1", "t1"),
        citation_relations=(
            ComparisonGoldCitationRelation(claim_id="a", chunk_ids=("c1",)),
            ComparisonGoldCitationRelation(claim_id="b", chunk_ids=("t1",)),
        ),
    )

    score = score_deterministic_case(
        _case(), gold, chunk_corpus_ids={"c1": "C001", "t1": "T001"}
    )

    assert score.final_target_correct is True
    assert score.evidence_recall_at_k == 1
    assert score.evidence_corpus_purity == 1
    assert score.citation_gold_chunk_rate == 1
