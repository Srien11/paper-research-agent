from __future__ import annotations

from paper_research_agent.evaluation.comparison_end_to_end import (
    CitationDiagnostic,
    ComparisonCaseDiagnostic,
    ComparisonEndToEndGold,
    ComparisonGoldCitationRelation,
    ComparisonGoldClaim,
    CompilationAttemptDiagnostic,
    CompilationAuditDiagnostic,
    CompilationRepairDiagnostic,
    RetrievalDiagnostic,
    aggregate_fact_lineage,
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
                outcome="schema_invalid",
                failure_code="schema_ledger_too_long",
                raw_ledger_cell_count=4,
                raw_fact_count=8,
            ),
        ),
        repair=CompilationRepairDiagnostic(
            applied=True,
            source_assessment_available=True,
            input_fact_count=8,
            retained_fact_count=8,
            dropped_chunk_scope_count=0,
            dropped_fact_mapping_count=0,
            missing_ledger_cell_count=0,
            fallback_empty_used=False,
        ),
    )

    case = _case(compilation_audit=audit)

    assert case.compilation_audit == audit
    assert "secret" not in str(case.compilation_audit.model_dump())


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
