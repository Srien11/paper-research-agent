from __future__ import annotations

from paper_research_agent.evaluation.comparison_end_to_end import (
    ComparisonCaseDiagnostic,
    FactLineageDiagnostic,
    question_sha256,
)
from scripts.summarize_hydration_ablation import render_markdown, summarize_run_payload


def test_summarizer_outputs_only_safe_aggregate_hydration_metrics() -> None:
    lineage = FactLineageDiagnostic(
        claim_id="F1",
        retrieved=True,
        hydrated=False,
        visible_to_compiler=False,
        exact_gold_chunk_compiled=False,
        same_paper_alternative_chunk_compiled=False,
        semantic_fact_compiled=False,
        compiled=False,
        in_generation_input=False,
        expressed=False,
        citation_correct=False,
        loss_stage="not_hydrated",
        best_final_rank=6,
        best_stage_ranks={"cross_route_rrf": 5, "final": 6},
        search_occurrences=2,
        same_page_top4=True,
    )
    case = ComparisonCaseDiagnostic(
        question_id="CPG001",
        split="dev",
        original_question_sha256=question_sha256("private question"),
        raw_question_preserved=True,
        rewrite_status="success",
        candidate_paper_ids_top8=("C001",),
        final_paper_ids=("C001",),
        planned_dimensions=("method",),
        retrievals=(),
        tool_call_count=4,
        generation_input_tokens=120,
        citations=(),
        fact_lineage=(lineage,),
        total_latency_ms=50,
    )
    payload = {
        "experiment_fingerprint": "a" * 64,
        "code_revision": "b" * 40,
        "retrieval_config_sha256": "c" * 64,
        "checkpoint_id": "d" * 64,
        "evidence_per_step": 4,
        "rerank_mode": "current",
        "cases": [case.model_dump(mode="json")],
        "provider_payload": "must never be copied",
    }

    summary = summarize_run_payload(payload)
    markdown = render_markdown((summary,))

    assert summary["hydration"]["remaining_by_cutoff"] == {4: 1, 6: 0, 8: 0, 10: 0}
    assert summary["mean_generation_input_tokens"] == 120
    assert "must never be copied" not in markdown
    assert "private question" not in markdown
    assert "Top 10" in markdown
