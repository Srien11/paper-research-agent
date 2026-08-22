from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from paper_research_agent.agent.models import (
    EvidenceRecord,
    EvidenceRequirement,
    GetEvidenceResult,
    PlannerAttemptAudit,
    ResearchDimension,
    ResearchObservation,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
    SearchCorpusResult,
)
from paper_research_agent.agent.planner import (
    ComparisonPlanningError,
    ComparisonTargetResolutionError,
)
from paper_research_agent.retrieval.contracts import (
    BilingualRetrievalRun,
    QueryRewriteTrace,
    SearchHit,
)
from scripts.evaluate_comparison_end_to_end import (
    _build_parser,
    _citation_correct,
    _diagnostic,
    _experiment_metadata,
    _fact_ranking,
    _load_resumed_cases,
    _QueryRecorder,
)


def test_exact_gold_citation_is_not_overridden_by_model_judge_variance() -> None:
    assert _citation_correct(
        exact_gold_citation=True,
        semantic_citation_supported=False,
    )
    assert _citation_correct(
        exact_gold_citation=False,
        semantic_citation_supported=True,
    )
    assert not _citation_correct(
        exact_gold_citation=False,
        semantic_citation_supported=False,
    )


def test_diagnostic_copies_safe_fact_requirement_id_from_executed_step() -> None:
    plan = ResearchPlan(
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
                step_id="fact-a",
                objective="Find Paper A method",
                query="safe query",
                corpus_id="C001",
                target_ids=("a",),
                dimension_ids=("method",),
                fact_requirement_id="a-method-primary",
            ),
            ResearchStep(
                step_id="b-method",
                objective="Find Paper B method",
                query="Paper B method",
                corpus_id="T001",
                target_ids=("b",),
                dimension_ids=("method",),
            ),
        ),
    )
    observation = ResearchObservation(
        step_id="fact-a",
        objective="Find Paper A method",
        search=SearchCorpusResult(
            query="safe query",
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
                    text="safe evidence",
                    text_sha256="a" * 64,
                    storage_class="internal_research_only",
                ),
            )
        ),
    )
    research = SimpleNamespace(
        plan=plan,
        observations=(observation,),
        assessments=(),
        step_budget=4,
        tool_call_count=2,
        tool_call_budget=8,
    )

    diagnostic = _diagnostic(
        SimpleNamespace(question_id="CPG001", split="dev", question="safe question"),
        query_call=None,
        candidate_call=None,
        research=research,
        answer=None,
        generation=None,
        elapsed_ms=1.0,
        error=None,
    )

    assert diagnostic.retrievals[0].fact_requirement_id == "a-method-primary"


def test_diagnostic_copies_planner_attempts_from_success_and_failure() -> None:
    success_attempts = (
        PlannerAttemptAudit(
            attempt=1,
            outcome="contract_invalid",
            failure_code="planner_grid_incomplete",
        ),
        PlannerAttemptAudit(attempt=2, outcome="validated"),
    )
    research = SimpleNamespace(
        plan=ResearchPlan(
            steps=(ResearchStep(step_id="one", objective="One", query="one"),),
            planner_attempts=success_attempts,
        ),
        observations=(),
        assessments=(),
        step_budget=1,
        tool_call_count=0,
        tool_call_budget=1,
    )
    question = SimpleNamespace(
        question_id="CPG001", split="dev", question="private question body"
    )

    success = _diagnostic(
        question,
        query_call=None,
        candidate_call=None,
        research=research,
        answer=None,
        generation=None,
        elapsed_ms=1.0,
        error=None,
    )
    failure_error = ComparisonPlanningError(
        "planner_fact_proposal_invalid",
        attempts=(
            PlannerAttemptAudit(
                attempt=1,
                outcome="contract_invalid",
                failure_code="planner_fact_proposal_invalid",
            ),
            PlannerAttemptAudit(
                attempt=2,
                outcome="contract_invalid",
                failure_code="planner_fact_proposal_invalid",
            ),
        ),
    )
    failure = _diagnostic(
        question,
        query_call=None,
        candidate_call=None,
        research=None,
        answer=None,
        generation=None,
        elapsed_ms=1.0,
        error=failure_error,
    )

    assert [item.outcome for item in success.planner_attempts] == [
        "contract_invalid",
        "validated",
    ]
    assert [item.failure_code for item in failure.planner_attempts] == [
        "planner_fact_proposal_invalid",
        "planner_fact_proposal_invalid",
    ]
    serialized = failure.model_dump(mode="json")
    assert serialized["error_reason_code"] == "planner_fact_proposal_invalid"
    for forbidden in ("private question body", "model-authored fact", "raw query"):
        assert forbidden not in str(serialized)

    target_failure = _diagnostic(
        question,
        query_call=None,
        candidate_call=None,
        research=None,
        answer=None,
        generation=None,
        elapsed_ms=1.0,
        error=ComparisonTargetResolutionError("unknown_explicit_corpus_id"),
    )
    assert target_failure.error_reason_code == "unknown_explicit_corpus_id"


def _hit(
    chunk_id: str,
    rank: int,
    *,
    page: int,
    section: str,
    bm25_rank: int | None = None,
    vector_rank: int | None = None,
    rrf_rank: int | None = None,
) -> SearchHit:
    ranks = {"final": rank}
    if bm25_rank is not None:
        ranks["zh.bm25"] = bm25_rank
    if vector_rank is not None:
        ranks["zh.vector"] = vector_rank
    if rrf_rank is not None:
        ranks["cross_route_rrf"] = rrf_rank
    return SearchHit(
        chunk_id=chunk_id,
        corpus_id="C001",
        asset_id="asset-1",
        section_id=section,
        page_start=page,
        page_end=page,
        text_sha256=hashlib.sha256(chunk_id.encode()).hexdigest(),
        ranks=ranks,
        final_score=float(100 - rank),
        final_rank=rank,
    )


def _run(gold_rank: int, *, stage_ranks: tuple[int, int, int]) -> BilingualRetrievalRun:
    hits = [
        _hit(f"competitor-{index}", index, page=2, section="methods")
        for index in range(1, gold_rank)
    ]
    hits.append(
        _hit(
            "gold",
            gold_rank,
            page=2,
            section="methods",
            bm25_rank=stage_ranks[0],
            vector_rank=stage_ranks[1],
            rrf_rank=stage_ranks[2],
        )
    )
    return BilingualRetrievalRun(
        pipeline_id="pipeline-v1",
        original_query="safe query",
        rewrite=QueryRewriteTrace(
            status="success",
            english_query="safe query",
            requested_model="model",
            actual_model="model",
            prompt_version="v1",
            latency_ms=1,
        ),
        degraded=False,
        top_k=20,
        hits=tuple(hits),
        index_id="index-v1",
        config_sha256="0" * 64,
        storage_classes={"C001": "internal_research_only"},
        rights_status="loaded",
    )


def test_fact_ranking_uses_stages_from_the_best_final_rank_occurrence() -> None:
    runs = (
        _run(9, stage_ranks=(11, 8, 8)),
        _run(6, stage_ranks=(9, 4, 5)),
        _run(8, stage_ranks=(7, 6, 7)),
    )

    ranking = _fact_ranking(("gold",), runs)

    assert ranking["best_final_rank"] == 6
    assert ranking["search_occurrences"] == 3
    assert ranking["best_stage_ranks"] == {
        "final": 6,
        "zh.bm25": 9,
        "zh.vector": 4,
        "cross_route_rrf": 5,
    }
    assert ranking["same_page_top4"] is True
    assert ranking["same_section_top4"] is True


class _Retriever:
    def __init__(self, run: BilingualRetrievalRun) -> None:
        self.run = run
        self.kwargs: dict[str, object] = {}

    async def search(self, *args: object, **kwargs: object) -> BilingualRetrievalRun:
        del args
        self.kwargs = kwargs
        return self.run


class _Config:
    def __init__(self, name: str) -> None:
        self.name = name

    def model_dump(self, *, mode: str) -> dict[str, str]:
        assert mode == "json"
        return {"name": self.name}


def test_query_recorder_retains_private_runs_only_in_memory() -> None:
    run = _run(6, stage_ranks=(9, 4, 5))
    recorder = _QueryRecorder(_Retriever(run))

    returned = asyncio.run(recorder.search("safe query", top_k=10))

    assert returned is run
    assert recorder.retrieval_runs == [run]


def test_query_recorder_can_force_reranking_off_for_ablation() -> None:
    run = _run(6, stage_ranks=(9, 4, 5))
    inner = _Retriever(run)
    recorder = _QueryRecorder(inner, rerank_mode="off")

    asyncio.run(recorder.search("safe query", top_k=10, rerank=True))

    assert inner.kwargs["rerank"] is False


def test_parser_accepts_reproducible_hydration_and_rerank_variants() -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "--evidence-per-step",
            "10",
            "--rerank-mode",
            "off",
            "--question-id",
            "CPG001",
            "--question-id",
            "CPG020",
        ]
    )

    assert args.evidence_per_step == 10
    assert args.rerank_mode == "off"
    assert args.question_ids == ["CPG001", "CPG020"]
    with pytest.raises(SystemExit):
        parser.parse_args(["--evidence-per-step", "21"])


def test_resume_rejects_a_different_experiment_fingerprint() -> None:
    with TemporaryDirectory() as directory:
        output = Path(directory) / "run.json"
        output.write_text(
            json.dumps({"experiment_fingerprint": "a" * 64, "cases": []}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="fingerprint"):
            _load_resumed_cases(
                output,
                experiment_fingerprint="b" * 64,
                selected_ids={"CPG001"},
            )


def test_experiment_metadata_identifies_cutoff_reranking_config_and_checkpoint() -> None:
    retriever = _Retriever(_run(6, stage_ranks=(9, 4, 5)))
    retriever.retrieval_config = _Config("retrieval")
    retriever.bilingual_config = _Config("bilingual")
    with TemporaryDirectory() as directory:
        args = argparse.Namespace(
            evidence_per_step=6,
            rerank_mode="off",
            checkpoint=Path(directory) / "checkpoint.sqlite3",
        )

        metadata = _experiment_metadata(args, retriever)

    assert metadata["evidence_per_step"] == 6
    assert metadata["rerank_mode"] == "off"
    assert len(str(metadata["code_revision"])) >= 7
    assert len(str(metadata["retrieval_config_sha256"])) == 64
    assert len(str(metadata["checkpoint_id"])) == 64
    assert len(str(metadata["experiment_fingerprint"])) == 64
