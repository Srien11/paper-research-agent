from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.agent.policy import ResearchRuntimePolicy
from paper_research_agent.answering.config import load_answering_config
from paper_research_agent.evaluation.candidate_gold import load_candidate_paper_gold
from paper_research_agent.evaluation.comparison_end_to_end import (
    CitationDiagnostic,
    ComparisonCaseDiagnostic,
    ComparisonEndToEndGold,
    ComparisonModelJudgeScore,
    RetrievalDiagnostic,
    aggregate_answer_scores,
    aggregate_fact_lineage,
    aggregate_smoke_cases,
    classify_fact_lineage,
    deterministic_rewrite_retention,
    question_sha256,
    score_deterministic_case,
)
from paper_research_agent.web.runtime import RAGRuntime


class _JudgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    item_id: str
    hit: bool
    citation_supported: bool = False


class _ForbiddenJudgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    item_id: str
    present: bool


class _JudgeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dimensions: tuple[_JudgeItem, ...]
    must_have: tuple[_JudgeItem, ...]
    forbidden: tuple[_ForbiddenJudgeItem, ...]
    answer_complete: bool
    supported_answer_claim_count: int = Field(ge=0)


class _ComparisonJudge:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self.model = model
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60),
        )
        self._endpoint = base_url.rstrip("/") + "/chat/completions"

    async def score(
        self,
        gold: ComparisonEndToEndGold,
        answer: Any,
        sources: tuple[Any, ...],
    ) -> ComparisonModelJudgeScore:
        dimension_ids = tuple(f"D{index}" for index, _ in enumerate(gold.expected_dimensions))
        must_ids = tuple(item.claim_id for item in gold.must_have_claims)
        forbidden_ids = tuple(f"F{index}" for index, _ in enumerate(gold.forbidden_claims))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evidence-grounded comparison evaluator. Return only the "
                        "requested JSON. Mark a required dimension or fact hit only if the answer "
                        "expresses it with material qualifiers intact. citation_supported is true "
                        "only when the answer's own cited excerpts entail that fact and belong to "
                        "the stated paper. Mark forbidden facts if stated or clearly implied. Use no "
                        "outside knowledge and return every supplied ID exactly once."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "expected_dimensions": [
                                {"item_id": item_id, "text": text}
                                for item_id, text in zip(
                                    dimension_ids, gold.expected_dimensions, strict=True
                                )
                            ],
                            "must_have": [
                                {
                                    "item_id": item.claim_id,
                                    "corpus_id": item.corpus_id,
                                    "fact": item.normalized_fact,
                                }
                                for item in gold.must_have_claims
                            ],
                            "forbidden": [
                                {"item_id": item_id, "fact": text}
                                for item_id, text in zip(
                                    forbidden_ids, gold.forbidden_claims, strict=True
                                )
                            ],
                            "answer_claims": [
                                {
                                    "index": index,
                                    "text": claim.text,
                                    "citation_ids": list(claim.citation_ids),
                                }
                                for index, claim in enumerate(answer.claims)
                            ],
                            "sources": [
                                {
                                    "citation_id": source.citation_id,
                                    "chunk_id": source.chunk_id,
                                    "corpus_id": source.corpus_id,
                                    "excerpt": source.excerpt,
                                }
                                for source in sources
                            ],
                            "required_output": {
                                "dimensions": [
                                    {"item_id": item_id, "hit": False, "citation_supported": False}
                                    for item_id in dimension_ids
                                ],
                                "must_have": [
                                    {"item_id": item_id, "hit": False, "citation_supported": False}
                                    for item_id in must_ids
                                ],
                                "forbidden": [
                                    {"item_id": item_id, "present": False}
                                    for item_id in forbidden_ids
                                ],
                                "answer_complete": False,
                                "supported_answer_claim_count": 0,
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "top_p": 0.7,
            "enable_thinking": False,
            "max_tokens": 1500,
        }
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = await self._client.post(self._endpoint, json=payload)
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]
                draft = _JudgeDraft.model_validate(json.loads(raw))
                if {item.item_id for item in draft.dimensions} != set(dimension_ids):
                    raise ValueError("judge returned invalid dimension IDs")
                if {item.item_id for item in draft.must_have} != set(must_ids):
                    raise ValueError("judge returned invalid must-have IDs")
                if {item.item_id for item in draft.forbidden} != set(forbidden_ids):
                    raise ValueError("judge returned invalid forbidden IDs")
                if draft.supported_answer_claim_count > len(answer.claims):
                    raise ValueError("judge supported claim count exceeds answer claims")
                return ComparisonModelJudgeScore(
                    dimension_hit=sum(item.hit for item in draft.dimensions),
                    dimension_total=len(dimension_ids),
                    must_have_hit=sum(item.hit for item in draft.must_have),
                    must_have_total=len(must_ids),
                    citation_supported_hit=sum(
                        item.hit and item.citation_supported for item in draft.must_have
                    ),
                    forbidden_present=sum(item.present for item in draft.forbidden),
                    forbidden_total=len(forbidden_ids),
                    answer_complete=draft.answer_complete,
                    supported_answer_claim_count=draft.supported_answer_claim_count,
                    answer_claim_count=len(answer.claims),
                )
            except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        return ComparisonModelJudgeScore(
            dimension_hit=0,
            dimension_total=len(dimension_ids),
            must_have_hit=0,
            must_have_total=len(must_ids),
            citation_supported_hit=0,
            forbidden_present=0,
            forbidden_total=len(forbidden_ids),
            answer_complete=False,
            supported_answer_claim_count=0,
            answer_claim_count=len(answer.claims),
            judge_error_type=type(last_error).__name__,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _load_local_env() -> None:
    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_answer_gold(path: Path) -> dict[str, ComparisonEndToEndGold]:
    with path.open(encoding="utf-8") as handle:
        rows = tuple(
            ComparisonEndToEndGold.model_validate_json(line)
            for line in handle
            if line.strip()
        )
    if len({item.question_id for item in rows}) != len(rows):
        raise ValueError("answer gold question IDs must be unique")
    return {item.question_id: item for item in rows}


class _QueryRecorder:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, object]] = []

    async def resolve_query(self, question: str):
        started = time.perf_counter()
        trace = await self.inner.resolve_query(question)
        self.calls.append(
            {
                "question_sha256": question_sha256(question),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "trace": trace,
            }
        )
        return trace

    async def search(self, *args: Any, **kwargs: Any):
        return await self.inner.search(*args, **kwargs)

    async def aclose(self) -> None:
        await self.inner.aclose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _CandidateRecorder:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, object]] = []

    async def search(self, query: Any, *, top_k: int):
        hits = await self.inner.search(query, top_k=top_k)
        self.calls.append(
            {
                "question_sha256": question_sha256(query.original_query),
                "top_k": top_k,
                "candidate_ids": tuple(hit.corpus_id for hit in hits),
            }
        )
        return hits


class _ResearchRecorder:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.results: list[Any] = []

    @property
    def dynamic_tools_enabled(self) -> bool:
        return bool(self.inner.dynamic_tools_enabled)

    @property
    def extended_tools_enabled(self) -> bool:
        return bool(self.inner.extended_tools_enabled)

    async def run(self, *args: Any, **kwargs: Any):
        result = await self.inner.run(*args, **kwargs)
        self.results.append(result)
        return result

    async def clear(self, *args: Any, **kwargs: Any):
        return await self.inner.clear(*args, **kwargs)

    async def aclose(self) -> None:
        await self.inner.aclose()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _diagnostic(
    question: Any,
    *,
    query_call: dict[str, object] | None,
    candidate_call: dict[str, object] | None,
    research: Any | None,
    answer: Any | None,
    elapsed_ms: float,
    error: Exception | None,
) -> ComparisonCaseDiagnostic:
    rewrite = query_call.get("trace") if query_call else None
    english_query = getattr(rewrite, "english_query", None)
    retained, required_count, retained_count = deterministic_rewrite_retention(
        question.question, english_query
    )
    candidate_ids = (
        tuple(candidate_call["candidate_ids"])[:8] if candidate_call is not None else ()
    )
    retrievals: list[RetrievalDiagnostic] = []
    final_ids: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    if research is not None:
        final_ids = tuple(target.corpus_id for target in research.plan.targets)
        dimensions = tuple(item.label for item in research.plan.dimensions)
        steps = {step.step_id: step for step in research.plan.steps}
        for observation in research.observations:
            step = steps[observation.step_id]
            retrievals.append(
                RetrievalDiagnostic(
                    step_id_hash=hashlib.sha256(step.step_id.encode("utf-8")).hexdigest(),
                    target_ids=step.target_ids,
                    dimension_ids=step.dimension_ids,
                    corpus_id_filter=observation.search.corpus_id,
                    search_hit_chunk_ids=tuple(
                        hit.chunk_id for hit in observation.search.hits
                    ),
                    evidence_chunk_ids=tuple(
                        item.chunk_id for item in observation.evidence.records
                    ),
                )
            )
    citations: list[CitationDiagnostic] = []
    if answer is not None:
        citation_by_id = {item.citation_id: item for item in answer.citations}
        for claim_index, claim in enumerate(answer.claims):
            for citation_id in claim.citation_ids:
                citation = citation_by_id[citation_id]
                citations.append(
                    CitationDiagnostic(
                        claim_index=claim_index,
                        citation_id=citation_id,
                        chunk_id=citation.chunk_id,
                        corpus_id=citation.corpus_id,
                    )
                )
    error_stage = None
    if error is not None:
        error_stage = (
            "before_rewrite"
            if query_call is None
            else "before_candidate"
            if candidate_call is None
            else "before_research_result"
            if research is None
            else "answer_generation"
        )
    return ComparisonCaseDiagnostic(
        question_id=question.question_id,
        split=question.split,
        original_question_sha256=question_sha256(question.question),
        raw_question_preserved=(
            candidate_call is not None
            and candidate_call["question_sha256"] == question_sha256(question.question)
        ),
        rewrite_status=str(getattr(rewrite, "status", "not_run")),
        rewrite_latency_ms=(
            float(query_call["latency_ms"]) if query_call is not None else None
        ),
        rewrite_information_retained=retained,
        rewrite_required_token_count=required_count,
        rewrite_retained_token_count=retained_count,
        candidate_paper_ids_top8=candidate_ids,
        final_paper_ids=final_ids,
        planned_dimensions=dimensions,
        retrievals=tuple(retrievals),
        step_budget=(int(research.step_budget) if research is not None else None),
        assessment_count=(len(research.assessments) if research is not None else 0),
        tool_call_count=(int(research.tool_call_count) if research is not None else 0),
        tool_call_budget=(
            int(research.tool_call_budget) if research is not None else None
        ),
        citations=tuple(citations),
        answer_status=(answer.status if answer is not None else None),
        total_latency_ms=elapsed_ms,
        error_type=type(error).__name__ if error is not None else None,
        error_reason_code=(
            str(reason_code)
            if error is not None
            and (reason_code := getattr(error, "reason_code", None)) is not None
            else None
        ),
        error_stage=error_stage,
    )


def _fact_lineage(
    gold: ComparisonEndToEndGold,
    *,
    research: Any | None,
    answer: Any | None,
) -> tuple[object, ...]:
    retrieved = tuple(
        hit.chunk_id
        for observation in (() if research is None else research.observations)
        for hit in observation.search.hits
    )
    hydrated = tuple(
        record.chunk_id
        for observation in (() if research is None else research.observations)
        for record in observation.evidence.records
    )
    final_ledger = (
        ()
        if research is None or not research.assessments
        else research.assessments[-1].ledger
    )
    compiled_facts = tuple(fact for cell in final_ledger for fact in cell.facts)
    compiler_visible_chunk_ids = (
        None
        if research is None
        or not research.assessments
        or not research.assessments[-1].compilation_visibility
        else tuple(
            chunk_id
            for item in research.assessments[-1].compilation_visibility
            for chunk_id in item.visible_chunk_ids
        )
    )
    answer_claims = () if answer is None else answer.claims
    answer_citations = (
        {} if answer is None else {item.citation_id: item for item in answer.citations}
    )
    relation_by_id = {item.claim_id: item for item in gold.citation_relations}
    result = []
    for gold_claim in gold.must_have_claims:
        relation = relation_by_id[gold_claim.claim_id]
        gold_chunks = set(relation.chunk_ids)
        matching_facts = tuple(
            fact for fact in compiled_facts if gold_chunks & set(fact.chunk_ids)
        )
        matching_fact_ids = {fact.fact_id for fact in matching_facts}
        matching_claims = tuple(
            claim for claim in answer_claims if matching_fact_ids & set(claim.fact_ids)
        )
        citation_correct = any(
            (citation := answer_citations.get(citation_id)) is not None
            and citation.chunk_id in gold_chunks
            and citation.corpus_id == gold_claim.corpus_id
            for claim in matching_claims
            for citation_id in claim.citation_ids
        )
        result.append(
            classify_fact_lineage(
                gold_claim.claim_id,
                gold_chunk_ids=gold_chunks,
                retrieved_chunk_ids=retrieved,
                hydrated_chunk_ids=hydrated,
                visible_chunk_ids=compiler_visible_chunk_ids,
                compiled_chunk_ids=(
                    chunk_id for fact in matching_facts for chunk_id in fact.chunk_ids
                ),
                in_generation_input=bool(matching_facts),
                expressed=bool(matching_claims),
                citation_correct=citation_correct,
            )
        )
    return tuple(result)


async def run(args: argparse.Namespace) -> dict[str, object]:
    questions = load_candidate_paper_gold(args.gold)[: args.limit]
    answer_gold = _load_answer_gold(args.answer_gold)
    missing_gold = {question.question_id for question in questions} - set(answer_gold)
    if missing_gold:
        raise ValueError("answer gold does not cover every selected question")
    runtime = RAGRuntime.from_environment()
    query_recorder = _QueryRecorder(runtime._retriever)
    candidate_recorder = _CandidateRecorder(runtime._paper_candidate_retriever)
    runtime._retriever = query_recorder
    runtime._paper_candidate_retriever = candidate_recorder
    answer_config = load_answering_config(args.answer_config)
    await runtime.enable_research_agent(
        model_id=answer_config.model,
        checkpoint_path=args.checkpoint,
        policy=ResearchRuntimePolicy(timeout_seconds=args.timeout),
        mode="always",
    )
    research_recorder = _ResearchRecorder(runtime._research_agent)
    runtime._research_agent = research_recorder
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("judge credentials are unavailable")
    judge = _ComparisonJudge(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        model=answer_config.model,
    )
    chunk_corpus_ids = {chunk.chunk_id: chunk.corpus_id for chunk in runtime._chunks}
    cases: list[ComparisonCaseDiagnostic] = []
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        selected_ids = {question.question_id for question in questions}
        cases = [
            ComparisonCaseDiagnostic.model_validate(item)
            for item in previous.get("cases", ())
            if item.get("question_id") in selected_ids
        ]
    completed_ids = {case.question_id for case in cases}
    try:
        for index, question in enumerate(questions, start=1):
            if question.question_id in completed_ids:
                continue
            query_index = len(query_recorder.calls)
            candidate_index = len(candidate_recorder.calls)
            research_index = len(research_recorder.results)
            started = time.perf_counter()
            result = None
            error: Exception | None = None
            try:
                result = await runtime.ask(
                    question.question,
                    session_id=f"comparison-e2e-{question.question_id}-{uuid.uuid4().hex}",
                    research_mode="planned",
                )
            except Exception as exc:  # noqa: BLE001 - one failed case must not abort the run
                error = exc
            query_call = (
                query_recorder.calls[query_index]
                if len(query_recorder.calls) > query_index
                else None
            )
            candidate_call = (
                candidate_recorder.calls[candidate_index]
                if len(candidate_recorder.calls) > candidate_index
                else None
            )
            research = (
                research_recorder.results[research_index]
                if len(research_recorder.results) > research_index
                else None
            )
            case = _diagnostic(
                question,
                query_call=query_call,
                candidate_call=candidate_call,
                research=research,
                answer=(result.answer if result is not None else None),
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error=error,
            )
            gold = answer_gold[question.question_id]
            deterministic_score = score_deterministic_case(
                case,
                gold,
                chunk_corpus_ids=chunk_corpus_ids,
            )
            model_judge_score = None
            if result is not None and result.answer.status == "answered":
                model_judge_score = await judge.score(
                    gold,
                    result.answer,
                    tuple(result.sources),
                )
            case = case.model_copy(
                update={
                    "deterministic_score": deterministic_score,
                    "model_judge_score": model_judge_score,
                    "fact_lineage": _fact_lineage(
                        gold,
                        research=research,
                        answer=(result.answer if result is not None else None),
                    ),
                }
            )
            cases.append(case)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {"schema_version": "comparison-e2e-run-v1", "cases": [
                        item.model_dump(mode="json") for item in cases
                    ]},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            progress = {
                "completed": index,
                "total": len(questions),
                "split": question.split,
                "status": "ok" if error is None else "error",
                "error_type": type(error).__name__ if error is not None else None,
                "latency_ms": round(case.total_latency_ms),
            }
            if question.split == "dev":
                progress["question_id"] = question.question_id
            print(
                json.dumps(
                    progress,
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        with suppress(Exception):
            await judge.aclose()
        with suppress(Exception):
            await runtime.aclose()
    chunks = runtime._chunks
    summary = aggregate_smoke_cases(
        cases,
        relevant_by_question={
            question.question_id: question.relevant_paper_ids for question in questions
        },
        chunk_corpus_ids={chunk.chunk_id: chunk.corpus_id for chunk in chunks},
    )
    split_counts: dict[str, int] = defaultdict(int)
    for case in cases:
        split_counts[case.split] += 1
    split_summaries = {
        split: {
            "pipeline": aggregate_smoke_cases(
                [case for case in cases if case.split == split],
                relevant_by_question={
                    question.question_id: question.relevant_paper_ids
                    for question in questions
                    if question.split == split
                },
                chunk_corpus_ids=chunk_corpus_ids,
            ),
            "answer_scores": aggregate_answer_scores(
                case for case in cases if case.split == split
            ),
            "fact_lineage": aggregate_fact_lineage(
                item
                for case in cases
                if case.split == split
                for item in case.fact_lineage
            ),
        }
        for split in split_counts
    }
    payload = {
        "schema_version": "comparison-e2e-run-v1",
        "code_revision": "dad78a943330075e95f50ca3aa91341abfe38e8a",
        "question_count": len(cases),
        "split_counts": dict(split_counts),
        "summary": summary,
        "answer_scores": aggregate_answer_scores(cases),
        "fact_lineage": aggregate_fact_lineage(
            item for case in cases for item in case.fact_lineage
        ),
        "split_summaries": split_summaries,
        "cases": [item.model_dump(mode="json") for item in cases],
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    _load_local_env()
    parser = argparse.ArgumentParser(description="Run private comparison E2E evaluation.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/candidate-paper-gold-v1.jsonl",
    )
    parser.add_argument(
        "--answer-config",
        type=Path,
        default=PROJECT_ROOT / "configs/answering/qwen-rag-v1.json",
    )
    parser.add_argument(
        "--answer-gold",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/evaluations/gold/comparison-end-to-end-gold-v1.jsonl"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "data/runtime/comparison-e2e-checkpoint.sqlite3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/runs/comparison-end-to-end-smoke5-v1.json",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    result = asyncio.run(run(args))
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
