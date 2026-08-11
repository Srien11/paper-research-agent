"""Privacy-safe RAG rubric evaluation over private answer/span labels."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.evaluation.gold_dataset import GoldQuestion


class MustHaveJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str
    satisfied: bool
    citation_supported: bool


class ForbiddenJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str
    present: bool


class RAGJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    must_have: tuple[MustHaveJudgment, ...] = ()
    forbidden: tuple[ForbiddenJudgment, ...] = ()
    supported_answer_claim_count: int = Field(ge=0)
    citation_supported_answer_claim_count: int = Field(ge=0)


class RAGGoldRuntime(Protocol):
    async def ask(
        self,
        question: str,
        *,
        session_id: str,
        research_mode: str = "single",
    ) -> object: ...


class RAGGoldJudge(Protocol):
    async def score(
        self,
        question: GoldQuestion,
        answer: object,
        sources: Sequence[object],
    ) -> RAGJudgeResult: ...


def _value(source: object, name: str, default: Any = None) -> Any:
    return source.get(name, default) if isinstance(source, dict) else getattr(source, name, default)


async def evaluate_rag_gold(
    runtime: RAGGoldRuntime,
    judge: RAGGoldJudge,
    questions: Sequence[GoldQuestion],
    output_path: Path,
    *,
    evaluation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for question in questions:
        started = time.perf_counter()
        base = {
            "question_id": question.question_id,
            "answerable": question.answerable,
            "annotation_status": question.annotation_status,
            "language": question.language,
            "task_type": question.task_type,
            "difficulty": question.difficulty,
            "must_have_total": len(question.must_have_claims),
            "forbidden_total": len(question.forbidden_claims),
        }
        try:
            result = await runtime.ask(
                question.question,
                session_id=f"rag-gold-evaluation-{uuid.uuid4().hex}",
                research_mode=(
                    "planned"
                    if question.task_type == "multi_paper_comparison"
                    else "single"
                ),
            )
            answer = _value(result, "answer")
            sources = tuple(_value(result, "sources", ()))
            generation = _value(result, "generation")
            status = str(_value(answer, "status"))
            answer_claims = tuple(_value(answer, "claims", ()))
            source_chunks = {str(_value(source, "chunk_id")) for source in sources}
            source_papers = {str(_value(source, "corpus_id")) for source in sources}
            expected_status = "answered" if question.answerable else "insufficient_evidence"
            judgment = (
                await judge.score(question, answer, sources)
                if status == "answered"
                else RAGJudgeResult(
                    supported_answer_claim_count=0,
                    citation_supported_answer_claim_count=0,
                )
            )
            _validate_judgment(question, judgment, len(answer_claims), status)
            must_hit = sum(item.satisfied for item in judgment.must_have)
            must_cited = sum(
                item.satisfied and item.citation_supported for item in judgment.must_have
            )
            forbidden_present = sum(item.present for item in judgment.forbidden)
            span_hits, span_total, group_hits, group_total = _retrieval_hits(
                question, source_chunks
            )
            gold_papers = (
                {span.paper_id for span in question.evidence_spans}
                if question.answerable
                else set()
            )
            base.update(
                {
                    "success": True,
                    "error_type": None,
                    "status": status,
                    "status_correct": status == expected_status,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "answer_claim_count": len(answer_claims),
                    "supported_answer_claim_count": judgment.supported_answer_claim_count,
                    "citation_supported_answer_claim_count": (
                        judgment.citation_supported_answer_claim_count
                    ),
                    "must_have_hit": must_hit,
                    "must_have_citation_supported": must_cited,
                    "forbidden_present": forbidden_present,
                    "span_hit": span_hits,
                    "span_total": span_total,
                    "required_group_hit": group_hits,
                    "required_group_total": group_total,
                    "paper_hit": len(source_papers & gold_papers),
                    "paper_total": len(gold_papers),
                    "source_count": len(sources),
                    "input_tokens": int(_value(generation, "input_tokens", 0)),
                    "output_tokens": int(_value(generation, "output_tokens", 0)),
                }
            )
        except Exception as error:  # noqa: BLE001 - persist only the safe class name
            base.update(
                {
                    "success": False,
                    "error_type": type(error).__name__,
                    "status": None,
                    "status_correct": False,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "answer_claim_count": 0,
                    "supported_answer_claim_count": 0,
                    "citation_supported_answer_claim_count": 0,
                    "must_have_hit": 0,
                    "must_have_citation_supported": 0,
                    "forbidden_present": 0,
                    "span_hit": 0,
                    "span_total": 0,
                    "required_group_hit": 0,
                    "required_group_total": 0,
                    "paper_hit": 0,
                    "paper_total": 0,
                    "source_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )
        records.append(base)

    safe_context = {
        key: value
        for key, value in (evaluation_context or {}).items()
        if key in {"model_id", "judge_model_id", "dataset_sha256", "index_id", "code_revision"}
        and isinstance(value, (str, int, float, bool))
    }
    result = {
        "schema_version": "rag-gold-evaluation-v1",
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(
                {
                    "context": safe_context,
                    "questions": [question.model_dump(mode="json") for question in questions],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "evaluation_context": safe_context,
        "question_count": len(questions),
        "answerable_count": sum(question.answerable for question in questions),
        "unanswerable_count": sum(not question.answerable for question in questions),
        "aggregates": _aggregates(records),
        "slice_aggregates": {
            dimension: {
                str(value): _aggregates(
                    [record for record in records if record[dimension] == value]
                )
                for value in sorted({record[dimension] for record in records})
            }
            for dimension in ("language", "task_type", "difficulty")
        },
        "records": records,
        "limitations": [
            "当前标签为模型生成银标草案，未经双人复核与仲裁，不进入正式总分",
            "rubric Judge 使用自动模型，只作为开发诊断，必须以人工校准结果修正",
            "结果不保存问题、答案、证据正文、Judge 理由或 Provider 原始载荷",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _validate_judgment(
    question: GoldQuestion,
    judgment: RAGJudgeResult,
    answer_claim_count: int,
    status: str,
) -> None:
    if judgment.supported_answer_claim_count > answer_claim_count:
        raise ValueError("judge supported claim count exceeds answer claims")
    if judgment.citation_supported_answer_claim_count > answer_claim_count:
        raise ValueError("judge citation claim count exceeds answer claims")
    expected_must = {claim.claim_id for claim in question.must_have_claims} if status == "answered" else set()
    expected_forbidden = (
        {claim.claim_id for claim in question.forbidden_claims} if status == "answered" else set()
    )
    if {item.claim_id for item in judgment.must_have} != expected_must:
        raise ValueError("judge must-have IDs do not match the rubric")
    if {item.claim_id for item in judgment.forbidden} != expected_forbidden:
        raise ValueError("judge forbidden IDs do not match the rubric")


def _retrieval_hits(
    question: GoldQuestion, source_chunks: set[str]
) -> tuple[int, int, int, int]:
    if not question.answerable:
        return 0, 0, 0, 0
    spans = {span.span_id: span for span in question.evidence_spans}
    referenced_ids = {
        relation.span_id
        for relation in question.citation_relations
        if relation.relation == "supports"
    }
    span_hit = sum(
        bool(set(spans[span_id].projected_chunk_ids) & source_chunks)
        for span_id in referenced_ids
    )
    groups: list[set[str]] = []
    for claim in question.must_have_claims:
        group = {
            relation.span_id
            for relation in question.citation_relations
            if relation.claim_id == claim.claim_id and relation.relation == "supports"
        }
        if group:
            groups.append(group)
    group_hit = sum(
        all(set(spans[span_id].projected_chunk_ids) & source_chunks for span_id in group)
        for group in groups
    )
    return span_hit, len(referenced_ids), group_hit, len(groups)


def _aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def total(field: str) -> int:
        return sum(int(record[field]) for record in records)

    must_total = total("must_have_total")
    answer_claims = total("answer_claim_count")
    claim_precision = _ratio(total("supported_answer_claim_count"), answer_claims)
    claim_recall = _ratio(total("must_have_hit"), must_total)
    citation_precision = _ratio(total("citation_supported_answer_claim_count"), answer_claims)
    citation_recall = _ratio(total("must_have_citation_supported"), must_total)
    true_positive = sum(
        not record["answerable"] and record["status"] == "insufficient_evidence"
        for record in records
    )
    false_positive = sum(
        record["answerable"] and record["status"] == "insufficient_evidence"
        for record in records
    )
    false_negative = sum(
        not record["answerable"] and record["status"] == "answered" for record in records
    )
    refusal_precision = _ratio(true_positive, true_positive + false_positive)
    refusal_recall = _ratio(true_positive, true_positive + false_negative)
    latencies = [float(record["latency_ms"]) for record in records if record["success"]]
    return {
        "run_success_rate": _ratio(sum(record["success"] for record in records), len(records)),
        "status_accuracy": _ratio(sum(record["status_correct"] for record in records), len(records)),
        "claim_precision": claim_precision,
        "claim_recall": claim_recall,
        "claim_f1": _f1(claim_precision, claim_recall),
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_f1": _f1(citation_precision, citation_recall),
        "forbidden_claim_rate": _ratio(total("forbidden_present"), total("forbidden_total")),
        "unsupported_claim_rate": _ratio(
            answer_claims - total("supported_answer_claim_count"), answer_claims
        ),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": _f1(refusal_precision, refusal_recall),
        "false_refusal_rate": _ratio(
            false_positive, sum(record["answerable"] for record in records)
        ),
        "span_recall": _ratio(total("span_hit"), total("span_total")),
        "required_group_recall": _ratio(
            total("required_group_hit"), total("required_group_total")
        ),
        "paper_recall": _ratio(total("paper_hit"), total("paper_total")),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "mean_input_tokens": statistics.fmean(record["input_tokens"] for record in records)
        if records
        else None,
        "mean_output_tokens": statistics.fmean(record["output_tokens"] for record in records)
        if records
        else None,
        "error_types": dict(Counter(record["error_type"] for record in records if record["error_type"])),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0 if precision is not None and recall is not None else None
    return 2 * precision * recall / (precision + recall)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def write_rag_gold_report(result: Mapping[str, Any], path: Path) -> None:
    values = result["aggregates"]

    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def ms(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f} ms"

    rows = [
        ("运行成功率", pct(values["run_success_rate"])),
        ("回答状态准确率", pct(values["status_accuracy"])),
        ("Must-have Claim F1", pct(values["claim_f1"])),
        ("Citation F1", pct(values["citation_f1"])),
        ("Forbidden Claim Rate", pct(values["forbidden_claim_rate"])),
        ("Unsupported Claim Rate", pct(values["unsupported_claim_rate"])),
        ("拒答 F1", pct(values["refusal_f1"])),
        ("错误拒答率", pct(values["false_refusal_rate"])),
        ("Span Recall", pct(values["span_recall"])),
        ("必要证据组 Recall", pct(values["required_group_recall"])),
        ("论文 Recall", pct(values["paper_recall"])),
        ("端到端延迟 P50", ms(values["latency_p50_ms"])),
        ("端到端延迟 P95", ms(values["latency_p95_ms"])),
    ]
    task_lines = []
    for name, metrics in result.get("slice_aggregates", {}).get("task_type", {}).items():
        task_lines.append(
            f"| `{name}` | {pct(metrics['status_accuracy'])} | "
            f"{pct(metrics['claim_f1'])} | {pct(metrics['span_recall'])} |"
        )
    text = "\n".join(
        [
            "# RAG 银标 Rubric 诊断",
            "",
            f"- 实验指纹：`{result['fingerprint_sha256']}`",
            f"- 题目：{result['question_count']}（可回答 {result['answerable_count']} / 不可回答 {result['unanswerable_count']}）",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            *[f"| {name} | {value} |" for name, value in rows],
            "",
            "## 题型切片",
            "",
            "| 题型 | 状态准确率 | Claim F1 | Span Recall |",
            "|---|---:|---:|---:|",
            *task_lines,
            "",
            "## 安全错误类型",
            "",
            json.dumps(values["error_types"], ensure_ascii=False, sort_keys=True),
            "",
            "## 解释边界",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
