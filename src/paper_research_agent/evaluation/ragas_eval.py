"""Ragas 0.4.x evaluation over live, private RAG results.

Private questions, answers, claims, and excerpts stay in memory. Persisted artifacts only
contain stable case identifiers, numeric scores, timings, and safe error type names.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from paper_research_agent.evaluation.gold_dataset import GoldQuestion

RAGAS_METRICS = (
    "selected_context_precision",
    "selected_context_recall",
    "response_faithfulness",
    "answer_relevancy",
    "citation_faithfulness",
)
ALL_METRICS = (*RAGAS_METRICS, "citation_validity", "paper_id_precision", "paper_id_recall")


class RagasEvaluationConfig(BaseModel):
    """Versioned configuration without credentials or local paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ragas-evaluation-config-v1"] = "ragas-evaluation-config-v1"
    ragas_version: Literal["0.4.3"] = "0.4.3"
    judge_model: str = Field(min_length=1, max_length=256)
    embedding_model: str | None = Field(default=None, min_length=1, max_length=256)
    max_concurrency: int = Field(default=4, ge=1, le=4)
    enabled_metrics: tuple[str, ...] = RAGAS_METRICS
    diagnostic_thresholds: dict[str, float] = Field(default_factory=dict)
    release_gate_enabled: bool = False

    @field_validator("enabled_metrics")
    @classmethod
    def validate_enabled_metrics(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("enabled Ragas metrics must be unique")
        unknown = set(values) - set(RAGAS_METRICS)
        if unknown:
            raise ValueError(f"unsupported Ragas metrics: {len(unknown)}")
        return values

    @field_validator("diagnostic_thresholds")
    @classmethod
    def validate_thresholds(cls, values: dict[str, float]) -> dict[str, float]:
        unknown = set(values) - set(ALL_METRICS)
        if unknown:
            raise ValueError(f"unsupported diagnostic thresholds: {len(unknown)}")
        if any(not 0 <= value <= 1 for value in values.values()):
            raise ValueError("diagnostic thresholds must be between zero and one")
        return values

    @model_validator(mode="after")
    def validate_embedding_requirement(self) -> RagasEvaluationConfig:
        if "answer_relevancy" in self.enabled_metrics and self.embedding_model is None:
            raise ValueError("answer relevancy requires an embedding model")
        return self

    @classmethod
    def from_path(cls, path: Path) -> RagasEvaluationConfig:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"cannot read Ragas config: {path.name}") from error


@dataclass(frozen=True)
class CitationClaimInput:
    """One answer claim and only the excerpts referenced by that claim."""

    response: str = field(repr=False)
    retrieved_contexts: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class RagasCaseInput:
    """Ephemeral metric input. Its text fields must never be serialized."""

    user_input: str = field(repr=False)
    response: str = field(repr=False)
    reference: str = field(repr=False)
    retrieved_contexts: tuple[str, ...] = field(repr=False)
    citation_claims: tuple[CitationClaimInput, ...] = field(repr=False)


class RagasCaseScorer(Protocol):
    model_id: str
    embedding_model_id: str | None
    ragas_version: str

    async def score(self, sample: RagasCaseInput) -> Mapping[str, float | None]: ...


class RagasMetricSuite:
    """Thin adapter around the Ragas 0.4.3 collections API."""

    def __init__(
        self,
        *,
        llm: Any,
        embeddings: Any | None,
        config: RagasEvaluationConfig,
    ) -> None:
        try:
            import ragas
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                'Ragas evaluation dependencies are unavailable; install ".[ragas-eval]"'
            ) from error

        if ragas.__version__ != config.ragas_version:
            raise RuntimeError(
                f"Ragas version mismatch: expected {config.ragas_version}, got {ragas.__version__}"
            )
        if "answer_relevancy" in config.enabled_metrics and embeddings is None:
            raise ValueError("answer relevancy is enabled but embeddings are unavailable")

        self.model_id: str = config.judge_model
        self.embedding_model_id: str | None = config.embedding_model
        self.ragas_version: str = config.ragas_version
        self._enabled = frozenset(config.enabled_metrics)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._context_precision = ContextPrecision(llm=llm)
        self._context_recall = ContextRecall(llm=llm)
        self._faithfulness = Faithfulness(llm=llm)
        self._answer_relevancy: Any | None
        if "answer_relevancy" in self._enabled:
            assert embeddings is not None
            self._answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        else:
            self._answer_relevancy = None

    async def _run(self, awaitable: Any) -> Any:
        async with self._semaphore:
            return await awaitable

    async def score(self, sample: RagasCaseInput) -> Mapping[str, float | None]:
        tasks: dict[str, Any] = {}
        if "selected_context_precision" in self._enabled:
            tasks["selected_context_precision"] = self._context_precision.ascore(
                user_input=sample.user_input,
                reference=sample.reference,
                retrieved_contexts=list(sample.retrieved_contexts),
            )
        if "selected_context_recall" in self._enabled:
            tasks["selected_context_recall"] = self._context_recall.ascore(
                user_input=sample.user_input,
                reference=sample.reference,
                retrieved_contexts=list(sample.retrieved_contexts),
            )
        if "response_faithfulness" in self._enabled:
            tasks["response_faithfulness"] = self._faithfulness.ascore(
                user_input=sample.user_input,
                response=sample.response,
                retrieved_contexts=list(sample.retrieved_contexts),
            )
        if self._answer_relevancy is not None:
            tasks["answer_relevancy"] = self._answer_relevancy.ascore(
                user_input=sample.user_input,
                response=sample.response,
            )

        try:
            names = tuple(tasks)
            values = await asyncio.gather(*(self._run(tasks[name]) for name in names))
            scores: dict[str, float | None] = {
                name: _metric_value(value) for name, value in zip(names, values, strict=True)
            }
            scores["citation_faithfulness"] = await self._citation_faithfulness(sample)
            return scores
        except Exception as error:  # noqa: BLE001 - text must not leak through exception chains
            raise RuntimeError(f"Ragas scoring failed: {type(error).__name__}") from None

    async def _citation_faithfulness(self, sample: RagasCaseInput) -> float | None:
        if "citation_faithfulness" not in self._enabled or not sample.citation_claims:
            return None
        tasks = [
            self._faithfulness.ascore(
                user_input=sample.user_input,
                response=claim.response,
                retrieved_contexts=list(claim.retrieved_contexts),
            )
            for claim in sample.citation_claims
        ]
        values = await asyncio.gather(*(self._run(task) for task in tasks))
        scores = [_metric_value(value) for value in values]
        return statistics.fmean(scores) if scores else None


def _metric_value(result: object) -> float:
    value = getattr(result, "value", result)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Ragas metric returned a non-numeric value")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Ragas metric returned a non-finite value")
    return numeric


def paper_id_precision_recall(
    retrieved_ids: Sequence[str], reference_ids: Sequence[str]
) -> tuple[float | None, float | None]:
    retrieved = set(retrieved_ids)
    reference = set(reference_ids)
    true_positive = len(retrieved & reference)
    precision = true_positive / len(retrieved) if retrieved else None
    recall = true_positive / len(reference) if reference else None
    return precision, recall


def _value(source: object, name: str, default: Any = None) -> Any:
    return source.get(name, default) if isinstance(source, dict) else getattr(source, name, default)


def _reference_text(question: GoldQuestion) -> str:
    if question.answerable:
        return "\n".join(claim.text for claim in question.must_have_claims)
    return question.unanswerable_reason or "证据不足，应该拒绝回答。"


def _hydrated_source_contexts(
    sources: Sequence[object], chunk_text_by_id: Mapping[str, str]
) -> tuple[str, ...]:
    contexts: list[str] = []
    for source in sources:
        chunk_id = str(_value(source, "chunk_id", ""))
        context = chunk_text_by_id.get(chunk_id, "")
        if not chunk_id or not context.strip():
            raise ValueError("selected source is missing its full evaluation chunk")
        contexts.append(context)
    return tuple(contexts)


def _case_input(
    question: GoldQuestion,
    answer: object,
    sources: Sequence[object],
    chunk_text_by_id: Mapping[str, str],
) -> RagasCaseInput:
    source_contexts = _hydrated_source_contexts(sources, chunk_text_by_id)
    source_by_citation = {
        str(_value(source, "citation_id")): context
        for source, context in zip(sources, source_contexts, strict=True)
        if str(_value(source, "citation_id", ""))
    }
    citation_claims = []
    for claim in tuple(_value(answer, "claims", ())):
        contexts = tuple(
            source_by_citation[citation_id]
            for citation_id in tuple(_value(claim, "citation_ids", ()))
            if citation_id in source_by_citation
        )
        if contexts:
            citation_claims.append(
                CitationClaimInput(
                    response=str(_value(claim, "text", "")),
                    retrieved_contexts=contexts,
                )
            )
    return RagasCaseInput(
        user_input=question.question,
        response=str(_value(answer, "answer_markdown", "")),
        reference=_reference_text(question),
        retrieved_contexts=source_contexts,
        citation_claims=tuple(citation_claims),
    )


def _reference_paper_ids(question: GoldQuestion) -> tuple[str, ...]:
    must_have_claim_ids = {claim.claim_id for claim in question.must_have_claims}
    supporting_span_ids = {
        relation.span_id
        for relation in question.citation_relations
        if relation.claim_id in must_have_claim_ids and relation.relation == "supports"
    }
    return tuple(
        span.paper_id for span in question.evidence_spans if span.span_id in supporting_span_ids
    )


def _citation_validity(answer: object, sources: Sequence[object]) -> float | None:
    claims = tuple(_value(answer, "claims", ()))
    expected = [
        str(identifier)
        for claim in claims
        for identifier in tuple(_value(claim, "citation_ids", ()))
    ]
    if not expected:
        return None
    available = {str(_value(source, "citation_id", "")) for source in sources}
    return sum(identifier in available for identifier in expected) / len(expected)


def _mean_present(records: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [float(record[name]) for record in records if record.get(name) is not None]
    return statistics.fmean(values) if values else None


def _aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record.get("ragas_scored")]
    return {
        "run_success_rate": statistics.fmean(bool(record["run_success"]) for record in records)
        if records
        else 0.0,
        "answer_status_accuracy": statistics.fmean(
            bool(record["answer_status_correct"]) for record in records
        )
        if records
        else 0.0,
        "ragas_scored_rate": statistics.fmean(bool(record["ragas_scored"]) for record in records)
        if records
        else 0.0,
        **{metric: _mean_present(scored, metric) for metric in ALL_METRICS},
        "mean_agent_latency_ms": _mean_present(records, "agent_latency_ms"),
        "mean_ragas_latency_ms": _mean_present(scored, "ragas_latency_ms"),
        "error_types": dict(
            Counter(str(record["error_type"]) for record in records if record.get("error_type"))
        ),
    }


def _slice_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        dimension: {
            str(value): _aggregates([record for record in records if record[dimension] == value])
            for value in sorted({record[dimension] for record in records})
        }
        for dimension in ("task_type", "difficulty", "language")
    }


def _write_safe_result(result: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def merge_ragas_results(
    batch_results: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    evaluation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge body-free isolated-process batches without loading private inputs."""

    if not batch_results:
        raise ValueError("at least one Ragas batch result is required")
    records: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    limitations: list[str] = []
    for batch in batch_results:
        if batch.get("schema_version") != "ragas-gold-evaluation-v1":
            raise ValueError("Ragas batch result schema is unsupported")
        batch_records = batch.get("records")
        if not isinstance(batch_records, list):
            raise TypeError("Ragas batch records are unavailable")
        records.extend(dict(record) for record in batch_records)
        fingerprints.append(str(batch.get("fingerprint_sha256", "")))
        for limitation in batch.get("limitations", ()):
            text = str(limitation)
            if text not in limitations:
                limitations.append(text)

    question_ids = [str(record.get("question_id", "")) for record in records]
    if any(not identifier for identifier in question_ids):
        raise ValueError("Ragas batch record is missing its question ID")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Ragas batch results contain duplicate question IDs")

    safe_context = dict(evaluation_context or batch_results[0].get("evaluation_context", {}))
    fingerprint_source = {
        "context": safe_context,
        "batch_fingerprints": fingerprints,
        "question_ids": question_ids,
    }
    limitations.append("评测按独立进程分批运行，以便在批次之间释放本地模型内存")
    result = {
        "schema_version": "ragas-gold-evaluation-v1",
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(fingerprint_source, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "evaluation_context": safe_context,
        "question_count": len(records),
        "answerable_count": sum(bool(record.get("answerable")) for record in records),
        "unanswerable_count": sum(not bool(record.get("answerable")) for record in records),
        "aggregates": _aggregates(records),
        "slice_aggregates": _slice_aggregates(records),
        "records": records,
        "limitations": limitations,
    }
    _write_safe_result(result, output_path)
    return result


async def _score_answered_case(
    *,
    record: dict[str, Any],
    question: GoldQuestion,
    answer: object,
    sources: Sequence[object],
    scorer: RagasCaseScorer,
    chunk_text_by_id: Mapping[str, str],
) -> None:
    ragas_started = time.perf_counter()
    try:
        scores = await scorer.score(_case_input(question, answer, sources, chunk_text_by_id))
        record.update(
            {
                "ragas_scored": True,
                "ragas_skip_reason": None,
                "ragas_latency_ms": (time.perf_counter() - ragas_started) * 1000,
                **{metric: scores.get(metric) for metric in RAGAS_METRICS},
                "error_type": None,
            }
        )
    except Exception as error:  # noqa: BLE001 - persist only the safe type name
        record.update(
            {
                "ragas_scored": False,
                "ragas_skip_reason": "scoring_error",
                "ragas_latency_ms": (time.perf_counter() - ragas_started) * 1000,
                **{metric: None for metric in RAGAS_METRICS},
                "error_type": type(error).__name__,
            }
        )


async def evaluate_ragas_gold(
    runtime: Any,
    scorer: RagasCaseScorer,
    questions: Sequence[GoldQuestion],
    output_path: Path,
    *,
    chunk_text_by_id: Mapping[str, str],
    case_concurrency: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
    evaluation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one local Agent and score body-free batches of at most four cases."""

    if not 1 <= case_concurrency <= 4:
        raise ValueError("case concurrency must be between one and four")

    records: list[dict[str, Any]] = []
    pending_scores: list[Awaitable[None]] = []
    for question_index, question in enumerate(questions, start=1):
        record: dict[str, Any] = {
            "question_id": question.question_id,
            "answerable": question.answerable,
            "annotation_status": question.annotation_status,
            "language": question.language,
            "task_type": question.task_type,
            "difficulty": question.difficulty,
        }
        agent_started = time.perf_counter()
        try:
            result = await runtime.ask(
                question.question,
                session_id=f"ragas-evaluation-{uuid.uuid4().hex}",
                research_mode=(
                    "planned" if question.task_type == "multi_paper_comparison" else "single"
                ),
            )
            agent_latency_ms = (time.perf_counter() - agent_started) * 1000
            answer = _value(result, "answer")
            sources = tuple(_value(result, "sources", ()))
            status = str(_value(answer, "status", ""))
            expected_status = "answered" if question.answerable else "insufficient_evidence"
            retrieved_papers = [str(_value(source, "corpus_id", "")) for source in sources]
            reference_papers = _reference_paper_ids(question)
            paper_precision, paper_recall = paper_id_precision_recall(
                retrieved_papers, reference_papers
            )
            record.update(
                {
                    "run_success": True,
                    "status": status,
                    "answer_status_correct": status == expected_status,
                    "source_count": len(sources),
                    "claim_count": len(tuple(_value(answer, "claims", ()))),
                    "agent_latency_ms": agent_latency_ms,
                    "citation_validity": _citation_validity(answer, sources),
                    "paper_id_precision": paper_precision,
                    "paper_id_recall": paper_recall,
                }
            )
            if status != "answered" or not sources:
                record.update(
                    {
                        "ragas_scored": False,
                        "ragas_skip_reason": "non_answer" if status != "answered" else "no_context",
                        "ragas_latency_ms": None,
                        **{metric: None for metric in RAGAS_METRICS},
                        "error_type": None,
                    }
                )
            else:
                pending_scores.append(
                    _score_answered_case(
                        record=record,
                        question=question,
                        answer=answer,
                        sources=sources,
                        scorer=scorer,
                        chunk_text_by_id=chunk_text_by_id,
                    )
                )
        except Exception as error:  # noqa: BLE001 - only the safe type name is persisted
            record.update(
                {
                    "run_success": False,
                    "status": None,
                    "answer_status_correct": False,
                    "ragas_scored": False,
                    "ragas_skip_reason": "error",
                    "source_count": 0,
                    "claim_count": 0,
                    "agent_latency_ms": (time.perf_counter() - agent_started) * 1000,
                    "ragas_latency_ms": None,
                    **{metric: None for metric in ALL_METRICS},
                    "error_type": type(error).__name__,
                }
            )
        records.append(record)
        if question_index % case_concurrency == 0:
            if pending_scores:
                await asyncio.gather(*pending_scores)
                pending_scores.clear()
            if progress_callback is not None:
                progress_callback(question_index, len(questions))

    if pending_scores:
        await asyncio.gather(*pending_scores)
    if questions and len(questions) % case_concurrency and progress_callback is not None:
        progress_callback(len(questions), len(questions))

    safe_context = dict(evaluation_context or {})
    fingerprint_source = {
        "context": safe_context,
        "questions": [
            {
                "question_id": question.question_id,
                "annotation_status": question.annotation_status,
                "corpus_version": question.corpus_version,
            }
            for question in questions
        ],
    }
    result = {
        "schema_version": "ragas-gold-evaluation-v1",
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(fingerprint_source, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "evaluation_context": safe_context,
        "question_count": len(questions),
        "answerable_count": sum(question.answerable for question in questions),
        "unanswerable_count": sum(not question.answerable for question in questions),
        "aggregates": _aggregates(records),
        "slice_aggregates": _slice_aggregates(records),
        "records": records,
        "limitations": [
            "Ragas 指标只对 answered 且存在引用来源的结果评分",
            "Ragas 评分使用实际入选 chunk 的完整正文；正文只驻留内存且不写入结果",
            "selected_context 指标评估最终入选引用，不等同于原始召回排序",
            "paper_id 指标的金标仅来自 must-have claim 的 supports 关系",
            "LLM 裁判分数必须用人工双标样本校准，当前诊断阈值未启用发布门禁",
            "结果不保存问题、回答、claim、证据摘录、Provider 原始响应或本地路径",
        ],
    }
    _write_safe_result(result, output_path)
    return result


def write_ragas_report(
    result: Mapping[str, Any],
    path: Path,
    *,
    thresholds: Mapping[str, float] | None = None,
    release_gate_enabled: bool = False,
) -> None:
    values = result["aggregates"]
    diagnostic = dict(thresholds or {})

    def display(name: str) -> str:
        value = values.get(name)
        if value is None:
            return "N/A"
        threshold = diagnostic.get(name)
        marker = ""
        if threshold is not None:
            marker = "（达标）" if value >= threshold else "（待改进）"
        return f"{value:.4f}{marker}"

    rows = [
        "| 指标 | 平均分 | 诊断阈值 |",
        "|---|---:|---:|",
        *[
            f"| {name} | {display(name)} | "
            f"{diagnostic.get(name, 'N/A') if name in diagnostic else 'N/A'} |"
            for name in ALL_METRICS
        ],
    ]
    text = "\n".join(
        [
            "# Ragas 论文研究 Agent 评测",
            "",
            f"- 实验指纹：`{result['fingerprint_sha256']}`",
            f"- 问题数：{result['question_count']}",
            f"- Ragas 实际评分率：{values['ragas_scored_rate']:.2%}",
            f"- 回答状态正确率：{values['answer_status_accuracy']:.2%}",
            f"- 发布门禁：{'启用' if release_gate_enabled else '未启用，仅作诊断'}",
            "",
            *rows,
            "",
            "## 解释边界",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
