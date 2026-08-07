"""Privacy-safe end-to-end RAG evaluation and reporting."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from paper_research_agent.evaluation.dataset import DiagnosticQuery


class RAGEvaluatorRuntime(Protocol):
    async def ask(self, question: str, *, session_id: str) -> object: ...


def _value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _mean_present(records: Sequence[dict[str, Any]], name: str) -> float | None:
    values = [float(record[name]) for record in records if record.get(name) is not None]
    return statistics.fmean(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


async def evaluate_end_to_end(
    runtime: RAGEvaluatorRuntime,
    queries: Sequence[DiagnosticQuery],
    output_path: Path,
    *,
    evaluation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate live RAG behavior without persisting questions, answers, or excerpts."""
    records: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "query_id": query.query_id,
            "answerable": query.answerable,
            "annotation_status": query.annotation_status,
        }
        try:
            result = await runtime.ask(
                query.query,
                session_id=f"evaluation-{uuid.uuid4().hex}",
            )
            latency_ms = (time.perf_counter() - started) * 1000
            answer = _value(result, "answer")
            sources = tuple(_value(result, "sources", ()))
            generation = _value(result, "generation")
            status = str(_value(answer, "status"))
            source_papers = {str(_value(source, "corpus_id")) for source in sources}
            source_chunks = {str(_value(source, "chunk_id")) for source in sources}
            relevant_papers = set(query.relevant_paper_ids)
            relevant_chunks = set(query.relevant_chunk_ids)
            claims = tuple(_value(answer, "claims", ()))
            citations = tuple(_value(answer, "citations", ()))
            expected_status = "answered" if query.answerable else "insufficient_evidence"
            record.update(
                {
                    "success": True,
                    "error_type": None,
                    "status": status,
                    "answer_status_correct": status == expected_status,
                    "latency_ms": latency_ms,
                    "source_count": len(sources),
                    "paper_recall": (
                        len(source_papers & relevant_papers) / len(relevant_papers)
                        if relevant_papers
                        else None
                    ),
                    "evidence_hit": (
                        float(bool(source_chunks & relevant_chunks))
                        if relevant_chunks
                        else None
                    ),
                    "citation_structure_valid": bool(
                        status == "insufficient_evidence" or (claims and citations)
                    ),
                    "claim_count": len(claims),
                    "citation_count": len(citations),
                    "input_tokens": int(_value(generation, "input_tokens", 0)),
                    "output_tokens": int(_value(generation, "output_tokens", 0)),
                    "attempts": int(_value(generation, "attempts", 0)),
                }
            )
        except Exception as error:  # noqa: BLE001 - only the safe type name is persisted
            record.update(
                {
                    "success": False,
                    "error_type": type(error).__name__,
                    "status": None,
                    "answer_status_correct": False,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "source_count": 0,
                    "paper_recall": None,
                    "evidence_hit": None,
                    "citation_structure_valid": False,
                    "claim_count": 0,
                    "citation_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "attempts": 0,
                }
            )
        records.append(record)

    latencies = [float(record["latency_ms"]) for record in records if record["success"]]
    successful = [record for record in records if record["success"]]
    context = evaluation_context or {}
    result = {
        "schema_version": "rag-end-to-end-evaluation-v1",
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(
                {
                    "context": context,
                    "queries": [query.model_dump(mode="json") for query in queries],
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "evaluation_context": context,
        "query_count": len(queries),
        "answerable_query_count": sum(query.answerable for query in queries),
        "unanswerable_query_count": sum(not query.answerable for query in queries),
        "evidence_labeled_query_count": sum(bool(query.relevant_chunk_ids) for query in queries),
        "aggregates": {
            "run_success_rate": statistics.fmean(record["success"] for record in records)
            if records
            else 0.0,
            "answer_status_accuracy": statistics.fmean(
                record["answer_status_correct"] for record in records
            )
            if records
            else 0.0,
            "paper_recall": _mean_present(successful, "paper_recall"),
            "evidence_hit_rate": _mean_present(successful, "evidence_hit"),
            "citation_structure_rate": statistics.fmean(
                record["citation_structure_valid"] for record in successful
            )
            if successful
            else 0.0,
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "mean_input_tokens": _mean_present(successful, "input_tokens"),
            "mean_output_tokens": _mean_present(successful, "output_tokens"),
            "error_types": dict(Counter(record["error_type"] for record in records if record["error_type"])),
        },
        "records": records,
        "limitations": [
            "问题集是开发期银标，不是密封金标测试集",
            "回答状态和引用结构是自动代理指标，不等同于事实正确性人工金标",
            "输出不保存问题、回答、证据正文、Provider 原始响应或本地路径",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def write_end_to_end_report(result: dict[str, Any], path: Path) -> None:
    """Write a compact Chinese report from a privacy-safe evaluation result."""
    values = result["aggregates"]

    def percentage(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def milliseconds(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2f} ms"

    text = "\n".join(
        [
            "# 端到端 RAG 自动评测",
            "",
            f"- 实验指纹：`{result['fingerprint_sha256']}`",
            f"- 查询数：{result['query_count']}",
            f"- 可回答查询数：{result['answerable_query_count']}",
            f"- 不可回答查询数：{result['unanswerable_query_count']}",
            f"- 证据级标注查询数：{result['evidence_labeled_query_count']}",
            "",
            "| 指标 | 结果 |",
            "|---|---:|",
            f"| 运行成功率 | {percentage(values['run_success_rate'])} |",
            f"| 回答状态一致率 | {percentage(values['answer_status_accuracy'])} |",
            f"| 论文召回率 | {percentage(values['paper_recall'])} |",
            f"| 证据命中率 | {percentage(values['evidence_hit_rate'])} |",
            f"| 引用结构合格率 | {percentage(values['citation_structure_rate'])} |",
            f"| 端到端延迟 P50 | {milliseconds(values['latency_p50_ms'])} |",
            f"| 端到端延迟 P95 | {milliseconds(values['latency_p95_ms'])} |",
            f"| 平均输入 Token | {values['mean_input_tokens'] or 0:.2f} |",
            f"| 平均输出 Token | {values['mean_output_tokens'] or 0:.2f} |",
            "",
            "## 解释边界",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
