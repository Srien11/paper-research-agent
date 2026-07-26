"""Run comparable A/B/C retrieval experiments and save a fingerprint."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from paper_research_agent.evaluation.dataset import DiagnosticQuery
from paper_research_agent.evaluation.metrics import (
    evidence_hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from paper_research_agent.retrieval.service import RetrievalService


def evaluate(
    service: RetrievalService,
    queries: list[DiagnosticQuery],
    output_path: Path,
    *,
    variants: tuple[str, ...] = ("A", "B", "C"),
    index_size_bytes: int | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for variant in variants:
        for query in queries:
            started = time.perf_counter()
            run = service.search(query.query, variant)
            latency_ms = (time.perf_counter() - started) * 1000
            paper_ids = [hit.corpus_id for hit in run.hits]
            chunk_ids = [hit.chunk_id for hit in run.hits]
            relevant_papers = set(query.relevant_paper_ids)
            relevant_chunks = set(query.relevant_chunk_ids)
            records.append(
                {
                    "query_id": query.query_id,
                    "variant": variant,
                    "latency_ms": latency_ms,
                    "paper_ranking": paper_ids,
                    "chunk_ranking": chunk_ids,
                    "recall_at_k": recall_at_k(paper_ids, relevant_papers, run.top_k),
                    "mrr": reciprocal_rank(paper_ids, relevant_papers),
                    "ndcg_at_k": ndcg_at_k(paper_ids, relevant_papers, run.top_k),
                    "evidence_hit_at_k": evidence_hit_at_k(
                        chunk_ids, relevant_chunks, run.top_k
                    ),
                }
            )
    def mean_present(variant: str, metric: str) -> float | None:
        values = [
            record[metric]
            for record in records
            if record["variant"] == variant and record[metric] is not None
        ]
        return statistics.fmean(values) if values else None

    aggregates = {
        variant: {
            metric: mean_present(variant, metric)
            for metric in ("recall_at_k", "mrr", "ndcg_at_k", "evidence_hit_at_k", "latency_ms")
        }
        for variant in variants
    }
    fingerprint_source = {
        "index_id": service.index_id,
        "config": service.config.model_dump(mode="json"),
        "queries": [query.model_dump(mode="json") for query in queries],
        "variants": variants,
    }
    result = {
        "fingerprint_sha256": hashlib.sha256(
            json.dumps(fingerprint_source, sort_keys=True).encode()
        ).hexdigest(),
        "index_id": service.index_id,
        "index_size_bytes": index_size_bytes,
        "query_count": len(queries),
        "evidence_labeled_query_count": sum(bool(query.relevant_chunk_ids) for query in queries),
        "aggregates": aggregates,
        "records": records,
        "limitations": [
            "单审阅者银标开发集，不是密封测试集",
            "图片与复杂表格仅覆盖解析出的标题或正文，未做完整 OCR",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def write_chinese_report(result: dict[str, Any], path: Path) -> None:
    rows = [
        "| 变体 | Recall@k | MRR | nDCG@k | 证据命中率 | 平均延迟(ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, values in result["aggregates"].items():
        evidence = values["evidence_hit_at_k"]
        evidence_text = f"{evidence:.4f}" if evidence is not None else "N/A"
        rows.append(
            f"| {variant} | {values['recall_at_k']:.4f} | {values['mrr']:.4f} | "
            f"{values['ndcg_at_k']:.4f} | {evidence_text} | "
            f"{values['latency_ms']:.2f} |"
        )
    text = "\n".join(
        [
            "# A/B/C 检索基线诊断报告",
            "",
            f"- 实验指纹：`{result['fingerprint_sha256']}`",
            f"- 索引：`{result['index_id']}`",
            f"- 索引大小：{result['index_size_bytes']} bytes",
            f"- 查询数：{result['query_count']}",
            f"- 已标证据片段的查询数：{result['evidence_labeled_query_count']}",
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
