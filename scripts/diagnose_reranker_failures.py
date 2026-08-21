from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.evaluation.comparison_end_to_end import ComparisonEndToEndGold
from paper_research_agent.evaluation.reranker_diagnostics import (
    RerankerFactDiagnostic,
    summarize_reranker_causes,
    token_coverage,
)
from paper_research_agent.retrieval.config import (
    load_bilingual_retrieval_config,
    load_retrieval_config,
)
from paper_research_agent.retrieval.model_adapters import FastEmbedReranker
from paper_research_agent.retrieval.query_rewrite import DashScopeQueryRewriter


@dataclass(frozen=True)
class _AuditRequest:
    request_id: str
    original_query: str | None
    rewritten_query: str | None
    rankings: dict[str, dict[str, tuple[int, float]]]


def _load_gold(path: Path) -> dict[str, ComparisonEndToEndGold]:
    rows = tuple(
        ComparisonEndToEndGold.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {item.question_id: item for item in rows}


def _load_chunks(path: Path) -> dict[str, dict[str, Any]]:
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {str(item["chunk_id"]): item for item in rows}


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


async def _translate_non_english_atomic_queries(
    run_path: Path,
    *,
    answer_gold_path: Path,
    model_id: str,
    timeout_seconds: float,
) -> dict[str, str]:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    gold_by_question = _load_gold(answer_gold_path)
    pending: list[tuple[str, str]] = []
    for case in payload.get("cases", ()):
        question_id = str(case["question_id"])
        claims = {
            item.claim_id: item.normalized_fact
            for item in gold_by_question[question_id].must_have_claims
        }
        for lineage in case.get("fact_lineage", ()):
            if (
                lineage.get("loss_stage") != "not_hydrated"
                or lineage.get("same_paper_alternative_chunk_compiled") is True
            ):
                continue
            claim_id = str(lineage["claim_id"])
            fact_text = claims[claim_id]
            if _ascii_ratio(fact_text) < 0.8:
                pending.append((f"{question_id}:{claim_id}", fact_text))

    rewriter = DashScopeQueryRewriter(
        model_id,
        timeout_seconds=timeout_seconds,
    )
    try:
        translations = await asyncio.gather(
            *(rewriter.rewrite(fact_text) for _fact_id, fact_text in pending)
        )
    finally:
        await rewriter.aclose()
    return {
        fact_id: result.english_query
        for (fact_id, _fact_text), result in zip(pending, translations, strict=True)
    }


def _select_request(
    request_ids: tuple[str, ...],
    *,
    runs: dict[str, tuple[datetime, str | None, str | None]],
    completed_at: datetime,
) -> str:
    eligible = tuple(
        request_id
        for request_id in request_ids
        if request_id in runs and runs[request_id][0] <= completed_at
    )
    if not eligible:
        raise ValueError("no matching audit request predates the evaluation run")
    return max(eligible, key=lambda request_id: runs[request_id][0])


def _load_audit_requests(
    path: Path,
    *,
    signatures: tuple[tuple[str, ...], ...],
    completed_at: datetime,
) -> tuple[dict[tuple[str, ...], _AuditRequest], int]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        final_by_request: dict[str, list[str]] = defaultdict(list)
        for request_id, chunk_id, _rank in connection.execute(
            """SELECT request_id, chunk_id, rank
               FROM rankings WHERE stage = 'final'
               ORDER BY request_id, rank"""
        ):
            final_by_request[str(request_id)].append(str(chunk_id))
        requests_by_signature: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for request_id, chunk_ids in final_by_request.items():
            requests_by_signature[tuple(chunk_ids)].append(request_id)

        candidate_ids = tuple(
            dict.fromkeys(
                request_id
                for signature in signatures
                for request_id in requests_by_signature.get(signature, ())
            )
        )
        if not candidate_ids:
            raise ValueError("evaluation retrievals do not match the query audit")
        placeholders = ",".join("?" for _ in candidate_ids)
        runs = {
            str(request_id): (
                datetime.fromisoformat(str(created_at)),
                str(original_query) if original_query is not None else None,
                str(rewritten_query) if rewritten_query is not None else None,
            )
            for request_id, created_at, original_query, rewritten_query in connection.execute(
                f"""SELECT request_id, created_at, original_query, rewritten_query
                    FROM runs WHERE request_id IN ({placeholders})""",
                candidate_ids,
            )
        }
        selected_by_signature = {
            signature: _select_request(
                tuple(requests_by_signature[signature]),
                runs=runs,
                completed_at=completed_at,
            )
            for signature in signatures
        }
        selected_ids = tuple(dict.fromkeys(selected_by_signature.values()))
        selected_placeholders = ",".join("?" for _ in selected_ids)
        rankings: dict[str, dict[str, dict[str, tuple[int, float]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for request_id, stage, chunk_id, rank, score in connection.execute(
            f"""SELECT request_id, stage, chunk_id, rank, score
                FROM rankings WHERE request_id IN ({selected_placeholders})""",
            selected_ids,
        ):
            rankings[str(request_id)][str(stage)][str(chunk_id)] = (
                int(rank),
                float(score),
            )
        matched = {
            signature: _AuditRequest(
                request_id=request_id,
                original_query=runs[request_id][1],
                rewritten_query=runs[request_id][2],
                rankings=dict(rankings[request_id]),
            )
            for signature, request_id in selected_by_signature.items()
        }
        ambiguity_count = sum(
            len(requests_by_signature[signature]) > 1 for signature in signatures
        )
        return matched, ambiguity_count
    finally:
        connection.close()


def diagnose_run(
    run_path: Path,
    *,
    answer_gold_path: Path,
    chunks_path: Path,
    audit_path: Path,
    reranker: Any | None = None,
    atomic_queries: dict[str, str] | None = None,
) -> tuple[tuple[RerankerFactDiagnostic, ...], dict[str, object]]:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    cases = tuple(payload.get("cases", ()))
    signatures = tuple(
        tuple(str(chunk_id) for chunk_id in retrieval["search_hit_chunk_ids"])
        for case in cases
        for retrieval in case.get("retrievals", ())
    )
    completed_at = datetime.fromtimestamp(run_path.stat().st_mtime, tz=UTC)
    audit_requests, ambiguity_count = _load_audit_requests(
        audit_path,
        signatures=signatures,
        completed_at=completed_at,
    )
    gold_by_question = _load_gold(answer_gold_path)
    chunks = _load_chunks(chunks_path)
    diagnostics: list[RerankerFactDiagnostic] = []

    for case in cases:
        question_id = str(case["question_id"])
        gold = gold_by_question[question_id]
        relation_by_id = {item.claim_id: item for item in gold.citation_relations}
        claim_by_id = {item.claim_id: item for item in gold.must_have_claims}
        requests = tuple(
            audit_requests[
                tuple(str(chunk_id) for chunk_id in item["search_hit_chunk_ids"])
            ]
            for item in case.get("retrievals", ())
        )
        for lineage in case.get("fact_lineage", ()):
            if (
                lineage.get("loss_stage") != "not_hydrated"
                or lineage.get("same_paper_alternative_chunk_compiled") is True
            ):
                continue
            claim_id = str(lineage["claim_id"])
            fact_id = f"{question_id}:{claim_id}"
            gold_chunk_ids = set(relation_by_id[claim_id].chunk_ids)
            occurrences: list[tuple[_AuditRequest, str, int]] = []
            for request in requests:
                final = request.rankings.get("final", {})
                matching = tuple(
                    (chunk_id, final[chunk_id][0])
                    for chunk_id in gold_chunk_ids
                    if chunk_id in final
                )
                if matching:
                    chunk_id, rank = min(matching, key=lambda item: item[1])
                    occurrences.append((request, chunk_id, rank))
            if not occurrences:
                raise ValueError("true hydration loss is absent from matched audit requests")
            request, gold_chunk_id, final_rank = min(
                occurrences, key=lambda item: item[2]
            )
            final = request.rankings["final"]
            pre = request.rankings["cross_route_rrf"]
            if gold_chunk_id not in pre:
                raise ValueError("final gold hit is missing its pre-rerank lineage")
            pre_rank = pre[gold_chunk_id][0]
            reranker_score = final[gold_chunk_id][1]
            top4 = tuple(
                chunk_id
                for chunk_id, (rank, _score) in sorted(
                    final.items(), key=lambda item: item[1][0]
                )
                if rank <= 4 and chunk_id not in gold_chunk_ids
            )
            if not top4:
                raise ValueError("ranked hydration loss requires Top 4 competitors")
            top4_scores = tuple(final[chunk_id][1] for chunk_id in top4)
            overtakers = tuple(
                chunk_id
                for chunk_id in top4
                if chunk_id in pre and pre[chunk_id][0] > pre_rank
            )
            gold_chunk = chunks[gold_chunk_id]
            competitor_chunks = tuple(chunks[chunk_id] for chunk_id in top4)
            same_page = tuple(
                chunk_id
                for chunk_id in top4
                if _same_page(gold_chunk, chunks[chunk_id])
            )
            same_section = tuple(
                chunk_id
                for chunk_id in top4
                if _same_section(gold_chunk, chunks[chunk_id])
            )
            query_text = request.rewritten_query or request.original_query
            gold_overlap = token_coverage(query_text, str(gold_chunk.get("text", "")))
            top4_overlaps = tuple(
                token_coverage(query_text, str(chunk.get("text", "")))
                for chunk in competitor_chunks
            )
            available_overlaps = tuple(
                value for value in top4_overlaps if value is not None
            )
            top4_max_overlap = max(available_overlaps) if available_overlaps else None
            top4_median_chars = float(
                statistics.median(len(str(chunk.get("text", ""))) for chunk in competitor_chunks)
            )
            gold_chars = len(str(gold_chunk.get("text", "")))
            broad_reproduced: bool | None = None
            atomic_eligible = False
            atomic_rank: int | None = None
            atomic_improvement: int | None = None
            atomic_reaches_top4: bool | None = None
            if reranker is not None and query_text:
                candidate_ids = tuple(
                    chunk_id
                    for chunk_id, (_rank, _score) in sorted(
                        pre.items(), key=lambda item: item[1][0]
                    )
                )
                candidate_texts = tuple(
                    str(chunks[chunk_id].get("text", "")) for chunk_id in candidate_ids
                )
                broad_ranked = _rank_with_model(
                    reranker,
                    query_text,
                    candidate_ids,
                    candidate_texts,
                )
                audited_final = tuple(
                    chunk_id
                    for chunk_id, (_rank, _score) in sorted(
                        final.items(), key=lambda item: item[1][0]
                    )
                )
                broad_reproduced = broad_ranked[: len(audited_final)] == audited_final
                fact_text = claim_by_id[claim_id].normalized_fact
                atomic_query = (
                    (atomic_queries or {}).get(fact_id)
                    or (fact_text if _ascii_ratio(fact_text) >= 0.8 else None)
                )
                atomic_eligible = atomic_query is not None
                if atomic_query is not None:
                    atomic_ranked = _rank_with_model(
                        reranker,
                        atomic_query,
                        candidate_ids,
                        candidate_texts,
                    )
                    atomic_rank = min(
                        index
                        for index, chunk_id in enumerate(atomic_ranked, start=1)
                        if chunk_id in gold_chunk_ids
                    )
                    atomic_improvement = final_rank - atomic_rank
                    atomic_reaches_top4 = atomic_rank <= 4
            diagnostics.append(
                RerankerFactDiagnostic(
                    fact_id=fact_id,
                    final_rank=final_rank,
                    pre_rerank_rank=pre_rank,
                    rank_delta=final_rank - pre_rank,
                    search_occurrences=len(occurrences),
                    reranker_score=reranker_score,
                    top4_score_floor=min(top4_scores),
                    score_margin_to_top4_floor=reranker_score - min(top4_scores),
                    top4_overtaker_count=len(overtakers),
                    same_page_top4=bool(same_page),
                    same_section_top4=bool(same_section),
                    same_page_overtaker_count=sum(
                        chunk_id in overtakers for chunk_id in same_page
                    ),
                    same_section_overtaker_count=sum(
                        chunk_id in overtakers for chunk_id in same_section
                    ),
                    gold_chunk_chars=gold_chars,
                    top4_median_chunk_chars=top4_median_chars,
                    chunk_length_ratio=(
                        gold_chars / top4_median_chars if top4_median_chars else None
                    ),
                    query_fact_token_coverage=token_coverage(
                        claim_by_id[claim_id].normalized_fact,
                        " ".join(
                            value
                            for value in (
                                request.original_query,
                                request.rewritten_query,
                            )
                            if value
                        ),
                    ),
                    gold_query_token_coverage=gold_overlap,
                    top4_max_query_token_coverage=top4_max_overlap,
                    query_overlap_gap=(
                        gold_overlap - top4_max_overlap
                        if gold_overlap is not None and top4_max_overlap is not None
                        else None
                    ),
                    broad_query_reproduced=broad_reproduced,
                    atomic_query_eligible=atomic_eligible,
                    atomic_query_rank=atomic_rank,
                    atomic_rank_improvement=atomic_improvement,
                    atomic_query_reaches_top4=atomic_reaches_top4,
                )
            )

    metadata = {
        "schema_version": "reranker-cause-diagnostic-v1",
        "source_code_revision": payload.get("code_revision"),
        "question_count": len(cases),
        "retrieval_count": len(signatures),
        "matched_retrieval_count": len(audit_requests),
        "ambiguous_signature_count": ambiguity_count,
    }
    return tuple(diagnostics), metadata


def _same_page(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("corpus_id") == right.get("corpus_id")
        and int(left["page_start"]) <= int(right["page_end"])
        and int(right["page_start"]) <= int(left["page_end"])
    )


def _rank_with_model(
    reranker: Any,
    query: str,
    candidate_ids: tuple[str, ...],
    candidate_texts: tuple[str, ...],
) -> tuple[str, ...]:
    scores = tuple(float(value) for value in reranker.score(query, candidate_texts))
    if len(scores) != len(candidate_ids):
        raise ValueError("reranker score count does not match candidate count")
    return tuple(
        chunk_id
        for chunk_id, _score in sorted(
            zip(candidate_ids, scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
    )


def _ascii_ratio(value: str) -> float:
    return sum(ord(character) < 128 for character in value) / max(1, len(value))


def _same_section(left: dict[str, Any], right: dict[str, Any]) -> bool:
    section_id = left.get("section_id")
    return bool(
        section_id
        and left.get("corpus_id") == right.get("corpus_id")
        and section_id == right.get("section_id")
    )


def render_report(
    diagnostics: tuple[RerankerFactDiagnostic, ...],
    metadata: dict[str, object],
) -> str:
    summary = summarize_reranker_causes(diagnostics)
    lines = [
        "# 当前重排器误排原因诊断",
        "",
        "本报告复用既有 30 题运行及本地查询审计，只输出安全 ID、排名、分数差和聚合特征，不包含查询、事实或论文正文。",
        "",
        "## 样本与血缘",
        "",
        f"- 代码版本：`{metadata.get('source_code_revision', 'unknown')}`",
        f"- 题目数：{metadata['question_count']}",
        f"- 检索次数：{metadata['retrieval_count']}",
        f"- 匹配审计的检索签名：{metadata['matched_retrieval_count']}",
        f"- 存在历史重复签名：{metadata['ambiguous_signature_count']}（按评测完成时间之前最近一次审计确定）",
        f"- 真实水化缺失事实：{summary.fact_count}",
        "",
        "## 聚合结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 重排升权 / 降权 / 不变 | {summary.reranker_promoted_count} / {summary.reranker_demoted_count} / {summary.reranker_unchanged_count} |",
        f"| 排名变化中位数（正数为降权） | {_number(summary.median_rank_delta)} |",
        f"| 被原本排在其后的 Top 4 块越位的事实 | {summary.facts_with_top4_overtakers}/{summary.fact_count} |",
        f"| Top 4 越位块总数 | {summary.total_top4_overtaker_count} |",
        f"| 同页 / 同节 Top 4 竞争事实 | {summary.same_page_top4_count} / {summary.same_section_top4_count} |",
        f"| 同页 / 同节越位块数量 | {summary.same_page_overtaker_count} / {summary.same_section_overtaker_count} |",
        f"| 金标块比 Top 4 中位块更长 | {summary.gold_longer_than_top4_median_count}/{summary.fact_count} |",
        f"| 金标块长度比中位数 | {_number(summary.median_chunk_length_ratio)} |",
        f"| 查询词覆盖劣于最佳 Top 4 块 | {summary.query_overlap_disadvantage_count}/{summary.fact_count} |",
        f"| 查询覆盖必要事实 Token 中位数 | {_percent(summary.median_query_fact_token_coverage)} |",
        f"| 金标块相对 Top 4 的查询覆盖差中位数 | {_percent(summary.median_query_overlap_gap)} |",
        f"| 金标块相对 Top 4 最低重排分数差中位数 | {_number(summary.median_score_margin_to_top4_floor, digits=4)} |",
        f"| 宽查询重排顺序离线复现 | {summary.broad_query_reproduction_count}/{summary.fact_count} |",
        f"| 可做原子查询消融 | {summary.atomic_query_eligible_count}/{summary.fact_count} |",
        f"| 原子查询后进入 Top 4 | {summary.atomic_query_top4_count}/{summary.atomic_query_eligible_count} |",
        f"| 原子查询排名提升中位数 | {_number(summary.median_atomic_rank_improvement)} |",
        "",
        "## 逐事实安全诊断",
        "",
        "| 事实 ID | 融合排名 | 最终排名 | Δ | 越位块 | 同页 | 同节 | 长度比 | 查询覆盖事实 | 查询覆盖差 | 分数差 | 原子查询排名 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in diagnostics:
        lines.append(
            f"| `{item.fact_id}` | {item.pre_rerank_rank} | {item.final_rank} | "
            f"{item.rank_delta:+d} | {item.top4_overtaker_count} | "
            f"{'是' if item.same_page_top4 else '否'} | "
            f"{'是' if item.same_section_top4 else '否'} | "
            f"{_number(item.chunk_length_ratio)} | "
            f"{_percent(item.query_fact_token_coverage)} | "
            f"{_percent(item.query_overlap_gap)} | "
            f"{_number(item.score_margin_to_top4_floor, digits=4)} | "
            f"{item.atomic_query_rank if item.atomic_query_rank is not None else '—'} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 越位是直接排名证据：Top 4 块在融合阶段位于金标块之后，但经重排后反超。",
            "- 同页、同节、长度和词覆盖属于相关性诊断，不能单独证明因果；下一步应只针对出现集中的特征做受控重排实验。",
            "- 原子查询消融使用私有金标事实作为 oracle（理想化查询），只用于验证查询粒度因果，不可直接作为生产查询。",
            "- 本报告不评价扩大 Top K，也不把关闭重排器当作最终方案。",
        ]
    )
    return "\n".join(lines) + "\n"


def _number(value: float | None, *, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose current reranker failures.")
    parser.add_argument(
        "--run",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/evaluations/runs/comparison-end-to-end-minimal-compiler-v2-full30-v2.json"
        ),
    )
    parser.add_argument(
        "--answer-gold",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/comparison-end-to-end-gold-v1.jsonl",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "data/processed/chunks/chunks.jsonl",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "data/runtime/query-audit-v1.sqlite3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/当前重排器误排原因诊断-v1.md",
    )
    parser.add_argument("--atomic-query-ablation", action="store_true")
    parser.add_argument("--translate-non-english-atomic", action="store_true")
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=PROJECT_ROOT / "configs/retrieval/hybrid-rerank-v1.json",
    )
    parser.add_argument(
        "--bilingual-config",
        type=Path,
        default=PROJECT_ROOT / "configs/retrieval/bilingual-qwen-v1.json",
    )
    return parser


def main() -> None:
    _load_local_env()
    args = _build_parser().parse_args()
    reranker = None
    if args.atomic_query_ablation:
        config = load_retrieval_config(args.retrieval_config)
        reranker = FastEmbedReranker(
            config.reranker_model,
            revision=config.reranker_revision,
        )
    atomic_queries: dict[str, str] = {}
    if args.translate_non_english_atomic:
        if not args.atomic_query_ablation:
            raise ValueError("atomic translation requires --atomic-query-ablation")
        bilingual = load_bilingual_retrieval_config(args.bilingual_config)
        atomic_queries = asyncio.run(
            _translate_non_english_atomic_queries(
                args.run,
                answer_gold_path=args.answer_gold,
                model_id=bilingual.rewrite_model,
                timeout_seconds=bilingual.rewrite_timeout_seconds,
            )
        )
    diagnostics, metadata = diagnose_run(
        args.run,
        answer_gold_path=args.answer_gold,
        chunks_path=args.chunks,
        audit_path=args.audit,
        reranker=reranker,
        atomic_queries=atomic_queries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(diagnostics, metadata), encoding="utf-8")
    print(
        json.dumps(
            {
                "fact_count": len(diagnostics),
                "summary": summarize_reranker_causes(diagnostics).model_dump(mode="json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
