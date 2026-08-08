from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk, PaperCard
from paper_research_agent.evaluation.candidate_gold import load_candidate_paper_gold
from paper_research_agent.evaluation.candidate_runner import (
    CandidatePaperEvaluationCase,
    summarize_candidate_paper_cases,
)
from paper_research_agent.retrieval.config import (
    load_bilingual_retrieval_config,
    load_retrieval_config,
)
from paper_research_agent.retrieval.model_adapters import FastEmbedEncoder
from paper_research_agent.retrieval.papers import (
    HybridPaperCandidateRetriever,
    PaperCandidateQuery,
    build_paper_candidate_documents,
)
from paper_research_agent.retrieval.query_rewrite import DashScopeQueryRewriter


def _read_jsonl(path: Path, model):
    return tuple(
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


async def run(args: argparse.Namespace) -> dict[str, object]:
    cards = _read_jsonl(args.paper_cards, PaperCard)
    chunks = _read_jsonl(args.chunks, EvidenceChunk)
    questions = load_candidate_paper_gold(args.gold)
    catalog = {card.corpus_id for card in cards}
    referenced = {
        paper_id
        for question in questions
        for paper_id in (
            *question.relevant_paper_ids,
            *question.nearest_distractor_paper_ids,
        )
    }
    if unknown := sorted(referenced - catalog):
        raise ValueError(f"candidate gold references {len(unknown)} unknown papers")

    retrieval_config = load_retrieval_config(args.retrieval_config)
    bilingual_config = load_bilingual_retrieval_config(args.bilingual_config)
    retriever = HybridPaperCandidateRetriever(
        build_paper_candidate_documents(cards, chunks),
        FastEmbedEncoder(
            retrieval_config.embedding_model,
            revision=retrieval_config.embedding_revision,
        ),
    )
    rewriter = DashScopeQueryRewriter(
        bilingual_config.rewrite_model,
        timeout_seconds=(
            bilingual_config.rewrite_timeout_seconds
            if args.rewrite_timeout is None
            else args.rewrite_timeout
        ),
    )
    cases: list[CandidatePaperEvaluationCase] = []
    try:
        for question in questions:
            english_query: str | None = None
            rewrite_status = "success"
            try:
                english_query = (await rewriter.rewrite(question.question)).english_query
            except (RuntimeError, TimeoutError):
                rewrite_status = "fallback_original"
            hits = await retriever.search(
                PaperCandidateQuery(
                    original_query=question.question,
                    english_query=english_query,
                ),
                top_k=args.top_k,
            )
            cases.append(
                CandidatePaperEvaluationCase(
                    question_id=question.question_id,
                    split=question.split,
                    clue_scope=question.clue_scope,
                    relevant_paper_ids=question.relevant_paper_ids,
                    candidate_paper_ids=tuple(hit.corpus_id for hit in hits),
                    rewrite_status=rewrite_status,
                )
            )
    finally:
        await rewriter.aclose()

    summary = summarize_candidate_paper_cases(cases, cutoffs=(5, 8))
    return {
        **summary,
        "candidate_top_k": args.top_k,
        "embedding_model": retrieval_config.embedding_model,
        "dev_cases": [
            case.model_dump(mode="json") for case in cases if case.split == "dev"
        ],
        "sealed_test_case_count": sum(case.split == "sealed_test" for case in cases),
        "limitations": (
        "封存测试题正文和逐题结果不进入报告；候选评测集经过题目构造与反向核对"
        "两轮一致性检查，未经过独立多人仲裁。"
        ),
    }


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _write_report(result: dict[str, object], path: Path) -> None:
    primary = result["primary"]
    rewrite_success = result["rewrite_success"]
    rewrite_fallback = result["rewrite_fallback"]
    dev = result["dev"]
    sealed = result["sealed_test"]
    assert (
        isinstance(primary, dict)
        and isinstance(rewrite_success, dict)
        and isinstance(rewrite_fallback, dict)
        and isinstance(dev, dict)
        and isinstance(sealed, dict)
    )
    lines = [
        "# RAG 候选论文召回评测基线 v1",
        "",
        f"- 金标题数：{result['question_count']}（开发集20，封存集10）",
        f"- 转化降级次数：{result['rewrite_fallback_count']}",
        f"- 总体 Recall@5：{_percent(primary['recall_at_5_macro'])}",
        f"- 总体 Recall@8：{_percent(primary['recall_at_8_macro'])}",
        f"- 总体 All-target Hit@8：{_percent(primary['all_target_hit_at_8'])}",
        f"- 转化成功子集 Recall@8：{_percent(rewrite_success['recall_at_8_macro'])}",
        f"- 转化降级子集 Recall@8：{_percent(rewrite_fallback['recall_at_8_macro'])}",
        f"- 开发集 Recall@8：{_percent(dev['recall_at_8_macro'])}",
        f"- 封存集 Recall@8：{_percent(sealed['recall_at_8_macro'])}",
        f"- 封存集 All-target Hit@8：{_percent(sealed['all_target_hit_at_8'])}",
        "",
        "## 数据边界",
        "",
        f"- {result['limitations']}",
        "- 金标问题、目标编号和短标注理由保存在 Git 忽略的私有目录。",
        "- 报告不包含论文摘要、原文证据或封存题正文。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate private candidate-paper gold.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/candidate-paper-gold-v1.jsonl",
    )
    parser.add_argument(
        "--paper-cards",
        type=Path,
        default=PROJECT_ROOT / "data/processed/chunks/paper_cards.jsonl",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "data/processed/chunks/chunks.jsonl",
    )
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
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rewrite-timeout", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/runs/candidate-paper-gold-v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/RAG候选论文人工金标基线-v1.md",
    )
    args = parser.parse_args()
    if args.top_k < 8:
        raise ValueError("top_k must be at least 8 for the frozen metrics")
    if args.rewrite_timeout is not None and args.rewrite_timeout <= 0:
        raise ValueError("rewrite_timeout must be positive")
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(result, args.report)
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
