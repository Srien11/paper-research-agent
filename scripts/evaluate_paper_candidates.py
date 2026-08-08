from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.agent.planner import parse_explicit_corpus_ids
from paper_research_agent.chunking.models import EvidenceChunk, PaperCard
from paper_research_agent.evaluation.gold_dataset import load_gold_dataset
from paper_research_agent.evaluation.metrics import (
    candidate_paper_recall,
    explicit_corpus_id_accuracy,
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
    documents = build_paper_candidate_documents(cards, chunks)
    config = load_retrieval_config(args.retrieval_config)
    bilingual_config = load_bilingual_retrieval_config(args.bilingual_config)
    retriever = HybridPaperCandidateRetriever(
        documents,
        FastEmbedEncoder(config.embedding_model, revision=config.embedding_revision),
    )
    questions = tuple(
        question
        for question in load_gold_dataset(args.dataset)
        if question.task_type == "multi_paper_comparison" and question.answerable
    )[: args.limit]
    cases: list[dict[str, object]] = []
    rewriter = DashScopeQueryRewriter(
        bilingual_config.rewrite_model,
        timeout_seconds=bilingual_config.rewrite_timeout_seconds,
    )
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
            relevant = {
                span.paper_id
                for span in question.evidence_spans
                if span.support_role != "distractor"
            }
            candidate_ids = [hit.corpus_id for hit in hits]
            cases.append(
                {
                    "question_id": question.question_id,
                    "rewrite_status": rewrite_status,
                    "relevant_paper_ids": sorted(relevant),
                    "candidate_paper_ids": candidate_ids,
                    "candidate_paper_recall": candidate_paper_recall(
                        candidate_ids, relevant
                    ),
                }
            )
    finally:
        await rewriter.aclose()
    recalls = [float(case["candidate_paper_recall"]) for case in cases]
    parser_inputs = (
        "Compare C001 and T001",
        "比较 c001、t001。",
        "C001 versus C001 and T001",
        "XC001 C0010",
    )
    parser_expected = (
        ("C001", "T001"),
        ("C001", "T001"),
        ("C001", "T001"),
        (),
    )
    result: dict[str, object] = {
        "candidate_top_k": args.top_k,
        "embedding_model": config.embedding_model,
        "question_count": len(cases),
        "candidate_paper_recall_macro": sum(recalls) / len(recalls) if recalls else None,
        "explicit_corpus_id_accuracy": explicit_corpus_id_accuracy(
            [parse_explicit_corpus_ids(value) for value in parser_inputs],
            parser_expected,
        ),
        "missing_abstract_fallback_count": sum(document.used_fallback for document in documents),
        "missing_abstract_fallback_corpus_ids": [
            document.corpus_id for document in documents if document.used_fallback
        ],
        "cases": cases,
        "limitations": (
            "银标仅用于诊断：GQ016 更接近单论文识别，GQ017 更接近跨论文多跳问答，"
            "两者都没有硬编码论文编号。当前向量模型的跨语言能力会影响中文问题的模型分数，"
            "不影响显式编号、候选边界和 corpus_id 隔离等结构保证。"
        ),
    }
    return result


def _write_report(result: dict[str, object], path: Path) -> None:
    lines = [
        "# RAG 多论文候选检索小样本诊断",
        "",
        (
            f"- Candidate Paper Recall@{result['candidate_top_k']}（macro）："
            f"{result['candidate_paper_recall_macro']:.2%}"
        ),
        f"- 显式编号解析准确率：{result['explicit_corpus_id_accuracy']:.2%}",
        f"- 摘要缺失回退论文数：{result['missing_abstract_fallback_count']}",
        f"- 论文候选向量模型：`{result['embedding_model']}`",
        "",
        "| 问题 | 标注论文 | 候选论文 | Recall |",
        "|---|---|---|---:|",
    ]
    for case in result["cases"]:
        lines.append(
            f"| `{case['question_id']}` | "
            f"{', '.join(case['relevant_paper_ids'])} | "
            f"{', '.join(case['candidate_paper_ids'])} | "
            f"{case['candidate_paper_recall']:.2%} |"
        )
    lines.extend(("", "## 解释边界", "", f"- {result['limitations']}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paper-level candidate retrieval.")
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
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/gold/rag-answer-candidates-v1.jsonl",
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
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/evaluations/runs/paper-candidates-sample5.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/RAG多论文候选检索-sample5.md",
    )
    args = parser.parse_args()
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
