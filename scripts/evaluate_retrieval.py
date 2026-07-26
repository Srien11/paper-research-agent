from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.evaluation.dataset import load_dataset
from paper_research_agent.evaluation.runner import evaluate, write_chinese_report
from paper_research_agent.retrieval.bm25 import BM25Index
from paper_research_agent.retrieval.config import load_retrieval_config
from paper_research_agent.retrieval.contracts import IndexManifest
from paper_research_agent.retrieval.model_adapters import (
    FastEmbedEncoder,
    FastEmbedReranker,
)
from paper_research_agent.retrieval.service import RetrievalService
from paper_research_agent.retrieval.vector import FaissVectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comparable A/B/C retrieval experiments.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation/datasets/dev-silver-v1.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/retrieval/hybrid-rerank-v1.json",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/evaluations/runs/baseline-v1.json"
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports/检索C基线诊断报告-v1.md"
    )
    args = parser.parse_args()
    config = load_retrieval_config(args.config)
    chunks = [
        EvidenceChunk.model_validate_json(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = IndexManifest.model_validate_json(
        (config.index_dir / "manifest.json").read_text(encoding="utf-8")
    )
    service = RetrievalService(
        BM25Index(chunks),
        FaissVectorIndex(
            chunks,
            FastEmbedEncoder(config.embedding_model, revision=config.embedding_revision),
            config.index_dir / "vectors.faiss",
        ),
        FastEmbedReranker(config.reranker_model, revision=config.reranker_revision),
        config,
        index_id=manifest.index_id,
    )
    index_size_bytes = sum(
        (config.index_dir / name).stat().st_size
        for name in ("vectors.faiss", "metadata.sqlite", "manifest.json")
    )
    result = evaluate(
        service,
        load_dataset(args.dataset),
        args.output,
        index_size_bytes=index_size_bytes,
    )
    write_chinese_report(result, args.report)
    print(args.output)
    print(args.report)


if __name__ == "__main__":
    main()
