from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
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
    parser = argparse.ArgumentParser(description="Search the A/B/C retrieval baseline.")
    parser.add_argument("query")
    parser.add_argument("--variant", choices=("A", "B", "C"), default="C")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/retrieval/hybrid-rerank-v1.json",
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
    encoder = FastEmbedEncoder(config.embedding_model, revision=config.embedding_revision)
    service = RetrievalService(
        BM25Index(chunks),
        FaissVectorIndex(chunks, encoder, config.index_dir / "vectors.faiss"),
        FastEmbedReranker(config.reranker_model, revision=config.reranker_revision),
        config,
        index_id=manifest.index_id,
    )
    print(json.dumps(service.search(args.query, args.variant).model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
