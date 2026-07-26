from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.retrieval.config import load_retrieval_config
from paper_research_agent.retrieval.indexer import build_index
from paper_research_agent.retrieval.model_adapters import FastEmbedEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local BM25/FAISS retrieval index.")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/retrieval/hybrid-rerank-v1.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_retrieval_config(args.config)
    chunks = [
        EvidenceChunk.model_validate_json(line)
        for line in args.chunks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = build_index(
        chunks,
        FastEmbedEncoder(config.embedding_model, revision=config.embedding_revision),
        args.output or config.index_dir,
        embedding_model=config.embedding_model,
        embedding_revision=config.embedding_revision,
        chunk_build_sha256=hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
