from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.retrieval.bilingual import BilingualRetrievalService
from paper_research_agent.retrieval.bm25 import BM25Index
from paper_research_agent.retrieval.config import (
    load_bilingual_retrieval_config,
    load_retrieval_config,
)
from paper_research_agent.retrieval.contracts import IndexManifest
from paper_research_agent.retrieval.model_adapters import (
    FastEmbedEncoder,
    FastEmbedReranker,
)
from paper_research_agent.retrieval.query_rewrite import (
    DashScopeQueryRewriter,
    UnavailableQueryRewriter,
)
from paper_research_agent.retrieval.query_store import (
    NullQueryRewriteCache,
    SQLiteQueryAuditLogger,
    SQLiteQueryRewriteCache,
)
from paper_research_agent.retrieval.rights import CorpusRightsMap
from paper_research_agent.retrieval.service import RetrievalService
from paper_research_agent.retrieval.vector import FaissVectorIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the A/B/C retrieval baseline.")
    parser.add_argument("query")
    parser.add_argument("--variant", choices=("A", "B", "C"), default="C")
    parser.add_argument(
        "--bilingual",
        action="store_true",
        help="Rewrite a Chinese query and run the production Chinese/English dual route.",
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/retrieval/hybrid-rerank-v1.json",
    )
    parser.add_argument(
        "--bilingual-config",
        type=Path,
        default=PROJECT_ROOT / "configs/retrieval/bilingual-qwen-v1.json",
    )
    corpus_dir = os.getenv("PRA_CORPUS_DIR")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(corpus_dir) if corpus_dir else None,
        help="Frozen manifest directory used to attach redistribution rights.",
    )
    args = parser.parse_args()
    if args.bilingual and args.variant != "C":
        parser.error("--variant A/B cannot be combined with --bilingual")
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
    sparse = BM25Index(chunks)
    vector = FaissVectorIndex(chunks, encoder, config.index_dir / "vectors.faiss")
    reranker = FastEmbedReranker(config.reranker_model, revision=config.reranker_revision)
    if not args.bilingual:
        service = RetrievalService(
            sparse,
            vector,
            reranker,
            config,
            index_id=manifest.index_id,
        )
        result = service.search(args.query, args.variant, top_k=args.top_k)
    else:
        result = asyncio.run(
            _search_bilingual(
                args.query,
                top_k=args.top_k,
                corpus_dir=args.corpus_dir,
                sparse=sparse,
                vector=vector,
                reranker=reranker,
                retrieval_config=config,
                bilingual_config_path=args.bilingual_config,
                index_id=manifest.index_id,
            )
        )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


async def _search_bilingual(
    query,
    *,
    top_k,
    corpus_dir,
    sparse,
    vector,
    reranker,
    retrieval_config,
    bilingual_config_path,
    index_id,
    privacy_ttl_days=None,
):
    bilingual_config = load_bilingual_retrieval_config(bilingual_config_path)
    try:
        rewriter = DashScopeQueryRewriter(
            bilingual_config.rewrite_model,
            timeout_seconds=bilingual_config.rewrite_timeout_seconds,
        )
    except RuntimeError:
        rewriter = UnavailableQueryRewriter(
            bilingual_config.rewrite_model,
            reason="query rewrite credentials are unavailable",
        )
    try:
        cache = SQLiteQueryRewriteCache(bilingual_config.cache_path)
    except (OSError, sqlite3.Error):
        cache = NullQueryRewriteCache()
    try:
        audit = SQLiteQueryAuditLogger(
            bilingual_config.audit_path,
            plaintext_days=bilingual_config.audit_plaintext_days,
        )
    except (OSError, sqlite3.Error):
        audit = None
    rights = CorpusRightsMap.from_corpus_dir(corpus_dir) if corpus_dir is not None else None
    service = BilingualRetrievalService(
        sparse,
        vector,
        reranker,
        rewriter,
        cache,
        audit,
        retrieval_config,
        bilingual_config,
        index_id=index_id,
        rights=rights,
    )
    try:
        return await service.search(
            query,
            top_k=top_k,
            privacy_ttl_days=privacy_ttl_days,
        )
    finally:
        await service.aclose()
        close = getattr(rewriter, "aclose", None)
        if close is not None:
            await close()


if __name__ == "__main__":
    main()
