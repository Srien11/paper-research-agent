from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from answer import _optional_audit
from search import _search_bilingual

from paper_research_agent.answering.config import load_answering_config
from paper_research_agent.answering.dashscope import (
    DashScopeAnswerGenerator,
    UnavailableAnswerGenerator,
)
from paper_research_agent.answering.models import RAGAnswer
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.memory.config import load_memory_config
from paper_research_agent.memory.context import contextualize_retrieval_query, to_context_memory
from paper_research_agent.memory.service import turn_from_answer
from paper_research_agent.memory.store import SQLiteShortTermMemory
from paper_research_agent.rag import answer_retrieval_run
from paper_research_agent.retrieval.bm25 import BM25Index
from paper_research_agent.retrieval.config import load_retrieval_config
from paper_research_agent.retrieval.contracts import IndexManifest
from paper_research_agent.retrieval.model_adapters import FastEmbedEncoder, FastEmbedReranker
from paper_research_agent.retrieval.vector import FaissVectorIndex

DEFAULT_RETRIEVAL_CONFIG = PROJECT_ROOT / "configs/retrieval/hybrid-rerank-v1.json"
DEFAULT_BILINGUAL_CONFIG = PROJECT_ROOT / "configs/retrieval/bilingual-qwen-v1.json"
DEFAULT_ANSWER_CONFIG = PROJECT_ROOT / "configs/answering/qwen-rag-v1.json"
DEFAULT_MEMORY_CONFIG = PROJECT_ROOT / "configs/memory/short-term-v1.json"
DEFAULT_ANSWER_AUDIT = PROJECT_ROOT / "data/runtime/answer-audit-v1.sqlite3"


async def run_rag(
    question: str,
    *,
    chunks_path: Path,
    corpus_dir: Path,
    retrieval_config_path: Path = DEFAULT_RETRIEVAL_CONFIG,
    bilingual_config_path: Path = DEFAULT_BILINGUAL_CONFIG,
    answer_config_path: Path = DEFAULT_ANSWER_CONFIG,
    top_k: int | None = None,
    token_budget: int = 8192,
    output_reserve_tokens: int = 1200,
    output_path: Path | None = None,
    audit_path: Path = DEFAULT_ANSWER_AUDIT,
    session_id: str | None = None,
    memory_config_path: Path = DEFAULT_MEMORY_CONFIG,
) -> RAGAnswer:
    retrieval_config = load_retrieval_config(retrieval_config_path)
    chunks = [
        EvidenceChunk.model_validate_json(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = IndexManifest.model_validate_json(
        (retrieval_config.index_dir / "manifest.json").read_text(encoding="utf-8")
    )
    encoder = FastEmbedEncoder(
        retrieval_config.embedding_model,
        revision=retrieval_config.embedding_revision,
    )
    sparse = BM25Index(chunks)
    vector = FaissVectorIndex(
        chunks,
        encoder,
        retrieval_config.index_dir / "vectors.faiss",
    )
    reranker = FastEmbedReranker(
        retrieval_config.reranker_model,
        revision=retrieval_config.reranker_revision,
    )
    memory_config = load_memory_config(memory_config_path)
    memory_store: SQLiteShortTermMemory | None = None
    memory_turns = ()
    if session_id is not None:
        memory_path = PROJECT_ROOT / memory_config.store_path
        try:
            memory_store = SQLiteShortTermMemory(memory_path, config=memory_config)
            memory_turns = memory_store.recent(session_id)
        except (OSError, sqlite3.Error):
            memory_store = None
            memory_turns = ()
    retrieval_question = contextualize_retrieval_query(
        question,
        memory_turns,
        max_question_chars=memory_config.follow_up_max_chars,
    )
    run = await _search_bilingual(
        retrieval_question,
        top_k=top_k,
        corpus_dir=corpus_dir,
        sparse=sparse,
        vector=vector,
        reranker=reranker,
        retrieval_config=retrieval_config,
        bilingual_config_path=bilingual_config_path,
        index_id=manifest.index_id,
        privacy_ttl_days=(
            max(1, math.ceil(memory_config.ttl_hours / 24)) if session_id is not None else None
        ),
    )
    answer_config = load_answering_config(answer_config_path)
    try:
        generator = DashScopeAnswerGenerator(answer_config)
    except RuntimeError:
        generator = UnavailableAnswerGenerator(answer_config.model, answer_config.prompt_version)
    try:
        result = await answer_retrieval_run(
            run,
            chunks=chunks,
            generator=generator,
            audit=_optional_audit(audit_path),
            token_budget=token_budget,
            output_reserve_tokens=output_reserve_tokens,
            user_question=question,
            short_term_memory=to_context_memory(memory_turns),
            memory_token_budget=(
                memory_config.context_token_budget if session_id is not None else 0
            ),
            protected_evidence_count=(
                memory_config.protected_evidence_count if session_id is not None else 1
            ),
        )
    finally:
        close = getattr(generator, "aclose", None)
        if close is not None:
            await close()
    if session_id is not None and memory_store is not None:
        try:
            turn = turn_from_answer(
                session_id,
                question,
                result,
                config=memory_config,
                standalone_question=retrieval_question,
            )
            memory_store.append(turn)
        except (OSError, sqlite3.Error, ValueError):
            pass
    payload = result.model_dump(mode="json")
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path is None:
        print(rendered)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chinese query rewrite, retrieval, context assembly, and answer validation."
    )
    parser.add_argument("question")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--retrieval-config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--bilingual-config", type=Path, default=DEFAULT_BILINGUAL_CONFIG)
    parser.add_argument("--answer-config", type=Path, default=DEFAULT_ANSWER_CONFIG)
    parser.add_argument("--memory-config", type=Path, default=DEFAULT_MEMORY_CONFIG)
    parser.add_argument(
        "--session-id",
        help="Enable 24-hour local short-term memory for this explicit session ID.",
    )
    configured_corpus = os.getenv("PRA_CORPUS_DIR")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(configured_corpus) if configured_corpus else None,
    )
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--output-reserve", type=int, default=1200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_ANSWER_AUDIT)
    args = parser.parse_args()
    if args.corpus_dir is None:
        parser.error("--corpus-dir or PRA_CORPUS_DIR is required for rights enforcement")
    try:
        asyncio.run(
            run_rag(
                args.question,
                chunks_path=args.chunks,
                corpus_dir=args.corpus_dir,
                retrieval_config_path=args.retrieval_config,
                bilingual_config_path=args.bilingual_config,
                answer_config_path=args.answer_config,
                top_k=args.top_k,
                token_budget=args.token_budget,
                output_reserve_tokens=args.output_reserve,
                output_path=args.output,
                audit_path=args.audit_path,
                session_id=args.session_id,
                memory_config_path=args.memory_config,
            )
        )
    except Exception as error:  # noqa: BLE001 - sanitize every CLI failure
        parser.exit(1, f"rag failed: {type(error).__name__}\n")


if __name__ == "__main__":
    main()
