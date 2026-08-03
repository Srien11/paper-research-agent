"""Persistent, single-flight Web runtime over the private local RAG pipeline.

Only the explicitly whitelisted models in this module cross the Web boundary.  In
particular, source paths, complete figure records, provider payloads, prompts, and
retrieval scores never appear in :class:`RuntimeExecutionResult`.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.answering.audit import SQLiteAnswerAuditLogger
from paper_research_agent.answering.config import load_answering_config
from paper_research_agent.answering.dashscope import (
    AsyncAnswerGenerator,
    DashScopeAnswerGenerator,
)
from paper_research_agent.answering.models import AnswerRequest, RAGAnswer
from paper_research_agent.answering.service import AnswerAuditLogger, answer_context
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.adapters import join_retrieval_evidence
from paper_research_agent.context.assembler import assemble_context
from paper_research_agent.context.models import AssembledContext, ContextRequest
from paper_research_agent.corpus import load_frozen_papers
from paper_research_agent.memory.config import ShortTermMemoryConfig, load_memory_config
from paper_research_agent.memory.context import contextualize_retrieval_query, to_context_memory
from paper_research_agent.memory.service import turn_from_answer
from paper_research_agent.memory.store import ShortTermMemoryStore, SQLiteShortTermMemory
from paper_research_agent.models import FrozenPaper
from paper_research_agent.rag import DEFAULT_RAG_SYSTEM_RULES, AsyncBilingualRetriever
from paper_research_agent.retrieval.bilingual import BilingualRetrievalService
from paper_research_agent.retrieval.bm25 import BM25Index
from paper_research_agent.retrieval.config import (
    RetrievalConfig,
    load_bilingual_retrieval_config,
    load_retrieval_config,
)
from paper_research_agent.retrieval.contracts import BilingualRetrievalRun, IndexManifest
from paper_research_agent.retrieval.model_adapters import FastEmbedEncoder, FastEmbedReranker
from paper_research_agent.retrieval.query_rewrite import (
    AsyncQueryRewriter,
    DashScopeQueryRewriter,
    UnavailableQueryRewriter,
)
from paper_research_agent.retrieval.query_store import (
    NullQueryRewriteCache,
    QueryRewriteCache,
    SQLiteQueryAuditLogger,
    SQLiteQueryRewriteCache,
)
from paper_research_agent.retrieval.rights import CorpusRightsMap
from paper_research_agent.retrieval.vector import FaissVectorIndex

StorageClass = Literal["redistributable", "internal_research_only"]
EvidenceType = Literal["text", "figure_summary"]
RewriteStatus = Literal["success", "cache_hit", "stale_cache", "timeout", "error"]


class _FrozenWebModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SafePaperMetadata(_FrozenWebModel):
    """The only corpus-manifest fields allowed to reach the owner UI."""

    corpus_id: str = Field(pattern=r"^[CT]\d{3}$")
    title: str = Field(min_length=1)
    official_url: str = Field(min_length=1)
    storage_class: StorageClass


class SafeRetrievalHit(_FrozenWebModel):
    chunk_id: str
    corpus_id: str
    final_rank: int = Field(gt=0)
    evidence_type: EvidenceType
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    route_ranks: dict[str, int]


class SafeRetrievalTrace(_FrozenWebModel):
    original_question: str
    resolved_question: str
    english_query: str | None
    rewrite_status: RewriteStatus
    degraded: bool
    degraded_reason: str | None
    index_id: str
    audit_persisted: bool
    hits: tuple[SafeRetrievalHit, ...]


class SafeContextTrace(_FrozenWebModel):
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(gt=0)
    output_reserve_tokens: int = Field(ge=0)
    included_memory_turn_count: int = Field(ge=0)
    omitted_memory_turn_count: int = Field(ge=0)
    included_evidence_count: int = Field(ge=0)
    omitted_evidence_count: int = Field(ge=0)
    evidence_insufficient: bool


class SafeGenerationTrace(_FrozenWebModel):
    requested_model: str
    actual_model: str | None
    prompt_version: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    attempts: int = Field(ge=0)
    audit_persisted: bool


class SafeEvidenceSource(_FrozenWebModel):
    citation_id: str
    chunk_id: str
    corpus_id: str
    title: str
    official_url: str
    section_id: str | None
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    evidence_type: EvidenceType
    storage_class: StorageClass
    excerpt: str
    final_rank: int = Field(gt=0)


class RuntimeExecutionResult(_FrozenWebModel):
    """Validated answer plus a deliberately small, owner-facing execution trace."""

    answer: RAGAnswer
    sources: tuple[SafeEvidenceSource, ...]
    retrieval: SafeRetrievalTrace
    context: SafeContextTrace
    generation: SafeGenerationTrace


class RuntimeBusyError(RuntimeError):
    """The single local model lane is already processing another question."""


class RuntimeClosedError(RuntimeError):
    """The runtime has been shut down and cannot accept new work."""


class RuntimeDependencies:
    """Injectable long-lived dependencies used by :class:`RAGRuntime`."""

    def __init__(
        self,
        *,
        chunks: Sequence[EvidenceChunk],
        papers: Mapping[str, SafePaperMetadata],
        retriever: AsyncBilingualRetriever,
        generator: AsyncAnswerGenerator,
        memory_store: ShortTermMemoryStore,
        memory_config: ShortTermMemoryConfig,
        answer_audit: AnswerAuditLogger | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        self.papers = dict(papers)
        self.retriever = retriever
        self.generator = generator
        self.memory_store = memory_store
        self.memory_config = memory_config
        self.answer_audit = answer_audit


class RAGRuntime:
    """Load expensive local models once and serialize complete question executions."""

    def __init__(
        self,
        dependencies: RuntimeDependencies,
        *,
        top_k: int | None = None,
        token_budget: int = 8192,
        output_reserve_tokens: int = 1200,
        system_rules: str = DEFAULT_RAG_SYSTEM_RULES,
        excerpt_chars: int = 360,
    ) -> None:
        if not dependencies.chunks:
            raise ValueError("runtime requires at least one evidence chunk")
        if token_budget <= 0 or output_reserve_tokens < 0:
            raise ValueError("runtime token budgets must be non-negative")
        if output_reserve_tokens >= token_budget:
            raise ValueError("output reserve must be smaller than token budget")
        if excerpt_chars < 32 or excerpt_chars > 2000:
            raise ValueError("excerpt_chars must be between 32 and 2000")
        chunk_ids = [chunk.chunk_id for chunk in dependencies.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("runtime chunks contain duplicate chunk IDs")
        corpus_ids = {chunk.corpus_id for chunk in dependencies.chunks}
        missing_papers = corpus_ids - set(dependencies.papers)
        if missing_papers:
            raise ValueError("runtime paper metadata does not cover all chunks")

        self._chunks = dependencies.chunks
        self._chunk_map = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._papers = dependencies.papers
        self._retriever = dependencies.retriever
        self._generator = dependencies.generator
        self._memory_store = dependencies.memory_store
        self._memory_config = dependencies.memory_config
        self._answer_audit = dependencies.answer_audit
        self._top_k = top_k
        self._token_budget = token_budget
        self._output_reserve_tokens = output_reserve_tokens
        self._system_rules = system_rules
        self._excerpt_chars = excerpt_chars
        self._execution_lock = asyncio.Lock()
        self._busy = False
        self._closed = False

    @property
    def is_ready(self) -> bool:
        return not self._closed

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @classmethod
    def load(
        cls,
        *,
        project_root: Path,
        corpus_dir: Path,
        chunks_path: Path | None = None,
        retrieval_config_path: Path | None = None,
        bilingual_config_path: Path | None = None,
        answer_config_path: Path | None = None,
        memory_config_path: Path | None = None,
        answer_audit_path: Path | None = None,
        top_k: int | None = None,
        token_budget: int = 8192,
        output_reserve_tokens: int = 1200,
        excerpt_chars: int = 360,
    ) -> RAGRuntime:
        """Construct production dependencies once from project-local artifacts.

        Provider credentials are resolved by the existing DashScope adapters from
        environment variables.  This method deliberately does not load ``.env``.
        """
        root = project_root.resolve()
        chunks_file = _project_path(
            root,
            chunks_path,
            "data/processed/chunks/chunks.jsonl",
        )
        retrieval_file = _project_path(
            root,
            retrieval_config_path,
            "configs/retrieval/hybrid-rerank-v1.json",
        )
        bilingual_file = _project_path(
            root,
            bilingual_config_path,
            "configs/retrieval/bilingual-qwen-v1.json",
        )
        answer_file = _project_path(
            root,
            answer_config_path,
            "configs/answering/qwen-rag-v1.json",
        )
        memory_file = _project_path(
            root,
            memory_config_path,
            "configs/memory/short-term-v1.json",
        )
        answer_audit_file = _project_path(
            root,
            answer_audit_path,
            "data/runtime/answer-audit-v1.sqlite3",
        )

        # Fail before allocating CPU models when the answer provider cannot work.
        # The secret is never logged or retained by this runtime.
        if not os.getenv("DASHSCOPE_API_KEY", "").strip():
            raise RuntimeError("answer generation credentials are unavailable")

        chunks = tuple(_load_chunks(chunks_file))
        retrieval_config = load_retrieval_config(retrieval_file)
        bilingual_config = load_bilingual_retrieval_config(bilingual_file)
        answer_config = load_answering_config(answer_file)
        memory_config = load_memory_config(memory_file)
        index_dir = root / retrieval_config.index_dir
        manifest = IndexManifest.model_validate_json(
            (index_dir / "manifest.json").read_text(encoding="utf-8")
        )
        _validate_index_manifest(manifest, retrieval_config, chunks)
        _validate_index_files(manifest, index_dir)

        papers = load_frozen_papers(
            [corpus_dir / "core_frozen.jsonl", corpus_dir / "challenge_frozen.jsonl"]
        )
        paper_metadata = _safe_paper_metadata(papers)
        rights = CorpusRightsMap(
            {paper.corpus_id: paper.storage_class for paper in papers}
        )

        encoder = FastEmbedEncoder(
            retrieval_config.embedding_model,
            revision=retrieval_config.embedding_revision,
        )
        sparse = BM25Index(chunks)
        vector = FaissVectorIndex(chunks, encoder, index_dir / "vectors.faiss")
        reranker = FastEmbedReranker(
            retrieval_config.reranker_model,
            revision=retrieval_config.reranker_revision,
        )
        rewriter: AsyncQueryRewriter
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
        cache: QueryRewriteCache
        try:
            cache = SQLiteQueryRewriteCache(root / bilingual_config.cache_path)
        except (OSError, sqlite3.Error):
            cache = NullQueryRewriteCache()
        try:
            query_audit = SQLiteQueryAuditLogger(
                root / bilingual_config.audit_path,
                plaintext_days=bilingual_config.audit_plaintext_days,
            )
        except (OSError, sqlite3.Error):
            query_audit = None
        retriever = BilingualRetrievalService(
            sparse,
            vector,
            reranker,
            rewriter,
            cache,
            query_audit,
            retrieval_config,
            bilingual_config,
            index_id=manifest.index_id,
            rights=rights,
        )
        generator = DashScopeAnswerGenerator(answer_config)
        memory_store = SQLiteShortTermMemory(
            root / memory_config.store_path,
            config=memory_config,
        )
        try:
            answer_audit: AnswerAuditLogger | None = SQLiteAnswerAuditLogger(answer_audit_file)
        except (OSError, sqlite3.Error):
            answer_audit = None
        return cls(
            RuntimeDependencies(
                chunks=chunks,
                papers=paper_metadata,
                retriever=retriever,
                generator=cast(AsyncAnswerGenerator, generator),
                memory_store=memory_store,
                memory_config=memory_config,
                answer_audit=answer_audit,
            ),
            top_k=top_k,
            token_budget=token_budget,
            output_reserve_tokens=output_reserve_tokens,
            excerpt_chars=excerpt_chars,
        )

    @classmethod
    def from_environment(cls) -> RAGRuntime:
        """Load production paths from environment without reading a dotenv file."""
        project_root = Path(
            os.getenv("PRA_PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))
        )
        corpus_value = os.getenv("PRA_CORPUS_DIR", "").strip()
        if not corpus_value:
            raise RuntimeError("PRA_CORPUS_DIR is required")
        corpus_dir = _project_path(project_root, Path(corpus_value), corpus_value)
        return cls.load(
            project_root=project_root,
            corpus_dir=corpus_dir,
            chunks_path=_optional_env_path("PRA_CHUNKS_PATH"),
            retrieval_config_path=_optional_env_path("PRA_RETRIEVAL_CONFIG"),
            bilingual_config_path=_optional_env_path("PRA_BILINGUAL_CONFIG"),
            answer_config_path=_optional_env_path("PRA_ANSWER_CONFIG"),
            memory_config_path=_optional_env_path("PRA_MEMORY_CONFIG"),
            answer_audit_path=_optional_env_path("PRA_ANSWER_AUDIT_PATH"),
        )

    async def ask(self, question: str, *, session_id: str) -> RuntimeExecutionResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question cannot be blank")
        if self._closed:
            raise RuntimeClosedError("RAG runtime is closed")
        if self._busy:
            raise RuntimeBusyError("RAG runtime is busy")

        # There is no await between checking and setting this flag.  Within one
        # asyncio event loop this is an atomic admission gate; the lock also makes
        # close wait for an admitted request to finish.
        self._busy = True
        try:
            async with self._execution_lock:
                if self._closed:
                    raise RuntimeClosedError("RAG runtime is closed")
                return await self._execute(normalized_question, session_id=session_id)
        finally:
            self._busy = False

    async def clear_conversation(self, session_id: str) -> int:
        if self._closed:
            raise RuntimeClosedError("RAG runtime is closed")
        if self._busy:
            raise RuntimeBusyError("RAG runtime is busy")
        clear = getattr(self._memory_store, "clear", None)
        if clear is None:
            return 0
        return int(await asyncio.to_thread(clear, session_id))

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._execution_lock:
            await _close_async(self._retriever)
            # BilingualRetrievalService does not own its provider adapter.
            rewriter = getattr(self._retriever, "rewriter", None)
            if rewriter is not None:
                await _close_async(rewriter)
            await _close_async(self._generator)

    async def _execute(
        self,
        question: str,
        *,
        session_id: str,
    ) -> RuntimeExecutionResult:
        policy = self._memory_config
        try:
            memory_turns = self._memory_store.recent(session_id)
        except (OSError, sqlite3.Error):
            memory_turns = ()
        resolved_question = contextualize_retrieval_query(
            question,
            memory_turns,
            max_question_chars=policy.follow_up_max_chars,
        )
        run = await self._retriever.search(
            resolved_question,
            top_k=self._top_k,
            privacy_ttl_days=max(1, math.ceil(policy.ttl_hours / 24)),
        )
        context = assemble_context(
            ContextRequest(
                system_rules=self._system_rules,
                user_question=question,
                evidence=join_retrieval_evidence(run, self._chunks),
                short_term_memory=to_context_memory(memory_turns),
                memory_token_budget=policy.context_token_budget,
                protected_evidence_count=policy.protected_evidence_count,
                token_budget=self._token_budget,
                output_reserve_tokens=self._output_reserve_tokens,
            )
        )
        answer = await answer_context(
            AnswerRequest(context=context),
            self._generator,
            audit=self._answer_audit,
        )
        try:
            self._memory_store.append(
                turn_from_answer(
                    session_id,
                    question,
                    answer,
                    config=policy,
                    standalone_question=resolved_question,
                )
            )
        except (OSError, sqlite3.Error, ValueError):
            pass
        return self._safe_result(
            question=question,
            resolved_question=resolved_question,
            run=run,
            context=context,
            answer=answer,
        )

    def _safe_result(
        self,
        *,
        question: str,
        resolved_question: str,
        run: BilingualRetrievalRun,
        context: AssembledContext,
        answer: RAGAnswer,
    ) -> RuntimeExecutionResult:
        hit_by_chunk = {hit.chunk_id: hit for hit in run.hits}
        sources: list[SafeEvidenceSource] = []
        for citation in context.citations:
            chunk = self._chunk_map[citation.chunk_id]
            paper = self._papers[citation.corpus_id]
            hit = hit_by_chunk[citation.chunk_id]
            if citation.storage_class is None:
                raise RuntimeError("selected evidence is missing storage rights")
            sources.append(
                SafeEvidenceSource(
                    citation_id=citation.citation_id,
                    chunk_id=citation.chunk_id,
                    corpus_id=citation.corpus_id,
                    title=paper.title,
                    official_url=paper.official_url,
                    section_id=citation.section_id,
                    page_start=citation.page_start,
                    page_end=citation.page_end,
                    evidence_type=citation.evidence_type,
                    storage_class=citation.storage_class,
                    excerpt=_excerpt(chunk.text, self._excerpt_chars),
                    final_rank=hit.final_rank,
                )
            )
        hits = tuple(
            SafeRetrievalHit(
                chunk_id=hit.chunk_id,
                corpus_id=hit.corpus_id,
                final_rank=hit.final_rank,
                evidence_type=hit.evidence_type,
                page_start=hit.page_start,
                page_end=hit.page_end,
                route_ranks=dict(hit.ranks),
            )
            for hit in run.hits
        )
        return RuntimeExecutionResult(
            answer=answer,
            sources=tuple(sources),
            retrieval=SafeRetrievalTrace(
                original_question=question,
                resolved_question=resolved_question,
                english_query=run.rewrite.english_query,
                rewrite_status=run.rewrite.status,
                degraded=run.degraded,
                degraded_reason=run.degraded_reason,
                index_id=run.index_id,
                audit_persisted=run.audit_persisted,
                hits=hits,
            ),
            context=SafeContextTrace(
                estimated_tokens=context.estimated_tokens,
                token_budget=context.token_budget,
                output_reserve_tokens=context.output_reserve_tokens,
                included_memory_turn_count=len(context.included_memory_turn_ids),
                omitted_memory_turn_count=context.omitted_memory_turn_count,
                included_evidence_count=len(context.citations),
                omitted_evidence_count=context.omitted_evidence_count,
                evidence_insufficient=context.evidence_insufficient,
            ),
            generation=SafeGenerationTrace(
                requested_model=answer.requested_model,
                actual_model=answer.actual_model,
                prompt_version=answer.prompt_version,
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
                latency_ms=answer.latency_ms,
                attempts=answer.attempts,
                audit_persisted=answer.audit_persisted,
            ),
        )


def _load_chunks(path: Path) -> list[EvidenceChunk]:
    return [
        EvidenceChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _project_path(root: Path, value: Path | None, default: str) -> Path:
    candidate = value if value is not None else Path(default)
    return candidate if candidate.is_absolute() else root / candidate


def _optional_env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


def _safe_paper_metadata(papers: Sequence[FrozenPaper]) -> dict[str, SafePaperMetadata]:
    result: dict[str, SafePaperMetadata] = {}
    for paper in papers:
        if paper.corpus_id in result:
            raise ValueError("corpus manifests contain duplicate corpus IDs")
        result[paper.corpus_id] = SafePaperMetadata(
            corpus_id=paper.corpus_id,
            title=paper.title,
            official_url=paper.official_url,
            storage_class=paper.storage_class,
        )
    return result


def _validate_index_manifest(
    manifest: IndexManifest,
    retrieval_config: RetrievalConfig,
    chunks: Sequence[EvidenceChunk],
) -> None:
    if manifest.chunk_count != len(chunks):
        raise ValueError("index manifest chunk count does not match chunk artifact")
    if manifest.embedding_model != retrieval_config.embedding_model:
        raise ValueError("index manifest embedding model does not match retrieval config")
    if manifest.embedding_revision != retrieval_config.embedding_revision:
        raise ValueError("index manifest embedding revision does not match retrieval config")


def _validate_index_files(manifest: IndexManifest, index_dir: Path) -> None:
    for name, expected in manifest.files_sha256.items():
        path = index_dir / name
        if not path.is_file():
            raise ValueError("index manifest references a missing artifact")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("index artifact checksum does not match manifest")


def _excerpt(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return "（空白证据）"
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


async def _close_async(value: object) -> None:
    close = getattr(value, "aclose", None)
    if close is not None:
        await close()
