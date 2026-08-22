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
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from paper_research_agent.agent.intent import requires_research_planning
from paper_research_agent.answering.audit import SQLiteAnswerAuditLogger
from paper_research_agent.answering.comparison import (
    answer_comparison,
    build_comparison_answer_request,
    compiler_failed_comparison_answer,
)
from paper_research_agent.answering.config import load_answering_config
from paper_research_agent.answering.dashscope import (
    AsyncAnswerGenerator,
    DashScopeAnswerGenerator,
)
from paper_research_agent.answering.models import AnswerRequest, RAGAnswer
from paper_research_agent.answering.service import AnswerAuditLogger, answer_context
from paper_research_agent.chunking.models import EvidenceChunk, PaperCard
from paper_research_agent.context.adapters import join_retrieval_evidence
from paper_research_agent.context.assembler import (
    assemble_comparison_context,
    assemble_context,
)
from paper_research_agent.context.models import (
    AssembledContext,
    ContextLongTermMemory,
    ContextMemoryTurn,
    ContextRequest,
)
from paper_research_agent.conversation.models import ConversationResolution
from paper_research_agent.corpus import load_frozen_papers
from paper_research_agent.ingestion.models import DocumentElement, SectionRecord
from paper_research_agent.memory.config import ShortTermMemoryConfig, load_memory_config
from paper_research_agent.memory.context import contextualize_retrieval_query, to_context_memory
from paper_research_agent.memory.service import turn_from_answer
from paper_research_agent.memory.store import ShortTermMemoryStore, SQLiteShortTermMemory
from paper_research_agent.models import FrozenPaper
from paper_research_agent.rag import DEFAULT_RAG_SYSTEM_RULES, AsyncBilingualRetriever
from paper_research_agent.retrieval.bilingual import (
    DEFAULT_LOCAL_RETRIEVAL_WORKERS,
    BilingualRetrievalService,
)
from paper_research_agent.retrieval.bm25 import BM25Index
from paper_research_agent.retrieval.config import (
    RetrievalConfig,
    load_bilingual_retrieval_config,
    load_retrieval_config,
)
from paper_research_agent.retrieval.contracts import BilingualRetrievalRun, IndexManifest
from paper_research_agent.retrieval.model_adapters import FastEmbedEncoder, FastEmbedReranker
from paper_research_agent.retrieval.papers import (
    AsyncPaperCandidateRetriever,
    HybridPaperCandidateRetriever,
    build_paper_candidate_documents,
)
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

if TYPE_CHECKING:
    from paper_research_agent.agent.dynamic.models import DynamicResearchResult
    from paper_research_agent.agent.runtime import (
        ResearchAgentRuntime,
        ResearchRuntimeResult,
    )

StorageClass = Literal["redistributable", "internal_research_only"]
EvidenceType = Literal["text", "figure_summary"]
RewriteStatus = Literal["success", "cache_hit", "stale_cache", "timeout", "error", "agent"]
ResearchAgentMode = Literal["auto", "always"]
ResearchRequestMode = Literal["single", "planned"]


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
    standalone_question: str
    chinese_query: str
    english_query: str | None
    rewrite_status: RewriteStatus
    degraded: bool
    degraded_reason: str | None
    index_id: str
    audit_persisted: bool
    conversation_memory_hit_count: int = Field(ge=0)
    selected_history_turn_ids: tuple[str, ...] = ()
    selected_history_questions: tuple[str, ...] = ()
    selected_history_relevances: tuple[float, ...] = ()
    inherited_across_route: bool = False
    rewrite_confidence: float = Field(default=1, ge=0, le=1)
    needs_clarification: bool = False
    recent_context_turn_count: int = Field(default=0, ge=0)
    recalled_candidate_count: int = Field(default=0, ge=0)
    interpretation_source: str = "deterministic"
    hits: tuple[SafeRetrievalHit, ...]


class SafeContextTrace(_FrozenWebModel):
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(gt=0)
    output_reserve_tokens: int = Field(ge=0)
    included_memory_turn_count: int = Field(ge=0)
    omitted_memory_turn_count: int = Field(ge=0)
    included_long_term_memory_count: int = Field(default=0, ge=0)
    omitted_long_term_memory_count: int = Field(default=0, ge=0)
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


class SafeComparisonTrace(_FrozenWebModel):
    requirement_count: int = Field(ge=0)
    fact_requirement_count: int = Field(default=0, ge=0)
    compiled_fact_ids: tuple[str, ...] = ()
    satisfied_fact_requirement_ids: tuple[str, ...] = ()
    missing_fact_requirement_ids: tuple[str, ...] = ()
    expressed_fact_ids: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()
    partial_requirement_ids: tuple[str, ...] = ()
    available_compiler_chunk_count: int = Field(default=0, ge=0)
    visible_compiler_chunk_count: int = Field(default=0, ge=0)
    compilation_attempt_outcomes: tuple[str, ...] = ()
    compilation_failure_codes: tuple[str, ...] = ()
    compilation_accepted_requirement_ids: tuple[str, ...] = ()
    compilation_failed_requirement_ids: tuple[str, ...] = ()
    compilation_failed_unit_count: int = Field(default=0, ge=0)
    compilation_repair_applied: bool = False
    compilation_input_fact_count: int = Field(default=0, ge=0)
    compilation_retained_fact_count: int = Field(default=0, ge=0)
    compilation_dropped_chunk_scope_count: int = Field(default=0, ge=0)
    compilation_dropped_fact_mapping_count: int = Field(default=0, ge=0)
    compilation_missing_ledger_cell_count: int = Field(default=0, ge=0)
    compilation_fallback_empty_used: bool = False


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
    comparison: SafeComparisonTrace | None = None


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
        research_agent: ResearchAgentRuntime | None = None,
        project_root: Path | None = None,
        frozen_papers: Sequence[FrozenPaper] = (),
        sections: Sequence[SectionRecord] = (),
        elements: Sequence[DocumentElement] = (),
        paper_cards: Sequence[PaperCard] = (),
        paper_candidate_retriever: AsyncPaperCandidateRetriever | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        self.papers = dict(papers)
        self.retriever = retriever
        self.generator = generator
        self.memory_store = memory_store
        self.memory_config = memory_config
        self.answer_audit = answer_audit
        self.research_agent = research_agent
        self.project_root = project_root
        self.frozen_papers = tuple(frozen_papers)
        self.sections = tuple(sections)
        self.elements = tuple(elements)
        self.paper_cards = tuple(paper_cards)
        self.paper_candidate_retriever = paper_candidate_retriever


class RAGRuntime:
    """Load expensive local models once and serialize complete question executions."""

    rag_available = True
    agent_available = True

    def __init__(
        self,
        dependencies: RuntimeDependencies,
        *,
        top_k: int | None = None,
        token_budget: int = 8192,
        output_reserve_tokens: int = 1200,
        system_rules: str = DEFAULT_RAG_SYSTEM_RULES,
        excerpt_chars: int = 360,
        research_agent_mode: ResearchAgentMode = "always",
    ) -> None:
        if not dependencies.chunks:
            raise ValueError("runtime requires at least one evidence chunk")
        if token_budget <= 0 or output_reserve_tokens < 0:
            raise ValueError("runtime token budgets must be non-negative")
        if output_reserve_tokens >= token_budget:
            raise ValueError("output reserve must be smaller than token budget")
        if excerpt_chars < 32 or excerpt_chars > 2000:
            raise ValueError("excerpt_chars must be between 32 and 2000")
        if research_agent_mode not in {"auto", "always"}:
            raise ValueError("research_agent_mode must be auto or always")
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
        self._research_agent = dependencies.research_agent
        self._research_agent_mode = research_agent_mode
        self._project_root = dependencies.project_root
        self._frozen_papers = dependencies.frozen_papers
        self._sections = dependencies.sections
        self._elements = dependencies.elements
        self._paper_cards = dependencies.paper_cards
        self._paper_candidate_retriever = dependencies.paper_candidate_retriever
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

    @property
    def research_planning_available(self) -> bool:
        return self._research_agent is not None

    @property
    def research_agent(self) -> ResearchAgentRuntime | None:
        """Expose the guarded child runtime to the main Agent without Web internals."""
        return self._research_agent

    @classmethod
    def load(
        cls,
        *,
        project_root: Path,
        corpus_dir: Path,
        chunks_path: Path | None = None,
        paper_cards_path: Path | None = None,
        retrieval_config_path: Path | None = None,
        bilingual_config_path: Path | None = None,
        answer_config_path: Path | None = None,
        memory_config_path: Path | None = None,
        answer_audit_path: Path | None = None,
        sections_path: Path | None = None,
        elements_path: Path | None = None,
        top_k: int | None = None,
        token_budget: int = 8192,
        output_reserve_tokens: int = 1200,
        excerpt_chars: int = 360,
        local_retrieval_workers: int = DEFAULT_LOCAL_RETRIEVAL_WORKERS,
    ) -> RAGRuntime:
        """Construct local runtime dependencies once from project-local artifacts.

        Provider credentials are resolved by the existing DashScope adapters from
        environment variables.  This method deliberately does not load ``.env``.
        """
        root = project_root.resolve()
        chunks_file = _project_path(
            root,
            chunks_path,
            "data/processed/chunks/chunks.jsonl",
        )
        paper_cards_file = _project_path(
            root,
            paper_cards_path,
            "data/processed/chunks/paper_cards.jsonl",
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
        paper_cards = tuple(_load_paper_cards(paper_cards_file))
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
        sections = (
            tuple(_load_sections(_project_path(root, sections_path, ""))) if sections_path else ()
        )
        elements = (
            tuple(_load_elements(_project_path(root, elements_path, ""))) if elements_path else ()
        )
        paper_metadata = _safe_paper_metadata(papers)
        rights = CorpusRightsMap({paper.corpus_id: paper.storage_class for paper in papers})

        encoder = FastEmbedEncoder(
            retrieval_config.embedding_model,
            revision=retrieval_config.embedding_revision,
        )
        paper_documents = build_paper_candidate_documents(paper_cards, chunks)
        if {item.corpus_id for item in paper_documents} != set(paper_metadata):
            raise ValueError("paper cards do not exactly cover the corpus catalog")
        paper_candidate_retriever = HybridPaperCandidateRetriever(
            paper_documents,
            encoder,
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
            local_workers=local_retrieval_workers,
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
                project_root=root,
                frozen_papers=papers,
                sections=sections,
                elements=elements,
                paper_cards=paper_cards,
                paper_candidate_retriever=paper_candidate_retriever,
            ),
            top_k=top_k,
            token_budget=token_budget,
            output_reserve_tokens=output_reserve_tokens,
            excerpt_chars=excerpt_chars,
        )

    @classmethod
    def from_environment(cls) -> RAGRuntime:
        """Load local paths from environment without reading a dotenv file."""
        project_root = Path(os.getenv("PRA_PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
        corpus_value = os.getenv("PRA_CORPUS_DIR", "").strip()
        if not corpus_value:
            raise RuntimeError("PRA_CORPUS_DIR is required")
        corpus_dir = _project_path(project_root, Path(corpus_value), corpus_value)
        return cls.load(
            project_root=project_root,
            corpus_dir=corpus_dir,
            chunks_path=_optional_env_path("PRA_CHUNKS_PATH"),
            paper_cards_path=_optional_env_path("PRA_PAPER_CARDS_PATH"),
            retrieval_config_path=_optional_env_path("PRA_RETRIEVAL_CONFIG"),
            bilingual_config_path=_optional_env_path("PRA_BILINGUAL_CONFIG"),
            answer_config_path=_optional_env_path("PRA_ANSWER_CONFIG"),
            memory_config_path=_optional_env_path("PRA_MEMORY_CONFIG"),
            answer_audit_path=_optional_env_path("PRA_ANSWER_AUDIT_PATH"),
            sections_path=_optional_env_path("PRA_SECTIONS_PATH"),
            elements_path=_optional_env_path("PRA_ELEMENTS_PATH"),
            local_retrieval_workers=_environment_int(
                "PRA_LOCAL_RETRIEVAL_WORKERS",
                DEFAULT_LOCAL_RETRIEVAL_WORKERS,
            ),
        )

    @classmethod
    def research_agent_enabled_from_environment(cls) -> bool:
        del cls
        return _environment_flag("PRA_RESEARCH_AGENT_ENABLED", default=False)

    @classmethod
    def research_agent_mode_from_environment(cls) -> ResearchAgentMode:
        del cls
        value = os.getenv("PRA_RESEARCH_AGENT_MODE", "auto").strip().casefold()
        if value not in {"auto", "always"}:
            raise ValueError("PRA_RESEARCH_AGENT_MODE must be auto or always")
        return cast(ResearchAgentMode, value)

    @classmethod
    async def from_environment_with_agent(cls) -> RAGRuntime:
        """Construct the normal runtime and then attach the optional Agent lane."""
        project_root = Path(os.getenv("PRA_PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
        answer_file = _project_path(
            project_root,
            _optional_env_path("PRA_ANSWER_CONFIG"),
            "configs/answering/qwen-rag-v1.json",
        )
        answer_config = load_answering_config(answer_file)
        checkpoint_path = _project_path(
            project_root,
            _optional_env_path("PRA_RESEARCH_AGENT_CHECKPOINT_PATH"),
            "data/runtime/research-agent-state-v1.sqlite3",
        )
        policy = _research_policy_from_environment()
        runtime = cls.from_environment()
        try:
            await runtime.enable_research_agent(
                model_id=answer_config.model,
                checkpoint_path=checkpoint_path,
                policy=policy,
                mode=cls.research_agent_mode_from_environment(),
            )
        except BaseException:
            await runtime.aclose()
            raise
        return runtime

    async def enable_research_agent(
        self,
        *,
        model_id: str,
        checkpoint_path: Path,
        policy: object,
        mode: ResearchAgentMode = "always",
    ) -> None:
        """Attach one durable, policy-gated Agent without exposing internals to Web."""
        if self._closed:
            raise RuntimeClosedError("RAG runtime is closed")
        if self._busy:
            raise RuntimeBusyError("RAG runtime is busy")
        if self._research_agent is not None:
            raise RuntimeError("research agent is already enabled")
        from paper_research_agent.agent.factory import create_research_agent_runtime
        from paper_research_agent.agent.planner import ComparisonQueryResolver
        from paper_research_agent.agent.policy import ResearchRuntimePolicy
        from paper_research_agent.agent.service import AsyncResearchRetriever

        runtime_policy = ResearchRuntimePolicy.model_validate(policy)
        if mode not in {"auto", "always"}:
            raise ValueError("research agent mode must be auto or always")
        storage_classes = {
            corpus_id: paper.storage_class for corpus_id, paper in self._papers.items()
        }
        self._research_agent = await create_research_agent_runtime(
            retriever=cast(AsyncResearchRetriever, self._retriever),
            paper_candidate_retriever=self._require_paper_candidate_retriever(),
            paper_candidate_query_resolver=cast(ComparisonQueryResolver, self._retriever),
            chunks=self._chunks,
            storage_classes=storage_classes,
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            policy=runtime_policy,
            project_root=self._project_root,
            papers=self._frozen_papers,
            sections=self._sections,
            elements=self._elements,
        )
        self._research_agent_mode = mode

    def _require_paper_candidate_retriever(self) -> AsyncPaperCandidateRetriever:
        if self._paper_candidate_retriever is None:
            raise RuntimeError("paper candidate retriever is unavailable")
        return self._paper_candidate_retriever

    async def ask(
        self,
        question: str,
        *,
        session_id: str,
        research_mode: ResearchRequestMode = "single",
        conversation_context: ConversationResolution | None = None,
        long_term_memory: tuple[ContextLongTermMemory, ...] = (),
    ) -> RuntimeExecutionResult:
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
                return await self._execute(
                    normalized_question,
                    session_id=session_id,
                    research_mode=research_mode,
                    conversation_context=conversation_context,
                    long_term_memory=long_term_memory,
                )
        finally:
            self._busy = False

    async def run_tool_research(
        self,
        question: str,
        *,
        session_id: str,
    ) -> DynamicResearchResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question cannot be blank")
        if self._research_agent is None or not self._research_agent.dynamic_tools_enabled:
            raise RuntimeError("dynamic research tools are unavailable")
        if self._closed:
            raise RuntimeClosedError("RAG runtime is closed")
        if self._busy:
            raise RuntimeBusyError("RAG runtime is busy")
        self._busy = True
        try:
            async with self._execution_lock:
                return await self._research_agent.run_dynamic_tools(
                    normalized_question,
                    thread_id=session_id,
                )
        finally:
            self._busy = False

    async def resume_tool_research(
        self,
        *,
        session_id: str,
        approved: bool,
    ) -> DynamicResearchResult:
        if self._research_agent is None or not self._research_agent.dynamic_tools_enabled:
            raise RuntimeError("dynamic research tools are unavailable")
        if self._closed:
            raise RuntimeClosedError("RAG runtime is closed")
        if self._busy:
            raise RuntimeBusyError("RAG runtime is busy")
        self._busy = True
        try:
            async with self._execution_lock:
                return await self._research_agent.resume_dynamic_tools(
                    thread_id=session_id,
                    approved=approved,
                )
        finally:
            self._busy = False

    async def list_long_term_memories(self, *, limit: int = 20) -> object:
        if self._research_agent is None or not self._research_agent.extended_tools_enabled:
            raise RuntimeError("long-term memory is unavailable")
        if self._closed:
            raise RuntimeClosedError("RAG runtime is closed")
        if self._busy:
            raise RuntimeBusyError("RAG runtime is busy")
        self._busy = True
        try:
            async with self._execution_lock:
                return await self._research_agent.list_long_term_memories(limit=limit)
        finally:
            self._busy = False

    async def clear_conversation(self, session_id: str) -> int:
        if self._closed:
            raise RuntimeClosedError("RAG runtime is closed")
        if self._busy:
            raise RuntimeBusyError("RAG runtime is busy")
        clear = getattr(self._memory_store, "clear", None)
        cleared = 0 if clear is None else int(await asyncio.to_thread(clear, session_id))
        if self._research_agent is not None:
            await self._research_agent.clear(session_id)
        return cleared

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._execution_lock:
            await _close_async(self._research_agent)
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
        research_mode: ResearchRequestMode = "single",
        conversation_context: ConversationResolution | None = None,
        long_term_memory: tuple[ContextLongTermMemory, ...] = (),
    ) -> RuntimeExecutionResult:
        policy = self._memory_config
        if conversation_context is None:
            try:
                memory_turns = self._memory_store.recent(session_id)
            except (OSError, sqlite3.Error):
                memory_turns = ()
            resolved_question = contextualize_retrieval_query(
                question,
                memory_turns,
                max_question_chars=policy.follow_up_max_chars,
            )
            context_memory = to_context_memory(memory_turns)
        else:
            memory_turns = ()
            resolved_question = conversation_context.standalone_question
            context_memory = _conversation_context_memory(conversation_context)
        research: ResearchRuntimeResult | None = None
        use_research_agent = self._research_agent is not None and (
            self._research_agent_mode == "always"
            or research_mode == "planned"
            or requires_research_planning(resolved_question)
        )
        if not use_research_agent:
            run = await self._retriever.search(
                resolved_question,
                top_k=self._top_k,
                privacy_ttl_days=max(1, math.ceil(policy.ttl_hours / 24)),
            )
            evidence = join_retrieval_evidence(run, self._chunks)
            task_state = None
            retrieval_trace = self._safe_retrieval_trace(
                question=question,
                resolved_question=resolved_question,
                run=run,
                conversation_context=conversation_context,
            )
        else:
            if self._research_agent is None:
                raise RuntimeError("research agent selection is inconsistent")
            research = await self._research_agent.run(
                resolved_question,
                thread_id=session_id,
                planning_required=(
                    research_mode == "planned"
                    or requires_research_planning(resolved_question)
                ),
            )
            evidence = research.evidence
            task_state = research.task_state
            retrieval_trace = self._safe_research_trace(
                question=question,
                resolved_question=resolved_question,
                research=research,
                conversation_context=conversation_context,
            )
        context_request = ContextRequest(
            system_rules=self._system_rules,
            user_question=question,
            standalone_question=resolved_question,
            evidence=evidence,
            task_state=task_state,
            allow_partial_answer=(
                research is not None
                and bool(evidence)
                and not research.evidence_sufficient
            ),
            short_term_memory=context_memory,
            memory_token_budget=policy.context_token_budget,
            long_term_memory=long_term_memory,
            long_term_memory_token_budget=policy.context_token_budget,
            protected_evidence_count=policy.protected_evidence_count,
            token_budget=self._token_budget,
            output_reserve_tokens=self._output_reserve_tokens,
        )
        comparison_assessment = (
            research.assessments[-1]
            if research is not None
            and research.plan.task_type == "comparison"
            and research.assessments
            else None
        )
        has_compiled_comparison_facts = (
            comparison_assessment is not None
            and any(cell.facts for cell in comparison_assessment.ledger)
        )
        if (
            comparison_assessment is not None
            and comparison_assessment.status == "compiler_failed"
            and not has_compiled_comparison_facts
        ):
            context_request = context_request.model_copy(update={"evidence": ()})
            context = assemble_context(context_request)
            answer = compiler_failed_comparison_answer(self._generator)
        elif comparison_assessment is not None and has_compiled_comparison_facts:
            assert research is not None
            context = assemble_comparison_context(
                context_request,
                plan=research.plan,
                assessment=comparison_assessment,
            )
            comparison_request = build_comparison_answer_request(
                resolved_question,
                plan=research.plan,
                assessment=comparison_assessment,
                context=context,
            )
            answer = await answer_comparison(
                comparison_request,
                self._generator,
                audit=self._answer_audit,
            )
        else:
            if comparison_assessment is not None:
                context_request = context_request.model_copy(update={"evidence": ()})
            context = assemble_context(context_request)
            answer = await answer_context(
                AnswerRequest(context=context),
                self._generator,
                audit=self._answer_audit,
            )
        if conversation_context is None:
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
            retrieval=retrieval_trace,
            context=context,
            answer=answer,
            research=research,
        )
    def _safe_retrieval_trace(
        self,
        *,
        question: str,
        resolved_question: str,
        run: BilingualRetrievalRun,
        conversation_context: ConversationResolution | None = None,
    ) -> SafeRetrievalTrace:
        candidates = conversation_context.candidates if conversation_context is not None else ()
        selected = (
            conversation_context.selected_candidates if conversation_context is not None else ()
        )
        return SafeRetrievalTrace(
            original_question=question,
            resolved_question=resolved_question,
            standalone_question=resolved_question,
            chinese_query=(
                conversation_context.chinese_query
                if conversation_context is not None
                else resolved_question
            ),
            english_query=run.rewrite.english_query,
            rewrite_status=run.rewrite.status,
            degraded=run.degraded,
            degraded_reason=run.degraded_reason,
            index_id=run.index_id,
            audit_persisted=run.audit_persisted,
            conversation_memory_hit_count=len(candidates),
            selected_history_turn_ids=tuple(item.turn_id for item in selected),
            selected_history_questions=tuple(item.user_question for item in selected),
            selected_history_relevances=tuple(item.relevance for item in selected),
            inherited_across_route=(
                conversation_context.inherited_across_route
                if conversation_context is not None
                else False
            ),
            rewrite_confidence=(
                conversation_context.confidence if conversation_context is not None else 1
            ),
            needs_clarification=(
                conversation_context.needs_clarification
                if conversation_context is not None
                else False
            ),
            recent_context_turn_count=(
                conversation_context.recent_context_turn_count
                if conversation_context is not None
                else 0
            ),
            recalled_candidate_count=(
                conversation_context.recalled_candidate_count
                if conversation_context is not None
                else 0
            ),
            interpretation_source=(
                conversation_context.interpretation_source
                if conversation_context is not None
                else "deterministic"
            ),
            hits=tuple(
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
            ),
        )

    def _safe_research_trace(
        self,
        *,
        question: str,
        resolved_question: str,
        research: ResearchRuntimeResult,
        conversation_context: ConversationResolution | None = None,
    ) -> SafeRetrievalTrace:
        index_ids = {item.search.index_id for item in research.observations}
        if len(index_ids) != 1:
            raise ValueError("research steps must use one immutable retrieval index")
        reasons = tuple(
            dict.fromkeys(
                item.search.degraded_reason
                for item in research.observations
                if item.search.degraded_reason is not None
            )
        )
        return SafeRetrievalTrace(
            original_question=question,
            resolved_question=resolved_question,
            standalone_question=resolved_question,
            chinese_query=(
                conversation_context.chinese_query
                if conversation_context is not None
                else resolved_question
            ),
            english_query=None,
            rewrite_status="agent",
            degraded=bool(reasons),
            degraded_reason="; ".join(reasons) or None,
            index_id=next(iter(index_ids)),
            audit_persisted=False,
            conversation_memory_hit_count=(
                len(conversation_context.candidates) if conversation_context is not None else 0
            ),
            selected_history_turn_ids=(
                conversation_context.selected_turn_ids if conversation_context is not None else ()
            ),
            selected_history_questions=(
                tuple(item.user_question for item in conversation_context.selected_candidates)
                if conversation_context is not None
                else ()
            ),
            selected_history_relevances=(
                tuple(item.relevance for item in conversation_context.selected_candidates)
                if conversation_context is not None
                else ()
            ),
            inherited_across_route=(
                conversation_context.inherited_across_route
                if conversation_context is not None
                else False
            ),
            rewrite_confidence=(
                conversation_context.confidence if conversation_context is not None else 1
            ),
            needs_clarification=(
                conversation_context.needs_clarification
                if conversation_context is not None
                else False
            ),
            recent_context_turn_count=(
                conversation_context.recent_context_turn_count
                if conversation_context is not None
                else 0
            ),
            recalled_candidate_count=(
                conversation_context.recalled_candidate_count
                if conversation_context is not None
                else 0
            ),
            interpretation_source=(
                conversation_context.interpretation_source
                if conversation_context is not None
                else "deterministic"
            ),
            hits=tuple(
                SafeRetrievalHit(
                    chunk_id=item.chunk_id,
                    corpus_id=item.corpus_id,
                    final_rank=item.final_rank,
                    evidence_type=item.evidence_type,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    route_ranks={"agent": item.final_rank},
                )
                for item in research.evidence
            ),
        )

    def _safe_result(
        self,
        *,
        question: str,
        retrieval: SafeRetrievalTrace,
        context: AssembledContext,
        answer: RAGAnswer,
        research: ResearchRuntimeResult | None = None,
    ) -> RuntimeExecutionResult:
        rank_by_chunk = {hit.chunk_id: hit.final_rank for hit in retrieval.hits}
        sources: list[SafeEvidenceSource] = []
        for citation in context.citations:
            chunk = self._chunk_map[citation.chunk_id]
            paper = self._papers[citation.corpus_id]
            final_rank = rank_by_chunk[citation.chunk_id]
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
                    final_rank=final_rank,
                )
            )
        comparison_trace = None
        if research is not None and research.plan.task_type == "comparison":
            final_assessment = research.assessments[-1]
            compilation_audit = final_assessment.compilation_audit
            repair_audit = (
                compilation_audit.repair if compilation_audit is not None else None
            )
            comparison_trace = SafeComparisonTrace(
                requirement_count=len(research.plan.requirements),
                fact_requirement_count=sum(
                    len(requirement.fact_requirements)
                    for requirement in research.plan.requirements
                ),
                compiled_fact_ids=tuple(
                    fact.fact_id
                    for cell in final_assessment.ledger
                    for fact in cell.facts
                ),
                satisfied_fact_requirement_ids=tuple(
                    dict.fromkeys(
                        fact_requirement_id
                        for cell in final_assessment.ledger
                        for fact in cell.facts
                        for fact_requirement_id in fact.fact_requirement_ids
                    )
                ),
                missing_fact_requirement_ids=tuple(
                    dict.fromkeys(
                        fact_requirement_id
                        for cell in final_assessment.ledger
                        for fact_requirement_id in cell.missing_fact_requirement_ids
                    )
                ),
                expressed_fact_ids=tuple(
                    dict.fromkeys(
                        fact_id for claim in answer.claims for fact_id in claim.fact_ids
                    )
                ),
                missing_requirement_ids=tuple(
                    cell.requirement_id
                    for cell in final_assessment.ledger
                    if cell.status in {"missing", "partial"}
                ),
                partial_requirement_ids=tuple(
                    cell.requirement_id
                    for cell in final_assessment.ledger
                    if cell.status == "partial"
                ),
                available_compiler_chunk_count=len(
                    {
                        chunk_id
                        for item in final_assessment.compilation_visibility
                        for chunk_id in item.available_chunk_ids
                    }
                ),
                visible_compiler_chunk_count=len(
                    {
                        chunk_id
                        for item in final_assessment.compilation_visibility
                        for chunk_id in item.visible_chunk_ids
                    }
                ),
                compilation_attempt_outcomes=(
                    tuple(item.outcome for item in compilation_audit.attempts)
                    if compilation_audit is not None
                    else ()
                ),
                compilation_failure_codes=(
                    tuple(
                        item.failure_code
                        for item in compilation_audit.attempts
                        if item.failure_code is not None
                    )
                    if compilation_audit is not None
                    else ()
                ),
                compilation_accepted_requirement_ids=(
                    tuple(
                        dict.fromkeys(
                            requirement_id
                            for item in compilation_audit.attempts
                            for requirement_id in item.accepted_requirement_ids
                        )
                    )
                    if compilation_audit is not None
                    else ()
                ),
                compilation_failed_requirement_ids=(
                    compilation_audit.attempts[-1].failed_requirement_ids
                    if compilation_audit is not None and compilation_audit.attempts
                    else ()
                ),
                compilation_failed_unit_count=(
                    len(compilation_audit.attempts[-1].failed_requirement_ids)
                    if compilation_audit is not None and compilation_audit.attempts
                    else 0
                ),
                compilation_repair_applied=(
                    repair_audit.applied if repair_audit is not None else False
                ),
                compilation_input_fact_count=(
                    repair_audit.input_fact_count if repair_audit is not None else 0
                ),
                compilation_retained_fact_count=(
                    repair_audit.retained_fact_count if repair_audit is not None else 0
                ),
                compilation_dropped_chunk_scope_count=(
                    repair_audit.dropped_chunk_scope_count
                    if repair_audit is not None
                    else 0
                ),
                compilation_dropped_fact_mapping_count=(
                    repair_audit.dropped_fact_mapping_count
                    if repair_audit is not None
                    else 0
                ),
                compilation_missing_ledger_cell_count=(
                    repair_audit.missing_ledger_cell_count
                    if repair_audit is not None
                    else 0
                ),
                compilation_fallback_empty_used=(
                    repair_audit.fallback_empty_used
                    if repair_audit is not None
                    else False
                ),
            )
        return RuntimeExecutionResult(
            answer=answer,
            sources=tuple(sources),
            retrieval=retrieval,
            context=SafeContextTrace(
                estimated_tokens=context.estimated_tokens,
                token_budget=context.token_budget,
                output_reserve_tokens=context.output_reserve_tokens,
                included_memory_turn_count=len(context.included_memory_turn_ids),
                omitted_memory_turn_count=context.omitted_memory_turn_count,
                included_long_term_memory_count=len(
                    context.included_long_term_memory_ids
                ),
                omitted_long_term_memory_count=(
                    context.omitted_long_term_memory_count
                ),
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
            comparison=comparison_trace,
        )


def _conversation_context_memory(
    resolution: ConversationResolution,
) -> tuple[ContextMemoryTurn, ...]:
    turns: list[ContextMemoryTurn] = []
    for candidate in resolution.selected_candidates:
        if candidate.status == "insufficient_evidence":
            turns.append(
                ContextMemoryTurn(
                    turn_id=candidate.turn_id,
                    user_question=candidate.user_question,
                    status="insufficient_evidence",
                )
            )
        elif candidate.assistant_summary:
            turns.append(
                ContextMemoryTurn(
                    turn_id=candidate.turn_id,
                    user_question=candidate.user_question,
                    status="answered",
                    assistant_claims=(candidate.assistant_summary,),
                )
            )
    return tuple(turns)


def _load_chunks(path: Path) -> list[EvidenceChunk]:
    return [
        EvidenceChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_paper_cards(path: Path) -> list[PaperCard]:
    return [
        PaperCard.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_sections(path: Path) -> list[SectionRecord]:
    return [
        SectionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_elements(path: Path) -> list[DocumentElement]:
    return [
        DocumentElement.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _project_path(root: Path, value: Path | None, default: str) -> Path:
    candidate = value if value is not None else Path(default)
    return candidate if candidate.is_absolute() else root / candidate


def _optional_env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


def _environment_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _environment_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None


def _environment_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number") from None


def _research_policy_from_environment() -> object:
    from paper_research_agent.agent.policy import (
        DEFAULT_COMPARISON_SEARCH_CONCURRENCY,
        ResearchRuntimePolicy,
    )

    legacy_name = "PRA_RESEARCH_AGENT_EVIDENCE_PER_STEP"
    cutoff_names = (
        "PRA_RESEARCH_AGENT_INITIAL_EVIDENCE_PER_STEP",
        "PRA_RESEARCH_AGENT_FIRST_FOLLOWUP_EVIDENCE_PER_STEP",
        "PRA_RESEARCH_AGENT_LATER_FOLLOWUP_EVIDENCE_PER_STEP",
    )
    legacy_configured = bool(os.getenv(legacy_name, "").strip())
    if legacy_configured and any(os.getenv(name, "").strip() for name in cutoff_names):
        raise ValueError(
            "legacy and adaptive research evidence cutoff variables cannot be combined"
        )
    if legacy_configured:
        initial_evidence = _environment_int(legacy_name, 4)
        first_followup_evidence = initial_evidence
        later_followup_evidence = initial_evidence
    else:
        initial_evidence = _environment_int(cutoff_names[0], 4)
        first_followup_evidence = _environment_int(cutoff_names[1], 6)
        later_followup_evidence = _environment_int(cutoff_names[2], 10)

    return ResearchRuntimePolicy(
        max_steps=_environment_int("PRA_RESEARCH_AGENT_MAX_STEPS", 24),
        max_followup_steps=_environment_int(
            "PRA_RESEARCH_AGENT_MAX_FOLLOWUP_STEPS",
            4,
        ),
        comparison_search_concurrency=_environment_int(
            "PRA_COMPARISON_SEARCH_CONCURRENCY",
            DEFAULT_COMPARISON_SEARCH_CONCURRENCY,
        ),
        adaptive_evidence_hydration_enabled=_environment_flag(
            "PRA_RESEARCH_AGENT_ADAPTIVE_EVIDENCE_HYDRATION_ENABLED",
            default=False,
        ),
        initial_evidence_per_step=initial_evidence,
        first_followup_evidence_per_step=first_followup_evidence,
        later_followup_evidence_per_step=later_followup_evidence,
        max_tool_calls=_environment_int("PRA_RESEARCH_AGENT_MAX_TOOL_CALLS", 48),
        timeout_seconds=_environment_float(
            "PRA_RESEARCH_AGENT_TIMEOUT_SECONDS",
            180,
        ),
    )


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
