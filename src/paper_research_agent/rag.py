"""Question-level orchestration for the complete private-research RAG path."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable
from typing import Protocol

from paper_research_agent.answering.dashscope import AsyncAnswerGenerator
from paper_research_agent.answering.models import AnswerRequest, RAGAnswer
from paper_research_agent.answering.service import AnswerAuditLogger, answer_context
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.adapters import join_retrieval_evidence
from paper_research_agent.context.assembler import assemble_context
from paper_research_agent.context.models import ContextMemoryTurn, ContextRequest
from paper_research_agent.memory.config import ShortTermMemoryConfig
from paper_research_agent.memory.context import contextualize_retrieval_query, to_context_memory
from paper_research_agent.memory.models import ShortTermMemoryTurn
from paper_research_agent.memory.service import turn_from_answer
from paper_research_agent.memory.store import ShortTermMemoryStore
from paper_research_agent.retrieval.contracts import BilingualRetrievalRun

DEFAULT_RAG_SYSTEM_RULES = (
    "Answer in Simplified Chinese only from supplied evidence. Preserve uncertainty, "
    "separate distinct findings into concise claims, and never reproduce long source passages."
)


class AsyncBilingualRetriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        privacy_ttl_days: int | None = None,
    ) -> BilingualRetrievalRun: ...


async def answer_question(
    question: str,
    *,
    retriever: AsyncBilingualRetriever,
    chunks: Iterable[EvidenceChunk],
    generator: AsyncAnswerGenerator,
    audit: AnswerAuditLogger | None = None,
    top_k: int | None = None,
    token_budget: int = 8192,
    output_reserve_tokens: int = 1200,
    system_rules: str = DEFAULT_RAG_SYSTEM_RULES,
    session_id: str | None = None,
    memory_store: ShortTermMemoryStore | None = None,
    memory_config: ShortTermMemoryConfig | None = None,
) -> RAGAnswer:
    """Run Chinese bilingual retrieval, context assembly, generation, and validation."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question cannot be blank")
    if (session_id is None) != (memory_store is None):
        raise ValueError("session_id and memory_store must be provided together")
    policy = memory_config or ShortTermMemoryConfig()
    memory_turns: tuple[ShortTermMemoryTurn, ...] = ()
    if session_id is not None and memory_store is not None:
        try:
            memory_turns = memory_store.recent(session_id)
        except (OSError, sqlite3.Error):
            memory_turns = ()
    retrieval_question = contextualize_retrieval_query(
        normalized_question,
        memory_turns,
        max_question_chars=policy.follow_up_max_chars,
    )
    run = await retriever.search(
        retrieval_question,
        top_k=top_k,
        privacy_ttl_days=(
            max(1, math.ceil(policy.ttl_hours / 24)) if session_id is not None else None
        ),
    )
    result = await answer_retrieval_run(
        run,
        chunks=chunks,
        generator=generator,
        audit=audit,
        token_budget=token_budget,
        output_reserve_tokens=output_reserve_tokens,
        system_rules=system_rules,
        user_question=normalized_question,
        short_term_memory=to_context_memory(memory_turns),
        memory_token_budget=policy.context_token_budget,
        protected_evidence_count=policy.protected_evidence_count,
    )
    if session_id is not None and memory_store is not None:
        try:
            turn = turn_from_answer(
                session_id,
                normalized_question,
                result,
                config=policy,
                standalone_question=retrieval_question,
            )
            memory_store.append(turn)
        except (OSError, sqlite3.Error, ValueError):
            pass
    return result


async def answer_retrieval_run(
    run: BilingualRetrievalRun,
    *,
    chunks: Iterable[EvidenceChunk],
    generator: AsyncAnswerGenerator,
    audit: AnswerAuditLogger | None = None,
    token_budget: int = 8192,
    output_reserve_tokens: int = 1200,
    system_rules: str = DEFAULT_RAG_SYSTEM_RULES,
    user_question: str | None = None,
    short_term_memory: tuple[ContextMemoryTurn, ...] = (),
    memory_token_budget: int = 0,
    protected_evidence_count: int = 1,
) -> RAGAnswer:
    """Continue a completed bilingual run through the trusted answer boundary."""
    context = assemble_context(
        ContextRequest(
            system_rules=system_rules,
            user_question=user_question or run.original_query,
            evidence=join_retrieval_evidence(run, chunks),
            short_term_memory=short_term_memory,
            memory_token_budget=memory_token_budget,
            protected_evidence_count=protected_evidence_count,
            token_budget=token_budget,
            output_reserve_tokens=output_reserve_tokens,
        )
    )
    return await answer_context(AnswerRequest(context=context), generator, audit=audit)
