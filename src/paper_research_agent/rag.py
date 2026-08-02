"""Question-level orchestration for the complete private-research RAG path."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from paper_research_agent.answering.dashscope import AsyncAnswerGenerator
from paper_research_agent.answering.models import AnswerRequest, RAGAnswer
from paper_research_agent.answering.service import AnswerAuditLogger, answer_context
from paper_research_agent.chunking.models import EvidenceChunk
from paper_research_agent.context.adapters import join_retrieval_evidence
from paper_research_agent.context.assembler import assemble_context
from paper_research_agent.context.models import ContextRequest
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
) -> RAGAnswer:
    """Run Chinese bilingual retrieval, context assembly, generation, and validation."""
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question cannot be blank")
    run = await retriever.search(normalized_question, top_k=top_k)
    return await answer_retrieval_run(
        run,
        chunks=chunks,
        generator=generator,
        audit=audit,
        token_budget=token_budget,
        output_reserve_tokens=output_reserve_tokens,
        system_rules=system_rules,
    )


async def answer_retrieval_run(
    run: BilingualRetrievalRun,
    *,
    chunks: Iterable[EvidenceChunk],
    generator: AsyncAnswerGenerator,
    audit: AnswerAuditLogger | None = None,
    token_budget: int = 8192,
    output_reserve_tokens: int = 1200,
    system_rules: str = DEFAULT_RAG_SYSTEM_RULES,
) -> RAGAnswer:
    """Continue a completed bilingual run through the trusted answer boundary."""
    context = assemble_context(
        ContextRequest(
            system_rules=system_rules,
            user_question=run.original_query,
            evidence=join_retrieval_evidence(run, chunks),
            token_budget=token_budget,
            output_reserve_tokens=output_reserve_tokens,
        )
    )
    return await answer_context(AnswerRequest(context=context), generator, audit=audit)
