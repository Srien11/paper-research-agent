"""Create safe memory turns only from locally validated RAG answers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from paper_research_agent.answering.models import RAGAnswer
from paper_research_agent.memory.config import ShortTermMemoryConfig
from paper_research_agent.memory.models import MemorySourceRef, ShortTermMemoryTurn


def turn_from_answer(
    session_id: str,
    question: str,
    answer: RAGAnswer,
    *,
    config: ShortTermMemoryConfig,
    standalone_question: str | None = None,
    now: datetime | None = None,
) -> ShortTermMemoryTurn:
    """Project a validated answer without old citation labels or evidence bodies."""
    created = (now or datetime.now(UTC)).astimezone(UTC)
    refs = tuple(
        MemorySourceRef(
            chunk_id=citation.chunk_id,
            corpus_id=citation.corpus_id,
            text_sha256=citation.text_sha256,
            storage_class=citation.storage_class,
        )
        for citation in answer.citations
    )
    return ShortTermMemoryTurn(
        turn_id=uuid.uuid4().hex,
        session_id=session_id,
        created_at=created,
        expires_at=created + timedelta(hours=config.ttl_hours),
        user_question=question,
        standalone_question=standalone_question or question,
        status=answer.status,
        assistant_claims=tuple(claim.text for claim in answer.claims),
        source_refs=refs,
    )
