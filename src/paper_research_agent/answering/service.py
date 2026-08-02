"""Orchestrate provider generation, strict parsing, citation validation, and audit."""

from __future__ import annotations

import sqlite3
from typing import Protocol

from pydantic import ValidationError

from paper_research_agent.answering.dashscope import AsyncAnswerGenerator
from paper_research_agent.answering.models import AnswerRequest, ProviderAnswer, RAGAnswer
from paper_research_agent.answering.validation import (
    INSUFFICIENT_ANSWER,
    AnswerValidationError,
    validate_and_render,
)


class AnswerAuditLogger(Protocol):
    def log(self, result: RAGAnswer) -> bool: ...


async def answer_context(
    request: AnswerRequest,
    generator: AsyncAnswerGenerator,
    *,
    audit: AnswerAuditLogger | None = None,
) -> RAGAnswer:
    """Create one validated answer, skipping the provider when no evidence fits."""
    if request.context.evidence_insufficient or not request.context.citations:
        result = RAGAnswer(
            status="insufficient_evidence",
            answer_markdown=INSUFFICIENT_ANSWER,
            claims=(),
            citations=(),
            requested_model=generator.model_id,
            actual_model=None,
            prompt_version=generator.prompt_version,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            attempts=0,
        )
        return _best_effort_audit(result, audit)

    generation = await generator.generate(request)
    try:
        draft = ProviderAnswer.model_validate_json(generation.content)
    except ValidationError:
        raise AnswerValidationError("provider answer schema validation failed") from None
    result = validate_and_render(draft, request, generation)
    return _best_effort_audit(result, audit)


def _best_effort_audit(
    result: RAGAnswer,
    audit: AnswerAuditLogger | None,
) -> RAGAnswer:
    if audit is None:
        return result
    try:
        persisted = audit.log(result)
    except (OSError, sqlite3.Error):
        persisted = False
    return result.model_copy(update={"audit_persisted": persisted})
