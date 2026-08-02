"""Deterministic citation allow-list validation and answer rendering."""

from __future__ import annotations

from typing import cast

from paper_research_agent.answering.models import (
    AnswerCitation,
    AnswerRequest,
    GenerationResult,
    ProviderAnswer,
    RAGAnswer,
    StorageClass,
)


class AnswerValidationError(ValueError):
    """A provider draft cannot be mapped to the trusted context."""


INSUFFICIENT_ANSWER = "当前检索上下文没有足够证据，无法可靠回答该问题。"


def validate_and_render(
    draft: ProviderAnswer,
    request: AnswerRequest,
    generation: GenerationResult,
) -> RAGAnswer:
    """Reject unknown citations and render markers from trusted local metadata."""
    if draft.status == "insufficient_evidence":
        return RAGAnswer(
            status="insufficient_evidence",
            answer_markdown=INSUFFICIENT_ANSWER,
            claims=(),
            citations=(),
            requested_model=generation.requested_model,
            actual_model=generation.actual_model,
            prompt_version=generation.prompt_version,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            latency_ms=generation.latency_ms,
            attempts=generation.attempts,
        )

    context_citations = {citation.citation_id: citation for citation in request.context.citations}
    used_ids = {identifier for claim in draft.claims for identifier in claim.citation_ids}
    unknown_ids = sorted(used_ids - set(context_citations))
    if unknown_ids:
        raise AnswerValidationError(f"provider answer contains unknown citation IDs: {unknown_ids}")

    rendered_claims = [
        f"{claim.text}{''.join(f'[{identifier}]' for identifier in claim.citation_ids)}"
        for claim in draft.claims
    ]
    citations = tuple(
        AnswerCitation(
            citation_id=citation.citation_id,
            chunk_id=citation.chunk_id,
            corpus_id=citation.corpus_id,
            asset_id=citation.asset_id,
            section_id=citation.section_id,
            page_start=citation.page_start,
            page_end=citation.page_end,
            text_sha256=citation.text_sha256,
            evidence_type=citation.evidence_type,
            storage_class=_required_storage_class(citation.storage_class),
        )
        for citation in request.context.citations
        if citation.citation_id in used_ids
    )
    return RAGAnswer(
        status="answered",
        answer_markdown="\n\n".join(rendered_claims),
        claims=draft.claims,
        citations=citations,
        requested_model=generation.requested_model,
        actual_model=generation.actual_model,
        prompt_version=generation.prompt_version,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        latency_ms=generation.latency_ms,
        attempts=generation.attempts,
    )


def _required_storage_class(
    value: str | None,
) -> StorageClass:
    if value not in {"redistributable", "internal_research_only"}:
        raise AnswerValidationError("citation storage class was not loaded")
    return cast(StorageClass, value)
