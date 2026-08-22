"""Fact-closed, per-dimension comparison answer generation."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Protocol

from paper_research_agent.agent.models import EvidenceAssessment, ResearchPlan
from paper_research_agent.answering.dashscope import AsyncAnswerGenerator
from paper_research_agent.answering.models import (
    AnswerCitation,
    AnswerClaim,
    ComparisonAnswerRequest,
    ComparisonDimension,
    ComparisonFact,
    ComparisonTarget,
    RAGAnswer,
)
from paper_research_agent.answering.validation import AnswerValidationError
from paper_research_agent.context.models import AssembledContext, CitationRef


class ComparisonAnswerAudit(Protocol):
    def log(self, result: RAGAnswer) -> bool: ...


def compiler_failed_comparison_answer(
    generator: AsyncAnswerGenerator,
) -> RAGAnswer:
    """Return a deterministic failure without misreporting missing evidence."""
    return RAGAnswer(
        status="compiler_failed",
        answer_markdown=(
            "证据编译失败：系统未能把已检索证据转换为满足约束的比较事实。"
            "这不表示论文证据不足，请稍后重试。"
        ),
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


def build_comparison_answer_request(
    question: str,
    *,
    plan: ResearchPlan,
    assessment: EvidenceAssessment,
    context: AssembledContext,
) -> ComparisonAnswerRequest:
    """Join a validated ledger to trusted citation IDs without exposing raw blocks."""
    if plan.task_type != "comparison" or not assessment.ledger:
        raise ValueError("comparison answer requires a compiled comparison ledger")
    requirement_by_id = {item.requirement_id: item for item in plan.requirements}
    citation_by_chunk = {item.chunk_id: item.citation_id for item in context.citations}
    facts: list[ComparisonFact] = []
    for cell in assessment.ledger:
        requirement = requirement_by_id[cell.requirement_id]
        for fact in cell.facts:
            try:
                citation_ids = tuple(citation_by_chunk[item] for item in fact.chunk_ids)
            except KeyError as exc:
                raise ValueError(
                    "comparison ledger fact is absent from trusted generation citations"
                ) from exc
            facts.append(
                ComparisonFact(
                    fact_id=fact.fact_id,
                    requirement_id=cell.requirement_id,
                    target_id=requirement.target_id,
                    dimension_id=requirement.dimension_id,
                    statement=fact.statement,
                    fact_requirement_ids=fact.fact_requirement_ids,
                    citation_ids=citation_ids,
                    qualifiers=tuple(
                        item.model_dump(mode="json") for item in fact.qualifiers
                    ),
                )
            )
    if not facts:
        raise ValueError("comparison answer requires at least one compiled fact")
    return ComparisonAnswerRequest(
        question=question.strip(),
        context=context,
        targets=tuple(
            ComparisonTarget(
                target_id=item.target_id,
                label=item.label,
                corpus_id=item.corpus_id,
            )
            for item in plan.targets
            if item.corpus_id is not None
        ),
        dimensions=tuple(
            ComparisonDimension(dimension_id=item.dimension_id, label=item.label)
            for item in plan.dimensions
        ),
        facts=tuple(facts),
    )


async def answer_comparison(
    request: ComparisonAnswerRequest,
    generator: AsyncAnswerGenerator,
    *,
    audit: ComparisonAnswerAudit | None = None,
    max_dimension_attempts: int = 2,
) -> RAGAnswer:
    """Render the trusted comparison ledger without making a Provider call."""
    if max_dimension_attempts <= 0 or max_dimension_attempts > 2:
        raise ValueError("max_dimension_attempts must be between 1 and 2")
    ordered_fact_ids = {
        dimension.dimension_id: tuple(
            fact.fact_id
            for target in request.targets
            for fact in request.facts
            if fact.dimension_id == dimension.dimension_id
            and fact.target_id == target.target_id
        )
        for dimension in request.dimensions
    }

    claims = _render_claims(request, ordered_fact_ids)
    citations_by_id = {item.citation_id: item for item in request.context.citations}
    used_citations = {citation_id for claim in claims for citation_id in claim.citation_ids}
    citations = tuple(
        _answer_citation(item)
        for item in request.context.citations
        if item.citation_id in used_citations
    )
    markdown = _render_markdown(request, claims)
    result = RAGAnswer(
        status="answered",
        answer_markdown=markdown,
        claims=claims,
        citations=citations,
        requested_model=generator.model_id,
        actual_model=None,
        prompt_version="comparison-ledger-render-v2",
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
        attempts=0,
    )
    if set(used_citations) - set(citations_by_id):
        raise AnswerValidationError("comparison renderer produced an unknown citation")
    return _best_effort_audit(result, audit)


def _render_claims(
    request: ComparisonAnswerRequest,
    ordered_fact_ids: Mapping[str, tuple[str, ...]],
) -> tuple[AnswerClaim, ...]:
    fact_by_id = {item.fact_id: item for item in request.facts}
    claims: list[AnswerClaim] = []
    for dimension in request.dimensions:
        for fact_id in ordered_fact_ids[dimension.dimension_id]:
            fact = fact_by_id[fact_id]
            claims.append(
                AnswerClaim(
                    text=fact.statement,
                    citation_ids=fact.citation_ids,
                    fact_ids=(fact.fact_id,),
                )
            )
    if {fact_id for claim in claims for fact_id in claim.fact_ids} != set(fact_by_id):
        raise AnswerValidationError("comparison renderer omitted compiled facts")
    return tuple(claims)


def _render_markdown(
    request: ComparisonAnswerRequest,
    claims: tuple[AnswerClaim, ...],
) -> str:
    claim_by_fact = {claim.fact_ids[0]: claim for claim in claims}
    fact_by_dimension_target: dict[tuple[str, str], list[ComparisonFact]] = {}
    for fact in request.facts:
        fact_by_dimension_target.setdefault((fact.dimension_id, fact.target_id), []).append(fact)
    sections: list[str] = []
    for dimension in request.dimensions:
        lines = [f"## {dimension.label}"]
        for target in request.targets:
            facts = fact_by_dimension_target.get((dimension.dimension_id, target.target_id), [])
            if not facts:
                lines.append(f"- {target.label}：暂无可靠事实。")
                continue
            for fact in facts:
                claim = claim_by_fact[fact.fact_id]
                markers = "".join(f"[{item}]" for item in claim.citation_ids)
                lines.append(f"- {target.label}：{fact.statement}{markers}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _answer_citation(value: CitationRef) -> AnswerCitation:
    if value.storage_class not in {"redistributable", "internal_research_only"}:
        raise AnswerValidationError("comparison citation storage class was not loaded")
    return AnswerCitation(
        citation_id=value.citation_id,
        chunk_id=value.chunk_id,
        corpus_id=value.corpus_id,
        asset_id=value.asset_id,
        section_id=value.section_id,
        page_start=value.page_start,
        page_end=value.page_end,
        text_sha256=value.text_sha256,
        evidence_type=value.evidence_type,
        storage_class=value.storage_class,
    )


def _best_effort_audit(
    result: RAGAnswer,
    audit: ComparisonAnswerAudit | None,
) -> RAGAnswer:
    if audit is None:
        return result
    try:
        persisted = audit.log(result)
    except (OSError, sqlite3.Error):
        persisted = False
    return result.model_copy(update={"audit_persisted": persisted})
