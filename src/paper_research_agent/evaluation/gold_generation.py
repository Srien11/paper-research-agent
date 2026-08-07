"""Convert model-generated silver drafts into the strict private gold contract."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from paper_research_agent.evaluation.gold_dataset import GoldQuestion
from paper_research_agent.evaluation.gold_selection import CandidateBlueprint


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceEvidence(FrozenModel):
    span_id: str = Field(pattern=r"^S\d{3}$")
    paper_id: str = Field(pattern=r"^[CT]\d{3}$")
    evidence_version_id: str = Field(min_length=1, max_length=256)
    page: int = Field(ge=1)
    element_id: str = Field(min_length=1, max_length=256)
    raw_quote: str = Field(min_length=1)
    span_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_role: Literal["required", "supporting", "distractor"]
    projected_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=20)


class GeneratedClaim(FrozenModel):
    claim_id: str = Field(pattern=r"^[MF]\d{1,3}$")
    text: str = Field(min_length=1, max_length=4000)
    span_ids: tuple[str, ...] = Field(default=(), max_length=20)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class GeneratedCandidate(FrozenModel):
    question: str = Field(min_length=1, max_length=10_000)
    must_have_claims: tuple[GeneratedClaim, ...] = ()
    forbidden_claims: tuple[GeneratedClaim, ...] = ()
    unanswerable_reason: str | None = Field(default=None, min_length=1, max_length=4000)

    @field_validator("question", "unanswerable_reason")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


def build_gold_question(
    blueprint: CandidateBlueprint,
    evidence: list[SourceEvidence],
    draft: GeneratedCandidate,
    *,
    corpus_version: str,
    knowledge_cutoff: date,
) -> GoldQuestion:
    """Build a validated ``silver_generated`` record; it is not adjudicated gold."""

    known_spans = {span.span_id for span in evidence}
    if len(known_spans) != len(evidence):
        raise ValueError("source evidence span IDs must be unique")
    if blueprint.answerable and not draft.must_have_claims:
        raise ValueError("answerable draft requires must-have claims")
    if not blueprint.answerable and draft.must_have_claims:
        raise ValueError("unanswerable draft cannot contain must-have claims")

    citation_relations: list[dict[str, str]] = []
    for claim in draft.must_have_claims:
        if not claim.span_ids:
            raise ValueError(f"must-have claim {claim.claim_id} has no evidence span")
        for span_id in claim.span_ids:
            if span_id not in known_spans:
                raise ValueError(f"generated claim references unknown evidence span: {span_id}")
            citation_relations.append(
                {"claim_id": claim.claim_id, "span_id": span_id, "relation": "supports"}
            )

    question_id = "GQ" + blueprint.case_id[1:]
    return GoldQuestion.model_validate(
        {
            "question_id": question_id,
            "split": "dev",
            "language": blueprint.language,
            "task_type": blueprint.task_type,
            "difficulty": blueprint.difficulty,
            "question": draft.question,
            "answerable": blueprint.answerable,
            "must_have_claims": [
                {"claim_id": claim.claim_id, "text": claim.text}
                for claim in draft.must_have_claims
            ],
            "forbidden_claims": [
                {"claim_id": claim.claim_id, "text": claim.text}
                for claim in draft.forbidden_claims
            ],
            "evidence_spans": [
                {
                    "span_id": span.span_id,
                    "paper_id": span.paper_id,
                    "evidence_version_id": span.evidence_version_id,
                    "page": span.page,
                    "element_id": span.element_id,
                    "raw_span_start": 0,
                    "raw_span_end": len(span.raw_quote),
                    "raw_quote": span.raw_quote,
                    "span_hash": span.span_hash,
                    "support_role": span.support_role,
                    "projected_chunk_ids": span.projected_chunk_ids,
                }
                for span in evidence
            ],
            "citation_relations": citation_relations,
            "unanswerable_reason": draft.unanswerable_reason,
            "nearest_distractor_paper_ids": blueprint.nearest_distractor_paper_ids,
            "annotation_status": "silver_generated",
            "reviewer_ids": [],
            "adjudicator_id": None,
            "corpus_version": corpus_version,
            "knowledge_cutoff": knowledge_cutoff,
        }
    )
