"""Scope adjudication for comparison evaluation gold references."""

from __future__ import annotations

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research_agent.evaluation.comparison_end_to_end import (
    ComparisonEndToEndGold,
)


class FrozenScopeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RequiredClaimScope(FrozenScopeModel):
    claim_id: str = Field(min_length=1)
    request_spans: tuple[str, ...] = Field(min_length=1)


class OptionalClaimScope(FrozenScopeModel):
    claim_id: str = Field(min_length=1)
    reason: Literal["not_explicit", "supporting_detail", "redundant"]


class ComparisonQuestionScope(FrozenScopeModel):
    question_id: str = Field(pattern=r"^CPG\d{3}$")
    expected_dimensions: tuple[str, ...] = Field(min_length=1, max_length=8)
    required_claims: tuple[RequiredClaimScope, ...] = Field(min_length=1)
    optional_claims: tuple[OptionalClaimScope, ...] = ()

    @model_validator(mode="after")
    def validate_unique_claim_ids(self) -> ComparisonQuestionScope:
        claim_ids = [item.claim_id for item in self.required_claims]
        claim_ids.extend(item.claim_id for item in self.optional_claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("scope claim IDs must be unique")
        return self


class ComparisonGoldScopeManifest(FrozenScopeModel):
    schema_version: Literal["comparison-gold-scope-v2"] = "comparison-gold-scope-v2"
    questions: tuple[ComparisonQuestionScope, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_question_ids(self) -> ComparisonGoldScopeManifest:
        question_ids = [item.question_id for item in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("scope question IDs must be unique")
        return self


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def apply_comparison_gold_scope(
    *,
    question: str,
    gold: ComparisonEndToEndGold,
    scope: ComparisonQuestionScope,
) -> ComparisonEndToEndGold:
    """Project one legacy gold row to explicitly requested primary facts."""
    if gold.question_id != scope.question_id:
        raise ValueError("scope question ID does not match gold")
    source_claim_ids = {item.claim_id for item in gold.must_have_claims}
    required_claim_ids = {item.claim_id for item in scope.required_claims}
    optional_claim_ids = {item.claim_id for item in scope.optional_claims}
    if required_claim_ids | optional_claim_ids != source_claim_ids:
        raise ValueError("scope must account for every source claim exactly once")

    normalized_question = _normalized(question)
    for claim in scope.required_claims:
        if any(_normalized(span) not in normalized_question for span in claim.request_spans):
            raise ValueError("required claim request span must occur verbatim in the question")

    claims = tuple(
        item for item in gold.must_have_claims if item.claim_id in required_claim_ids
    )
    relations = tuple(
        item for item in gold.citation_relations if item.claim_id in required_claim_ids
    )
    evidence_chunk_ids = tuple(
        dict.fromkeys(chunk_id for item in relations for chunk_id in item.chunk_ids)
    )
    evidence_span_hashes = tuple(
        dict.fromkeys(span_hash for item in relations for span_hash in item.span_hashes)
    )
    return ComparisonEndToEndGold(
        question_id=gold.question_id,
        split=gold.split,
        relevant_paper_ids=gold.relevant_paper_ids,
        expected_dimensions=scope.expected_dimensions,
        must_have_claims=claims,
        evidence_chunk_ids=evidence_chunk_ids,
        evidence_span_hashes=evidence_span_hashes,
        forbidden_claims=gold.forbidden_claims,
        citation_relations=relations,
    )
