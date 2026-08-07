"""Strict private gold-dataset contracts and deterministic source replay.

Gold files may contain copyrighted source spans and therefore belong under the
Git-ignored ``data/evaluations/gold`` directory.  Validation reports contain
only counts and stable identifiers, never source bodies or local paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Sha256 = str
EvaluationSplit = Literal["dev", "regression", "sealed_test"]
QuestionLanguage = Literal["zh", "en", "mixed"]
QuestionDifficulty = Literal["easy", "medium", "hard"]
TaskType = Literal[
    "definition_scope",
    "method_mechanism",
    "experimental_result",
    "multi_paper_comparison",
    "conflicting_evidence",
    "multi_hop_synthesis",
    "figure_table_explanation",
]
AnnotationStatus = Literal["silver_generated", "silver_reviewed", "gold_adjudicated"]
SupportRole = Literal["required", "supporting", "distractor"]
CitationRelationKind = Literal["supports", "contradicts", "distractor"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GoldClaim(FrozenModel):
    claim_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("claim text cannot be blank")
        return normalized


class EvidenceSpan(FrozenModel):
    """One authoritative location plus its current chunk projection.

    ``evidence_version_id`` is the ingestion ``asset_id``.  Span offsets are
    zero-based character offsets inside ``DocumentElement.raw_text``.
    """

    span_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    paper_id: str = Field(pattern=r"^[CT]\d{3}$")
    evidence_version_id: str = Field(min_length=1, max_length=256)
    page: int = Field(ge=1)
    element_id: str = Field(min_length=1, max_length=256)
    raw_span_start: int = Field(ge=0)
    raw_span_end: int = Field(gt=0)
    raw_quote: str = Field(min_length=1)
    span_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_role: SupportRole
    projected_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=20)

    @field_validator(
        "evidence_version_id",
        "element_id",
        "raw_quote",
    )
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("span text fields cannot be blank")
        return value

    @field_validator("projected_chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("projected chunk IDs cannot be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("projected chunk IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_range(self) -> EvidenceSpan:
        if self.raw_span_end <= self.raw_span_start:
            raise ValueError("raw span end must be greater than start")
        return self


class CitationRelation(FrozenModel):
    claim_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    span_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    relation: CitationRelationKind


class GoldQuestion(FrozenModel):
    schema_version: Literal["rag-gold-question-v1"] = "rag-gold-question-v1"
    question_id: str = Field(pattern=r"^GQ\d{3,}$")
    split: EvaluationSplit
    language: QuestionLanguage
    task_type: TaskType
    difficulty: QuestionDifficulty
    question: str = Field(min_length=1, max_length=10_000)
    answerable: bool
    must_have_claims: tuple[GoldClaim, ...] = ()
    forbidden_claims: tuple[GoldClaim, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    citation_relations: tuple[CitationRelation, ...] = ()
    unanswerable_reason: str | None = Field(default=None, min_length=1, max_length=4000)
    nearest_distractor_paper_ids: tuple[str, ...] = Field(default=(), max_length=20)
    annotation_status: AnnotationStatus
    reviewer_ids: tuple[str, ...] = Field(default=(), max_length=20)
    adjudicator_id: str | None = Field(default=None, min_length=1, max_length=128)
    corpus_version: str = Field(min_length=1, max_length=256)
    knowledge_cutoff: date

    @field_validator("question", "unanswerable_reason", "adjudicator_id", "corpus_version")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("gold question text fields cannot be blank")
        return normalized

    @field_validator("nearest_distractor_paper_ids")
    @classmethod
    def validate_distractor_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("nearest distractor paper IDs must be unique")
        if any(re.fullmatch(r"[CT]\d{3}", value) is None for value in values):
            raise ValueError("nearest distractor paper ID is invalid")
        return values

    @field_validator("reviewer_ids")
    @classmethod
    def validate_reviewer_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value or len(value) > 128 for value in normalized):
            raise ValueError("reviewer IDs must contain between 1 and 128 characters")
        if len(normalized) != len(set(normalized)):
            raise ValueError("reviewer IDs must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_question_contract(self) -> GoldQuestion:
        if self.answerable and not self.must_have_claims:
            raise ValueError("answerable question requires at least one must-have claim")
        if not self.answerable and self.must_have_claims:
            raise ValueError("unanswerable question cannot contain unconditional must-have claims")
        all_claims = (*self.must_have_claims, *self.forbidden_claims)
        claim_ids = [claim.claim_id for claim in all_claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim IDs must be unique across the question")
        span_ids = [span.span_id for span in self.evidence_spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("span IDs must be unique within the question")
        relation_keys = [
            (relation.claim_id, relation.span_id, relation.relation)
            for relation in self.citation_relations
        ]
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("citation relations must be unique")

        known_claims = set(claim_ids)
        known_spans = set(span_ids)
        for relation in self.citation_relations:
            if relation.claim_id not in known_claims:
                raise ValueError(f"citation relation references unknown claim: {relation.claim_id}")
            if relation.span_id not in known_spans:
                raise ValueError(f"citation relation references unknown span: {relation.span_id}")

        if self.answerable:
            self._validate_answerable(known_spans)
        else:
            self._validate_unanswerable()
        self._validate_annotation_state()
        return self

    def _validate_answerable(self, known_spans: set[str]) -> None:
        if not self.evidence_spans:
            raise ValueError("answerable question requires at least one evidence span")
        if self.unanswerable_reason is not None:
            raise ValueError("answerable question cannot contain an unanswerable reason")
        required_or_supporting = {
            span.span_id
            for span in self.evidence_spans
            if span.support_role in {"required", "supporting"}
        }
        for claim in self.must_have_claims:
            supporting = {
                relation.span_id
                for relation in self.citation_relations
                if relation.claim_id == claim.claim_id and relation.relation == "supports"
            }
            if not supporting or not supporting.issubset(known_spans):
                raise ValueError(f"must-have claim has no supporting citation: {claim.claim_id}")
            if not supporting & required_or_supporting:
                raise ValueError(
                    f"must-have claim is supported only by distractor spans: {claim.claim_id}"
                )

    def _validate_unanswerable(self) -> None:
        if self.unanswerable_reason is None:
            raise ValueError("unanswerable question requires an unanswerable reason")
        if any(span.support_role != "distractor" for span in self.evidence_spans):
            raise ValueError("unanswerable question evidence spans must be distractors")
        if any(relation.relation == "supports" for relation in self.citation_relations):
            raise ValueError("unanswerable question cannot contain supporting citations")

    def _validate_annotation_state(self) -> None:
        if self.annotation_status == "silver_generated":
            if self.reviewer_ids or self.adjudicator_id is not None:
                raise ValueError("generated silver question cannot contain completed reviews")
            return
        if len(self.reviewer_ids) < 2:
            raise ValueError("reviewed questions require two independent reviewers")
        if self.annotation_status == "silver_reviewed":
            if self.adjudicator_id is not None:
                raise ValueError("silver reviewed question cannot contain an adjudicator")
            return
        if self.adjudicator_id is None:
            raise ValueError("gold adjudicated question requires an adjudicator")
        if self.adjudicator_id in self.reviewer_ids:
            raise ValueError("adjudicator must be distinct from the independent reviewers")


class SourceReplayReport(FrozenModel):
    schema_version: Literal["gold-source-replay-v1"] = "gold-source-replay-v1"
    question_count: int = Field(ge=0)
    span_count: int = Field(ge=0)
    projected_chunk_count: int = Field(ge=0)
    question_ids: tuple[str, ...]


class SourceReplayError(ValueError):
    """A private gold span cannot be reproduced from the frozen local artifacts."""


def load_gold_dataset(path: Path) -> list[GoldQuestion]:
    """Load strict JSONL without exposing file contents in error messages."""
    questions: list[GoldQuestion] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    questions.append(GoldQuestion.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(
                        f"invalid gold record at line {line_number} ({type(error).__name__})"
                    ) from error
    except OSError as error:
        raise ValueError(f"cannot read gold dataset file: {path.name}") from error
    identifiers = [question.question_id for question in questions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("gold dataset question_id values must be unique")
    return questions


def validate_source_replay(
    questions: Sequence[GoldQuestion],
    elements_path: Path,
    chunks_path: Path,
) -> SourceReplayReport:
    """Replay authoritative spans and their current chunk mappings, fail closed."""
    required_element_ids = {
        span.element_id for question in questions for span in question.evidence_spans
    }
    required_chunk_ids = {
        chunk_id
        for question in questions
        for span in question.evidence_spans
        for chunk_id in span.projected_chunk_ids
    }
    elements = _records_by_unique_id(
        elements_path,
        "element_id",
        "element",
        required_element_ids,
    )
    chunks = _records_by_unique_id(
        chunks_path,
        "chunk_id",
        "chunk",
        required_chunk_ids,
    )
    seen_chunk_ids: set[str] = set()
    span_count = 0
    for question in questions:
        for span in question.evidence_spans:
            span_count += 1
            _replay_span(question.question_id, span, elements, chunks)
            seen_chunk_ids.update(span.projected_chunk_ids)
    return SourceReplayReport(
        question_count=len(questions),
        span_count=span_count,
        projected_chunk_count=len(seen_chunk_ids),
        question_ids=tuple(question.question_id for question in questions),
    )


def _replay_span(
    question_id: str,
    span: EvidenceSpan,
    elements: Mapping[str, Mapping[str, Any]],
    chunks: Mapping[str, Mapping[str, Any]],
) -> None:
    element = elements.get(span.element_id)
    if element is None:
        raise SourceReplayError(f"{question_id}/{span.span_id}: unknown element")
    if element.get("corpus_id") != span.paper_id:
        raise SourceReplayError(f"{question_id}/{span.span_id}: element paper mismatch")
    if element.get("asset_id") != span.evidence_version_id:
        raise SourceReplayError(f"{question_id}/{span.span_id}: evidence version mismatch")
    if element.get("page_number") != span.page:
        raise SourceReplayError(f"{question_id}/{span.span_id}: element page mismatch")
    raw_text = element.get("raw_text")
    if not isinstance(raw_text, str):
        raise SourceReplayError(f"{question_id}/{span.span_id}: element has no raw text")
    if span.raw_span_end > len(raw_text):
        raise SourceReplayError(f"{question_id}/{span.span_id}: raw span is out of bounds")
    replayed = raw_text[span.raw_span_start : span.raw_span_end]
    if replayed != span.raw_quote:
        raise SourceReplayError(f"{question_id}/{span.span_id}: raw span quote mismatch")
    if _span_sha256(replayed) != span.span_hash:
        raise SourceReplayError(f"{question_id}/{span.span_id}: span hash mismatch")

    for chunk_id in span.projected_chunk_ids:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise SourceReplayError(f"{question_id}/{span.span_id}: unknown chunk projection")
        if chunk.get("corpus_id") != span.paper_id:
            raise SourceReplayError(f"{question_id}/{span.span_id}: chunk paper mismatch")
        if chunk.get("asset_id") != span.evidence_version_id:
            raise SourceReplayError(f"{question_id}/{span.span_id}: chunk evidence version mismatch")
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if (
            not isinstance(page_start, int)
            or not isinstance(page_end, int)
            or not page_start <= span.page <= page_end
        ):
            raise SourceReplayError(f"{question_id}/{span.span_id}: chunk page mismatch")
        element_ids = chunk.get("element_ids")
        if not isinstance(element_ids, list) or span.element_id not in element_ids:
            raise SourceReplayError(f"{question_id}/{span.span_id}: chunk element mismatch")


def _records_by_unique_id(
    path: Path,
    identifier_field: str,
    label: str,
    required_ids: set[str],
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SourceReplayError(
                        f"invalid {label} JSON at line {line_number}"
                    ) from error
                if not isinstance(value, dict):
                    raise SourceReplayError(f"invalid {label} record at line {line_number}")
                identifier = value.get(identifier_field)
                if not isinstance(identifier, str) or not identifier:
                    raise SourceReplayError(f"{label} record has no stable ID at line {line_number}")
                if identifier not in required_ids:
                    continue
                if identifier in records:
                    raise SourceReplayError(f"ambiguous duplicate {label} ID: {identifier}")
                records[identifier] = value
    except OSError as error:
        raise SourceReplayError(f"cannot read {label} source file: {path.name}") from error
    missing_count = len(required_ids - set(records))
    if missing_count:
        raise SourceReplayError(f"unknown {label} source IDs: {missing_count}")
    return records


def _span_sha256(value: str) -> Sha256:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dataset_summary(questions: Iterable[GoldQuestion]) -> dict[str, object]:
    """Return a body-free summary suitable for command output and reports."""
    materialized = tuple(questions)
    return {
        "schema_version": "rag-gold-summary-v1",
        "question_count": len(materialized),
        "answerable_count": sum(question.answerable for question in materialized),
        "unanswerable_count": sum(not question.answerable for question in materialized),
        "gold_adjudicated_count": sum(
            question.annotation_status == "gold_adjudicated" for question in materialized
        ),
        "span_count": sum(len(question.evidence_spans) for question in materialized),
        "question_ids": [question.question_id for question in materialized],
    }
