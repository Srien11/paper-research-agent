"""Strict, body-free contracts for private candidate-paper retrieval gold."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CandidateGoldSplit = Literal["dev", "sealed_test"]
CandidateTaskType = Literal["single_paper_identification", "multi_paper_comparison"]
CandidateClueScope = Literal["title_abstract", "full_text_detail"]


class CandidatePaperGoldQuestion(BaseModel):
    """One manually reviewed mapping from a natural question to local papers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["candidate-paper-gold-v1"] = "candidate-paper-gold-v1"
    question_id: str = Field(pattern=r"^CPG\d{3,}$")
    split: CandidateGoldSplit
    language: Literal["zh", "en", "mixed"]
    task_type: CandidateTaskType
    difficulty: Literal["easy", "medium", "hard"]
    clue_scope: CandidateClueScope
    question: str = Field(min_length=1, max_length=2000)
    relevant_paper_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    nearest_distractor_paper_ids: tuple[str, ...] = Field(default=(), max_length=12)
    annotation_reasons: dict[str, str]
    annotation_status: Literal["delegated_expert_reviewed"]
    reviewer_id: str = Field(min_length=1, max_length=128)
    review_passes: tuple[Literal["authorship", "reverse_verification"], ...]
    corpus_version: str = Field(min_length=1, max_length=256)

    @field_validator("question", "reviewer_id", "corpus_version")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("candidate gold text fields cannot be blank")
        return normalized

    @field_validator("question")
    @classmethod
    def reject_explicit_corpus_ids(cls, value: str) -> str:
        if re.search(r"(?<![A-Za-z0-9])[CT]\d{3}(?!\d)", value, flags=re.IGNORECASE):
            raise ValueError("candidate gold question must not contain corpus IDs")
        return value

    @field_validator("relevant_paper_ids", "nearest_distractor_paper_ids")
    @classmethod
    def validate_paper_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("candidate gold paper IDs must be unique")
        if any(re.fullmatch(r"[CT]\d{3}", value) is None for value in values):
            raise ValueError("candidate gold paper ID is invalid")
        return values

    @field_validator("review_passes")
    @classmethod
    def require_two_review_passes(
        cls,
        values: tuple[Literal["authorship", "reverse_verification"], ...],
    ) -> tuple[Literal["authorship", "reverse_verification"], ...]:
        if set(values) != {"authorship", "reverse_verification"} or len(values) != 2:
            raise ValueError("candidate gold requires authorship and reverse verification")
        return values

    @field_validator("annotation_reasons")
    @classmethod
    def normalize_reasons(cls, values: dict[str, str]) -> dict[str, str]:
        normalized = {key: " ".join(value.split()) for key, value in values.items()}
        if any(not value or len(value) > 500 for value in normalized.values()):
            raise ValueError("candidate gold reasons must contain 1 to 500 characters")
        return normalized

    @model_validator(mode="after")
    def validate_mapping(self) -> CandidatePaperGoldQuestion:
        relevant = set(self.relevant_paper_ids)
        if set(self.annotation_reasons) != relevant:
            raise ValueError("candidate gold reasons must match relevant papers")
        if relevant & set(self.nearest_distractor_paper_ids):
            raise ValueError("candidate gold distractors must not overlap relevant papers")
        expected_task = (
            "single_paper_identification"
            if len(self.relevant_paper_ids) == 1
            else "multi_paper_comparison"
        )
        if self.task_type != expected_task:
            raise ValueError("candidate gold task type does not match target count")
        return self


def load_candidate_paper_gold(path: Path) -> tuple[CandidatePaperGoldQuestion, ...]:
    questions: list[CandidatePaperGoldQuestion] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    questions.append(CandidatePaperGoldQuestion.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(
                        f"invalid candidate gold record at line {line_number}"
                    ) from error
    except OSError as error:
        raise ValueError(f"cannot read candidate gold file: {path.name}") from error
    identifiers = [question.question_id for question in questions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate gold question_id values must be unique")
    return tuple(questions)


def candidate_gold_summary(
    questions: Iterable[CandidatePaperGoldQuestion],
) -> dict[str, object]:
    materialized = tuple(questions)
    return {
        "schema_version": "candidate-paper-gold-summary-v1",
        "question_count": len(materialized),
        "dev_count": sum(question.split == "dev" for question in materialized),
        "sealed_test_count": sum(
            question.split == "sealed_test" for question in materialized
        ),
        "title_abstract_count": sum(
            question.clue_scope == "title_abstract" for question in materialized
        ),
        "full_text_detail_count": sum(
            question.clue_scope == "full_text_detail" for question in materialized
        ),
        "single_paper_count": sum(
            question.task_type == "single_paper_identification"
            for question in materialized
        ),
        "multi_paper_count": sum(
            question.task_type == "multi_paper_comparison" for question in materialized
        ),
        "question_ids": [question.question_id for question in materialized],
    }
