"""Deterministic quarantine controls for model-generated silver evaluation data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research_agent.evaluation.gold_dataset import GoldQuestion

SilverIssueCode = Literal[
    "cross_paper_attribution",
    "degraded_ocr_evidence",
    "incomplete_reference",
    "non_atomic_reference",
    "question_reference_mismatch",
]
SilverReviewStatus = Literal["confirmed", "high_risk"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SilverQuarantineEntry(FrozenModel):
    question_id: str = Field(pattern=r"^GQ\d{3,}$")
    reason_code: SilverIssueCode
    review_status: SilverReviewStatus


class SilverQuarantineManifest(FrozenModel):
    schema_version: Literal["silver-quarantine-v1"] = "silver-quarantine-v1"
    dataset_id: str = Field(min_length=1, max_length=128)
    scope_question_ids: tuple[str, ...] = ()
    entries: tuple[SilverQuarantineEntry, ...]

    @model_validator(mode="after")
    def validate_unique_question_ids(self) -> SilverQuarantineManifest:
        identifiers = [entry.question_id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("silver quarantine question IDs must be unique")
        if len(self.scope_question_ids) != len(set(self.scope_question_ids)):
            raise ValueError("silver quarantine scope question IDs must be unique")
        if (
            any(entry.question_id not in self.scope_question_ids for entry in self.entries)
            and self.scope_question_ids
        ):
            raise ValueError("silver quarantine entries must belong to the frozen scope")
        return self

    @classmethod
    def from_path(cls, path: Path) -> SilverQuarantineManifest:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError("cannot read silver quarantine manifest") from error


class SilverQuarantineReport(FrozenModel):
    schema_version: Literal["silver-quarantine-report-v1"] = "silver-quarantine-report-v1"
    dataset_id: str
    input_question_count: int = Field(ge=0)
    scope_question_count: int = Field(ge=0)
    outside_scope_question_count: int = Field(ge=0)
    retained_question_count: int = Field(ge=0)
    excluded_question_count: int = Field(ge=0)
    excluded_question_ids: tuple[str, ...]
    reason_counts: dict[str, int]


def apply_silver_quarantine(
    questions: Sequence[GoldQuestion],
    manifest: SilverQuarantineManifest,
) -> tuple[tuple[GoldQuestion, ...], SilverQuarantineReport]:
    """Remove quarantined IDs and return a body-free, persistable audit report."""

    question_ids = {question.question_id for question in questions}
    manifest_ids = {entry.question_id for entry in manifest.entries}
    scope_ids = set(manifest.scope_question_ids) or question_ids
    unknown_ids = (manifest_ids | scope_ids) - question_ids
    if unknown_ids:
        raise ValueError(f"silver quarantine contains {len(unknown_ids)} unknown question IDs")

    scoped = tuple(question for question in questions if question.question_id in scope_ids)
    retained = tuple(question for question in scoped if question.question_id not in manifest_ids)
    excluded_ids = tuple(
        question.question_id for question in scoped if question.question_id in manifest_ids
    )
    reason_counts: dict[str, int] = {}
    for entry in manifest.entries:
        reason_counts[entry.reason_code] = reason_counts.get(entry.reason_code, 0) + 1
    report = SilverQuarantineReport(
        dataset_id=manifest.dataset_id,
        input_question_count=len(questions),
        scope_question_count=len(scoped),
        outside_scope_question_count=len(questions) - len(scoped),
        retained_question_count=len(retained),
        excluded_question_count=len(excluded_ids),
        excluded_question_ids=excluded_ids,
        reason_counts=dict(sorted(reason_counts.items())),
    )
    return retained, report
