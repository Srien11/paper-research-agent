"""Deterministic, metadata-only blueprint selection for the private gold dataset."""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANSWERABLE_TASK_QUOTAS = {
    "definition_scope": 10,
    "method_mechanism": 10,
    "experimental_result": 8,
    "multi_paper_comparison": 10,
    "conflicting_evidence": 8,
    "multi_hop_synthesis": 8,
    "figure_table_explanation": 6,
}
ANSWERABLE_LANGUAGE_QUOTAS = {"zh": 36, "en": 18, "mixed": 6}
ANSWERABLE_DIFFICULTY_QUOTAS = {"easy": 18, "medium": 26, "hard": 16}
ANSWERABLE_EVIDENCE_QUOTAS = {
    "body": 28,
    "table_appendix": 12,
    "figure_caption": 8,
    "mixed_sources": 12,
}
UNANSWERABLE_REASON_QUOTAS = {
    "corpus_absent": 5,
    "false_premise": 4,
    "missing_precise_value": 3,
    "time_cutoff": 3,
    "missing_artifact": 3,
    "missing_comparison_arm": 2,
}

MULTI_PAPER_TASKS = frozenset(
    {"multi_paper_comparison", "conflicting_evidence", "multi_hop_synthesis"}
)
CandidateTaskType = Literal[
    "definition_scope",
    "method_mechanism",
    "experimental_result",
    "multi_paper_comparison",
    "conflicting_evidence",
    "multi_hop_synthesis",
    "figure_table_explanation",
]


class CandidateBlueprint(BaseModel):
    """One selection slot; it intentionally contains no evidence or governance prompts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^G\d{3}$")
    answerable: bool
    task_type: CandidateTaskType
    language: Literal["zh", "en", "mixed"]
    difficulty: Literal["easy", "medium", "hard"]
    evidence_source: Literal["body", "table_appendix", "figure_caption", "mixed_sources"]
    primary_split: Literal["core", "challenge"]
    target_paper_ids: tuple[str, ...] = Field(min_length=1, max_length=3)
    expected_status: Literal["answered", "insufficient_evidence"]
    unanswerable_reason: str | None = None
    nearest_distractor_paper_ids: tuple[str, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def validate_answerability(self) -> CandidateBlueprint:
        if self.answerable:
            if self.expected_status != "answered" or self.unanswerable_reason is not None:
                raise ValueError("answerable blueprints require answered status and no reason")
        elif (
            self.expected_status != "insufficient_evidence"
            or self.unanswerable_reason not in UNANSWERABLE_REASON_QUOTAS
            or not self.nearest_distractor_paper_ids
        ):
            raise ValueError("unanswerable blueprints require a reason and distractors")
        return self


def build_candidate_blueprint(
    papers: Sequence[Mapping[str, object]],
    *,
    seed: int = 20260806,
) -> list[CandidateBlueprint]:
    """Build the frozen 60+20 selection matrix using only safe bibliography fields."""

    safe_rows = [_safe_paper(row) for row in papers]
    identifiers = [row["corpus_id"] for row in safe_rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("corpus IDs must be unique")
    core = sorted((row for row in safe_rows if row["dataset_split"] == "core"), key=_paper_id)
    challenge = sorted(
        (row for row in safe_rows if row["dataset_split"] == "challenge"), key=_paper_id
    )
    if len(core) < 42 or len(challenge) < 18:
        raise ValueError("candidate selection requires at least 42 core and 18 challenge papers")

    rng = random.Random(seed)
    rng.shuffle(core)
    rng.shuffle(challenge)
    primary_rows = core[:42] + challenge[:18]
    rng.shuffle(primary_rows)

    tasks = _expanded(ANSWERABLE_TASK_QUOTAS)
    languages = _expanded(ANSWERABLE_LANGUAGE_QUOTAS)
    difficulties = _expanded(ANSWERABLE_DIFFICULTY_QUOTAS)
    evidence_sources = _expanded(ANSWERABLE_EVIDENCE_QUOTAS)
    for values in (tasks, languages, difficulties, evidence_sources):
        rng.shuffle(values)

    all_rows = sorted(safe_rows, key=_paper_id)
    answerable: list[CandidateBlueprint] = []
    for index, primary in enumerate(primary_rows):
        task_type = tasks[index]
        target_ids = [str(primary["corpus_id"])]
        if task_type in MULTI_PAPER_TASKS:
            target_ids.append(_secondary_paper(primary, all_rows, offset=index + seed))
        answerable.append(
            CandidateBlueprint(
                case_id=f"G{index + 1:03d}",
                answerable=True,
                task_type=task_type,
                language=languages[index],
                difficulty=difficulties[index],
                evidence_source=evidence_sources[index],
                primary_split=primary["dataset_split"],
                target_paper_ids=tuple(target_ids),
                expected_status="answered",
            )
        )

    negative_reasons = _expanded(UNANSWERABLE_REASON_QUOTAS)
    rng.shuffle(negative_reasons)
    negative_languages = _expanded({"zh": 13, "en": 5, "mixed": 2})
    negative_difficulties = _expanded({"medium": 8, "hard": 12})
    rng.shuffle(negative_languages)
    rng.shuffle(negative_difficulties)
    negative_rows = list(all_rows)
    rng.shuffle(negative_rows)
    negatives: list[CandidateBlueprint] = []
    for offset, reason in enumerate(negative_reasons):
        anchor = negative_rows[offset]
        distractors = [
            str(anchor["corpus_id"]),
            _secondary_paper(anchor, all_rows, offset=offset + seed + 101),
        ]
        negatives.append(
            CandidateBlueprint(
                case_id=f"G{61 + offset:03d}",
                answerable=False,
                task_type=_negative_task_type(reason, offset),
                language=negative_languages[offset],
                difficulty=negative_difficulties[offset],
                evidence_source="body",
                primary_split=anchor["dataset_split"],
                target_paper_ids=(str(anchor["corpus_id"]),),
                expected_status="insufficient_evidence",
                unanswerable_reason=reason,
                nearest_distractor_paper_ids=tuple(distractors),
            )
        )
    return answerable + negatives


def _safe_paper(row: Mapping[str, object]) -> dict[str, str]:
    """Project only fields allowed for sampling; never copy challenge-design metadata."""

    corpus_id = str(row.get("corpus_id", ""))
    split = str(row.get("dataset_split", ""))
    if not re.fullmatch(r"[CT]\d{3}", corpus_id):
        raise ValueError(f"invalid corpus ID: {corpus_id!r}")
    if split not in {"core", "challenge"}:
        raise ValueError(f"invalid dataset split for {corpus_id}")
    title = str(row.get("title", "")).strip()
    storage_class = str(row.get("storage_class", "")).strip()
    if not title or not storage_class:
        raise ValueError(f"paper {corpus_id} lacks safe sampling metadata")
    return {
        "corpus_id": corpus_id,
        "title": title,
        "dataset_split": split,
        "storage_class": storage_class,
    }


def _paper_id(row: Mapping[str, str]) -> str:
    return row["corpus_id"]


def _expanded(quotas: Mapping[str, int]) -> list[str]:
    return [name for name, count in quotas.items() for _ in range(count)]


def _secondary_paper(
    primary: Mapping[str, str],
    papers: Sequence[Mapping[str, str]],
    *,
    offset: int,
) -> str:
    """Choose a stable nearby-title candidate without consulting challenge annotations."""

    primary_tokens = _title_tokens(primary["title"])
    candidates = [row for row in papers if row["corpus_id"] != primary["corpus_id"]]
    ranked = sorted(
        candidates,
        key=lambda row: (
            -len(primary_tokens & _title_tokens(row["title"])),
            row["corpus_id"],
        ),
    )
    window = min(5, len(ranked))
    return ranked[offset % window]["corpus_id"]


def _title_tokens(title: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", title.lower()) if len(token) > 2}


def _negative_task_type(reason: str, offset: int) -> CandidateTaskType:
    if reason == "missing_precise_value":
        return "experimental_result"
    if reason == "missing_artifact":
        return "figure_table_explanation"
    if reason == "missing_comparison_arm":
        return "multi_paper_comparison"
    if reason == "false_premise":
        return "conflicting_evidence"
    if reason == "time_cutoff":
        return "method_mechanism"
    return ("definition_scope", "method_mechanism")[offset % 2]
