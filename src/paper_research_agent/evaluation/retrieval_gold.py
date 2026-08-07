"""Private span-level retrieval view derived from reviewed RAG questions."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from paper_research_agent.evaluation.gold_dataset import EvidenceSpan, GoldQuestion

RetrievalCategory = Literal["single_one_span", "single_multi_span", "cross_paper", "figure"]
CATEGORY_QUOTAS: dict[RetrievalCategory, int] = {
    "single_one_span": 12,
    "single_multi_span": 8,
    "cross_paper": 6,
    "figure": 4,
}


class RetrievalGoldQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["retrieval-span-gold-v1"] = "retrieval-span-gold-v1"
    query_id: str = Field(pattern=r"^RG\d{3}$")
    source_question_id: str = Field(pattern=r"^GQ\d{3,}$")
    category: RetrievalCategory
    query: str = Field(min_length=1, max_length=10_000)
    language: Literal["zh", "en", "mixed"]
    difficulty: Literal["easy", "medium", "hard"]
    relevant_paper_ids: tuple[str, ...] = Field(min_length=1)
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
    required_span_groups: tuple[tuple[str, ...], ...] = Field(min_length=1)
    relevant_chunk_ids: tuple[str, ...] = Field(min_length=1)
    annotation_status: Literal["silver_generated", "silver_reviewed", "gold_adjudicated"]
    corpus_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_relations(self) -> RetrievalGoldQuery:
        span_ids = {span.span_id for span in self.evidence_spans}
        if any(not group or not set(group).issubset(span_ids) for group in self.required_span_groups):
            raise ValueError("required span groups must reference known non-empty evidence")
        projected = {
            chunk_id for span in self.evidence_spans for chunk_id in span.projected_chunk_ids
        }
        if set(self.relevant_chunk_ids) != projected:
            raise ValueError("relevant chunk IDs must equal the evidence projection")
        if {span.paper_id for span in self.evidence_spans} != set(self.relevant_paper_ids):
            raise ValueError("relevant paper IDs must equal the evidence papers")
        return self


def build_retrieval_gold(
    questions: Sequence[GoldQuestion],
    *,
    seed: int = 20260806,
) -> list[RetrievalGoldQuery]:
    """Select the frozen 12/8/6/4 task mix from answerable source questions."""

    eligible = [question for question in questions if question.answerable]
    buckets: dict[RetrievalCategory, list[GoldQuestion]] = {
        category: [] for category in CATEGORY_QUOTAS
    }
    for question in eligible:
        buckets[_category(question)].append(question)
    for category, quota in CATEGORY_QUOTAS.items():
        if len(buckets[category]) < quota:
            raise ValueError(f"insufficient {category} candidates: {len(buckets[category])}/{quota}")

    rng = random.Random(seed)
    best: list[GoldQuestion] | None = None
    best_score: tuple[int, int] | None = None
    for _ in range(20_000):
        selected = []
        for category, quota in CATEGORY_QUOTAS.items():
            selected.extend(rng.sample(buckets[category], quota))
        if len({item.question_id for item in selected}) != 30:
            continue
        score = _selection_score(selected)
        if best_score is None or score < best_score:
            best, best_score = selected, score
        if score == (0, 0):
            break
    if best is None:
        raise ValueError("retrieval gold selection could not satisfy disjoint category quotas")
    best.sort(key=lambda item: item.question_id)
    return [_to_retrieval_query(index, question) for index, question in enumerate(best, start=1)]


def _category(question: GoldQuestion) -> RetrievalCategory:
    if question.task_type == "figure_table_explanation":
        return "figure"
    spans = _referenced_spans(question)
    if len({span.paper_id for span in spans}) > 1:
        return "cross_paper"
    return "single_one_span" if len(spans) == 1 else "single_multi_span"


def _referenced_spans(question: GoldQuestion) -> list[EvidenceSpan]:
    referenced = {
        relation.span_id for relation in question.citation_relations if relation.relation == "supports"
    }
    return [span for span in question.evidence_spans if span.span_id in referenced]


def _selection_score(selected: Sequence[GoldQuestion]) -> tuple[int, int]:
    language_target = {"zh": 18, "en": 9, "mixed": 3}
    difficulty_target = {"easy": 10, "medium": 12, "hard": 8}
    language = Counter(item.language for item in selected)
    difficulty = Counter(item.difficulty for item in selected)
    primary_split = Counter(
        "challenge" if _referenced_spans(item)[0].paper_id.startswith("T") else "core"
        for item in selected
    )
    deviation = sum(abs(language[key] - value) for key, value in language_target.items())
    deviation += sum(abs(difficulty[key] - value) for key, value in difficulty_target.items())
    deviation += abs(primary_split["core"] - 20) + abs(primary_split["challenge"] - 10)
    coverage = len(
        {span.paper_id for item in selected for span in _referenced_spans(item)}
    )
    return deviation, max(0, 24 - coverage)


def _to_retrieval_query(index: int, question: GoldQuestion) -> RetrievalGoldQuery:
    spans = _referenced_spans(question)
    span_by_id = {span.span_id: span for span in spans}
    groups: list[tuple[str, ...]] = []
    for claim in question.must_have_claims:
        group = tuple(
            relation.span_id
            for relation in question.citation_relations
            if relation.claim_id == claim.claim_id
            and relation.relation == "supports"
            and relation.span_id in span_by_id
        )
        if group:
            groups.append(group)
    relevant_chunks = tuple(
        sorted({chunk_id for span in spans for chunk_id in span.projected_chunk_ids})
    )
    return RetrievalGoldQuery(
        query_id=f"RG{index:03d}",
        source_question_id=question.question_id,
        category=_category(question),
        query=question.question,
        language=question.language,
        difficulty=question.difficulty,
        relevant_paper_ids=tuple(sorted({span.paper_id for span in spans})),
        evidence_spans=tuple(spans),
        required_span_groups=tuple(groups),
        relevant_chunk_ids=relevant_chunks,
        annotation_status=question.annotation_status,
        corpus_version=question.corpus_version,
    )
