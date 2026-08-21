"""Deterministic construction and validation for fact-level search queries."""

from __future__ import annotations

import re
import unicodedata

from paper_research_agent.agent.models import (
    EvidenceFactRequirement,
    ResearchPlan,
    ResearchStep,
)

MAX_FACT_SEARCH_QUERY_LENGTH = 2000


def _normalized_match_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def question_anchor_catalog(question: str) -> tuple[str, ...]:
    """Expose exact question clauses so the planner can select rather than rewrite."""
    normalized = unicodedata.normalize("NFKC", question)
    full_question = " ".join(question.split())
    clauses = [
        " ".join(part.split())
        for part in re.split(r"[，,。\.；;：:\n！？!?]+", normalized)
        if " ".join(part.split())
    ]
    derived: list[str] = []
    for clause in clauses:
        major_parts = [
            part.strip()
            for part in re.split(r"同时|以及|并(?:且|以)?|且", clause)
            if part.strip()
        ]
        if len(major_parts) > 1:
            derived.extend(major_parts)
        for part in major_parts:
            minor_parts = [
                item.strip()
                for item in re.split(r"和|与|及", part)
                if item.strip()
            ]
            if len(minor_parts) > 1:
                derived.extend(minor_parts)
    return tuple(dict.fromkeys((full_question, *clauses, *derived)))[:40]


def bind_question_anchors(
    plan: ResearchPlan,
    anchor_catalog: tuple[str, ...],
) -> ResearchPlan:
    """Resolve model-selected catalog IDs to trusted question text."""
    if plan.task_type != "comparison":
        return plan
    if not anchor_catalog:
        raise ValueError("protected anchor catalog must not be empty")
    bound_requirements = []
    for requirement in plan.requirements:
        bound_facts = []
        for fact in requirement.fact_requirements:
            if fact.origin != "planned" and not fact.protected_anchor_ids:
                bound_facts.append(fact)
                continue
            fallback = not fact.protected_anchor_ids or any(
                anchor_id < 0 or anchor_id >= len(anchor_catalog)
                for anchor_id in fact.protected_anchor_ids
            )
            selected_ids = (0,) if fallback else fact.protected_anchor_ids
            anchors = tuple(
                dict.fromkeys(anchor_catalog[index] for index in selected_ids)
            )
            bound_facts.append(
                fact.model_copy(
                    update={
                        "protected_anchor_ids": selected_ids,
                        "protected_anchors": anchors,
                        "protected_anchor_fallback": fallback,
                    }
                )
            )
        bound_requirements.append(
            requirement.model_copy(update={"fact_requirements": tuple(bound_facts)})
        )
    return ResearchPlan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "requirements": [
                item.model_dump(mode="json") for item in bound_requirements
            ],
        }
    )


def validate_question_anchors(
    question: str,
    fact_requirement: EvidenceFactRequirement,
) -> None:
    """Require every protected anchor to be a verbatim span of the user question."""
    normalized_question = _normalized_match_text(question)
    for anchor in fact_requirement.protected_anchors:
        if _normalized_match_text(anchor) not in normalized_question:
            raise ValueError("protected anchor must be a verbatim span of the question")


def compose_fact_search_query(
    target_label: str,
    fact_requirement: EvidenceFactRequirement,
) -> str:
    """Append expansions to protected anchors without rewriting either collection."""
    ordered_terms = (
        " ".join(target_label.split()),
        *fact_requirement.protected_anchors,
        *fact_requirement.retrieval_expansions,
    )
    unique_terms: list[str] = []
    seen: set[str] = set()
    for term in ordered_terms:
        key = _normalized_match_text(term)
        if not key or key in seen:
            continue
        seen.add(key)
        unique_terms.append(term)
    query = " ".join(unique_terms)
    if len(query) > MAX_FACT_SEARCH_QUERY_LENGTH:
        raise ValueError("fact search query exceeds maximum length")
    return query


def materialize_atomic_fact_steps(plan: ResearchPlan) -> ResearchPlan:
    """Replace new comparison discovery steps with deterministic fact-level steps."""
    if plan.task_type != "comparison":
        return plan
    facts = tuple(
        fact
        for requirement in plan.requirements
        for fact in requirement.fact_requirements
    )
    if not facts or any(
        fact.origin != "planned" or not fact.protected_anchors for fact in facts
    ):
        return plan

    target_by_id = {item.target_id: item for item in plan.targets}
    dimension_by_id = {item.dimension_id: item for item in plan.dimensions}
    step_by_pair = {
        (step.target_ids[0], step.dimension_ids[0]): step for step in plan.steps
    }
    atomic_steps: list[ResearchStep] = []
    for index, requirement in enumerate(plan.requirements, 1):
        target = target_by_id[requirement.target_id]
        dimension = dimension_by_id[requirement.dimension_id]
        template = step_by_pair[(requirement.target_id, requirement.dimension_id)]
        for fact in requirement.fact_requirements:
            objective = (
                f"Find {target.label} {dimension.label}: {fact.description}"
            )[:500]
            atomic_steps.append(
                ResearchStep(
                    step_id=f"fact-{index:02d}-{len(atomic_steps) + 1:02d}",
                    objective=objective,
                    query=compose_fact_search_query(target.label, fact),
                    top_k=template.top_k,
                    corpus_id=target.corpus_id,
                    target_ids=(requirement.target_id,),
                    dimension_ids=(requirement.dimension_id,),
                    fact_requirement_id=fact.fact_requirement_id,
                )
            )
    return ResearchPlan.model_validate(
        {
            **plan.model_dump(mode="json"),
            "steps": [item.model_dump(mode="json") for item in atomic_steps],
        }
    )
