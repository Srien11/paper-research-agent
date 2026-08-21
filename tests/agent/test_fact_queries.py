from __future__ import annotations

import pytest

from paper_research_agent.agent.fact_queries import (
    bind_question_anchors,
    compose_fact_search_query,
    materialize_atomic_fact_steps,
    question_anchor_catalog,
    validate_question_anchors,
)
from paper_research_agent.agent.models import (
    EvidenceFactRequirement,
    EvidenceRequirement,
    ResearchDimension,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
)


def _fact(
    *,
    anchors: tuple[str, ...],
    expansions: tuple[str, ...] = (),
) -> EvidenceFactRequirement:
    return EvidenceFactRequirement(
        fact_requirement_id="paper-limit-reasoning",
        description="Explain why long-context reasoning fails",
        protected_anchors=anchors,
        retrieval_expansions=expansions,
    )


def test_compose_fact_query_preserves_anchors_and_only_appends_expansions() -> None:
    fact = _fact(
        anchors=("长上下文", "推理", "失败"),
        expansions=(
            "long-context",
            "reasoning",
            "failure",
            "limitation",
            "LongBench",
        ),
    )

    query = compose_fact_search_query("LongBench", fact)

    assert query == (
        "LongBench 长上下文 推理 失败 "
        "long-context reasoning failure limitation"
    )
    assert all(anchor in query for anchor in fact.protected_anchors)


def test_validate_question_anchors_accepts_verbatim_normalized_spans() -> None:
    fact = _fact(anchors=("深度理解", "推理", "7"))

    validate_question_anchors(
        "请比较论文在  深度理解 / 推理 任务上的 7 个失败案例",
        fact,
    )


def test_validate_question_anchors_rejects_semantic_generalization() -> None:
    fact = _fact(anchors=("task characteristics", "evaluation"))

    with pytest.raises(ValueError, match="verbatim span"):
        validate_question_anchors("解释论文为何在深度理解和推理上失败", fact)


def test_question_anchor_catalog_exposes_exact_anonymous_paper_clauses() -> None:
    question = (
        "在逻辑推理自我修正研究中，一篇认为没有外部反馈时模型的内在自我修正经常无效，"
        "另一篇区分找错与改错；指出给出错误位置后模型能够纠正。请找出两篇论文。"
    )

    assert question_anchor_catalog(question) == (
        question,
        "在逻辑推理自我修正研究中",
        "一篇认为没有外部反馈时模型的内在自我修正经常无效",
        "另一篇区分找错与改错",
        "指出给出错误位置后模型能够纠正",
        "请找出两篇论文",
        "另一篇区分找错",
        "改错",
    )


def test_question_anchor_catalog_adds_shorter_verbatim_connector_spans() -> None:
    catalog = question_anchor_catalog(
        "一项系统讨论位置、冗长和自我增强偏差并以多轮问答和竞技场验证。"
    )

    assert "一项系统讨论位置、冗长和自我增强偏差" in catalog
    assert "自我增强偏差" in catalog
    assert "多轮问答" in catalog
    assert "竞技场验证" in catalog

def test_bind_question_anchors_resolves_catalog_ids_without_model_rewrite() -> None:
    plan = _structured_plan()
    requirements = tuple(
        requirement.model_copy(
            update={
                "fact_requirements": tuple(
                    fact.model_copy(
                        update={
                            "protected_anchor_ids": (0,) if index == 0 else (1,),
                            "protected_anchors": (),
                        }
                    )
                    for index, fact in enumerate(requirement.fact_requirements)
                )
            }
        )
        for requirement in plan.requirements
    )
    unbound = plan.model_copy(update={"requirements": requirements})

    bound = bind_question_anchors(unbound, ("长上下文推理失败", "缓解方法"))

    assert [
        fact.protected_anchors
        for requirement in bound.requirements
        for fact in requirement.fact_requirements
    ] == [("长上下文推理失败",), ("缓解方法",), ("长上下文推理失败",)]


def test_bind_question_anchors_falls_back_to_full_question_for_unknown_id() -> None:
    plan = _structured_plan()
    requirement = plan.requirements[0]
    invalid_fact = requirement.fact_requirements[0].model_copy(
        update={"protected_anchor_ids": (99,), "protected_anchors": ()}
    )
    invalid_requirement = requirement.model_copy(
        update={"fact_requirements": (invalid_fact, *requirement.fact_requirements[1:])}
    )
    invalid_plan = plan.model_copy(
        update={"requirements": (invalid_requirement, *plan.requirements[1:])}
    )

    bound = bind_question_anchors(
        invalid_plan,
        ("完整用户问题", "长上下文推理失败", "缓解方法"),
    )

    repaired = bound.requirements[0].fact_requirements[0]
    assert repaired.protected_anchor_ids == (0,)
    assert repaired.protected_anchors == ("完整用户问题",)
    assert repaired.protected_anchor_fallback is True


def test_compose_fact_query_fails_closed_instead_of_truncating_anchors() -> None:
    fact = _fact(
        anchors=("a" * 1000, "b" * 1000),
        expansions=("reasoning",),
    )

    with pytest.raises(ValueError, match="maximum length"):
        compose_fact_search_query("Paper", fact)


def _structured_plan() -> ResearchPlan:
    return ResearchPlan(
        task_type="comparison",
        targets=(
            ResearchTarget(target_id="a", label="Paper A", corpus_id="C001"),
            ResearchTarget(target_id="b", label="Paper B", corpus_id="T001"),
        ),
        dimensions=(ResearchDimension(dimension_id="limits", label="Limits"),),
        requirements=(
            EvidenceRequirement(
                requirement_id="a-limits",
                target_id="a",
                dimension_id="limits",
                description="Paper A limits",
                fact_requirements=(
                    EvidenceFactRequirement(
                        fact_requirement_id="a-long-reasoning",
                        description="Long-context reasoning failure",
                        protected_anchors=("长上下文", "推理", "失败"),
                        retrieval_expansions=("long-context", "reasoning", "failure"),
                    ),
                    EvidenceFactRequirement(
                        fact_requirement_id="a-mitigation",
                        description="Mitigation",
                        protected_anchors=("缓解方法",),
                        retrieval_expansions=("mitigation",),
                    ),
                ),
            ),
            EvidenceRequirement(
                requirement_id="b-limits",
                target_id="b",
                dimension_id="limits",
                description="Paper B limits",
                fact_requirements=(
                    EvidenceFactRequirement(
                        fact_requirement_id="b-long-reasoning",
                        description="Long-context reasoning failure",
                        protected_anchors=("长上下文", "推理", "失败"),
                        retrieval_expansions=("long-context", "reasoning", "failure"),
                    ),
                ),
            ),
        ),
        steps=(
            ResearchStep(
                step_id="a-limits",
                objective="Discover Paper A limits",
                query="broad A query",
                corpus_id="C001",
                target_ids=("a",),
                dimension_ids=("limits",),
            ),
            ResearchStep(
                step_id="b-limits",
                objective="Discover Paper B limits",
                query="broad B query",
                corpus_id="T001",
                target_ids=("b",),
                dimension_ids=("limits",),
            ),
        ),
    )


def test_materialize_atomic_fact_steps_replaces_broad_cell_queries() -> None:
    materialized = materialize_atomic_fact_steps(_structured_plan())

    assert [step.fact_requirement_id for step in materialized.steps] == [
        "a-long-reasoning",
        "a-mitigation",
        "b-long-reasoning",
    ]
    assert [step.corpus_id for step in materialized.steps] == ["C001", "C001", "T001"]
    assert [step.query for step in materialized.steps] == [
        "Paper A 长上下文 推理 失败 long-context reasoning failure",
        "Paper A 缓解方法 mitigation",
        "Paper B 长上下文 推理 失败 long-context reasoning failure",
    ]
    assert all("broad" not in step.query for step in materialized.steps)


def test_materialize_atomic_fact_steps_preserves_legacy_plan() -> None:
    plan = _structured_plan()
    legacy_requirements = tuple(
        requirement.model_copy(
            update={
                "fact_requirements": tuple(
                    fact.model_copy(
                        update={"protected_anchors": (), "retrieval_expansions": ()}
                    )
                    for fact in requirement.fact_requirements
                )
            }
        )
        for requirement in plan.requirements
    )
    legacy = plan.model_copy(update={"requirements": legacy_requirements})

    assert materialize_atomic_fact_steps(legacy) == legacy
