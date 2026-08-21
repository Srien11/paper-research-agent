"""LangChain structured-output adapter for bounded research planning."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from paper_research_agent.agent.fact_queries import (
    bind_question_anchors,
    question_anchor_catalog,
    validate_question_anchors,
)
from paper_research_agent.agent.models import (
    ComparisonFactProposal,
    ComparisonPlanProposal,
    EvidenceFactRequirement,
    EvidenceRequirement,
    PlannerAttemptAudit,
    ResearchDimension,
    ResearchPlan,
    ResearchStep,
    ResearchTarget,
)
from paper_research_agent.agent.policy import MAX_INITIAL_PLAN_STEPS
from paper_research_agent.retrieval.contracts import QueryRewriteTrace
from paper_research_agent.retrieval.papers import (
    AsyncPaperCandidateRetriever,
    PaperCandidateQuery,
)


class ComparisonTargetResolutionError(RuntimeError):
    """Comparison targets could not be resolved without leaving local scope."""

    def __init__(
        self,
        reason_code: str,
        *,
        attempts: tuple[PlannerAttemptAudit, ...] = (),
    ) -> None:
        self.reason_code = reason_code
        self.attempts = attempts
        super().__init__(f"comparison target resolution failed: {reason_code}")


class ComparisonTargetResolver(Protocol):
    async def resolve(self, question: str) -> Mapping[str, str]: ...


class ComparisonQueryResolver(Protocol):
    async def resolve_query(self, question: str) -> QueryRewriteTrace: ...


_PLANNER_FAILURE_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("planned research requires a comparison plan", "planner_task_type_invalid"),
    ("task_type", "planner_task_type_invalid"),
    ("requires at least two targets", "planner_target_count_invalid"),
    ("targets must use distinct corpus ids", "planner_target_duplicate"),
    ("target ids must be unique", "planner_target_duplicate"),
    ("left the resolved candidate set", "planner_target_outside_candidate_set"),
    ("requires at least one dimension", "planner_dimension_invalid"),
    ("requires valid dimensions", "planner_dimension_invalid"),
    ("requires unique dimensions", "planner_dimension_invalid"),
    ("requirements must form a complete target-dimension grid", "planner_grid_incomplete"),
    ("requirement references an unknown target or dimension", "planner_requirement_reference_invalid"),
    ("step fact does not belong to its cell", "planner_requirement_reference_invalid"),
    ("steps require one target and one dimension", "planner_step_scope_invalid"),
    ("step corpus scope does not match its target", "planner_step_scope_invalid"),
    ("steps do not cover every requirement cell", "planner_step_grid_incomplete"),
    ("step ids must be unique", "planner_id_duplicate"),
    ("dimension ids must be unique", "planner_id_duplicate"),
    ("requirement ids must be unique", "planner_id_duplicate"),
    ("fact requirement ids must be globally unique", "planner_id_duplicate"),
    ("fact-bound research steps must be unique", "planner_id_duplicate"),
    ("exceeds the requested step budget", "planner_step_budget_invalid"),
    ("fact requirements exceed", "planner_fact_budget_invalid"),
    ("protected anchor", "planner_anchor_selection_invalid"),
    ("protected_anchor_ids", "planner_anchor_selection_invalid"),
)


_PLANNER_REPAIR_INSTRUCTIONS: dict[str, str] = {
    "planner_task_type_invalid": (
        "Return task_type=comparison and include the complete comparison metadata."
    ),
    "planner_target_count_invalid": "Return between two and four resolved targets.",
    "planner_target_duplicate": (
        "Return unique targets whose corpus IDs are distinct."
    ),
    "planner_target_outside_candidate_set": (
        "Use corpus IDs only from LOCAL_CORPUS_CATALOG_JSON."
    ),
    "planner_dimension_invalid": (
        "Return one or more unique dimensions explicitly requested by the question."
    ),
    "planner_grid_incomplete": (
        "Return exactly one requirement for every resolved target-by-dimension pair."
    ),
    "planner_requirement_reference_invalid": (
        "Bind every requirement and fact reference to declared target, dimension, and cell IDs."
    ),
    "planner_step_scope_invalid": (
        "Bind each step to exactly one target, one dimension, and that target's corpus ID."
    ),
    "planner_step_grid_incomplete": (
        "Return one discovery step for every resolved target-by-dimension pair."
    ),
    "planner_id_duplicate": "Return globally unique target, dimension, requirement, fact, and step IDs.",
    "planner_step_budget_invalid": "Return the full grid within the supplied step budget.",
    "planner_fact_budget_invalid": "Return no more atomic fact requirements than the supplied budget.",
    "planner_anchor_selection_invalid": (
        "Use valid, non-duplicate protected_anchor_ids from VERBATIM_ANCHOR_SOURCE_JSON."
    ),
    "planner_schema_invalid": "Return every field required by the ResearchPlan schema with valid types.",
}


def _planner_failure_code(error: ValueError) -> str:
    """Classify a planner failure without exposing model-authored text."""
    diagnostics: list[str] = []
    if isinstance(error, ValidationError):
        for item in error.errors():
            diagnostics.append(str(item.get("msg", "")))
            diagnostics.extend(str(part) for part in item.get("loc", ()))
            diagnostics.append(str(item.get("type", "")))
            location = tuple(str(part) for part in item.get("loc", ()))
            error_type = str(item.get("type", ""))
            if location in {("targets",), ("selected_corpus_ids",)} and error_type in {
                "too_short",
                "too_long",
            }:
                return "planner_target_count_invalid"
            if location in {("dimensions",), ("dimension_labels",)} and error_type in {
                "too_short",
                "too_long",
            }:
                return "planner_dimension_invalid"
    else:
        diagnostics.append(str(error))
    normalized = " ".join(diagnostics).casefold()
    for fragment, code in _PLANNER_FAILURE_FRAGMENTS:
        if fragment in normalized:
            return code
    return "planner_schema_invalid"


def parse_explicit_corpus_ids(question: str) -> tuple[str, ...]:
    """Parse exact local corpus identifiers without accepting partial IDs."""
    return tuple(
        dict.fromkeys(
            match.group(0).upper()
            for match in re.finditer(
                r"(?<![A-Za-z0-9])[CT]\d{3}(?!\d)",
                question,
                flags=re.IGNORECASE,
            )
        )
    )


def comparison_dimension_hints(question: str) -> tuple[str, ...]:
    """Derive high-level comparison topics only from explicit question clauses."""
    normalized = " ".join(question.split())
    scoped = re.split(r"[：:]", normalized, maxsplit=1)[-1]
    primary = (
        part.strip()
        for part in re.split(r"[，,；;。.!?！？]+", scoped)
        if part.strip()
    )
    hints: list[str] = []
    for part in primary:
        if re.match(r"^(?:请|找出|identify\b|find\b|which\b)", part, re.IGNORECASE):
            continue
        subparts = re.split(
            r"(?:同时|以及|并(?=以|且|指出|展示|验证|评估|比较|分析|说明))",
            part,
        )
        hints.extend(item.strip() for item in subparts if item.strip())
    return tuple(dict.fromkeys(hints))[:5] or (normalized,)


def build_comparison_research_plan(
    proposal: ComparisonPlanProposal,
    *,
    resolved_catalog: Mapping[str, str],
    max_steps: int,
    expected_dimension_count: int | None = None,
) -> ResearchPlan:
    """Build all comparison control-plane IDs and scopes deterministically."""
    if len(proposal.selected_corpus_ids) < 2:
        raise ValueError("comparison research plan requires at least two targets")
    if not set(proposal.selected_corpus_ids) <= set(resolved_catalog):
        raise ValueError("comparison plan left the resolved candidate set")
    if not proposal.dimension_labels:
        raise ValueError("comparison research plan requires at least one dimension")
    if (
        expected_dimension_count is not None
        and len(proposal.dimension_labels) != expected_dimension_count
    ):
        raise ValueError("comparison research plan requires valid dimensions")
    cell_count = len(proposal.selected_corpus_ids) * len(proposal.dimension_labels)
    if cell_count > max_steps:
        raise ValueError("research plan exceeds the requested step budget")
    materialized_fact_count = len(proposal.facts) * len(proposal.selected_corpus_ids)
    if materialized_fact_count > max_steps:
        raise ValueError(
            "comparison fact requirements exceed the requested step budget"
        )

    grouped: dict[int, list[ComparisonFactProposal]] = {}
    for fact in proposal.facts:
        if fact.dimension_index >= len(proposal.dimension_labels):
            raise ValueError(
                "comparison requirement references an unknown target or dimension"
            )
        grouped.setdefault(fact.dimension_index, []).append(fact)
    if set(grouped) != set(range(len(proposal.dimension_labels))):
        raise ValueError(
            "comparison requirements must form a complete target-dimension grid"
        )

    targets = tuple(
        ResearchTarget(
            target_id=f"target-{target_index:02d}",
            label=resolved_catalog[corpus_id],
            corpus_id=corpus_id,
        )
        for target_index, corpus_id in enumerate(proposal.selected_corpus_ids, 1)
    )
    dimensions = tuple(
        ResearchDimension(
            dimension_id=f"dimension-{dimension_index:02d}",
            label=label,
        )
        for dimension_index, label in enumerate(proposal.dimension_labels, 1)
    )
    requirements: list[EvidenceRequirement] = []
    steps: list[ResearchStep] = []
    for target_index, target in enumerate(targets, 1):
        assert target.corpus_id is not None
        for dimension_offset, dimension in enumerate(dimensions):
            requirement_id = (
                f"requirement-{target_index:02d}-{dimension_offset + 1:02d}"
            )
            fact_requirements = tuple(
                EvidenceFactRequirement(
                    fact_requirement_id=(
                        f"fact-{target_index:02d}-{dimension_offset + 1:02d}-"
                        f"{fact_index:02d}"
                    ),
                    description=fact.description,
                    protected_anchor_ids=fact.protected_anchor_ids,
                    retrieval_expansions=fact.retrieval_expansions,
                    required_qualifier_kinds=fact.required_qualifier_kinds,
                )
                for fact_index, fact in enumerate(
                    grouped[dimension_offset], 1
                )
            )
            requirements.append(
                EvidenceRequirement(
                    requirement_id=requirement_id,
                    target_id=target.target_id,
                    dimension_id=dimension.dimension_id,
                    description=f"{target.label}: {dimension.label}",
                    fact_requirements=fact_requirements,
                )
            )
            steps.append(
                ResearchStep(
                    step_id=f"discovery-{target_index:02d}-{dimension_offset + 1:02d}",
                    objective=f"Find {target.label}: {dimension.label}"[:500],
                    query=f"{target.label} {dimension.label}"[:2000],
                    corpus_id=target.corpus_id,
                    target_ids=(target.target_id,),
                    dimension_ids=(dimension.dimension_id,),
                )
            )
    return ResearchPlan(
        task_type="comparison",
        targets=targets,
        dimensions=dimensions,
        requirements=tuple(requirements),
        steps=tuple(steps),
    )


class LangChainComparisonTargetResolver:
    """Resolve an exact or paper-level-retrieved local candidate set."""

    def __init__(
        self,
        *,
        candidate_retriever: AsyncPaperCandidateRetriever,
        query_resolver: ComparisonQueryResolver,
        corpus_catalog: Mapping[str, str],
        candidate_limit: int = 8,
    ) -> None:
        if candidate_limit < 2 or candidate_limit > 80:
            raise ValueError("comparison candidate limit must be between 2 and 80")
        self._candidate_retriever = candidate_retriever
        self._query_resolver = query_resolver
        self._corpus_catalog = dict(sorted(corpus_catalog.items()))
        self._candidate_limit = candidate_limit

    async def resolve(self, question: str) -> Mapping[str, str]:
        explicit_ids = parse_explicit_corpus_ids(question)
        unknown_ids = [item for item in explicit_ids if item not in self._corpus_catalog]
        if unknown_ids:
            raise ComparisonTargetResolutionError("unknown_explicit_corpus_id")
        if len(explicit_ids) > 4:
            raise ComparisonTargetResolutionError("too_many_explicit_corpus_ids")
        if len(explicit_ids) >= 2:
            return {item: self._corpus_catalog[item] for item in explicit_ids}

        rewrite = await self._query_resolver.resolve_query(question)
        hits = await self._candidate_retriever.search(
            PaperCandidateQuery(
                original_query=question,
                english_query=rewrite.english_query,
            ),
            top_k=self._candidate_limit,
        )
        candidate_ids = list(explicit_ids)
        for hit in hits:
            if hit.corpus_id not in self._corpus_catalog:
                raise ComparisonTargetResolutionError("candidate_outside_catalog")
            if hit.corpus_id not in candidate_ids:
                candidate_ids.append(hit.corpus_id)
            if len(candidate_ids) >= self._candidate_limit:
                break
        if len(candidate_ids) < 2:
            raise ComparisonTargetResolutionError("insufficient_retrieval_candidates")
        return {
            corpus_id: self._corpus_catalog[corpus_id]
            for corpus_id in candidate_ids[: self._candidate_limit]
        }


class LangChainResearchPlanner:
    """Ask a chat model only for ordered corpus-search subquestions."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        corpus_catalog: Mapping[str, str] | None = None,
        target_resolver: ComparisonTargetResolver | None = None,
    ):
        self._model = model
        self._structured_model = model.with_structured_output(
            ResearchPlan,
            method="function_calling",
        )
        self._comparison_model: Any | None = None
        self._corpus_catalog = dict(sorted((corpus_catalog or {}).items()))
        self._target_resolver = target_resolver

    async def plan(
        self,
        question: str,
        *,
        max_steps: int,
        planning_required: bool = False,
    ) -> ResearchPlan:
        if max_steps <= 0 or max_steps > MAX_INITIAL_PLAN_STEPS:
            raise ValueError(
                f"max_steps must be between 1 and {MAX_INITIAL_PLAN_STEPS}"
            )
        resolved_catalog: Mapping[str, str] = self._corpus_catalog
        if planning_required and self._target_resolver is not None:
            resolved_catalog = await self._target_resolver.resolve(question)
        catalog = json.dumps(
            dict(sorted(resolved_catalog.items())),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        anchor_source = question_anchor_catalog(question)
        anchor_catalog = json.dumps(
            [
                {"anchor_id": index, "text": text}
                for index, text in enumerate(anchor_source)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        dimension_hints = comparison_dimension_hints(question)
        dimension_hint_catalog = json.dumps(
            [
                {"dimension_index": index, "text": text}
                for index, text in enumerate(dimension_hints)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system = SystemMessage(
            content=(
                "你是论文研究任务规划器。只拆分需要在现有本地论文库中检索的子问题，"
                f"最多 {max_steps} 步。每一步必须给出稳定 step_id、简短目标、可独立检索的"
                "查询和 1 到 20 的 top_k。若问题要求比较两个或更多论文、方法、模型或实验，"
                "必须使用 task_type=comparison：明确列出比较对象 targets 和比较维度 dimensions，"
                "为每个比较对象×维度建立完整证据网格 requirements，并生成逐对象、可独立命中的"
                "检索步骤；禁止用一个宽泛查询代替所有对象的检索。每个 comparison step 必须填写"
                "对应 target_ids 和 dimension_ids；每一步只能对应一个 target 和一个 dimension，"
                "每个 requirement 单元都必须有自己的独立查询，不能合并多个维度。"
                "comparison 的 steps 数必须等于 targets 数×dimensions 数，且不得超过步数预算；"
                "因此维度数不得超过 floor(步数预算/targets 数)。只保留用户问题明确要求且预算"
                "允许的最高优先级维度，不得生成无法放入完整网格的额外维度。"
                "若比较对象能对应下方本地论文目录，target 和其逐对象 step 都必须填写相同的 "
                "corpus_id，使 BM25 与向量召回在该论文范围内执行；不得把一个论文范围用于另一个对象。"
                "非比较问题使用 task_type=direct，且不填写比较元数据。不要回答问题，不要请求"
                "网络、代码执行、文件写入或数据库修改。"
                "The step budget accommodates the full supported comparison schema. Preserve "
                "all comparison dimensions explicitly required by the user; do not compress, "
                "merge, or discard them to save steps. For every comparison requirement, derive "
                "one to six atomic fact_requirements only from the user's question and that "
                "dimension. Give each a globally unique fact_requirement_id and a precise "
                "description. Split materially distinct needs such as mechanism, input or "
                "dependency, dataset, metric, numeric result, baseline, time, limitation, and "
                "applicability condition when the question asks for them. Include "
                "required_qualifier_kinds when omission would change the meaning. Do not use "
                "private references, gold answers, or facts not requested by the question. Keep "
                "one bounded discovery step per target-by-dimension cell. For every planned "
                "fact_requirement, emit protected_anchor_ids and retrieval_expansions, but do not "
                "emit protected anchor text or a final search query. For protected_anchor_ids, "
                "select the shortest available catalog spans that together identify exactly that "
                "fact; use multiple IDs when a relation or qualifier would otherwise be lost. "
                "Preserve requested entities, numbers, datasets, metrics, conditions, comparison "
                "directions, causal relations, failures, limitations, mitigations, reasoning, "
                "understanding, and explicit qualifiers. retrieval_expansions may add the paper "
                "name, English translations, academic synonyms, method names, dataset names, and "
                "related retrieval terms. Use an augment strategy: selected catalog text remains intact "
                "and expansions are only appended. Do not replace a protected anchor with a broader "
                "concept, do not combine sibling fact requirements, do not answer the question, "
                "and do not add facts absent from the user's request. Across the full plan, the "
                f"number of fact_requirements must not exceed {max_steps}. The cell-level discovery "
                "query and objective may cover all fact requirements in that cell, but the initial "
                "target-by-dimension grid must not expand. "
                f"\nVERBATIM_ANCHOR_SOURCE_JSON={anchor_catalog}"
                f"\nLOCAL_CORPUS_CATALOG_JSON={catalog}"
                + (
                    "上游语义路由已确认本题必须进行多对象研究规划，因此必须返回 "
                    "task_type=comparison，不能降为 direct。"
                    if planning_required
                    else ""
                )
            )
        )
        if planning_required:
            system = SystemMessage(
                content=(
                    "You propose only semantic content for a private-paper comparison plan. "
                    f"Select two to four corpus IDs only from LOCAL_CORPUS_CATALOG_JSON and "
                    f"use no more than {max_steps} total fact proposals. Return a distinct "
                    "dimension label for every explicit comparison topic, clue cluster, method, "
                    "result, limitation, condition, or manipulation requested by the user; do "
                    "not merge separate requested topics. Return at least one atomic fact for "
                    f"exactly {len(dimension_hints)} dimensions, preserving the one-to-one order "
                    "of EXPLICIT_DIMENSION_HINTS_JSON. Do not merge, drop, or add hint entries. "
                    "Return one or more atomic fact intents for every zero-based dimension_index. "
                    "Do not repeat facts per corpus: local code applies each dimension's intents "
                    "symmetrically to every selected corpus. Each fact description must "
                    "come only from the user's question. Select non-empty protected_anchor_ids "
                    "from VERBATIM_ANCHOR_SOURCE_JSON and put translations or academic synonyms "
                    "only in retrieval_expansions. Include required_qualifier_kinds when omission "
                    "would change meaning. Do not generate target IDs, dimension IDs, requirement "
                    "IDs, fact IDs, step IDs, corpus scopes, queries, objectives, or a ResearchPlan; "
                    "local deterministic code creates them. Do not answer the question and do not "
                    "use private references or gold answers."
                    f"\nVERBATIM_ANCHOR_SOURCE_JSON={anchor_catalog}"
                    f"\nEXPLICIT_DIMENSION_HINTS_JSON={dimension_hint_catalog}"
                    f"\nLOCAL_CORPUS_CATALOG_JSON={catalog}"
                )
            )
        messages = [system, HumanMessage(content=question)]
        failure_reason = "planner_schema_invalid"
        attempt_audits: list[PlannerAttemptAudit] = []
        structured_model = self._structured_model
        if planning_required:
            if self._comparison_model is None:
                self._comparison_model = self._model.with_structured_output(
                    ComparisonPlanProposal,
                    method="function_calling",
                )
            structured_model = self._comparison_model
        for attempt in range(2):
            try:
                raw = await structured_model.ainvoke(messages)
                if planning_required:
                    proposal = ComparisonPlanProposal.model_validate(raw)
                    plan = build_comparison_research_plan(
                        proposal,
                        resolved_catalog=resolved_catalog,
                        max_steps=max_steps,
                        expected_dimension_count=len(dimension_hints),
                    )
                else:
                    plan = ResearchPlan.model_validate(raw)
                if len(plan.steps) > max_steps:
                    raise ValueError("research plan exceeds the requested step budget")
                if planning_required and plan.task_type != "comparison":
                    raise ValueError("planned research requires a comparison plan")
                if plan.task_type == "comparison" and any(
                    fact_requirement.origin != "planned"
                    for requirement in plan.requirements
                    for fact_requirement in requirement.fact_requirements
                ):
                    raise ValueError(
                        "comparison plan requires explicit atomic fact requirements"
                    )
                if plan.task_type == "comparison":
                    plan = bind_question_anchors(plan, anchor_source)
                    planned_facts = tuple(
                        fact_requirement
                        for requirement in plan.requirements
                        for fact_requirement in requirement.fact_requirements
                    )
                    if len(planned_facts) > max_steps:
                        raise ValueError(
                            "comparison fact requirements exceed the requested step budget"
                        )
                    for fact_requirement in planned_facts:
                        validate_question_anchors(question, fact_requirement)
                if planning_required and self._target_resolver is not None:
                    planned_corpora = {target.corpus_id for target in plan.targets}
                    if not planned_corpora <= set(resolved_catalog):
                        raise ValueError("comparison plan left the resolved candidate set")
                attempt_audits.append(
                    PlannerAttemptAudit(attempt=attempt + 1, outcome="validated")
                )
                return plan.model_copy(
                    update={"planner_attempts": tuple(attempt_audits)}
                )
            except ValueError as exc:
                failure_reason = _planner_failure_code(exc)
                attempt_audits.append(
                    PlannerAttemptAudit(
                        attempt=attempt + 1,
                        outcome=(
                            "schema_invalid"
                            if failure_reason == "planner_schema_invalid"
                            else "contract_invalid"
                        ),
                        failure_code=failure_reason,
                    )
                )
                if attempt == 0:
                    messages = [
                        *messages,
                        HumanMessage(
                            content=(
                                f"FAILURE_CODE={failure_reason}. "
                                f"{_PLANNER_REPAIR_INSTRUCTIONS[failure_reason]} "
                                "Return the full corrected plan. Do not include model output from "
                                "the previous attempt."
                            )
                        ),
                    ]

        if planning_required:
            raise ComparisonTargetResolutionError(
                failure_reason,
                attempts=tuple(attempt_audits),
            )
        return ResearchPlan(
            steps=(
                ResearchStep(
                    step_id="fallback",
                    objective="Retrieve evidence for the requested research question",
                    query=question,
                    top_k=10,
                ),
            )
        )
