"""LangChain structured-output adapter for bounded research planning."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from paper_research_agent.agent.fact_queries import (
    bind_question_anchors,
    question_anchor_catalog,
    validate_question_anchors,
)
from paper_research_agent.agent.models import ResearchPlan, ResearchStep
from paper_research_agent.agent.policy import MAX_INITIAL_PLAN_STEPS
from paper_research_agent.retrieval.contracts import QueryRewriteTrace
from paper_research_agent.retrieval.papers import (
    AsyncPaperCandidateRetriever,
    PaperCandidateQuery,
)


class ComparisonTargetResolutionError(RuntimeError):
    """Comparison targets could not be resolved without leaving local scope."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"comparison target resolution failed: {reason_code}")


class ComparisonTargetResolver(Protocol):
    async def resolve(self, question: str) -> Mapping[str, str]: ...


class ComparisonQueryResolver(Protocol):
    async def resolve_query(self, question: str) -> QueryRewriteTrace: ...


def _planner_contract_reason(error: ValueError) -> str:
    message = str(error)
    if "protected anchor" in message or "protected_anchor_ids" in message:
        return "planner_anchor_selection_invalid"
    if "fact requirements exceed" in message:
        return "planner_fact_budget_invalid"
    return "planner_contract_invalid"


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
        self._structured_model = model.with_structured_output(
            ResearchPlan,
            method="function_calling",
        )
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
        messages = [system, HumanMessage(content=question)]
        failure_reason = "planner_contract_invalid"
        for attempt in range(2):
            try:
                raw = await self._structured_model.ainvoke(messages)
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
                return plan
            except ValueError as exc:
                failure_reason = _planner_contract_reason(exc)
                if attempt == 0:
                    messages = [
                        *messages,
                        HumanMessage(
                            content=(
                                "The previous plan violated the structured planning contract. "
                                "Return a valid full target-by-dimension comparison grid within "
                                f"the {max_steps}-step budget. The number of steps must equal "
                                "the target count multiplied by the dimension count. Preserve "
                                "every comparison dimension required by the question; never "
                                "return a partial grid. Every requirement must include one to six "
                                "explicit, question-derived atomic fact_requirements with globally "
                                "unique IDs and non-empty protected_anchor_ids selected from "
                                "VERBATIM_ANCHOR_SOURCE_JSON. Do not emit anchor text. Put "
                                "translations and academic synonyms only in "
                                "retrieval_expansions. Do not replace anchors with broader concepts, "
                                "rely on a derived compatibility intent, or copy sibling facts. The "
                                f"total fact_requirement count must not exceed {max_steps}."
                            )
                        ),
                    ]

        if planning_required:
            raise ComparisonTargetResolutionError(failure_reason)
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
