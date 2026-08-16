"""Bounded ReAct orchestration over the private, read-only research tools."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from paper_research_agent.agent.coverage import validate_evidence_assessment
from paper_research_agent.agent.langchain_tools import build_langchain_tools
from paper_research_agent.agent.models import (
    TERMINATION_REASONS,
    EvidenceAssessment,
    EvidenceFollowup,
    GetEvidenceResult,
    ResearchActionRecord,
    ResearchObservation,
    ResearchPlan,
    ResearchStep,
    SearchCorpusResult,
)
from paper_research_agent.agent.observability import (
    AgentEvent,
    AgentEventSink,
    emit_agent_event,
    safe_fingerprint,
)
from paper_research_agent.agent.policy import (
    ABSOLUTE_MAX_RESEARCH_STEPS,
    MAX_INITIAL_PLAN_STEPS,
    ResearchRuntimePolicy,
    ResearchToolName,
)
from paper_research_agent.agent.service import ResearchToolService

if TYPE_CHECKING:
    from paper_research_agent.agent.tooling.service import ExtendedResearchToolkit


class ResearchPlanner(Protocol):
    async def plan(
        self,
        question: str,
        *,
        max_steps: int,
        planning_required: bool = False,
    ) -> ResearchPlan: ...


class ResearchReasoner(Protocol):
    async def assess(
        self,
        question: str,
        *,
        plan: ResearchPlan,
        observations: tuple[ResearchObservation, ...],
        remaining_steps: int,
    ) -> EvidenceAssessment: ...


class ResearchGraphInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1, max_length=2000)
    planning_required: bool = False

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("research question must not be blank")
        return normalized


class ResearchGraphState(TypedDict, total=False):
    run_id: str
    question: str
    planning_required: bool
    plan: dict[str, Any]
    current_step: int
    step_budget: int
    active_step: dict[str, Any] | None
    tool_call_count: int
    tool_call_budget: int
    observations: list[dict[str, Any]]
    evidence_records: list[dict[str, Any]]
    assessments: list[dict[str, Any]]
    assessment_observation_counts: list[int]
    assessment_durations_ms: list[float]
    pending_followups: list[dict[str, Any]]
    action_history: list[dict[str, Any]]
    replan_count: int
    consecutive_no_new_evidence: int
    next_action: str
    evidence_sufficient: bool
    termination_reason: str | None
    started_at_epoch_seconds: float


class _StepExecutionResult(BaseModel):
    """One independently executed comparison step, returned in plan order."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    step: ResearchStep
    search: SearchCorpusResult
    evidence: GetEvidenceResult


def build_research_graph(
    *,
    planner: ResearchPlanner,
    reasoner: ResearchReasoner,
    service: ResearchToolService,
    max_steps: int = ABSOLUTE_MAX_RESEARCH_STEPS,
    evidence_per_step: int = 4,
    policy: ResearchRuntimePolicy | None = None,
    checkpointer: Any | None = None,
    event_sink: AgentEventSink | None = None,
    extended_toolkit: ExtendedResearchToolkit | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile a fail-closed reason-act-observe-assess loop over two local tools."""
    if max_steps <= 0 or max_steps > ABSOLUTE_MAX_RESEARCH_STEPS:
        raise ValueError(
            f"max_steps must be between 1 and {ABSOLUTE_MAX_RESEARCH_STEPS}"
        )
    if evidence_per_step <= 0 or evidence_per_step > 20:
        raise ValueError("evidence_per_step must be between 1 and 20")
    runtime_policy = policy or ResearchRuntimePolicy(
        max_steps=max_steps,
        evidence_per_step=evidence_per_step,
    )
    initial_plan_steps = max(
        1,
        min(
            MAX_INITIAL_PLAN_STEPS,
            runtime_policy.max_steps,
            runtime_policy.max_tool_calls // 2,
        ),
    )
    evidence_per_step = runtime_policy.evidence_per_step
    tools = {tool.name: tool for tool in build_langchain_tools(service, extended_toolkit)}
    search_tool = tools["search_corpus"]
    evidence_tool = tools["get_evidence"]

    async def create_plan(state: ResearchGraphState) -> ResearchGraphState:
        started_at_epoch_seconds = state.get("started_at_epoch_seconds")
        if not isinstance(started_at_epoch_seconds, (int, float)) or isinstance(
            started_at_epoch_seconds, bool
        ):
            started_at_epoch_seconds = time.time()
        graph_input = ResearchGraphInput(
            question=state["question"],
            planning_required=state.get("planning_required", False),
        )
        plan = await planner.plan(
            graph_input.question,
            max_steps=initial_plan_steps,
            planning_required=graph_input.planning_required,
        )
        if len(plan.steps) > initial_plan_steps:
            raise ValueError("research plan exceeds the graph step budget")
        if plan.task_type == "comparison":
            plan = plan.model_copy(
                update={
                    "steps": tuple(
                        step.model_copy(update={"top_k": 20}) for step in plan.steps
                    )
                }
            )
        step_budget, tool_call_budget = runtime_policy.freeze_invocation_budget(
            len(plan.steps)
        )
        return {
            "run_id": state.get("run_id", ""),
            "question": graph_input.question,
            "planning_required": graph_input.planning_required,
            "plan": plan.model_dump(mode="json"),
            "current_step": 0,
            "step_budget": step_budget,
            "active_step": None,
            "tool_call_count": 0,
            "tool_call_budget": tool_call_budget,
            "observations": [],
            "evidence_records": [],
            "assessments": [],
            "assessment_observation_counts": [],
            "assessment_durations_ms": [],
            "pending_followups": [],
            "action_history": [],
            "replan_count": 0,
            "consecutive_no_new_evidence": 0,
            "next_action": "reason",
            "evidence_sufficient": False,
            "termination_reason": None,
            "started_at_epoch_seconds": float(started_at_epoch_seconds),
        }

    async def reason(state: ResearchGraphState) -> ResearchGraphState:
        plan = ResearchPlan.model_validate(state["plan"])
        is_comparison = plan.task_type == "comparison"
        current_step = state["current_step"]
        observations = tuple(
            ResearchObservation.model_validate(value) for value in state["observations"]
        )

        def finish_or_assess(
            termination_reason: str,
            *,
            pending_followups: list[EvidenceFollowup] | None = None,
        ) -> ResearchGraphState:
            if _has_unassessed_comparison_evidence(state, plan, observations):
                update: ResearchGraphState = {
                    "active_step": None,
                    "next_action": "assess_evidence",
                    "termination_reason": None,
                }
                if pending_followups is not None:
                    update["pending_followups"] = [
                        item.model_dump(mode="json") for item in pending_followups
                    ]
                return update
            return _finish_update(termination_reason)

        if not observations:
            return {
                "active_step": plan.steps[0].model_dump(mode="json"),
                "next_action": "execute_tools",
                "termination_reason": None,
            }

        if (
            is_comparison
            and not state["assessments"]
            and current_step < len(plan.steps)
        ):
            initial_planned_next = plan.steps[current_step]
            executed_queries = {
                _query_scope_key(item.search.query, item.search.corpus_id)
                for item in observations
            }
            if _step_query_key(initial_planned_next) in executed_queries:
                return finish_or_assess("repeated_query")
            return {
                "active_step": initial_planned_next.model_dump(mode="json"),
                "next_action": "execute_tools",
                "termination_reason": None,
            }

        assessments = tuple(
            EvidenceAssessment.model_validate(value) for value in state["assessments"]
        )
        if not is_comparison and len(assessments) != len(observations):
            raise ValueError("research assessments do not match observations")
        if is_comparison and (
            not assessments or len(assessments) > len(observations)
        ):
            raise ValueError("comparison assessments do not match observation batches")
        latest = assessments[-1]
        if latest.evidence_sufficient:
            return finish_or_assess("evidence_sufficient")
        if latest.status == "compiler_failed":
            return finish_or_assess("compiler_failed")
        if _remaining_runtime_seconds(
            state, runtime_policy
        ) < _supplemental_completion_reserve_seconds(state, runtime_policy):
            return finish_or_assess("time_budget")
        if state["consecutive_no_new_evidence"] >= 2:
            return finish_or_assess("no_new_evidence")
        if current_step >= _state_step_budget(state, runtime_policy):
            return finish_or_assess("step_budget")
        if _state_tool_call_budget(state, runtime_policy) - state["tool_call_count"] < 2:
            return finish_or_assess("tool_budget")

        executed_queries = {
            _query_scope_key(item.search.query, item.search.corpus_id)
            for item in observations
        }
        pending_followups = [
            EvidenceFollowup.model_validate(item)
            for item in state.get("pending_followups", [])
        ]
        if is_comparison and pending_followups:
            requirement_by_id = {
                item.requirement_id: item for item in plan.requirements
            }
            while pending_followups:
                followup = pending_followups.pop(0)
                requested = requirement_by_id[followup.requirement_id]
                corpus_id = _comparison_corpus_id(plan, (requested.target_id,))
                if _query_scope_key(followup.query, corpus_id) in executed_queries:
                    continue
                replan_count = state["replan_count"] + 1
                replacement = ResearchStep(
                    step_id=f"replan-{replan_count}",
                    objective=followup.objective,
                    query=followup.query,
                    top_k=min(20, max(10, evidence_per_step)),
                    corpus_id=corpus_id,
                    target_ids=(requested.target_id,),
                    dimension_ids=(requested.dimension_id,),
                )
                updated_plan = plan.model_copy(
                    update={"steps": (*plan.steps, replacement)}
                )
                action_history = _append_action(
                    state,
                    action="replan",
                    step_id=replacement.step_id,
                    query=replacement.query,
                )
                return {
                    "plan": updated_plan.model_dump(mode="json"),
                    "active_step": replacement.model_dump(mode="json"),
                    "pending_followups": [
                        item.model_dump(mode="json") for item in pending_followups
                    ],
                    "action_history": action_history,
                    "replan_count": replan_count,
                    "next_action": "execute_tools",
                    "termination_reason": None,
                }
            return finish_or_assess(
                "repeated_query", pending_followups=pending_followups
            )
        planned_next = plan.steps[current_step] if current_step < len(plan.steps) else None
        if is_comparison and planned_next is not None:
            if _step_query_key(planned_next) in executed_queries:
                return finish_or_assess("repeated_query")
            return {
                "active_step": planned_next.model_dump(mode="json"),
                "next_action": "execute_tools",
                "termination_reason": None,
            }
        if latest.next_query is not None:
            replan_target_ids: tuple[str, ...] = ()
            replan_dimension_ids: tuple[str, ...] = ()
            if is_comparison:
                if len(latest.next_requirement_ids) != 1:
                    raise ValueError(
                        "comparison follow-up requires exactly one requirement cell"
                    )
                requirement_by_id = {
                    item.requirement_id: item for item in plan.requirements
                }
                requested = requirement_by_id[latest.next_requirement_ids[0]]
                replan_target_ids = (requested.target_id,)
                replan_dimension_ids = (requested.dimension_id,)
            replan_corpus_id = _comparison_corpus_id(plan, replan_target_ids)
            next_query_key = _query_scope_key(latest.next_query, replan_corpus_id)
            if next_query_key in executed_queries:
                if planned_next is None or _step_query_key(planned_next) in executed_queries:
                    return finish_or_assess("repeated_query")
            else:
                replan_count = state["replan_count"] + 1
                replacement = ResearchStep(
                    step_id=f"replan-{replan_count}",
                    objective=latest.next_objective or "Refine the evidence search",
                    query=latest.next_query,
                    top_k=(
                        planned_next.top_k
                        if planned_next is not None
                        else min(20, max(10, evidence_per_step))
                    ),
                    corpus_id=replan_corpus_id,
                    target_ids=replan_target_ids,
                    dimension_ids=replan_dimension_ids,
                )
                steps = list(plan.steps)
                if current_step < len(steps):
                    steps[current_step] = replacement
                else:
                    steps.append(replacement)
                updated_plan = ResearchPlan.model_validate(
                    {
                        **plan.model_dump(mode="json"),
                        "steps": [item.model_dump(mode="json") for item in steps],
                    }
                )
                action_history = _append_action(
                    state,
                    action="replan",
                    step_id=replacement.step_id,
                    query=replacement.query,
                )
                return {
                    "plan": updated_plan.model_dump(mode="json"),
                    "active_step": replacement.model_dump(mode="json"),
                    "action_history": action_history,
                    "replan_count": replan_count,
                    "next_action": "execute_tools",
                    "termination_reason": None,
                }

        if planned_next is None:
            return finish_or_assess("plan_exhausted")
        if _step_query_key(planned_next) in executed_queries:
            return finish_or_assess("repeated_query")
        return {
            "active_step": planned_next.model_dump(mode="json"),
            "next_action": "execute_tools",
            "termination_reason": None,
        }

    async def run_search(
        state: ResearchGraphState,
        step: ResearchStep,
        *,
        tool_call_count: int,
    ) -> SearchCorpusResult:
        search_started = time.perf_counter()
        _emit_graph_event(
            state,
            event_sink,
            event_type="tool_started",
            status="started",
            component="tool",
            name="search_corpus",
            step_id=step.step_id,
            query=step.query,
            requested_count=step.top_k,
            tool_call_count=tool_call_count,
        )
        try:
            raw_search = await search_tool.ainvoke(
                {
                    "query": step.query,
                    "top_k": step.top_k,
                    "corpus_id": step.corpus_id,
                }
            )
            search = SearchCorpusResult.model_validate(raw_search)
            if search.query != step.query:
                raise ValueError("search tool returned a mismatched query")
            if search.corpus_id != step.corpus_id:
                raise ValueError("search tool returned a mismatched corpus scope")
        except Exception as exc:
            _emit_graph_event(
                state,
                event_sink,
                event_type="tool_failed",
                status="failed",
                component="tool",
                name="search_corpus",
                duration_ms=_elapsed_ms(search_started),
                step_id=step.step_id,
                query=step.query,
                error_type=type(exc).__name__,
                reason_code="tool_execution_failed",
                requested_count=step.top_k,
                tool_call_count=tool_call_count,
            )
            raise
        _emit_graph_event(
            state,
            event_sink,
            event_type="tool_completed",
            status="succeeded",
            component="tool",
            name="search_corpus",
            duration_ms=_elapsed_ms(search_started),
            step_id=step.step_id,
            query=step.query,
            degraded=search.degraded,
            hit_count=len(search.hits),
            requested_count=step.top_k,
            returned_count=len(search.hits),
            tool_call_count=tool_call_count,
        )
        return search

    async def run_evidence(
        state: ResearchGraphState,
        step: ResearchStep,
        search: SearchCorpusResult,
        *,
        tool_call_count: int,
    ) -> GetEvidenceResult:
        selected_ids = tuple(hit.chunk_id for hit in search.hits[:evidence_per_step])
        if not selected_ids:
            return GetEvidenceResult(records=())
        evidence_started = time.perf_counter()
        _emit_graph_event(
            state,
            event_sink,
            event_type="tool_started",
            status="started",
            component="tool",
            name="get_evidence",
            step_id=step.step_id,
            requested_count=len(selected_ids),
            tool_call_count=tool_call_count,
        )
        try:
            raw_evidence = await evidence_tool.ainvoke({"chunk_ids": selected_ids})
            evidence = GetEvidenceResult.model_validate(raw_evidence)
            returned_ids = tuple(record.chunk_id for record in evidence.records)
            resolved_ids = set(returned_ids) | set(evidence.missing_chunk_ids)
            if resolved_ids != set(selected_ids):
                raise ValueError("evidence tool result does not match requested chunk IDs")
        except Exception as exc:
            _emit_graph_event(
                state,
                event_sink,
                event_type="tool_failed",
                status="failed",
                component="tool",
                name="get_evidence",
                duration_ms=_elapsed_ms(evidence_started),
                step_id=step.step_id,
                error_type=type(exc).__name__,
                reason_code="tool_execution_failed",
                requested_count=len(selected_ids),
                tool_call_count=tool_call_count,
            )
            raise
        _emit_graph_event(
            state,
            event_sink,
            event_type="tool_completed",
            status="succeeded",
            component="tool",
            name="get_evidence",
            duration_ms=_elapsed_ms(evidence_started),
            step_id=step.step_id,
            requested_count=len(selected_ids),
            returned_count=len(evidence.records),
            tool_call_count=tool_call_count,
        )
        return evidence

    async def execute_tools(state: ResearchGraphState) -> ResearchGraphState:
        plan = ResearchPlan.model_validate(state["plan"])
        first_step = ResearchStep.model_validate(state.get("active_step"))
        current_step = state["current_step"]
        batch_size = 1
        if plan.task_type == "comparison" and not state["assessments"]:
            batch_size = min(
                runtime_policy.comparison_search_concurrency,
                len(plan.steps) - current_step,
            )
        steps = plan.steps[current_step : current_step + batch_size]
        if not steps or steps[0].step_id != first_step.step_id:
            raise ValueError("active research step does not match the planned batch")
        executed_keys = {
            _query_scope_key(item["search"]["query"], item["search"].get("corpus_id"))
            for item in state["observations"]
        }
        batch_keys = tuple(_step_query_key(step) for step in steps)
        if len(set(batch_keys)) != len(batch_keys) or any(
            key in executed_keys for key in batch_keys
        ):
            raise ValueError("comparison search batch contains a repeated scoped query")

        tool_call_count = state["tool_call_count"]
        search_call_counts: list[int] = []
        for _step in steps:
            tool_call_count = _consume_tool(
                state,
                event_sink,
                runtime_policy,
                "search_corpus",
                tool_call_count,
            )
            search_call_counts.append(tool_call_count)
        searches = await asyncio.gather(
            *(
                run_search(state, step, tool_call_count=call_count)
                for step, call_count in zip(steps, search_call_counts, strict=True)
            )
        )

        evidence_call_counts: list[int | None] = []
        for search in searches:
            if search.hits:
                tool_call_count = _consume_tool(
                    state,
                    event_sink,
                    runtime_policy,
                    "get_evidence",
                    tool_call_count,
                )
                evidence_call_counts.append(tool_call_count)
            else:
                evidence_call_counts.append(None)
        evidence_results = await asyncio.gather(
            *(
                run_evidence(
                    state,
                    step,
                    search,
                    tool_call_count=call_count or tool_call_count,
                )
                for step, search, call_count in zip(
                    steps, searches, evidence_call_counts, strict=True
                )
            )
        )

        batch_results = tuple(
            _StepExecutionResult(step=step, search=search, evidence=evidence)
            for step, search, evidence in zip(
                steps, searches, evidence_results, strict=True
            )
        )
        action_history = list(state["action_history"])
        for result in batch_results:
            action_history = _append_action_from_history(
                action_history,
                action="search_corpus",
                step_id=result.step.step_id,
                query=result.step.query,
            )
            selected_ids = tuple(
                hit.chunk_id for hit in result.search.hits[:evidence_per_step]
            )
            if selected_ids:
                action_history = _append_action_from_history(
                    action_history,
                    action="get_evidence",
                    step_id=result.step.step_id,
                    chunk_ids=selected_ids,
                )

        merged = list(state["evidence_records"])
        seen = {str(record["chunk_id"]) for record in merged}
        previous_evidence_count = len(seen)
        for result in batch_results:
            for record in result.evidence.records:
                if record.chunk_id not in seen:
                    merged.append(record.model_dump(mode="json"))
                    seen.add(record.chunk_id)
        consecutive_no_new_evidence = (
            0
            if len(seen) > previous_evidence_count
            else state["consecutive_no_new_evidence"] + 1
        )
        return {
            "current_step": current_step + len(batch_results),
            "active_step": None,
            "tool_call_count": tool_call_count,
            "observations": [
                *state["observations"],
                *(
                    ResearchObservation(
                        step_id=result.step.step_id,
                        objective=result.step.objective,
                        search=result.search,
                        evidence=result.evidence,
                    ).model_dump(mode="json")
                    for result in batch_results
                ),
            ],
            "evidence_records": merged,
            "consecutive_no_new_evidence": consecutive_no_new_evidence,
            "action_history": action_history,
            "next_action": "assess_evidence",
        }

    async def assess_evidence(state: ResearchGraphState) -> ResearchGraphState:
        plan = ResearchPlan.model_validate(state["plan"])
        observations = tuple(
            ResearchObservation.model_validate(value) for value in state["observations"]
        )
        assessment_started = time.perf_counter()
        assessment = await reasoner.assess(
            state["question"],
            plan=plan,
            observations=observations,
            remaining_steps=min(
                max(0, _state_step_budget(state, runtime_policy) - state["current_step"]),
                max(
                    0,
                    (
                        _state_tool_call_budget(state, runtime_policy)
                        - state["tool_call_count"]
                    )
                    // 2,
                ),
            ),
        )
        assessment_duration_ms = _elapsed_ms(assessment_started)
        assessment = validate_evidence_assessment(plan, observations, assessment)
        followups = assessment.followups
        if (
            plan.task_type == "comparison"
            and not followups
            and assessment.next_query is not None
        ):
            followups = (
                EvidenceFollowup(
                    requirement_id=assessment.next_requirement_ids[0],
                    query=assessment.next_query,
                    objective=assessment.next_objective or "Refine the evidence search",
                ),
            )
        action_history = _append_action(
            state,
            action="assess_evidence",
            step_id=observations[-1].step_id,
            outcome=assessment.status,
        )
        return {
            "assessments": [
                *state["assessments"],
                assessment.model_dump(mode="json"),
            ],
            "assessment_observation_counts": [
                *state.get("assessment_observation_counts", []),
                len(observations),
            ],
            "assessment_durations_ms": [
                *state.get("assessment_durations_ms", []),
                assessment_duration_ms,
            ],
            "pending_followups": [
                item.model_dump(mode="json") for item in followups
            ],
            "action_history": action_history,
            "evidence_sufficient": assessment.evidence_sufficient,
            "next_action": "reason",
        }

    async def finalize(state: ResearchGraphState) -> ResearchGraphState:
        termination_reason = state.get("termination_reason")
        if not isinstance(termination_reason, str) or termination_reason not in TERMINATION_REASONS:
            raise ValueError("research graph cannot finalize without a termination reason")
        return {
            "active_step": None,
            "action_history": _append_action(
                state,
                action="finish",
                outcome=termination_reason,
            ),
            "next_action": "finish",
        }

    def route_after_reason(state: ResearchGraphState) -> str:
        return state["next_action"]

    def route_after_execute_tools(state: ResearchGraphState) -> str:
        plan = ResearchPlan.model_validate(state["plan"])
        if (
            plan.task_type == "comparison"
            and (
                bool(state.get("pending_followups"))
                or (
                    not state["assessments"]
                    and state["current_step"] < len(plan.steps)
                )
            )
        ):
            return "reason"
        return "assess_evidence"

    builder = StateGraph(ResearchGraphState)
    builder.add_node(
        "plan", cast(Any, _instrument_node("plan", create_plan, event_sink))
    )
    builder.add_node(
        "reason", cast(Any, _instrument_node("reason", reason, event_sink))
    )
    builder.add_node(
        "execute_tools",
        cast(Any, _instrument_node("execute_tools", execute_tools, event_sink)),
    )
    builder.add_node(
        "assess_evidence",
        cast(Any, _instrument_node("assess_evidence", assess_evidence, event_sink)),
    )
    builder.add_node(
        "finalize", cast(Any, _instrument_node("finalize", finalize, event_sink))
    )
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "reason")
    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {
            "execute_tools": "execute_tools",
            "assess_evidence": "assess_evidence",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "execute_tools",
        route_after_execute_tools,
        {"reason": "reason", "assess_evidence": "assess_evidence"},
    )
    builder.add_edge("assess_evidence", "reason")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="paper_research_react_v2")


ResearchNode = Callable[[ResearchGraphState], Awaitable[ResearchGraphState]]


def _state_step_budget(
    state: ResearchGraphState,
    policy: ResearchRuntimePolicy,
) -> int:
    value = state.get("step_budget", policy.max_steps)
    return value if isinstance(value, int) and value > 0 else policy.max_steps


def _state_tool_call_budget(
    state: ResearchGraphState,
    policy: ResearchRuntimePolicy,
) -> int:
    value = state.get("tool_call_budget", policy.max_tool_calls)
    return value if isinstance(value, int) and value > 0 else policy.max_tool_calls


def _remaining_runtime_seconds(
    state: ResearchGraphState,
    policy: ResearchRuntimePolicy,
) -> float:
    started = state.get("started_at_epoch_seconds")
    if not isinstance(started, (int, float)) or isinstance(started, bool):
        return policy.timeout_seconds
    return max(0.0, policy.timeout_seconds - (time.time() - float(started)))


def _supplemental_completion_reserve_seconds(
    state: ResearchGraphState,
    policy: ResearchRuntimePolicy,
) -> float:
    """Reserve enough time for follow-up tools and one more evidence compilation."""
    durations = state.get("assessment_durations_ms", [])
    valid_durations = [
        float(value)
        for value in durations
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    ]
    latest_assessment_seconds = valid_durations[-1] / 1000 if valid_durations else 20.0
    remaining_step_slots = max(
        0, _state_step_budget(state, policy) - state.get("current_step", 0)
    )
    remaining_tool_slots = max(
        0,
        (_state_tool_call_budget(state, policy) - state.get("tool_call_count", 0)) // 2,
    )
    pending_count = len(state.get("pending_followups", []))
    expected_followups = min(pending_count, remaining_step_slots, remaining_tool_slots)

    # Compilation dominates the observed tail latency. Use the latest measured
    # duration with headroom, plus a bounded allowance for local retrieval and
    # final graph/answer serialization. The 45-second floor protects runs that
    # have no trustworthy duration history yet.
    predicted_seconds = (
        latest_assessment_seconds * 1.25
        + expected_followups * 6.0
        + 10.0
    )
    return min(float(policy.timeout_seconds), max(45.0, predicted_seconds))


def _instrument_node(
    name: str,
    function: ResearchNode,
    sink: AgentEventSink | None,
) -> ResearchNode:
    async def instrumented(state: ResearchGraphState) -> ResearchGraphState:
        started = time.perf_counter()
        _emit_graph_event(
            state,
            sink,
            event_type="node_started",
            status="started",
            component="node",
            name=name,
        )
        try:
            update = await function(state)
        except Exception as exc:
            _emit_graph_event(
                state,
                sink,
                event_type="node_failed",
                status="failed",
                component="node",
                name=name,
                duration_ms=_elapsed_ms(started),
                error_type=type(exc).__name__,
                reason_code="node_execution_failed",
            )
            raise
        _emit_graph_event(
            state,
            sink,
            event_type="node_completed",
            status="succeeded",
            component="node",
            name=name,
            duration_ms=_elapsed_ms(started),
        )
        return update

    return instrumented


def _consume_tool(
    state: ResearchGraphState,
    sink: AgentEventSink | None,
    policy: ResearchRuntimePolicy,
    tool_name: ResearchToolName,
    current_calls: int,
) -> int:
    invocation_budget = _state_tool_call_budget(state, policy)
    try:
        if current_calls + 1 > invocation_budget:
            raise RuntimeError("research invocation tool call budget exceeded")
        return policy.consume(tool_name, current_calls)
    except PermissionError as exc:
        _emit_graph_event(
            state,
            sink,
            event_type="runtime_intercepted",
            status="intercepted",
            component="tool",
            name=tool_name,
            error_type=type(exc).__name__,
            reason_code="tool_not_allowed",
            tool_call_count=current_calls,
            max_tool_calls=invocation_budget,
        )
        raise
    except RuntimeError as exc:
        _emit_graph_event(
            state,
            sink,
            event_type="runtime_intercepted",
            status="intercepted",
            component="tool",
            name=tool_name,
            error_type=type(exc).__name__,
            reason_code="tool_budget_exceeded",
            tool_call_count=current_calls,
            max_tool_calls=invocation_budget,
        )
        raise


def _emit_graph_event(
    state: ResearchGraphState,
    sink: AgentEventSink | None,
    *,
    event_type: Literal[
        "node_started",
        "node_completed",
        "node_failed",
        "tool_started",
        "tool_completed",
        "tool_failed",
        "runtime_intercepted",
    ],
    status: Literal["started", "succeeded", "failed", "intercepted"],
    component: Literal["node", "tool"],
    name: str,
    duration_ms: float | None = None,
    step_id: str | None = None,
    query: str | None = None,
    error_type: str | None = None,
    reason_code: str | None = None,
    degraded: bool | None = None,
    hit_count: int | None = None,
    requested_count: int | None = None,
    returned_count: int | None = None,
    tool_call_count: int | None = None,
    max_tool_calls: int | None = None,
) -> bool:
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or len(run_id) != 32:
        return False
    question = state.get("question")
    event = AgentEvent(
        run_id=run_id,
        occurred_at=datetime.now(UTC),
        event_type=event_type,
        status=status,
        component=component,
        name=name,
        duration_ms=duration_ms,
        question_sha256=(safe_fingerprint(question) if isinstance(question, str) else None),
        step_id_sha256=safe_fingerprint(step_id) if step_id is not None else None,
        query_sha256=safe_fingerprint(query) if query is not None else None,
        error_type=error_type,
        reason_code=reason_code,
        degraded=degraded,
        hit_count=hit_count,
        requested_count=requested_count,
        returned_count=returned_count,
        tool_call_count=tool_call_count,
        max_tool_calls=max_tool_calls,
    )
    return emit_agent_event(sink, event)


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000)


def _comparison_corpus_id(
    plan: ResearchPlan,
    target_ids: tuple[str, ...],
) -> str | None:
    if len(target_ids) != 1:
        return None
    target_id = target_ids[0]
    return next(
        (target.corpus_id for target in plan.targets if target.target_id == target_id),
        None,
    )


def _query_key(query: str) -> str:
    return " ".join(query.casefold().split())


def _query_scope_key(query: str, corpus_id: str | None) -> tuple[str | None, str]:
    return corpus_id, _query_key(query)


def _step_query_key(step: ResearchStep) -> tuple[str | None, str]:
    return _query_scope_key(step.query, step.corpus_id)


def _finish_update(reason: str) -> ResearchGraphState:
    if reason not in TERMINATION_REASONS:
        raise ValueError("invalid research termination reason")
    return {
        "active_step": None,
        "next_action": "finalize",
        "termination_reason": reason,
    }


def _has_unassessed_comparison_evidence(
    state: ResearchGraphState,
    plan: ResearchPlan,
    observations: tuple[ResearchObservation, ...],
) -> bool:
    if plan.task_type != "comparison" or not observations:
        return False
    counts = state.get("assessment_observation_counts", [])
    last_assessed_count = counts[-1] if counts else 0
    return last_assessed_count < len(observations)


def _append_action(
    state: ResearchGraphState,
    *,
    action: str,
    step_id: str | None = None,
    query: str | None = None,
    chunk_ids: tuple[str, ...] = (),
    outcome: str | None = None,
) -> list[dict[str, Any]]:
    return _append_action_from_history(
        state["action_history"],
        action=action,
        step_id=step_id,
        query=query,
        chunk_ids=chunk_ids,
        outcome=outcome,
    )


def _append_action_from_history(
    history: list[dict[str, Any]],
    *,
    action: str,
    step_id: str | None = None,
    query: str | None = None,
    chunk_ids: tuple[str, ...] = (),
    outcome: str | None = None,
) -> list[dict[str, Any]]:
    record = ResearchActionRecord.model_validate(
        {
            "sequence": len(history) + 1,
            "action": action,
            "step_id": step_id,
            "query": query,
            "chunk_ids": chunk_ids,
            "outcome": outcome,
        }
    )
    return [*history, record.model_dump(mode="json")]
