"""Bounded ReAct orchestration over the private, read-only research tools."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator

from paper_research_agent.agent.coverage import validate_evidence_assessment
from paper_research_agent.agent.langchain_tools import build_langchain_tools
from paper_research_agent.agent.models import (
    TERMINATION_REASONS,
    EvidenceAssessment,
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
from paper_research_agent.agent.policy import ResearchRuntimePolicy, ResearchToolName
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
    active_step: dict[str, Any] | None
    tool_call_count: int
    observations: list[dict[str, Any]]
    evidence_records: list[dict[str, Any]]
    assessments: list[dict[str, Any]]
    action_history: list[dict[str, Any]]
    replan_count: int
    consecutive_no_new_evidence: int
    next_action: str
    evidence_sufficient: bool
    termination_reason: str | None


def build_research_graph(
    *,
    planner: ResearchPlanner,
    reasoner: ResearchReasoner,
    service: ResearchToolService,
    max_steps: int = 4,
    evidence_per_step: int = 4,
    policy: ResearchRuntimePolicy | None = None,
    checkpointer: Any | None = None,
    event_sink: AgentEventSink | None = None,
    extended_toolkit: ExtendedResearchToolkit | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile a fail-closed reason-act-observe-assess loop over two local tools."""
    if max_steps <= 0 or max_steps > 6:
        raise ValueError("max_steps must be between 1 and 6")
    if evidence_per_step <= 0 or evidence_per_step > 20:
        raise ValueError("evidence_per_step must be between 1 and 20")
    runtime_policy = policy or ResearchRuntimePolicy(
        max_steps=max_steps,
        evidence_per_step=evidence_per_step,
    )
    initial_plan_steps = runtime_policy.max_steps
    evidence_per_step = runtime_policy.evidence_per_step
    tools = {tool.name: tool for tool in build_langchain_tools(service, extended_toolkit)}
    search_tool = tools["search_corpus"]
    evidence_tool = tools["get_evidence"]

    async def create_plan(state: ResearchGraphState) -> ResearchGraphState:
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
        return {
            "run_id": state.get("run_id", ""),
            "question": graph_input.question,
            "planning_required": graph_input.planning_required,
            "plan": plan.model_dump(mode="json"),
            "current_step": 0,
            "active_step": None,
            "tool_call_count": 0,
            "observations": [],
            "evidence_records": [],
            "assessments": [],
            "action_history": [],
            "replan_count": 0,
            "consecutive_no_new_evidence": 0,
            "next_action": "reason",
            "evidence_sufficient": False,
            "termination_reason": None,
        }

    async def reason(state: ResearchGraphState) -> ResearchGraphState:
        plan = ResearchPlan.model_validate(state["plan"])
        current_step = state["current_step"]
        observations = tuple(
            ResearchObservation.model_validate(value) for value in state["observations"]
        )

        if not observations:
            return {
                "active_step": plan.steps[0].model_dump(mode="json"),
                "next_action": "execute_tools",
                "termination_reason": None,
            }

        assessments = tuple(
            EvidenceAssessment.model_validate(value) for value in state["assessments"]
        )
        if len(assessments) != len(observations):
            raise ValueError("research assessments do not match observations")
        latest = assessments[-1]
        if latest.evidence_sufficient:
            return _finish_update("evidence_sufficient")
        if state["consecutive_no_new_evidence"] >= 2:
            return _finish_update("no_new_evidence")
        if runtime_policy.max_tool_calls - state["tool_call_count"] < 2:
            return _finish_update("tool_budget")

        executed_queries = {_query_key(item.search.query) for item in observations}
        planned_next = plan.steps[current_step] if current_step < len(plan.steps) else None
        if plan.task_type == "comparison" and planned_next is not None:
            if _query_key(planned_next.query) in executed_queries:
                return _finish_update("repeated_query")
            return {
                "active_step": planned_next.model_dump(mode="json"),
                "next_action": "execute_tools",
                "termination_reason": None,
            }
        if latest.next_query is not None:
            if _query_key(latest.next_query) in executed_queries:
                if planned_next is None or _query_key(planned_next.query) in executed_queries:
                    return _finish_update("repeated_query")
            else:
                replan_count = state["replan_count"] + 1
                target_ids: tuple[str, ...] = ()
                dimension_ids: tuple[str, ...] = ()
                if plan.task_type == "comparison":
                    requirement_by_id = {
                        item.requirement_id: item for item in plan.requirements
                    }
                    requested = [
                        requirement_by_id[item]
                        for item in latest.next_requirement_ids
                    ]
                    target_ids = tuple(dict.fromkeys(item.target_id for item in requested))
                    dimension_ids = tuple(
                        dict.fromkeys(item.dimension_id for item in requested)
                    )
                replacement = ResearchStep(
                    step_id=f"replan-{replan_count}",
                    objective=latest.next_objective or "Refine the evidence search",
                    query=latest.next_query,
                    top_k=(
                        planned_next.top_k
                        if planned_next is not None
                        else min(20, max(10, evidence_per_step))
                    ),
                    target_ids=target_ids,
                    dimension_ids=dimension_ids,
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
            return _finish_update("plan_exhausted")
        if _query_key(planned_next.query) in executed_queries:
            return _finish_update("repeated_query")
        return {
            "active_step": planned_next.model_dump(mode="json"),
            "next_action": "execute_tools",
            "termination_reason": None,
        }

    async def execute_tools(state: ResearchGraphState) -> ResearchGraphState:
        step = ResearchStep.model_validate(state.get("active_step"))
        tool_call_count = _consume_tool(
            state,
            event_sink,
            runtime_policy,
            "search_corpus",
            state["tool_call_count"],
        )
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
            raw_search = await search_tool.ainvoke({"query": step.query, "top_k": step.top_k})
            search = SearchCorpusResult.model_validate(raw_search)
            if search.query != step.query:
                raise ValueError("search tool returned a mismatched query")
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
        action_history = _append_action(
            state,
            action="search_corpus",
            step_id=step.step_id,
            query=step.query,
        )

        selected_ids = tuple(hit.chunk_id for hit in search.hits[:evidence_per_step])
        if selected_ids:
            tool_call_count = _consume_tool(
                state,
                event_sink,
                runtime_policy,
                "get_evidence",
                tool_call_count,
            )
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
            action_history = _append_action_from_history(
                action_history,
                action="get_evidence",
                step_id=step.step_id,
                chunk_ids=selected_ids,
            )
        else:
            evidence = GetEvidenceResult(records=())

        observation = ResearchObservation(
            step_id=step.step_id,
            objective=step.objective,
            search=search,
            evidence=evidence,
        )
        merged = list(state["evidence_records"])
        seen = {str(record["chunk_id"]) for record in merged}
        previous_evidence_count = len(seen)
        for record in evidence.records:
            if record.chunk_id not in seen:
                merged.append(record.model_dump(mode="json"))
                seen.add(record.chunk_id)
        consecutive_no_new_evidence = (
            0
            if len(seen) > previous_evidence_count
            else state["consecutive_no_new_evidence"] + 1
        )
        return {
            "current_step": state["current_step"] + 1,
            "active_step": None,
            "tool_call_count": tool_call_count,
            "observations": [
                *state["observations"],
                observation.model_dump(mode="json"),
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
        assessment = await reasoner.assess(
            state["question"],
            plan=plan,
            observations=observations,
            remaining_steps=min(
                6,
                max(
                    0,
                    (runtime_policy.max_tool_calls - state["tool_call_count"]) // 2,
                ),
            ),
        )
        assessment = validate_evidence_assessment(plan, observations, assessment)
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
        return "execute_tools" if state["next_action"] == "execute_tools" else "finalize"

    builder = StateGraph(ResearchGraphState)
    builder.add_node("plan", _instrument_node("plan", create_plan, event_sink))
    builder.add_node("reason", _instrument_node("reason", reason, event_sink))
    builder.add_node(
        "execute_tools",
        _instrument_node("execute_tools", execute_tools, event_sink),
    )
    builder.add_node(
        "assess_evidence",
        _instrument_node("assess_evidence", assess_evidence, event_sink),
    )
    builder.add_node("finalize", _instrument_node("finalize", finalize, event_sink))
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "reason")
    builder.add_conditional_edges(
        "reason",
        route_after_reason,
        {"execute_tools": "execute_tools", "finalize": "finalize"},
    )
    builder.add_edge("execute_tools", "assess_evidence")
    builder.add_edge("assess_evidence", "reason")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="paper_research_react_v2")


ResearchNode = Callable[[ResearchGraphState], Awaitable[ResearchGraphState]]


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
    try:
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
            max_tool_calls=policy.max_tool_calls,
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
            max_tool_calls=policy.max_tool_calls,
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


def _query_key(query: str) -> str:
    return " ".join(query.casefold().split())


def _finish_update(reason: str) -> ResearchGraphState:
    if reason not in TERMINATION_REASONS:
        raise ValueError("invalid research termination reason")
    return {
        "active_step": None,
        "next_action": "finalize",
        "termination_reason": reason,
    }


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
