"""Independent LangGraph loop for model-selected research tools and approval interrupts."""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from paper_research_agent.agent.dynamic.memory import (
    DynamicMemoryProposer,
    has_explicit_memory_intent,
)
from paper_research_agent.agent.dynamic.models import (
    ApprovalDecision,
    PendingApproval,
    ToolDecision,
    ToolObservation,
)
from paper_research_agent.agent.dynamic.router import DynamicToolRouter
from paper_research_agent.agent.observability import AgentEvent, AgentEventSink, emit_agent_event
from paper_research_agent.agent.tooling.contracts import ToolExecutionResult
from paper_research_agent.agent.tooling.service import ExtendedResearchToolkit


class DynamicToolState(TypedDict, total=False):
    run_id: str
    question: str
    observations: list[dict[str, Any]]
    decision_fingerprints: list[str]
    pending_decision: dict[str, Any] | None
    pending_approval: dict[str, Any] | None
    final_summary: str | None
    termination_reason: str | None
    next_action: str
    memory_context: list[dict[str, Any]]
    memory_supplied: bool
    child_context: dict[str, Any]
    memory_proposal_completed: bool
    resume_after_execute: str | None


def build_dynamic_tool_graph(
    *,
    router: DynamicToolRouter,
    toolkit: ExtendedResearchToolkit,
    max_steps: int = 6,
    checkpointer: Any | None = None,
    event_sink: AgentEventSink | None = None,
    memory_proposer: DynamicMemoryProposer | None = None,
    memory_scope_id: str = "global",
) -> CompiledStateGraph[Any, Any, Any, Any]:
    if max_steps <= 0 or max_steps > 12:
        raise ValueError("dynamic tool max_steps must be between 1 and 12")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", memory_scope_id):
        raise ValueError("dynamic memory scope is invalid")

    async def recall_memory(state: DynamicToolState) -> DynamicToolState:
        if state.get("memory_supplied", False):
            return {"next_action": "route"}
        question = _required_text(state, "question")
        action = "list" if has_explicit_memory_intent(question) else "search"
        arguments: dict[str, Any] = {
            "action": action,
            "scope_id": memory_scope_id,
            "limit": 5,
        }
        if action == "search":
            arguments["query"] = question[:500]
        try:
            result = await toolkit.execute(
                "manage_long_term_memory",
                arguments,
                run_id=_required_text(state, "run_id"),
            )
        except (OSError, TimeoutError, sqlite3.Error):
            return {"memory_context": [], "next_action": "route"}
        if result.status not in {"ok", "not_found"}:
            raise RuntimeError("long-term memory recall returned an invalid status")
        return {
            "memory_context": [dict(item) for item in result.items],
            "next_action": "route",
        }

    async def route(state: DynamicToolState) -> DynamicToolState:
        observations = _observations(state)
        if len(observations) >= max_steps:
            return _finished("Tool-call budget reached.", "max_steps")
        decision = await router.decide(
            _required_text(state, "question"),
            observations,
            _memory_context(state),
            remaining_steps=max_steps - len(observations),
            child_context=state.get("child_context"),
        )
        if decision.action == "finish":
            return {
                "final_summary": cast(str, decision.final_summary),
                "termination_reason": "router_finished",
                "pending_decision": None,
                "pending_approval": None,
                "next_action": "propose_memory" if memory_proposer else "finalize",
            }
        if decision.tool_name == "manage_long_term_memory" and decision.arguments.get("action") in {
            "add",
            "update",
            "delete",
        }:
            raise PermissionError("router cannot directly mutate long-term memory")
        if decision.tool_name in {"save_research_note", "export_research_report"} and not _has_explicit_write_intent(
            _required_text(state, "question"), decision.tool_name
        ):
            return _finished(
                "我可以直接和你交流；当前请求没有要求保存或导出内容，因此没有调用敏感工具。",
                "router_finished",
            )
        fingerprints = _fingerprints(state)
        if decision.fingerprint in fingerprints:
            return _finished(
                "Stopped because the router repeated an identical tool call.",
                "repeated_tool_call",
            )
        return {
            "pending_decision": decision.model_dump(mode="json"),
            "resume_after_execute": "route",
            "next_action": "execute",
        }

    async def propose_memory(state: DynamicToolState) -> DynamicToolState:
        if memory_proposer is None or state.get("memory_proposal_completed", False):
            return {"next_action": "finalize"}
        proposal = await memory_proposer.propose(
            _required_text(state, "question"),
            _memory_context(state),
            _observations(state),
        )
        if proposal.action == "none":
            return {"memory_proposal_completed": True, "next_action": "finalize"}
        decision = ToolDecision(
            action="call_tool",
            tool_name="manage_long_term_memory",
            arguments=proposal.tool_arguments(scope_id=memory_scope_id),
            purpose=proposal.rationale,
        )
        return {
            "memory_proposal_completed": True,
            "pending_decision": decision.model_dump(mode="json"),
            "resume_after_execute": "finalize",
            "next_action": "execute",
        }

    async def execute(state: DynamicToolState) -> DynamicToolState:
        decision = ToolDecision.model_validate(state.get("pending_decision"))
        tool_name = cast(str, decision.tool_name)
        result = await toolkit.execute(
            tool_name,
            decision.arguments,
            run_id=_required_text(state, "run_id"),
        )
        if result.status == "approval_required":
            pending = PendingApproval(
                tool_name=tool_name,
                arguments=decision.arguments,
                purpose=decision.purpose,
                decision_fingerprint=decision.fingerprint,
                approval_request_id=str(result.summary["approval_request_id"]),
                arguments_sha256=str(result.summary["arguments_sha256"]),
                expires_at_epoch=float(result.summary["expires_at_epoch"]),
            )
            return {
                "pending_approval": pending.model_dump(mode="json"),
                "next_action": "approval",
            }
        return _record_result(state, decision, result)

    async def approval(state: DynamicToolState) -> DynamicToolState:
        pending = PendingApproval.model_validate(state.get("pending_approval"))
        raw_decision = interrupt(
            {
                "kind": "tool_approval_v1",
                "tool_name": pending.tool_name,
                "purpose": pending.purpose,
                "arguments_sha256": pending.arguments_sha256,
                "expires_at_epoch": pending.expires_at_epoch,
            }
        )
        approval_decision = ApprovalDecision.model_validate(raw_decision)
        tool_decision = ToolDecision(
            action="call_tool",
            tool_name=pending.tool_name,
            arguments=pending.arguments,
            purpose=pending.purpose,
        )
        if not approval_decision.approved:
            denied = ToolExecutionResult(
                tool_name=pending.tool_name,
                status="denied",
                trust="side_effect",
                summary={"reason": "user_denied"},
            )
            update = _record_result(state, tool_decision, denied)
            update.update(_finished("Sensitive tool request was denied.", "approval_denied"))
            return update
        if pending.expires_at_epoch <= time.time():
            update = _record_result(
                state,
                tool_decision,
                ToolExecutionResult(
                    tool_name=pending.tool_name,
                    status="denied",
                    trust="side_effect",
                    summary={"reason": "approval_expired"},
                ),
            )
            update.update(_finished("Sensitive tool approval expired.", "approval_expired"))
            return update
        try:
            approval_token = toolkit.approve(pending.approval_request_id)
        except ValueError:
            update = _record_result(
                state,
                tool_decision,
                ToolExecutionResult(
                    tool_name=pending.tool_name,
                    status="denied",
                    trust="side_effect",
                    summary={"reason": "approval_expired"},
                ),
            )
            update.update(_finished("Sensitive tool approval expired.", "approval_expired"))
            return update
        approved_arguments = {**pending.arguments, "approval_token": approval_token}
        result = await toolkit.execute(
            pending.tool_name,
            approved_arguments,
            run_id=_required_text(state, "run_id"),
        )
        if result.status == "approval_required":
            raise RuntimeError("approved write tool requested approval a second time")
        return _record_result(state, tool_decision, result)

    def route_next(
        state: DynamicToolState,
    ) -> Literal["execute", "propose_memory", "finalize"]:
        action = state.get("next_action")
        if action in {"execute", "propose_memory"}:
            return cast(Literal["execute", "propose_memory"], action)
        return "finalize"

    def execute_next(state: DynamicToolState) -> Literal["approval", "route", "finalize"]:
        action = state.get("next_action")
        if action in {"approval", "finalize"}:
            return cast(Literal["approval", "finalize"], action)
        return "route"

    def approval_next(state: DynamicToolState) -> Literal["route", "finalize"]:
        return "finalize" if state.get("next_action") == "finalize" else "route"

    async def finalize(state: DynamicToolState) -> DynamicToolState:
        del state
        return {
            "pending_decision": None,
            "pending_approval": None,
            "resume_after_execute": None,
            "next_action": "finish",
        }

    builder = StateGraph(DynamicToolState)
    builder.add_node(
        "recall_memory",
        cast(Any, _instrument_node("dynamic_recall_memory", recall_memory, event_sink)),
    )
    builder.add_node(
        "route", cast(Any, _instrument_node("dynamic_route", route, event_sink))
    )
    builder.add_node(
        "propose_memory",
        cast(
            Any,
            _instrument_node("dynamic_propose_memory", propose_memory, event_sink),
        ),
    )
    builder.add_node(
        "execute", cast(Any, _instrument_node("dynamic_execute", execute, event_sink))
    )
    builder.add_node(
        "approval",
        cast(Any, _instrument_node("dynamic_approval", approval, event_sink)),
    )
    builder.add_node(
        "finalize",
        cast(Any, _instrument_node("dynamic_finalize", finalize, event_sink)),
    )
    builder.add_edge(START, "recall_memory")
    builder.add_edge("recall_memory", "route")
    builder.add_conditional_edges(
        "route",
        route_next,
        {
            "execute": "execute",
            "propose_memory": "propose_memory",
            "finalize": "finalize",
        },
    )
    builder.add_conditional_edges(
        "propose_memory",
        lambda state: "execute" if state.get("next_action") == "execute" else "finalize",
        {"execute": "execute", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "execute",
        execute_next,
        {"approval": "approval", "route": "route", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "approval", approval_next, {"route": "route", "finalize": "finalize"}
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="paper_research_dynamic_tools_v1")


def _record_result(
    state: DynamicToolState,
    decision: ToolDecision,
    result: ToolExecutionResult,
) -> DynamicToolState:
    observations = list(_observations(state))
    observations.append(
        ToolObservation(
            sequence=len(observations) + 1,
            decision_fingerprint=decision.fingerprint,
            tool_name=cast(str, decision.tool_name),
            purpose=decision.purpose,
            result=result,
        )
    )
    next_action = state.get("resume_after_execute") or "route"
    return {
        "observations": [item.model_dump(mode="json") for item in observations],
        "decision_fingerprints": [*_fingerprints(state), decision.fingerprint],
        "pending_decision": None,
        "pending_approval": None,
        "resume_after_execute": None,
        "next_action": next_action,
    }


def _finished(summary: str, reason: str) -> DynamicToolState:
    return {
        "final_summary": summary,
        "termination_reason": reason,
        "pending_decision": None,
        "pending_approval": None,
        "next_action": "finalize",
    }


def _has_explicit_write_intent(question: str, tool_name: str) -> bool:
    normalized = question.casefold()
    markers: tuple[str, ...]
    if tool_name == "save_research_note":
        markers = ("保存", "记下", "记录", "写入", "存下", "save", "record", "write down")
    else:
        markers = ("导出", "生成报告", "下载报告", "export", "create a report", "generate a report")
    return any(marker in normalized for marker in markers)


def _observations(state: DynamicToolState) -> tuple[ToolObservation, ...]:
    raw = state.get("observations", [])
    return tuple(ToolObservation.model_validate(item) for item in raw)


def _fingerprints(state: DynamicToolState) -> list[str]:
    raw = state.get("decision_fingerprints", [])
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise TypeError("dynamic graph decision fingerprints are invalid")
    return raw


def _memory_context(state: DynamicToolState) -> tuple[dict[str, Any], ...]:
    raw = state.get("memory_context", [])
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise TypeError("dynamic graph memory context is invalid")
    return tuple(raw)


def _required_text(state: DynamicToolState, key: str) -> str:
    value = state.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"dynamic graph {key} is missing")
    return value


DynamicNode = Callable[[DynamicToolState], Awaitable[DynamicToolState]]


def _instrument_node(
    name: str,
    function: DynamicNode,
    sink: AgentEventSink | None,
) -> DynamicNode:
    async def instrumented(state: DynamicToolState) -> DynamicToolState:
        run_id = _required_text(state, "run_id")
        started = time.perf_counter()
        emit_agent_event(
            sink,
            AgentEvent(
                run_id=run_id,
                occurred_at=datetime.now(UTC),
                event_type="node_started",
                status="started",
                component="node",
                name=name,
            ),
        )
        try:
            update = await function(state)
        except Exception as exc:
            emit_agent_event(
                sink,
                AgentEvent(
                    run_id=run_id,
                    occurred_at=datetime.now(UTC),
                    event_type="node_failed",
                    status="failed",
                    component="node",
                    name=name,
                    duration_ms=max(0.0, (time.perf_counter() - started) * 1000),
                    error_type=type(exc).__name__,
                    reason_code="node_execution_failed",
                ),
            )
            raise
        emit_agent_event(
            sink,
            AgentEvent(
                run_id=run_id,
                occurred_at=datetime.now(UTC),
                event_type="node_completed",
                status="succeeded",
                component="node",
                name=name,
                duration_ms=max(0.0, (time.perf_counter() - started) * 1000),
            ),
        )
        return update

    return instrumented
