"""Public start/resume boundary for the checkpointed dynamic-tool graph."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any, Literal, Protocol, cast

from langgraph.types import Command

from paper_research_agent.agent.dynamic.models import (
    ApprovalDecision,
    DynamicResearchResult,
    PendingApproval,
    ToolObservation,
)
from paper_research_agent.agent.orchestrator.identifiers import (
    dynamic_checkpoint_thread_id,
)


class AsyncDynamicGraph(Protocol):
    async def ainvoke(
        self,
        input: Any,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any: ...


class DynamicResearchRuntime:
    def __init__(
        self,
        *,
        graph: AsyncDynamicGraph,
        max_steps: int = 6,
        timeout_seconds: float = 90,
    ) -> None:
        if max_steps <= 0 or max_steps > 12:
            raise ValueError("dynamic runtime max_steps must be between 1 and 12")
        if timeout_seconds <= 0 or timeout_seconds > 300:
            raise ValueError("dynamic runtime timeout must be between 0 and 300 seconds")
        self._graph = graph
        self._max_steps = max_steps
        self._timeout_seconds = timeout_seconds

    async def run(
        self,
        question: str,
        *,
        thread_id: str,
        memory_context: tuple[dict[str, object], ...] | None = None,
        child_context: dict[str, object] | None = None,
    ) -> DynamicResearchResult:
        normalized_question = _question(question)
        normalized_thread = _thread(thread_id)
        run_id = uuid.uuid4().hex
        initial = {
            "run_id": run_id,
            "question": normalized_question,
            "observations": [],
            "decision_fingerprints": [],
            "pending_decision": None,
            "pending_approval": None,
            "final_summary": None,
            "termination_reason": None,
            "memory_context": (
                list(memory_context) if memory_context is not None else []
            ),
            "memory_supplied": memory_context is not None,
            "child_context": dict(child_context) if child_context else {},
            "memory_proposal_completed": False,
            "resume_after_execute": None,
            "next_action": "route",
        }
        raw = await self._invoke(initial, normalized_thread)
        return self._project(raw, normalized_thread)

    async def resume(
        self,
        *,
        thread_id: str,
        approved: bool,
    ) -> DynamicResearchResult:
        normalized_thread = _thread(thread_id)
        command: Command[Any] = Command(
            resume=ApprovalDecision(approved=approved).model_dump(mode="json")
        )
        raw = await self._invoke(command, normalized_thread)
        return self._project(raw, normalized_thread)

    async def _invoke(self, value: Any, thread_id: str) -> Mapping[str, Any]:
        config: dict[str, Any] = {
            "configurable": {"thread_id": dynamic_checkpoint_thread_id(thread_id)},
            "recursion_limit": max(30, self._max_steps * 5 + 10),
        }
        async with asyncio.timeout(self._timeout_seconds):
            raw = await self._graph.ainvoke(value, config=config)
        if not isinstance(raw, Mapping):
            raise TypeError("dynamic tool graph returned an invalid state")
        return raw

    def _project(
        self,
        state: Mapping[str, Any],
        thread_id: str,
    ) -> DynamicResearchResult:
        run_id = state.get("run_id")
        if not isinstance(run_id, str):
            raise TypeError("dynamic tool graph returned no run_id")
        raw_observations = state.get("observations", [])
        if not isinstance(raw_observations, list):
            raise TypeError("dynamic tool observations are invalid")
        observations = tuple(ToolObservation.model_validate(item) for item in raw_observations)
        raw_pending = state.get("pending_approval")
        pending = PendingApproval.model_validate(raw_pending) if raw_pending is not None else None
        if pending is not None and state.get("next_action") == "approval":
            return DynamicResearchResult(
                run_id=run_id,
                thread_id=thread_id,
                status="approval_required",
                observations=observations,
                pending_approval=pending,
            )
        reason = state.get("termination_reason")
        summary = state.get("final_summary")
        if state.get("next_action") != "finish" or not isinstance(reason, str):
            raise ValueError("dynamic tool graph did not finish or pause for approval")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("dynamic tool graph returned no final summary")
        return DynamicResearchResult(
            run_id=run_id,
            thread_id=thread_id,
            status="completed",
            observations=observations,
            final_summary=summary,
            termination_reason=cast(
                Literal[
                    "router_finished",
                    "max_steps",
                    "repeated_tool_call",
                    "approval_denied",
                    "approval_expired",
                ],
                reason,
            ),
        )


def _thread(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 240:
        raise ValueError("thread_id must contain between 1 and 240 characters")
    return normalized


def _question(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 10_000:
        raise ValueError("dynamic research question must contain between 1 and 10000 characters")
    return normalized
