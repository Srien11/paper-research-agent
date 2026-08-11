"""Main Agent runtime: per-conversation locking, timeout, and approval resume."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any, cast

from paper_research_agent.agent.observability import (
    AgentEvent,
    AgentEventSink,
    AgentEventStatus,
    AgentEventType,
    ChildStatus,
    MainCapability,
    emit_agent_event,
    safe_fingerprint,
)
from paper_research_agent.agent.orchestrator.models import (
    ChildTaskResult,
    MainAgentRequest,
    MainAgentResult,
    MainAgentResumeRequest,
    RunStatus,
)
from paper_research_agent.conversation.store import ConversationStore

ApprovalResumer = Callable[[str, bool], Awaitable[MainAgentResult]]
Closer = Callable[[], Awaitable[None]]
ConversationClearer = Callable[[str], Awaitable[None]]
_MAIN_CAPABILITIES = frozenset(
    {"direct_chat", "local_rag", "dynamic_tools", "attachment_qa", "file_edit"}
)


@dataclass(frozen=True, slots=True)
class _EventContext:
    run_id: str
    question_sha256: str
    thread_sha256: str


class MainAgentRuntime:
    """Serialize one conversation at a time; different conversations run in parallel."""

    def __init__(
        self,
        *,
        graph: Any,
        repository: ConversationStore,
        approval_resumer: ApprovalResumer | None = None,
        timeout_seconds: float = 180,
        close: Closer | None = None,
        clear: ConversationClearer | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 3600:
            raise ValueError("main agent timeout must be between 0 and 3600 seconds")
        self._graph = graph
        self._repository = repository
        self._approval_resumer = approval_resumer
        self._timeout_seconds = timeout_seconds
        self._close = close
        self._clear = clear
        self._event_sink = event_sink
        self._locks: dict[str, asyncio.Lock] = {}
        self._inflight: dict[str, asyncio.Task[MainAgentResult]] = {}
        self._inflight_conversations: dict[str, str] = {}
        self._guard = asyncio.Lock()
        self._closed = False
        self._emit(
            AgentEvent(
                run_id="0" * 32,
                occurred_at=datetime.now(UTC),
                event_type="main_runtime_built",
                status="succeeded",
                component="runtime",
                name="main_agent",
                timeout_seconds=self._timeout_seconds,
            )
        )

    @property
    def event_sink(self) -> AgentEventSink | None:
        return self._event_sink

    async def run(self, request: MainAgentRequest) -> MainAgentResult:
        async with self._guard:
            if self._closed:
                raise RuntimeError("main agent runtime is closed")
            task = self._inflight.get(request.request_id)
            if task is not None:
                if (
                    self._inflight_conversations[request.request_id]
                    != request.conversation_id
                ):
                    raise ValueError("request_id belongs to another conversation")
            else:
                task = asyncio.create_task(
                    self._run_serialized(request),
                    name=f"main-agent::{request.request_id}",
                )
                self._inflight[request.request_id] = task
                self._inflight_conversations[request.request_id] = (
                    request.conversation_id
                )
                task.add_done_callback(partial(self._task_done, request.request_id))
        return await asyncio.shield(task)

    async def _run_serialized(self, request: MainAgentRequest) -> MainAgentResult:
        started_at = time.perf_counter()
        lock = await self._lock_for(request.conversation_id)
        async with lock:
            start = await asyncio.to_thread(
                self._repository.begin_agent_run,
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                user_question=request.message,
            )
            should_emit = start.outcome == "created"
            common = _EventContext(
                run_id=start.run_id,
                question_sha256=safe_fingerprint(request.message),
                thread_sha256=safe_fingerprint(request.conversation_id),
            )
            if should_emit:
                self._emit(
                    _runtime_event(
                        common,
                        event_type="main_run_started",
                        status="started",
                        timeout_seconds=self._timeout_seconds,
                    )
                )
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    state = await self._graph.ainvoke(
                        {
                            "request": request.model_dump(mode="json"),
                            "run_start": start.model_dump(mode="json"),
                        },
                        config={
                            "configurable": {
                                "thread_id": (
                                    f"main::{request.conversation_id}::{start.run_id}"
                                )
                            }
                        },
                    )
            except TimeoutError:
                await asyncio.to_thread(
                    self._repository.fail_agent_run,
                    run_id=start.run_id,
                    turn_id=start.turn_id,
                    reason_code="runtime_timeout",
                )
                if should_emit:
                    self._emit(
                        _runtime_event(
                            common,
                            event_type="runtime_intercepted",
                            status="intercepted",
                            duration_ms=_elapsed_ms(started_at),
                            error_type="TimeoutError",
                            reason_code="runtime_timeout",
                            timeout_seconds=self._timeout_seconds,
                        )
                    )
                raise TimeoutError("main agent run exceeded its total deadline") from None
            except asyncio.CancelledError:
                await asyncio.to_thread(
                    self._repository.fail_agent_run,
                    run_id=start.run_id,
                    turn_id=start.turn_id,
                    reason_code="runtime_cancelled",
                )
                raise
            except Exception:
                await asyncio.to_thread(
                    self._repository.fail_agent_run,
                    run_id=start.run_id,
                    turn_id=start.turn_id,
                    reason_code="runtime_error",
                )
                if should_emit:
                    self._emit(
                        _runtime_event(
                            common,
                            event_type="run_failed",
                            status="failed",
                            duration_ms=_elapsed_ms(started_at),
                            reason_code="runtime_error",
                        )
                    )
                raise
            result = _result_from_state(state, request)
            if should_emit:
                self._emit_result_events(
                    result=result,
                    state=state,
                    common=common,
                    started_at=started_at,
                )
            return result

    async def resume_approval(
        self, *, request_id: str, approved: bool
    ) -> MainAgentResult:
        if self._closed:
            raise RuntimeError("main agent runtime is closed")
        resumer = self._approval_resumer
        if resumer is None:
            raise RuntimeError("approval resume is unavailable")
        request = MainAgentResumeRequest(request_id=request_id, approved=approved)
        started_at = time.perf_counter()
        result = await resumer(request.request_id, request.approved)
        event_type: AgentEventType = (
            "main_run_paused"
            if result.status == "waiting_approval"
            else "main_run_completed"
            if result.status == "completed"
            else "run_failed"
        )
        event_status: AgentEventStatus = (
            "failed" if event_type == "run_failed" else "succeeded"
        )
        self._emit(
            AgentEvent(
                run_id=_observable_run_id(result.run_id),
                occurred_at=datetime.now(UTC),
                event_type=event_type,
                status=event_status,
                component="runtime",
                name="main_agent",
                duration_ms=_elapsed_ms(started_at),
                thread_sha256=safe_fingerprint(result.conversation_id),
                returned_count=len(result.child_results),
                workspace_version=result.workspace_version,
                termination_reason=result.status,
                reason_code=(
                    "approval_resume_failed" if event_type == "run_failed" else None
                ),
            )
        )
        return result

    async def clear(self, conversation_id: str) -> None:
        if self._closed:
            raise RuntimeError("main agent runtime is closed")
        lock = await self._lock_for(conversation_id)
        async with lock:
            if self._clear is not None:
                await self._clear(conversation_id)
            async with self._guard:
                if self._locks.get(conversation_id) is lock:
                    self._locks.pop(conversation_id, None)

    async def aclose(self) -> None:
        async with self._guard:
            if self._closed:
                return
            self._closed = True
            tasks = tuple(self._inflight.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._guard:
            self._inflight.clear()
            self._inflight_conversations.clear()
            self._locks.clear()
        if self._close is not None:
            await self._close()

    def _task_done(
        self, request_id: str, task: asyncio.Task[MainAgentResult]
    ) -> None:
        if not task.cancelled():
            task.exception()
        asyncio.create_task(self._release_inflight(request_id, task))

    async def _release_inflight(
        self, request_id: str, task: asyncio.Task[MainAgentResult]
    ) -> None:
        async with self._guard:
            if self._inflight.get(request_id) is task:
                self._inflight.pop(request_id, None)
                self._inflight_conversations.pop(request_id, None)

    def _emit_result_events(
        self,
        *,
        result: MainAgentResult,
        state: dict[str, Any],
        common: _EventContext,
        started_at: float,
    ) -> None:
        routes = tuple(
            dict.fromkeys(
                cast(MainCapability, route)
                for route in result.route_trace
                if route in _MAIN_CAPABILITIES
            )
        )
        for capability in routes:
            self._emit(
                _runtime_event(
                    common,
                    event_type="capability_routed",
                    status="succeeded",
                    capability=capability,
                )
            )
        for child in result.child_results:
            self._emit(
                _runtime_event(
                    common,
                    event_type="child_completed",
                    status="succeeded",
                    capability=child.capability,
                    child_status=child.status,
                    evidence_count=len(child.source_ids),
                )
            )
        duration_ms = _elapsed_ms(started_at)
        if result.status == "waiting_approval":
            self._emit(
                _runtime_event(
                    common,
                    event_type="main_run_paused",
                    status="succeeded",
                    duration_ms=duration_ms,
                    requested_count=len(routes),
                    returned_count=len(result.child_results),
                    workspace_version=result.workspace_version,
                    termination_reason="waiting_approval",
                )
            )
            return
        raw_validation_errors = state.get("validation_errors", ())
        validation_error_count = (
            len(raw_validation_errors)
            if isinstance(raw_validation_errors, (list, tuple))
            else 0
        )
        if result.status == "failed" and validation_error_count:
            self._emit(
                _runtime_event(
                    common,
                    event_type="main_commit_rejected",
                    status="failed",
                    duration_ms=duration_ms,
                    reason_code="commit_validation_failed",
                    validation_error_count=validation_error_count,
                    workspace_version=result.workspace_version,
                )
            )
            return
        if result.status == "completed":
            self._emit(
                _runtime_event(
                    common,
                    event_type="main_run_completed",
                    status="succeeded",
                    duration_ms=duration_ms,
                    requested_count=len(routes),
                    returned_count=len(result.child_results),
                    workspace_version=result.workspace_version,
                    termination_reason="completed",
                )
            )

    def _emit(self, event: AgentEvent) -> bool:
        return emit_agent_event(self._event_sink, event)

    async def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[conversation_id] = lock
            return lock


def _runtime_event(
    context: _EventContext,
    *,
    event_type: AgentEventType,
    status: AgentEventStatus,
    duration_ms: float | None = None,
    error_type: str | None = None,
    reason_code: str | None = None,
    termination_reason: str | None = None,
    requested_count: int | None = None,
    returned_count: int | None = None,
    evidence_count: int | None = None,
    timeout_seconds: float | None = None,
    capability: MainCapability | None = None,
    child_status: ChildStatus | None = None,
    workspace_version: int | None = None,
    validation_error_count: int | None = None,
) -> AgentEvent:
    return AgentEvent(
        run_id=context.run_id,
        occurred_at=datetime.now(UTC),
        event_type=event_type,
        status=status,
        component="runtime",
        name="main_agent",
        duration_ms=duration_ms,
        question_sha256=context.question_sha256,
        thread_sha256=context.thread_sha256,
        error_type=error_type,
        reason_code=reason_code,
        termination_reason=termination_reason,
        requested_count=requested_count,
        returned_count=returned_count,
        evidence_count=evidence_count,
        timeout_seconds=timeout_seconds,
        capability=capability,
        child_status=child_status,
        workspace_version=workspace_version,
        validation_error_count=validation_error_count,
    )


def _result_from_state(
    state: dict[str, Any], request: MainAgentRequest
) -> MainAgentResult:
    reason = state.get("termination_reason")
    statuses: dict[object, RunStatus] = {
        "completed": "completed",
        "cached": "completed",
        "waiting_approval": "waiting_approval",
        "waiting_approval_cached": "waiting_approval",
        "running_reused": "running",
        "failed": "failed",
    }
    status = statuses.get(reason, "failed")
    base_workspace_version = int(state.get("base_workspace_version", 0))
    workspace_version = (
        base_workspace_version + 1
        if status in {"completed", "waiting_approval"}
        and reason not in {"cached", "waiting_approval_cached"}
        else base_workspace_version
    )
    return MainAgentResult(
        run_id=str(state.get("run_id", "")),
        request_id=request.request_id,
        conversation_id=request.conversation_id,
        status=status,
        answer=str(state.get("final_answer", "")),
        route_trace=tuple(str(item) for item in state.get("route_trace", [])),
        child_results=tuple(
            ChildTaskResult.model_validate(item)
            for item in state.get("child_results", [])
        ),
        pending_approval=state.get("pending_approval"),
        workspace_version=workspace_version,
    )


def _elapsed_ms(started_at: float) -> float:
    return max(0.0, (time.perf_counter() - started_at) * 1000)


def _observable_run_id(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) == 32 and all(char in "0123456789abcdef" for char in normalized):
        return normalized
    return safe_fingerprint(normalized)[:32]
