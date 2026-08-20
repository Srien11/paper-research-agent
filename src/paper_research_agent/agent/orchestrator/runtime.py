"""Main Agent runtime: per-conversation locking, timeout, and approval resume."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Any, Protocol, cast

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
from paper_research_agent.agent.orchestrator.control import (
    AgentRunControl,
    PlanEdit,
    RunControlCommand,
)
from paper_research_agent.agent.orchestrator.identifiers import main_checkpoint_thread_id
from paper_research_agent.agent.orchestrator.models import (
    AgentRunStart,
    ChildTaskResult,
    ConversationWorkspace,
    MainAgentRequest,
    MainAgentResult,
    MainAgentResumeRequest,
    RunStatus,
)
from paper_research_agent.conversation.store import ConversationStore
from paper_research_agent.web.events import (
    AgentStreamEvent,
    AgentStreamEventDraft,
    AgentStreamEventType,
    RunNodeStatus,
    SafeRunEventDetail,
)

ApprovalResumer = Callable[[str, bool], Awaitable[MainAgentResult]]
Closer = Callable[[], Awaitable[None]]
ConversationClearer = Callable[[str], Awaitable[None]]


class RunEventPublisherLike(Protocol):
    async def publish(
        self,
        event: AgentStreamEventDraft,
        *,
        idempotency_key: str | None = None,
    ) -> AgentStreamEvent: ...
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
        run_event_publisher: RunEventPublisherLike | None = None,
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
        self._run_event_publisher = run_event_publisher
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

    @property
    def run_event_publisher(self) -> RunEventPublisherLike | None:
        return self._run_event_publisher

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
                request=request,
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
            await self._publish_run_start(request, start)
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
                                    main_checkpoint_thread_id(
                                        request.conversation_id,
                                        start.run_id,
                                    )
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
                await self._publish_run_boundary(
                    request,
                    start,
                    event_type="run_failed",
                    status="failed",
                    summary="运行超过总时限",
                    reason_code="runtime_timeout",
                )
                raise TimeoutError("main agent run exceeded its total deadline") from None
            except asyncio.CancelledError:
                await asyncio.to_thread(
                    self._repository.fail_agent_run,
                    run_id=start.run_id,
                    turn_id=start.turn_id,
                    reason_code="runtime_cancelled",
                )
                await self._publish_run_boundary(
                    request,
                    start,
                    event_type="run_cancelled",
                    status="cancelled",
                    summary="运行已取消",
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
                await self._publish_run_boundary(
                    request,
                    start,
                    event_type="run_failed",
                    status="failed",
                    summary="运行未能完成",
                    reason_code="runtime_error",
                )
                raise
            result = _result_from_state(state, request)
            await self._publish_result(request, start, result)
            if should_emit:
                self._emit_result_events(
                    result=result,
                    state=state,
                    common=common,
                    started_at=started_at,
                )
            return result

    async def _publish_run_start(
        self, request: MainAgentRequest, start: AgentRunStart
    ) -> None:
        if self._run_event_publisher is None:
            return
        existing = await asyncio.to_thread(
            self._repository.run_events, request.request_id, limit=10_000
        )
        if start.outcome == "created":
            await self._publish_product_event(
                request,
                start,
                event_type="run_started",
                status="running",
                title="开始运行",
                summary="正在准备对话上下文",
                idempotency_key="lifecycle:start",
            )
        elif start.outcome == "resuming":
            await self._publish_product_event(
                request,
                start,
                event_type="run_resumed",
                status="running",
                title="继续运行",
                summary="从已保存的计划继续",
                idempotency_key=f"lifecycle:resume:{len(existing) + 1}",
            )
        elif not existing and start.outcome != "running_reused":
            await self._publish_product_event(
                request,
                start,
                event_type="run_reused",
                status="running",
                title="恢复已有运行",
                summary="正在恢复已有结果",
                idempotency_key="lifecycle:reused",
            )

    async def _publish_result(
        self,
        request: MainAgentRequest,
        start: AgentRunStart,
        result: MainAgentResult,
    ) -> None:
        if self._run_event_publisher is None:
            return
        existing = await asyncio.to_thread(
            self._repository.run_events, request.request_id, limit=10_000
        )
        if any(item.to_stream_event().is_terminal for item in existing):
            return
        event_types = {item.event_type for item in existing}
        if result.status == "completed" and "reasoning_completed" not in event_types:
            await self._publish_product_event(
                request,
                start,
                event_type="reasoning_completed",
                status="completed",
                title="研究过程完成",
                summary="已完成计划内的研究与回答整理",
                idempotency_key="reasoning:completed",
                node_id="reasoning:main",
            )
        if result.status == "completed" and result.answer and "answer_delta" not in event_types:
            await self._publish_product_event(
                request,
                start,
                event_type="answer_started",
                status="running",
                title="整理回答",
                detail=SafeRunEventDetail(delivery_mode="validated_replay"),
                idempotency_key="answer:start",
                node_id="answer:main",
            )
            for index, chunk in enumerate(_answer_chunks(result.answer)):
                await self._publish_product_event(
                    request,
                    start,
                    event_type="answer_delta",
                    status="running",
                    detail=SafeRunEventDetail(delivery_mode="validated_replay"),
                    delta=chunk,
                    idempotency_key=f"answer:delta:{index}",
                    node_id="answer:main",
                )
            await self._publish_product_event(
                request,
                start,
                event_type="answer_completed",
                status="completed",
                title="回答完成",
                detail=SafeRunEventDetail(delivery_mode="validated_replay"),
                idempotency_key="answer:completed",
                node_id="answer:main",
            )

        if result.status == "waiting_approval":
            tool_name = None
            purpose = None
            arguments_sha256 = None
            expires_at_epoch = None
            if result.pending_approval is not None:
                candidate = result.pending_approval.get("tool_name")
                tool_name = candidate if isinstance(candidate, str) else None
                candidate = result.pending_approval.get("purpose")
                purpose = candidate if isinstance(candidate, str) else None
                candidate = result.pending_approval.get("arguments_sha256")
                arguments_sha256 = candidate if isinstance(candidate, str) else None
                candidate = result.pending_approval.get("expires_at_epoch")
                expires_at_epoch = (
                    float(candidate) if isinstance(candidate, (int, float)) else None
                )
            await self._publish_product_event(
                request,
                start,
                event_type="interaction_required",
                status="waiting_approval",
                title="需要工具审批",
                summary="敏感工具已暂停，等待批准或拒绝",
                detail=SafeRunEventDetail(
                    tool_name=tool_name,
                    purpose=purpose,
                    arguments_sha256=arguments_sha256,
                    expires_at_epoch=expires_at_epoch,
                ),
                idempotency_key="interaction:approval",
                node_id="interaction:approval",
            )
            await self._publish_run_boundary(
                request,
                start,
                event_type="run_waiting_approval",
                status="waiting_approval",
                summary="等待工具审批",
            )
        elif result.status == "waiting_user":
            await self._publish_product_event(
                request,
                start,
                event_type="interaction_required",
                status="waiting_user",
                title="需要补充信息",
                summary="请补充必要信息后继续",
                idempotency_key="interaction:user",
                node_id="interaction:user",
            )
            await self._publish_run_boundary(
                request,
                start,
                event_type="run_waiting_user",
                status="waiting_user",
                summary="等待用户补充信息",
            )
        elif result.status == "paused":
            await self._publish_run_boundary(
                request,
                start,
                event_type="run_paused",
                status="paused",
                summary="运行已暂停，进度已保存",
            )
        elif result.status == "completed":
            await self._publish_run_boundary(
                request,
                start,
                event_type="run_completed",
                status="completed",
                summary="运行完成",
            )
        elif result.status == "cancelled":
            await self._publish_run_boundary(
                request,
                start,
                event_type="run_cancelled",
                status="cancelled",
                summary="运行已取消",
            )
        elif result.status == "conflict":
            await self._publish_run_boundary(
                request,
                start,
                event_type="run_conflict",
                status="conflict",
                summary="工作区版本发生冲突",
                reason_code="workspace_conflict",
            )
        else:
            await self._publish_run_boundary(
                request,
                start,
                event_type="run_failed",
                status="failed",
                summary="运行未能完成",
                reason_code="run_failed",
            )

    async def _publish_run_boundary(
        self,
        request: MainAgentRequest,
        start: AgentRunStart,
        *,
        event_type: AgentStreamEventType,
        status: RunNodeStatus,
        summary: str,
        reason_code: str | None = None,
    ) -> None:
        await self._publish_product_event(
            request,
            start,
            event_type=event_type,
            status=status,
            title=summary,
            summary=summary,
            detail=SafeRunEventDetail(reason_code=reason_code),
            idempotency_key=f"boundary:{event_type}",
        )

    async def _publish_product_event(
        self,
        request: MainAgentRequest,
        start: AgentRunStart,
        *,
        event_type: AgentStreamEventType,
        status: RunNodeStatus | None = None,
        title: str | None = None,
        summary: str | None = None,
        detail: SafeRunEventDetail | None = None,
        delta: str | None = None,
        idempotency_key: str | None = None,
        node_id: str | None = None,
    ) -> None:
        publisher = self._run_event_publisher
        if publisher is None:
            return
        await publisher.publish(
            AgentStreamEventDraft(
                type=event_type,
                occurred_at=datetime.now(UTC),
                request_id=request.request_id,
                run_id=start.run_id,
                turn_id=start.turn_id,
                node_id=node_id or f"run:{start.run_id}",
                status=status,
                title=title,
                summary=summary,
                detail=detail or SafeRunEventDetail(delivery_mode="event_only"),
                delta=delta,
            ),
            idempotency_key=idempotency_key,
        )

    async def load_control(self, request_id: str) -> AgentRunControl | None:
        """Read the durable control state for polling and optimistic commands."""

        return await asyncio.to_thread(
            self._repository.load_agent_control, request_id=request_id
        )

    async def command_run(
        self, *, request_id: str, command: RunControlCommand
    ) -> AgentRunControl:
        """Pause/cancel cooperatively, or resume from the last committed step boundary."""

        if self._closed:
            raise RuntimeError("main agent runtime is closed")
        previous = await self.load_control(request_id)
        control = await asyncio.to_thread(
            self._repository.command_agent_run,
            request_id=request_id,
            command=command,
        )
        restart_for_cancel = (
            command.action == "cancel"
            and previous is not None
            and previous.status in {"paused", "waiting_approval"}
        )
        if command.action == "resume" or restart_for_cancel:
            request = await asyncio.to_thread(
                self._repository.load_agent_request, request_id
            )
            if request is None:
                raise ValueError("main agent request not found")
            resumed = asyncio.create_task(
                self._run_after_previous_release(request),
                name=f"main-agent-resume::{request_id}",
            )
            resumed.add_done_callback(_consume_background_result)
        return control

    async def _run_after_previous_release(
        self, request: MainAgentRequest
    ) -> MainAgentResult:
        """Avoid reusing the just-finished paused task during an immediate resume."""

        async with self._guard:
            previous = self._inflight.get(request.request_id)
        if previous is not None:
            await asyncio.shield(previous)
            await self._release_inflight(request.request_id, previous)
        return await self.run(request)

    async def edit_plan(
        self, *, request_id: str, edit: PlanEdit
    ) -> ConversationWorkspace:
        """Edit a paused plan atomically while keeping completed task facts immutable."""

        if self._closed:
            raise RuntimeError("main agent runtime is closed")
        return await asyncio.to_thread(
            self._repository.edit_agent_plan, request_id=request_id, edit=edit
        )

    async def load_workspace_for_run(
        self, request_id: str
    ) -> tuple[AgentRunControl, ConversationWorkspace] | None:
        control = await self.load_control(request_id)
        if control is None:
            return None
        state_reader = getattr(self._graph, "aget_state", None)
        if callable(state_reader):
            try:
                snapshot = await state_reader(
                    {
                        "configurable": {
                            "thread_id": (
                                main_checkpoint_thread_id(
                                    control.conversation_id,
                                    control.run_id,
                                )
                            )
                        }
                    }
                )
                values = getattr(snapshot, "values", {})
                draft = values.get("workspace_draft") if isinstance(values, dict) else None
                if draft is not None:
                    live_workspace = ConversationWorkspace.model_validate(draft)
                    if live_workspace.conversation_id == control.conversation_id:
                        return control, live_workspace
            except Exception:  # noqa: BLE001 - checkpoints are a best-effort live view
                snapshot = None
        workspace = await asyncio.to_thread(
            self._repository.load_workspace, control.conversation_id
        )
        return control, workspace

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
        original = await asyncio.to_thread(
            self._repository.load_agent_request, request.request_id
        )
        run_start = None
        if self._run_event_publisher is not None and original is not None:
            run_start = await asyncio.to_thread(
                self._repository.begin_agent_run,
                request_id=original.request_id,
                conversation_id=original.conversation_id,
                user_question=original.message,
                request=original,
            )
            existing = await asyncio.to_thread(
                self._repository.run_events, original.request_id, limit=10_000
            )
            await self._publish_product_event(
                original,
                run_start,
                event_type="interaction_resolved",
                status="completed",
                title="审批已处理",
                summary="已批准工具调用" if approved else "已拒绝工具调用",
                idempotency_key=f"interaction:approval:resolved:{len(existing) + 1}",
                node_id="interaction:approval",
            )
            await self._publish_product_event(
                original,
                run_start,
                event_type="run_resumed",
                status="running",
                title="继续运行",
                summary="审批处理后继续",
                idempotency_key=f"lifecycle:approval-resume:{len(existing) + 2}",
            )
        result = await resumer(request.request_id, request.approved)
        if original is not None and run_start is not None:
            await self._publish_result(original, run_start, result)
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
        "paused": "paused",
        "paused_cached": "paused",
        "cancelled": "cancelled",
        "cancelled_cached": "cancelled",
        "running_reused": "running",
        "failed": "failed",
    }
    status = statuses.get(reason, "failed")
    base_workspace_version = int(state.get("base_workspace_version", 0))
    workspace_version = (
        base_workspace_version + 1
        if status in {"completed", "waiting_approval", "paused", "cancelled"}
        and reason
        not in {"cached", "waiting_approval_cached", "paused_cached", "cancelled_cached"}
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


def _answer_chunks(text: str, *, chunk_size: int = 192) -> tuple[str, ...]:
    """Coalesce output into durable chunks instead of per-token transactions."""

    return tuple(text[index : index + chunk_size] for index in range(0, len(text), chunk_size))


def _consume_background_result(task: asyncio.Task[MainAgentResult]) -> None:
    if not task.cancelled():
        task.exception()


def _observable_run_id(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) == 32 and all(char in "0123456789abcdef" for char in normalized):
        return normalized
    return safe_fingerprint(normalized)[:32]
