"""Main Agent runtime: per-conversation locking, timeout, and approval resume."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

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
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 3600:
            raise ValueError("main agent timeout must be between 0 and 3600 seconds")
        self._graph = graph
        self._repository = repository
        self._approval_resumer = approval_resumer
        self._timeout_seconds = timeout_seconds
        self._close = close
        self._clear = clear
        self._locks: dict[str, asyncio.Lock] = {}
        self._inflight: dict[str, asyncio.Task[MainAgentResult]] = {}
        self._inflight_conversations: dict[str, str] = {}
        self._guard = asyncio.Lock()
        self._closed = False

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
                task.add_done_callback(
                    lambda completed, request_id=request.request_id: self._task_done(
                        request_id, completed
                    )
                )
        return await asyncio.shield(task)

    async def _run_serialized(self, request: MainAgentRequest) -> MainAgentResult:
        lock = await self._lock_for(request.conversation_id)
        async with lock:
            start = await asyncio.to_thread(
                self._repository.begin_agent_run,
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                user_question=request.message,
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
                raise
            return _result_from_state(state, request)

    async def resume_approval(
        self, *, request_id: str, approved: bool
    ) -> MainAgentResult:
        if self._closed:
            raise RuntimeError("main agent runtime is closed")
        resumer = self._approval_resumer
        if resumer is None:
            raise RuntimeError("approval resume is unavailable")
        request = MainAgentResumeRequest(request_id=request_id, approved=approved)
        return await resumer(request.request_id, request.approved)

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

    async def _lock_for(self, conversation_id: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[conversation_id] = lock
            return lock


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
