"""Main Agent runtime: per-conversation locking, timeout, and approval resume."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from paper_research_agent.agent.orchestrator.models import (
    ChildTaskResult,
    MainAgentRequest,
    MainAgentResult,
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
        self._guard = asyncio.Lock()
        self._closed = False

    async def run(self, request: MainAgentRequest) -> MainAgentResult:
        if self._closed:
            raise RuntimeError("main agent runtime is closed")
        lock = await self._lock_for(request.conversation_id)
        async with lock:
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    state = await self._graph.ainvoke(
                        {"request": request.model_dump(mode="json")}
                    )
            except TimeoutError:
                raise TimeoutError("main agent run exceeded its total deadline") from None
            return _result_from_state(state, request)

    async def resume_approval(
        self, *, request_id: str, approved: bool
    ) -> MainAgentResult:
        if self._closed:
            raise RuntimeError("main agent runtime is closed")
        resumer = self._approval_resumer
        if resumer is None:
            raise RuntimeError("approval resume is unavailable")
        return await resumer(request_id, approved)

    async def clear(self, conversation_id: str) -> None:
        if self._closed:
            raise RuntimeError("main agent runtime is closed")
        async with self._guard:
            self._locks.pop(conversation_id, None)
        if self._clear is not None:
            await self._clear(conversation_id)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._locks.clear()
        if self._close is not None:
            await self._close()

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
    status: Literal["completed", "waiting_approval"] = (
        "completed" if reason in {"completed", "cached"} else "waiting_approval"
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
        workspace_version=int(state.get("base_workspace_version", 0)) + 1,
    )
