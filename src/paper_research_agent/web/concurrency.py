"""Bounded, per-session admission control for long-running Web operations."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

DEFAULT_WEB_MAX_INFLIGHT_RUNS = 2
MAX_WEB_MAX_INFLIGHT_RUNS = 16


class RuntimeBusyError(RuntimeError):
    """The same conversation already has an operation in progress."""


class RuntimeCapacityError(RuntimeError):
    """The bounded Web execution lane has no remaining capacity."""


class RuntimeClosedError(RuntimeError):
    """The execution lane is closing and cannot accept new work."""


def web_max_inflight_runs_from_environment(
    environ: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if environ is None else environ
    raw = source.get(
        "PRA_WEB_MAX_INFLIGHT_RUNS",
        str(DEFAULT_WEB_MAX_INFLIGHT_RUNS),
    ).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("PRA_WEB_MAX_INFLIGHT_RUNS must be an integer") from error
    if not 1 <= value <= MAX_WEB_MAX_INFLIGHT_RUNS:
        raise ValueError(
            "PRA_WEB_MAX_INFLIGHT_RUNS must be between "
            f"1 and {MAX_WEB_MAX_INFLIGHT_RUNS}"
        )
    return value


class SessionExecutionGate:
    """Reject overlap within one session while allowing bounded cross-session work."""

    def __init__(self, *, max_inflight_runs: int = DEFAULT_WEB_MAX_INFLIGHT_RUNS) -> None:
        if not 1 <= max_inflight_runs <= MAX_WEB_MAX_INFLIGHT_RUNS:
            raise ValueError(
                "max_inflight_runs must be between "
                f"1 and {MAX_WEB_MAX_INFLIGHT_RUNS}"
            )
        self._max_inflight_runs = max_inflight_runs
        self._active_sessions: set[str] = set()
        self._guard = asyncio.Lock()
        self._drained = asyncio.Event()
        self._drained.set()
        self._closed = False

    @property
    def is_busy(self) -> bool:
        return bool(self._active_sessions)

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def inflight_count(self) -> int:
        return len(self._active_sessions)

    @property
    def max_inflight_runs(self) -> int:
        return self._max_inflight_runs

    @asynccontextmanager
    async def admit(self, session_id: str) -> AsyncIterator[None]:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id cannot be blank")
        await self._acquire(normalized)
        try:
            yield
        finally:
            await self._release(normalized)

    async def close(self) -> None:
        async with self._guard:
            self._closed = True
            drained = self._drained
        await drained.wait()

    async def _acquire(self, session_id: str) -> None:
        async with self._guard:
            if self._closed:
                raise RuntimeClosedError("runtime is closed")
            if session_id in self._active_sessions:
                raise RuntimeBusyError("conversation is already running")
            if len(self._active_sessions) >= self._max_inflight_runs:
                raise RuntimeCapacityError("runtime capacity is exhausted")
            self._active_sessions.add(session_id)
            self._drained.clear()

    async def _release(self, session_id: str) -> None:
        async with self._guard:
            self._active_sessions.discard(session_id)
            if not self._active_sessions:
                self._drained.set()
