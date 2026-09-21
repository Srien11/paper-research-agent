from __future__ import annotations

import asyncio
import unittest

from paper_research_agent.web.concurrency import (
    RuntimeBusyError,
    RuntimeCapacityError,
    RuntimeClosedError,
    SessionExecutionGate,
)


class SessionExecutionGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_allows_distinct_sessions_and_rejects_same_session_and_overflow(
        self,
    ) -> None:
        gate = SessionExecutionGate(max_inflight_runs=2)
        release = asyncio.Event()
        started = {"a": asyncio.Event(), "b": asyncio.Event()}

        async def hold(session_id: str) -> None:
            async with gate.admit(session_id):
                started[session_id].set()
                await release.wait()

        first = asyncio.create_task(hold("a"))
        second = asyncio.create_task(hold("b"))
        await asyncio.wait_for(
            asyncio.gather(started["a"].wait(), started["b"].wait()),
            timeout=1,
        )

        with self.assertRaises(RuntimeBusyError):
            async with gate.admit("a"):
                self.fail("same-session work must not overlap")
        with self.assertRaises(RuntimeCapacityError):
            async with gate.admit("c"):
                self.fail("capacity overflow must not be admitted")

        release.set()
        await asyncio.gather(first, second)
        self.assertFalse(gate.is_busy)

    async def test_close_rejects_new_work_and_waits_for_admitted_work(self) -> None:
        gate = SessionExecutionGate(max_inflight_runs=2)
        started = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> None:
            async with gate.admit("a"):
                started.set()
                await release.wait()

        active = asyncio.create_task(hold())
        await asyncio.wait_for(started.wait(), timeout=1)
        closing = asyncio.create_task(gate.close())
        await asyncio.sleep(0)

        self.assertFalse(closing.done())
        with self.assertRaises(RuntimeClosedError):
            async with gate.admit("b"):
                self.fail("closing gate must reject new work")

        release.set()
        await active
        await asyncio.wait_for(closing, timeout=1)
        self.assertTrue(gate.is_closed)


if __name__ == "__main__":
    unittest.main()
