from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from paper_research_agent.conversation.store import InMemoryConversationStore
from paper_research_agent.web.events import AgentStreamEventDraft
from paper_research_agent.web.run_event_bus import RunEventBus


class RunEventBusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.store = InMemoryConversationStore()
        self.bus = RunEventBus(self.store, subscriber_queue_size=1)
        self.request_id = "req_bus_123456789012"
        self.started = None

    def begin(self):
        self.started = self.store.begin_agent_run(
            request_id=self.request_id,
            conversation_id="conversation-bus",
            user_question="stream",
        )
        return self.started

    def draft(self, event_type: str, *, status: str | None = None, summary: str | None = None):
        assert self.started is not None
        return AgentStreamEventDraft(
            type=event_type,
            occurred_at=datetime.now(UTC),
            request_id=self.started.request_id,
            run_id=self.started.run_id,
            turn_id=self.started.turn_id,
            node_id=f"run:{self.started.run_id}",
            status=status,
            summary=summary,
        )

    async def asyncTearDown(self) -> None:
        await self.bus.aclose()

    async def test_subscribe_before_run_creation_receives_first_event(self) -> None:
        subscription = await self.bus.subscribe(self.request_id)
        self.begin()
        await self.bus.publisher.publish(
            self.draft("run_started"), idempotency_key="start"
        )

        event = await asyncio.wait_for(anext(subscription), timeout=1)

        self.assertEqual(event.type, "run_started")
        self.assertEqual(event.event_id, 1)
        await subscription.aclose()

    async def test_durable_backfill_and_slow_subscriber_do_not_drop_events(self) -> None:
        self.begin()
        subscription = await self.bus.subscribe(self.request_id)
        for index in range(5):
            await self.bus.publisher.publish(
                self.draft("reasoning_summary", summary=f"阶段 {index}"),
                idempotency_key=f"summary-{index}",
            )

        events = [await asyncio.wait_for(anext(subscription), timeout=1) for _ in range(5)]

        self.assertEqual([item.event_id for item in events], [1, 2, 3, 4, 5])
        self.assertEqual(events[-1].summary, "阶段 4")
        await subscription.aclose()

        replay = await self.bus.subscribe(self.request_id, after_event_id=3)
        replayed = [await anext(replay), await anext(replay)]
        self.assertEqual([item.event_id for item in replayed], [4, 5])
        await replay.aclose()

    async def test_segment_boundary_closes_subscription_but_run_can_resume(self) -> None:
        self.begin()
        subscription = await self.bus.subscribe(self.request_id)
        await self.bus.publisher.publish(
            self.draft("run_paused", status="paused"), idempotency_key="pause"
        )

        paused = await anext(subscription)
        self.assertEqual(paused.type, "run_paused")
        with self.assertRaises(StopAsyncIteration):
            await anext(subscription)

        resumed_subscription = await self.bus.subscribe(
            self.request_id, after_event_id=paused.event_id
        )
        await self.bus.publisher.publish(
            self.draft("run_resumed", status="running"), idempotency_key="resume"
        )
        resumed = await anext(resumed_subscription)
        self.assertEqual(resumed.event_id, 2)
        await resumed_subscription.aclose()

    async def test_subscription_after_latest_terminal_event_closes_immediately(self) -> None:
        self.begin()
        terminal = await self.bus.publisher.publish(
            self.draft("run_completed", status="completed"),
            idempotency_key="completed",
        )

        subscription = await self.bus.subscribe(
            self.request_id,
            after_event_id=terminal.event_id,
        )

        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(anext(subscription), timeout=1)


if __name__ == "__main__":
    unittest.main()
