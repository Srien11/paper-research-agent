"""Durable-first fan-out for recoverable main-Agent product events."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

from paper_research_agent.conversation.models import PersistedRunEvent
from paper_research_agent.conversation.store import ConversationStore
from paper_research_agent.web.events import AgentStreamEvent, AgentStreamEventDraft

_CLOSED = object()


class RunEventPublisher:
    """Persist an event before notifying any process-local subscriber."""

    def __init__(
        self,
        store: ConversationStore,
        fanout: Callable[[PersistedRunEvent], Awaitable[None]],
    ) -> None:
        self._store = store
        self._fanout = fanout

    async def publish(
        self,
        event: AgentStreamEventDraft,
        *,
        idempotency_key: str | None = None,
    ) -> AgentStreamEvent:
        persisted = await asyncio.to_thread(
            self._store.append_run_event,
            event,
            idempotency_key=idempotency_key,
        )
        await self._fanout(persisted)
        return persisted.to_stream_event()


class RunEventSubscription:
    def __init__(
        self,
        *,
        bus: RunEventBus,
        token: str,
        request_id: str,
        after_event_id: int,
        queue_size: int,
    ) -> None:
        self._bus = bus
        self.token = token
        self.request_id = request_id
        self.last_event_id = after_event_id
        self.queue: asyncio.Queue[AgentStreamEvent | object] = asyncio.Queue(
            maxsize=queue_size
        )
        self.backfill: list[AgentStreamEvent] = []
        self.needs_backfill = False
        self._segment_closed = False
        self._closed = False

    def __aiter__(self) -> RunEventSubscription:
        return self

    async def __anext__(self) -> AgentStreamEvent:
        if self._closed or self._segment_closed:
            await self.aclose()
            raise StopAsyncIteration
        while True:
            if self.backfill:
                event = self.backfill.pop(0)
            elif self.needs_backfill:
                self.needs_backfill = False
                self.backfill.extend(await self._bus._load_after(self))
                continue
            else:
                item = await self.queue.get()
                if item is _CLOSED:
                    await self.aclose()
                    raise StopAsyncIteration
                assert isinstance(item, AgentStreamEvent)
                event = item
            if event.event_id <= self.last_event_id:
                continue
            self.last_event_id = event.event_id
            if event.closes_delivery_segment:
                self._segment_closed = True
            return event

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._bus._unsubscribe(self.token)


class RunEventBus:
    """Request-keyed subscriptions that recover overflow from the durable ledger."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        subscriber_queue_size: int = 64,
    ) -> None:
        if subscriber_queue_size < 1:
            raise ValueError("subscriber_queue_size must be positive")
        self._store = store
        self._queue_size = subscriber_queue_size
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, RunEventSubscription] = {}
        self._closed = False
        self.publisher = RunEventPublisher(store, self._fanout)

    async def subscribe(
        self,
        request_id: str,
        *,
        after_event_id: int = 0,
    ) -> RunEventSubscription:
        if after_event_id < 0:
            raise ValueError("after_event_id must be non-negative")
        token = uuid.uuid4().hex
        subscription = RunEventSubscription(
            bus=self,
            token=token,
            request_id=request_id,
            after_event_id=after_event_id,
            queue_size=self._queue_size,
        )
        async with self._lock:
            if self._closed:
                raise RuntimeError("run event bus is closed")
            self._subscribers[token] = subscription
        subscription.backfill.extend(await self._load_after(subscription))
        if (
            not subscription.backfill
            and after_event_id > 0
            and await self._cursor_is_terminal(subscription)
        ):
            subscription._segment_closed = True
        return subscription

    async def _cursor_is_terminal(
        self, subscription: RunEventSubscription
    ) -> bool:
        persisted = await asyncio.to_thread(
            self._store.run_events,
            subscription.request_id,
            after_event_id=subscription.last_event_id - 1,
            limit=1,
        )
        return bool(
            persisted
            and persisted[0].event_id == subscription.last_event_id
            and persisted[0].to_stream_event().is_terminal
        )

    async def _load_after(
        self, subscription: RunEventSubscription
    ) -> list[AgentStreamEvent]:
        persisted = await asyncio.to_thread(
            self._store.run_events,
            subscription.request_id,
            after_event_id=subscription.last_event_id,
            limit=10_000,
        )
        return [item.to_stream_event() for item in persisted]

    async def _fanout(self, persisted: PersistedRunEvent) -> None:
        event = persisted.to_stream_event()
        async with self._lock:
            targets = tuple(
                item
                for item in self._subscribers.values()
                if item.request_id == persisted.request_id and not item._closed
            )
        for subscription in targets:
            try:
                subscription.queue.put_nowait(event)
            except asyncio.QueueFull:
                subscription.needs_backfill = True

    async def _unsubscribe(self, token: str) -> None:
        async with self._lock:
            self._subscribers.pop(token, None)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            subscriptions = tuple(self._subscribers.values())
            self._subscribers.clear()
        for subscription in subscriptions:
            subscription._closed = True
            if subscription.queue.full():
                subscription.queue.get_nowait()
            subscription.queue.put_nowait(_CLOSED)
