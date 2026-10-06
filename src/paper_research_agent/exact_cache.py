"""Bounded process-local memoization; keys are digests, never private input text."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


def exact_key(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ExactCache(Generic[T]):
    """LRU with absolute TTL and a zero-capacity off switch; callers copy mutable values."""

    def __init__(
        self,
        capacity: int = 256,
        ttl_seconds: float = 300,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 0 or ttl_seconds <= 0:
            raise ValueError("cache capacity must be non-negative and TTL positive")
        enabled = os.getenv("PRA_EXACT_CACHE_ENABLED", "true").strip().lower()
        if enabled not in {"true", "false", "1", "0"}:
            raise ValueError("PRA_EXACT_CACHE_ENABLED must be true/false or 1/0")
        self.capacity = capacity if enabled in {"true", "1"} else 0
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def _expire(self, now: float) -> None:
        for key in [key for key, (deadline, _) in self._entries.items() if deadline <= now]:
            del self._entries[key]

    def get(self, key: str) -> T | None:
        with self._lock:
            self._expire(self._clock())
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self.hits += 1
            self._entries.move_to_end(key)
            return entry[1]

    def put(self, key: str, value: T) -> None:
        with self._lock:
            now = self._clock()
            self._expire(now)
            if not self.capacity:
                return
            self._entries[key] = (now + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
