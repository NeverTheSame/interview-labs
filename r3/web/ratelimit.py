"""A small in-process sliding-window limiter for the sign-in and sign-up forms.

State lives in this process only: with several instances each keeps its own
count, which still bounds guessing per instance. The clock is injectable so
tests do not sleep."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Iterable


class RateLimiter:
    """At most ``limit`` recorded hits per key in the last ``window`` seconds."""

    def __init__(self, limit: int = 10, window: float = 900.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = {}
        self._swept = clock()

    def _live(self, key: str, now: float) -> deque[float] | None:
        hits = self._hits.get(key)
        if hits is None:
            return None
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if not hits:
            del self._hits[key]
            return None
        return hits

    def _sweep(self, now: float) -> None:
        if now - self._swept >= self.window:
            self._swept = now
            for key in list(self._hits):
                self._live(key, now)

    def retry_after(self, keys: Iterable[str]) -> int:
        """Seconds until a request under all of ``keys`` would be allowed
        again; 0 when none of them is at the limit."""
        now = self.clock()
        self._sweep(now)
        wait = 0.0
        for key in keys:
            hits = self._live(key, now)
            if hits is not None and len(hits) >= self.limit:
                wait = max(wait, hits[0] + self.window - now)
        return int(wait + 0.999) if wait > 0 else 0

    def record(self, keys: Iterable[str]) -> None:
        now = self.clock()
        for key in keys:
            self._live(key, now)
            self._hits.setdefault(key, deque()).append(now)
