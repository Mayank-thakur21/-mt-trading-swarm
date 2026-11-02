"""A small thread-safe TTL cache (no external dependency).

Market data is requested repeatedly within seconds — the API's ``/quote``, ``/history`` and
``/analyze`` endpoints and the scheduler all hit the same symbols — so caching downloads for
``CACHE_TTL_SECONDS`` saves latency and keeps us well inside Yahoo's rate limits.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Final

_MISSING: Final = object()


class TTLCache[V]:
    """Least-recently-used cache whose entries expire ``ttl_seconds`` after insertion.

    A ``ttl_seconds`` of 0 disables caching entirely (every lookup misses).
    """

    def __init__(
        self,
        ttl_seconds: float,
        maxsize: int = 512,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.ttl_seconds = float(ttl_seconds)
        self.maxsize = maxsize
        self._clock = clock
        self._data: OrderedDict[Hashable, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable, default: V | None = None) -> V | None:
        """Return the cached value for ``key`` or ``default`` if absent/expired."""
        value = self._lookup(key)
        return default if value is _MISSING else value  # type: ignore[return-value]

    def set(self, key: Hashable, value: V) -> None:
        if self.ttl_seconds <= 0:
            return
        with self._lock:
            self._data[key] = (self._clock() + self.ttl_seconds, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def get_or_set(self, key: Hashable, factory: Callable[[], V]) -> V:
        """Return the cached value, computing and storing it with ``factory`` on a miss.

        The factory runs outside the lock (it usually does network I/O), so two threads
        missing simultaneously may both compute; the last one wins. That is acceptable for
        idempotent market-data reads and avoids serialising all downloads.
        """
        value = self._lookup(key)
        if value is not _MISSING:
            return value  # type: ignore[return-value]
        fresh = factory()
        self.set(key, fresh)
        return fresh

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def _lookup(self, key: Hashable) -> object:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return _MISSING
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._data[key]
                return _MISSING
            self._data.move_to_end(key)
            return value


__all__ = ["TTLCache"]
