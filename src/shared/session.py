"""In-memory session store with LRU eviction."""

from __future__ import annotations

import time
from typing import Any


class SessionStore:
    """Simple in-memory session store.

    Each session holds a dict of key-value pairs.  Stale sessions are
    evicted LRU-style when the store exceeds *max_sessions*.
    """

    def __init__(self, max_sessions: int = 100) -> None:
        self._max = max_sessions
        self._data: dict[str, dict[str, Any]] = {}
        self._accessed: dict[str, float] = {}

    def get(self, session_id: str, key: str, default: Any = None) -> Any:
        self._touch(session_id)
        return self._data.get(session_id, {}).get(key, default)

    def set(self, session_id: str, key: str, value: Any) -> None:
        if session_id not in self._data and len(self._data) >= self._max:
            self._evict()
        self._data.setdefault(session_id, {})[key] = value
        self._touch(session_id)

    def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)
        self._accessed.pop(session_id, None)

    def clear(self, session_id: str) -> None:
        if session_id in self._data:
            self._data[session_id].clear()
            self._touch(session_id)

    def _touch(self, session_id: str) -> None:
        self._accessed[session_id] = time.monotonic()

    def _evict(self) -> None:
        oldest = min(self._accessed, key=lambda k: self._accessed[k])
        self.delete(oldest)
