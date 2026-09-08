"""Response caching service for reproducible RAG outputs."""

from __future__ import annotations

import logging
import time
from typing import Any

from src.shared.utils.crypto_utils import compute_sha256_str

logger = logging.getLogger(__name__)

DEFAULT_TTL = 300  # 5 minutes
MAX_CACHE_SIZE = 100


class ResponseCache:
    """In-memory cache with TTL for RAG responses."""

    def __init__(self, ttl: int = DEFAULT_TTL, max_size: int = MAX_CACHE_SIZE):
        """Initialize cache.

        Args:
            ttl: Time-to-live in seconds
            max_size: Maximum number of cached entries
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: dict[str, dict[str, Any]] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> str | None:
        """Get cached response if not expired.

        Args:
            key: Cache key

        Returns:
            Cached response or None if not found/expired
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return None

        self._update_access_order(key)
        return entry["response"]  # type: ignore[no-any-return]

    def set(self, key: str, response: str) -> None:
        """Cache a response.

        Args:
            key: Cache key
            response: Response text to cache
        """
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
        }
        self._update_access_order(key)

    def invalidate(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Cache invalidated")

    def remove(self, key: str) -> None:
        """Remove specific key from cache.

        Args:
            key: Cache key to remove
        """
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry["timestamp"] > self.ttl  # type: ignore[no-any-return]

    def _evict_oldest(self) -> None:
        """Evict oldest accessed entry (FIFO)."""
        if self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]

    def _update_access_order(self, key: str) -> None:
        """Update access order for FIFO eviction."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    @staticmethod
    def generate_key(query: str, context: str = "") -> str:
        """Generate cache key from query and context.

        Args:
            query: User query
            context: Additional context (search results, rule data)

        Returns:
            SHA-256 hash as cache key
        """
        content = f"{query}:{context}"
        return compute_sha256_str(content)[:16]
