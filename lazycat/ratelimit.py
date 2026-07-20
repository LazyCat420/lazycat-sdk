"""
Rate-limiting primitives for external calls.

Two complementary shapes, both keyed by an arbitrary string (a domain, an API
name, a tenant) so one instance covers every upstream a service talks to:

    KeyedRateLimiter — minimum interval between calls sharing a key.
                       Use when an upstream limits requests per second.

    KeyedSemaphore   — maximum concurrent calls sharing a key.
                       Use when an upstream limits simultaneous connections,
                       or to stop parallel workers from stampeding one API.

Usage:
    from lazycat.ratelimit import KeyedRateLimiter, KeyedSemaphore

    limiter = KeyedRateLimiter({"reddit.com": 0.5}, default_rate=1.0)
    async with limiter.acquire("reddit.com"):
        await client.get(url)

    sem = KeyedSemaphore({"yfinance": 4}, default_limit=3)
    async with sem.acquire("yfinance"):
        await collect_price_history(ticker)

    @sem.limit("reddit")
    async def scrape_reddit(ticker):
        ...

Rate/limit tables stay with the application — this module holds no defaults
for any particular upstream.
"""

import asyncio
import contextlib
import functools
import logging
import time
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

__all__ = ["KeyedRateLimiter", "KeyedSemaphore"]


class KeyedRateLimiter:
    """Per-key async rate limiter enforcing a minimum interval between calls.

    Each key gets its own lock, so calls under different keys never block each
    other. The minimum interval between two calls sharing a key is 1/rate
    seconds.

    Args:
        rates: Mapping of key → requests per second. Stored by reference, so
            later mutations by the caller take effect on the next acquire().
        default_rate: Requests per second for keys absent from `rates`.
    """

    def __init__(
        self,
        rates: dict[str, float] | None = None,
        default_rate: float = 1.0,
    ):
        self._rates: dict[str, float] = rates if rates is not None else {}
        self._default_rate = default_rate
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last: dict[str, float] = {}

    def set_rate(self, key: str, rate: float) -> None:
        """Set the requests-per-second budget for a single key."""
        self._rates[key] = rate

    @contextlib.asynccontextmanager
    async def acquire(self, key: str):
        rate = self._rates.get(key, self._default_rate)
        min_interval = 1.0 / rate

        async with self._locks[key]:
            elapsed = time.monotonic() - self._last.get(key, 0)
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last[key] = time.monotonic()
            yield


class KeyedSemaphore:
    """Per-key semaphore manager capping concurrent calls.

    Each key gets its own asyncio.Semaphore that caps concurrency across ALL
    parallel callers in the process — without it, N parallel workers × M
    upstreams produces N×M simultaneous requests and trips rate limits or IP
    bans.

    Args:
        limits: Mapping of key → max concurrent calls.
        default_limit: Concurrency cap for keys absent from `limits`.
    """

    def __init__(
        self,
        limits: dict[str, int] | None = None,
        default_limit: int = 3,
    ):
        self._limits: dict[str, int] = limits if limits is not None else {}
        self._default_limit = default_limit
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._burst_mode: bool = False

    def enable_burst_mode(self, enabled: bool = True) -> None:
        """Temporarily bypass all semaphores for high-intensity runs."""
        self._burst_mode = enabled
        if enabled:
            logger.info("[ratelimit] Burst mode enabled. Semaphores bypassed.")
        else:
            logger.info("[ratelimit] Burst mode disabled.")

    def _get_semaphore(self, key: str) -> asyncio.Semaphore:
        """Lazy-init the semaphore for a key."""
        if key not in self._semaphores:
            limit = self._limits.get(key, self._default_limit)
            self._semaphores[key] = asyncio.Semaphore(limit)
            logger.debug("[ratelimit] Created semaphore for %s (max=%d)", key, limit)
        return self._semaphores[key]

    @contextlib.asynccontextmanager
    async def acquire(self, key: str):
        """Hold a slot for `key` for the duration of the block."""
        if self._burst_mode:
            yield
            return

        sem = self._get_semaphore(key)
        async with sem:
            yield

    def limit(self, key: str) -> Callable:
        """Decorator form of acquire() for an async function."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self.acquire(key):
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    def status(self) -> dict[str, dict]:
        """Return the current state of all known semaphores for monitoring."""
        result = {}
        for key, limit in self._limits.items():
            sem = self._semaphores.get(key)
            if sem:
                # Semaphore._value shows remaining slots
                # (not part of the public API but stable in CPython)
                available = getattr(sem, "_value", "?")
                result[key] = {
                    "max": limit,
                    "available": available,
                    "in_use": limit - available if isinstance(available, int) else "?",
                }
            else:
                result[key] = {"max": limit, "available": limit, "in_use": 0}
        return result
