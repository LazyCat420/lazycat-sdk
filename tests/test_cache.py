"""
Tests for lazycat.cache — TTL cache with bounded LRU eviction.
"""
import pytest

from lazycat import cache as cache_mod
from lazycat.cache import (
    clear_cache,
    get_cache_stats,
    invalidate_cache,
    timed_cache,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    # The store is module-global by design, so tests must not inherit state.
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def clock(monkeypatch):
    """Controllable monotonic clock so TTL expiry needs no real waiting."""
    now = {"t": 1000.0}
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now["t"])
    return now


# ── basic hit/miss ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_result_is_cached():
    calls = []

    @timed_cache(ttl_seconds=60, group="g")
    async def f(x):
        calls.append(x)
        return x * 2

    assert await f(2) == 4
    assert await f(2) == 4
    assert len(calls) == 1
    stats = get_cache_stats()
    assert stats["hits"] == 1 and stats["misses"] == 1


def test_sync_result_is_cached():
    calls = []

    @timed_cache(ttl_seconds=60, group="g")
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert len(calls) == 1


def test_distinct_arguments_are_cached_separately():
    calls = []

    @timed_cache(ttl_seconds=60, group="g")
    def f(x, y=0):
        calls.append((x, y))
        return x + y

    f(1)
    f(2)
    f(1, y=5)
    f(1)
    assert len(calls) == 3  # only the repeat of f(1) hit


# ── TTL expiry ──────────────────────────────────────────────────────────


def test_entry_expires_after_ttl(clock):
    calls = []

    @timed_cache(ttl_seconds=10, group="g")
    def f():
        calls.append(1)
        return "v"

    f()
    clock["t"] += 5
    f()
    assert len(calls) == 1  # still fresh
    clock["t"] += 6  # now 11s old, past the 10s TTL
    f()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_async_entry_expires_after_ttl(clock):
    calls = []

    @timed_cache(ttl_seconds=10, group="g")
    async def f():
        calls.append(1)
        return "v"

    await f()
    clock["t"] += 11
    await f()
    assert len(calls) == 2


# ── invalidation ────────────────────────────────────────────────────────


def test_invalidate_clears_only_the_named_group():
    @timed_cache(ttl_seconds=60, group="a")
    def fa():
        return "a"

    @timed_cache(ttl_seconds=60, group="b")
    def fb():
        return "b"

    fa()
    fb()
    assert invalidate_cache("a") == 1
    assert get_cache_stats()["entries"] == 1
    assert invalidate_cache("a") == 0  # already gone
    assert invalidate_cache("b") == 1


def test_invalidation_counter_only_moves_on_real_clears():
    @timed_cache(ttl_seconds=60, group="a")
    def fa():
        return "a"

    fa()
    invalidate_cache("nonexistent")
    assert get_cache_stats()["invalidations"] == 0
    invalidate_cache("a")
    assert get_cache_stats()["invalidations"] == 1


# ── LRU bound ───────────────────────────────────────────────────────────


def test_cache_is_bounded_and_evicts_oldest(monkeypatch):
    monkeypatch.setattr(cache_mod, "MAX_CACHE_SIZE", 3)

    @timed_cache(ttl_seconds=600, group="g")
    def f(x):
        return x

    for i in range(10):
        f(i)

    # Never grows past the bound (eviction happens before insert).
    assert get_cache_stats()["entries"] <= 3


def test_hit_refreshes_lru_position(monkeypatch):
    monkeypatch.setattr(cache_mod, "MAX_CACHE_SIZE", 3)
    calls = []

    @timed_cache(ttl_seconds=600, group="g")
    def f(x):
        calls.append(x)
        return x

    f(1)
    f(2)
    f(1)  # hit — moves key 1 to the most-recent end
    f(3)
    f(4)  # forces an eviction; key 2 is the stale one, not key 1

    calls.clear()
    f(1)
    assert calls == []  # key 1 survived


# ── stats ───────────────────────────────────────────────────────────────


def test_stats_shape_and_hit_rate():
    @timed_cache(ttl_seconds=60, group="g")
    def f():
        return 1

    f()
    f()
    f()
    stats = get_cache_stats()
    assert stats["hits"] == 2 and stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(66.7)
    assert stats["entries"] == 1


def test_stats_hit_rate_is_zero_when_unused():
    assert get_cache_stats()["hit_rate"] == 0
