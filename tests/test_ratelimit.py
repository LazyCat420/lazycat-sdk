"""
Tests for lazycat.ratelimit — keyed min-interval and keyed concurrency caps.
"""
import asyncio

import pytest

from lazycat.ratelimit import KeyedRateLimiter, KeyedSemaphore


# ── KeyedRateLimiter ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_min_interval_is_enforced_between_calls_on_one_key():
    # 50 req/s → 20ms minimum spacing. Real sleeps, kept tiny.
    limiter = KeyedRateLimiter({"a": 50.0})
    stamps = []

    for _ in range(3):
        async with limiter.acquire("a"):
            stamps.append(asyncio.get_event_loop().time())

    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(g >= 0.015 for g in gaps), gaps


@pytest.mark.asyncio
async def test_different_keys_do_not_block_each_other():
    limiter = KeyedRateLimiter({"slow": 2.0}, default_rate=1000.0)
    start = asyncio.get_event_loop().time()

    async with limiter.acquire("slow"):
        pass
    # A different key must not wait behind "slow"'s 500ms interval.
    async with limiter.acquire("other"):
        pass

    assert asyncio.get_event_loop().time() - start < 0.2


@pytest.mark.asyncio
async def test_default_rate_applies_to_unknown_keys():
    limiter = KeyedRateLimiter({}, default_rate=1000.0)
    async with limiter.acquire("anything"):
        pass  # would hang noticeably at a slow default


@pytest.mark.asyncio
async def test_set_rate_overrides_a_key():
    limiter = KeyedRateLimiter({}, default_rate=1000.0)
    limiter.set_rate("x", 500.0)
    async with limiter.acquire("x"):
        pass
    assert limiter._rates["x"] == 500.0


@pytest.mark.asyncio
async def test_rates_dict_is_held_by_reference():
    # The trading shim keeps its DOMAIN_LIMITS table module-side and expects
    # runtime edits to take effect without rebuilding the limiter.
    rates = {"a": 1000.0}
    limiter = KeyedRateLimiter(rates, default_rate=1000.0)
    rates["b"] = 250.0
    assert limiter._rates["b"] == 250.0


# ── KeyedSemaphore ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrency_is_capped_per_key():
    sem = KeyedSemaphore({"api": 2})
    in_flight = 0
    peak = 0

    async def worker():
        nonlocal in_flight, peak
        async with sem.acquire("api"):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(10)))
    assert peak == 2


@pytest.mark.asyncio
async def test_default_limit_used_for_unknown_keys():
    sem = KeyedSemaphore({}, default_limit=1)
    in_flight = 0
    peak = 0

    async def worker():
        nonlocal in_flight, peak
        async with sem.acquire("unknown"):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.005)
            in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(4)))
    assert peak == 1


@pytest.mark.asyncio
async def test_separate_keys_get_separate_budgets():
    sem = KeyedSemaphore({"a": 1, "b": 1})
    order = []

    async def worker(key):
        async with sem.acquire(key):
            order.append(f"{key}-in")
            await asyncio.sleep(0.01)
            order.append(f"{key}-out")

    await asyncio.gather(worker("a"), worker("b"))
    # Both entered before either exited — they did not serialize.
    assert order[:2] == ["a-in", "b-in"] or order[:2] == ["b-in", "a-in"]


@pytest.mark.asyncio
async def test_burst_mode_bypasses_the_cap():
    sem = KeyedSemaphore({"api": 1})
    sem.enable_burst_mode(True)
    in_flight = 0
    peak = 0

    async def worker():
        nonlocal in_flight, peak
        async with sem.acquire("api"):
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(5)))
    assert peak == 5

    sem.enable_burst_mode(False)
    assert sem._burst_mode is False


@pytest.mark.asyncio
async def test_limit_decorator_applies_the_cap():
    sem = KeyedSemaphore({"api": 2})
    in_flight = 0
    peak = 0

    @sem.limit("api")
    async def work():
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "done"

    results = await asyncio.gather(*(work() for _ in range(6)))
    assert peak == 2
    assert results == ["done"] * 6


@pytest.mark.asyncio
async def test_status_reports_configured_and_live_semaphores():
    sem = KeyedSemaphore({"a": 3, "b": 2})

    # Before use: reported from config, nothing in flight.
    assert sem.status()["a"] == {"max": 3, "available": 3, "in_use": 0}

    async with sem.acquire("a"):
        live = sem.status()
    assert live["a"]["max"] == 3
    assert live["a"]["in_use"] == 1
    assert live["b"] == {"max": 2, "available": 2, "in_use": 0}


@pytest.mark.asyncio
async def test_slot_is_released_when_the_body_raises():
    sem = KeyedSemaphore({"api": 1})

    with pytest.raises(RuntimeError):
        async with sem.acquire("api"):
            raise RuntimeError("boom")

    # If the slot leaked, this would deadlock rather than return.
    await asyncio.wait_for(sem.acquire("api").__aenter__(), timeout=0.5)
