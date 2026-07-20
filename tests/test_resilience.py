"""
Tests for lazycat.resilience.

Several behaviors here are quirky but long-standing (a FATAL classification on
the FIRST attempt is still retried once; a failing sync on_failure is swallowed
silently). They are pinned deliberately — callers depend on them, so they are
characterized rather than "fixed".
"""
import asyncio
import json

import pytest

from lazycat import resilience
from lazycat.resilience import (
    AttemptRecord,
    FailureType,
    ResilientCallError,
    aresilient_call,
    classify_exception,
    resilient_call,
    set_failure_emitter,
)


@pytest.fixture(autouse=True)
def _reset_module_policy():
    """Keep application-level policy from leaking between tests."""
    resilience.NON_RETRYABLE_EXCEPTION_NAMES.clear()
    set_failure_emitter(None)
    yield
    resilience.NON_RETRYABLE_EXCEPTION_NAMES.clear()
    set_failure_emitter(None)


@pytest.fixture
def no_sleep(monkeypatch):
    """Make backoff instantaneous while recording the delays requested."""
    slept: list[float] = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(resilience.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(resilience.time, "sleep", lambda d: slept.append(d))
    return slept


# ── classification ──────────────────────────────────────────────────────


def test_classifies_transient_and_degraded_and_fatal():
    assert classify_exception(asyncio.TimeoutError()) is FailureType.TRANSIENT
    assert classify_exception(ConnectionError()) is FailureType.TRANSIENT
    assert classify_exception(KeyError("k")) is FailureType.DEGRADED
    assert classify_exception(ValueError("model not found")) is FailureType.FATAL
    assert classify_exception(RuntimeError("something odd")) is FailureType.FATAL


def test_vllm_offline_runtimeerror_is_transient():
    # Model resolution raises a bare RuntimeError; without this branch a node
    # blip was treated as fatal and killed the turn after two attempts.
    assert classify_exception(RuntimeError("VLLM endpoint offline")) is FailureType.TRANSIENT
    assert classify_exception(RuntimeError("no models found")) is FailureType.TRANSIENT


def test_registered_non_retryable_name_classifies_fatal():
    class DoomLoopException(Exception):
        pass

    assert classify_exception(DoomLoopException()) is FailureType.FATAL  # default anyway
    resilience.NON_RETRYABLE_EXCEPTION_NAMES.add("DoomLoopException")
    assert classify_exception(DoomLoopException()) is FailureType.FATAL


# ── async retry behavior ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retries_transient_then_succeeds(no_sleep):
    calls = []

    @aresilient_call(retries=3)
    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise asyncio.TimeoutError()
        return "ok"

    assert await flaky() == "ok"
    assert len(calls) == 3
    assert no_sleep == [1.0, 2.0]  # exponential


@pytest.mark.asyncio
async def test_exhausted_retries_raise_with_attempt_history(no_sleep):
    @aresilient_call(retries=3)
    async def always_transient():
        raise asyncio.TimeoutError()

    with pytest.raises(ResilientCallError) as exc:
        await always_transient()
    assert len(exc.value.attempts) == 3
    assert all(isinstance(a, AttemptRecord) for a in exc.value.attempts)
    assert exc.value.last_failure_type is FailureType.TRANSIENT
    assert "always_transient" in str(exc.value)


@pytest.mark.asyncio
async def test_fatal_is_retried_once_then_stops(no_sleep):
    # Characterization of the `attempt > 1` condition: a FATAL on the first
    # attempt still gets a second chance, then stops short of the budget.
    calls = []

    @aresilient_call(retries=5)
    async def fatal():
        calls.append(1)
        raise ValueError("model not found")

    with pytest.raises(ResilientCallError):
        await fatal()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_registered_non_retryable_stops_immediately(no_sleep):
    class DoomLoopException(Exception):
        pass

    resilience.NON_RETRYABLE_EXCEPTION_NAMES.add("DoomLoopException")
    calls = []

    @aresilient_call(retries=5)
    async def doomed():
        calls.append(1)
        raise DoomLoopException("loop")

    with pytest.raises(ResilientCallError):
        await doomed()
    assert len(calls) == 1  # not retried even once


@pytest.mark.asyncio
async def test_retryable_types_reraises_others_immediately(no_sleep):
    calls = []

    @aresilient_call(retries=3, retryable_types=(TimeoutError,))
    async def picky():
        calls.append(1)
        raise KeyError("nope")

    with pytest.raises(KeyError):
        await picky()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_on_failure_sync_and_async_fallbacks(no_sleep):
    @aresilient_call(retries=2, on_failure=lambda: {"action": "HOLD"})
    async def sync_fb():
        raise asyncio.TimeoutError()

    assert await sync_fb() == {"action": "HOLD"}

    async def async_fallback():
        return {"action": "ASYNC"}

    @aresilient_call(retries=2, on_failure=async_fallback)
    async def async_fb():
        raise asyncio.TimeoutError()

    assert await async_fb() == {"action": "ASYNC"}


@pytest.mark.asyncio
async def test_failing_on_failure_still_raises_resilient_error(no_sleep):
    def boom():
        raise RuntimeError("fallback broke")

    @aresilient_call(retries=2, on_failure=boom)
    async def f():
        raise asyncio.TimeoutError()

    with pytest.raises(ResilientCallError):
        await f()


# ── backoff strategies ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backoff_strategies_and_max_delay(no_sleep):
    @aresilient_call(retries=4, backoff="linear", base_delay=2.0)
    async def lin():
        raise asyncio.TimeoutError()

    with pytest.raises(ResilientCallError):
        await lin()
    assert no_sleep == [2.0, 4.0, 6.0]

    no_sleep.clear()

    @aresilient_call(retries=3, backoff="none")
    async def none_():
        raise asyncio.TimeoutError()

    with pytest.raises(ResilientCallError):
        await none_()
    assert no_sleep == []  # zero delays are never awaited

    no_sleep.clear()

    @aresilient_call(retries=4, backoff="exponential", base_delay=10.0, max_delay=15.0)
    async def capped():
        raise asyncio.TimeoutError()

    with pytest.raises(ResilientCallError):
        await capped()
    assert no_sleep == [10.0, 15.0, 15.0]  # capped


# ── monitoring hook ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failure_emitter_receives_events(no_sleep):
    seen = []
    set_failure_emitter(lambda *a: seen.append(a))

    @aresilient_call(retries=2)
    async def f():
        raise asyncio.TimeoutError()

    with pytest.raises(ResilientCallError):
        await f()

    assert len(seen) == 2
    func_name, attempt, max_attempts, failure_type, exc, elapsed_ms = seen[0]
    assert attempt == 1 and max_attempts == 2
    assert failure_type is FailureType.TRANSIENT
    assert isinstance(elapsed_ms, int)


@pytest.mark.asyncio
async def test_emitter_exceptions_never_break_the_call_path(no_sleep):
    def bad_emitter(*a):
        raise RuntimeError("monitoring is down")

    set_failure_emitter(bad_emitter)
    calls = []

    @aresilient_call(retries=2)
    async def f():
        calls.append(1)
        if len(calls) < 2:
            raise asyncio.TimeoutError()
        return "ok"

    assert await f() == "ok"


# ── sync variant ────────────────────────────────────────────────────────


def test_sync_retries_then_succeeds(no_sleep):
    calls = []

    @resilient_call(retries=3)
    def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise ConnectionError()
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 2


def test_sync_exhausted_raises(no_sleep):
    @resilient_call(retries=2)
    def always():
        raise ConnectionError()

    with pytest.raises(ResilientCallError):
        always()


def test_sync_failing_on_failure_is_swallowed_then_raises(no_sleep):
    def boom():
        raise RuntimeError("fallback broke")

    @resilient_call(retries=2, on_failure=boom)
    def f():
        raise ConnectionError()

    # Characterization: the sync path swallows the fallback error silently
    # (unlike the async path, which logs it) and raises the original error.
    with pytest.raises(ResilientCallError):
        f()


def test_resilience_config_is_introspectable():
    @aresilient_call(retries=7, backoff="linear")
    async def f():
        return 1

    assert f._resilience_config["retries"] == 7
    assert f._resilience_config["backoff"] == "linear"
