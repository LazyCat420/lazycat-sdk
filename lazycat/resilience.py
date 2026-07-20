"""
Resilience decorators — uniform retry/backoff for LLM and external calls.

Gives every tool call and LLM call a configurable recovery wrapper as a
first-class primitive, replacing scattered per-file retry logic.

Usage:
    from lazycat.resilience import aresilient_call, resilient_call

    @aresilient_call(retries=3, backoff="exponential")
    async def call_llm(...):
        ...

    @resilient_call(retries=3, backoff="exponential")
    def fetch_data(...):
        ...

    # With fallback
    @aresilient_call(retries=2, on_failure=lambda *a, **kw: {"action": "HOLD"})
    async def risky_call(...):
        ...

Applications customize two things without editing this module:

    NON_RETRYABLE_EXCEPTION_NAMES.add("MyDoomLoopException")   # stop immediately
    set_failure_emitter(my_emit_fn)                            # monitoring hook
"""

import asyncio
import functools
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "FailureType",
    "AttemptRecord",
    "ResilientCallError",
    "classify_exception",
    "aresilient_call",
    "resilient_call",
    "set_failure_emitter",
    "NON_RETRYABLE_EXCEPTION_NAMES",
]


# ── Application-supplied policy ─────────────────────────────────────────

# Exception class names that must never be retried, by name so applications can
# register their own without this module importing them.
NON_RETRYABLE_EXCEPTION_NAMES: set[str] = set()

_failure_emitter: Callable | None = None


def set_failure_emitter(fn: Callable | None) -> None:
    """Register a monitoring hook invoked on every failed attempt.

    The callable receives (func_name, attempt, max_attempts, failure_type,
    exc, elapsed_ms). Exceptions raised by the emitter are swallowed — the
    retry path must never fail because of monitoring.
    """
    global _failure_emitter
    _failure_emitter = fn


# ── Failure Classification ──────────────────────────────────────────────


class FailureType(str, Enum):
    """Categorizes failures for the recovery engine.

    TRANSIENT   — network blip, timeout, rate limit. Action: retry with backoff.
    DEGRADED    — LLM returned empty/invalid/low-confidence output.
                  Action: retry with simplified prompt or reduced context.
    FATAL       — data missing entirely, unrecoverable parse error, DB failure.
                  Action: skip this unit of work, log full context, continue.
    """

    TRANSIENT = "transient"
    DEGRADED = "degraded"
    FATAL = "fatal"


@dataclass
class AttemptRecord:
    """Record of a single retry attempt."""

    attempt: int
    exception_type: str
    exception_msg: str
    failure_type: FailureType
    elapsed_ms: int
    timestamp: float


class ResilientCallError(Exception):
    """Raised when all retries are exhausted.

    Contains the full attempt history for debugging and the recovery engine.
    """

    def __init__(
        self,
        message: str,
        attempts: list[AttemptRecord],
        last_failure_type: FailureType,
        func_name: str = "",
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_failure_type = last_failure_type
        self.func_name = func_name

    def __str__(self):
        return (
            f"ResilientCallError({self.func_name}): {super().__str__()} "
            f"[{len(self.attempts)} attempts, last_type={self.last_failure_type.value}]"
        )


# ── Exception Classifier ───────────────────────────────────────────────


def classify_exception(exc: Exception) -> FailureType:
    """Map common exceptions to a FailureType.

    This classification drives the recovery decision:
    - TRANSIENT failures get retried with backoff
    - DEGRADED failures get retried with simplified context
    - FATAL failures are logged and skipped
    """
    exc_msg = str(exc).lower()

    # ── httpx network errors → TRANSIENT ──
    try:
        import httpx

        if isinstance(
            exc,
            (
                httpx.TimeoutException,
                # NetworkError covers ConnectError, ReadError, WriteError,
                # CloseError — dropped connections (e.g. a gateway container
                # restarting mid-stream) recover within a minute and must be
                # retried, not treated as fatal.
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        ):
            return FailureType.TRANSIENT

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            # Rate limit or service unavailable → TRANSIENT
            if status in (429, 502, 503, 504):
                return FailureType.TRANSIENT
            # Bad request or unprocessable → DEGRADED (likely bad prompt)
            if status in (400, 422):
                return FailureType.DEGRADED
            # Not found → FATAL (model/endpoint missing)
            if status == 404:
                return FailureType.FATAL
            # Server error → TRANSIENT (might recover)
            if 500 <= status < 600:
                return FailureType.TRANSIENT
    except ImportError:
        pass

    # ── asyncio errors → TRANSIENT ──
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
        return FailureType.TRANSIENT

    # ── vLLM node blips → TRANSIENT ──
    # Model resolution raises bare RuntimeError("VLLM endpoint offline…");
    # without this branch it fell through to FATAL and killed the agent turn
    # after 2 attempts while a proxy blip got 5 retries.
    if isinstance(exc, RuntimeError) and (
        "offline" in exc_msg or "no models found" in exc_msg
    ):
        return FailureType.TRANSIENT

    # ── Application-registered non-retryables → FATAL ──
    if type(exc).__name__ in NON_RETRYABLE_EXCEPTION_NAMES:
        return FailureType.FATAL

    # ── JSON/parse errors → DEGRADED (LLM output was malformed) ──
    if isinstance(exc, (json.JSONDecodeError, KeyError)):
        return FailureType.DEGRADED

    # ── Model not found → FATAL ──
    if isinstance(exc, ValueError) and "not found" in exc_msg:
        return FailureType.FATAL
    if isinstance(exc, ValueError) and "not hosted" in exc_msg:
        return FailureType.FATAL

    # ── Everything else → FATAL ──
    return FailureType.FATAL


# Backwards-compatible private alias (this function was _classify_exception).
_classify_exception = classify_exception


# ── Backoff Calculators ─────────────────────────────────────────────────


def _calculate_delay(
    attempt: int,
    backoff: str,
    base_delay: float,
    max_delay: float,
) -> float:
    """Calculate the delay before the next retry attempt."""
    if backoff == "none":
        return 0.0
    elif backoff == "linear":
        delay = base_delay * attempt
    else:  # exponential (default)
        delay = base_delay * (2 ** (attempt - 1))
    return min(delay, max_delay)


# ── Emit Helper (non-blocking) ──────────────────────────────────────────


def _emit_failure_event(
    func_name: str,
    attempt: int,
    max_attempts: int,
    failure_type: FailureType,
    exc: Exception,
    elapsed_ms: int,
):
    """Forward a structured failure event to the registered emitter, if any.

    Non-blocking — if no emitter is registered or it raises, we continue.
    The resilience decorator must never fail because of monitoring.
    """
    if _failure_emitter is None:
        return
    try:
        _failure_emitter(
            func_name, attempt, max_attempts, failure_type, exc, elapsed_ms
        )
    except Exception:
        pass  # Monitoring must never break the call path


def _should_stop(exc: Exception, failure_type: FailureType, attempt: int) -> bool:
    """Whether to abandon retries for this exception.

    NOTE the `attempt > 1` condition: a FATAL classification on the FIRST
    attempt is still retried once. This is long-standing behavior that callers
    depend on (a misclassified first failure gets a second chance); it is
    preserved deliberately rather than "fixed".
    """
    if type(exc).__name__ in NON_RETRYABLE_EXCEPTION_NAMES:
        return True
    return failure_type == FailureType.FATAL and attempt > 1


# ── Async Resilient Call Decorator ──────────────────────────────────────


def aresilient_call(
    retries: int = 3,
    backoff: str = "exponential",
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_failure: Callable | None = None,
    failure_type_hint: str | None = None,
    retryable_types: tuple[type, ...] | None = None,
):
    """Async decorator that wraps a function with structured retry behavior.

    Args:
        retries: Maximum number of attempts (default 3).
        backoff: Backoff strategy — "exponential", "linear", or "none".
        base_delay: Initial wait in seconds (default 1.0).
        max_delay: Ceiling on wait time (default 30.0).
        on_failure: Optional fallback called if all retries are exhausted.
                    Receives the same args/kwargs as the original function.
                    If provided and it returns a value, that value is returned
                    instead of raising ResilientCallError.
        failure_type_hint: Optional tag emitted with failure events.
        retryable_types: If provided, only retry on these exception types.
                         Others are re-raised immediately.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempts: list[AttemptRecord] = []
            func_name = func.__qualname__
            last_failure_type = FailureType.FATAL

            for attempt in range(1, retries + 1):
                start = time.monotonic()
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    elapsed = int((time.monotonic() - start) * 1000)
                    failure_type = classify_exception(exc)
                    last_failure_type = failure_type

                    # If retryable_types specified, only retry those
                    if retryable_types and not isinstance(exc, retryable_types):
                        logger.warning(
                            "[RESILIENCE] %s attempt %d/%d: non-retryable %s — re-raising",
                            func_name,
                            attempt,
                            retries,
                            type(exc).__name__,
                        )
                        raise

                    record = AttemptRecord(
                        attempt=attempt,
                        exception_type=type(exc).__name__,
                        exception_msg=str(exc)[:300],
                        failure_type=failure_type,
                        elapsed_ms=elapsed,
                        timestamp=time.time(),
                    )

                    if _should_stop(exc, failure_type, attempt):
                        logger.warning(
                            "[RESILIENCE] %s attempt %d/%d: unrecoverable %s — stopping retries",
                            func_name,
                            attempt,
                            retries,
                            type(exc).__name__,
                        )
                        attempts.append(record)
                        break

                    attempts.append(record)

                    logger.warning(
                        "[RESILIENCE] %s attempt %d/%d failed: %s [%s] (%dms)",
                        func_name,
                        attempt,
                        retries,
                        type(exc).__name__,
                        failure_type.value,
                        elapsed,
                    )

                    _emit_failure_event(
                        func_name, attempt, retries, failure_type, exc, elapsed
                    )

                    # Wait before next attempt (unless last attempt)
                    if attempt < retries:
                        delay = _calculate_delay(
                            attempt, backoff, base_delay, max_delay
                        )
                        if delay > 0:
                            logger.info(
                                "[RESILIENCE] %s: waiting %.1fs before retry %d/%d",
                                func_name,
                                delay,
                                attempt + 1,
                                retries,
                            )
                            await asyncio.sleep(delay)

            # All retries exhausted
            logger.error(
                "[RESILIENCE] %s: all %d attempts failed (last: %s)",
                func_name,
                retries,
                last_failure_type.value,
            )

            if on_failure is not None:
                try:
                    fallback_result = on_failure(*args, **kwargs)
                    if asyncio.iscoroutine(fallback_result):
                        fallback_result = await fallback_result
                    logger.info(
                        "[RESILIENCE] %s: on_failure callback returned result", func_name
                    )
                    return fallback_result
                except Exception as fb_exc:
                    logger.error(
                        "[RESILIENCE] %s: on_failure callback also failed: %s",
                        func_name,
                        fb_exc,
                    )

            raise ResilientCallError(
                message=f"All {retries} attempts failed for {func_name}",
                attempts=attempts,
                last_failure_type=last_failure_type,
                func_name=func_name,
            )

        # Attach metadata for introspection
        wrapper._resilience_config = {
            "retries": retries,
            "backoff": backoff,
            "base_delay": base_delay,
            "max_delay": max_delay,
            "failure_type_hint": failure_type_hint,
        }
        return wrapper

    return decorator


# ── Sync Resilient Call Decorator ───────────────────────────────────────


def resilient_call(
    retries: int = 3,
    backoff: str = "exponential",
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_failure: Callable | None = None,
    failure_type_hint: str | None = None,
    retryable_types: tuple[type, ...] | None = None,
):
    """Sync version of aresilient_call for non-async functions.

    Same parameters and behavior as aresilient_call, but uses time.sleep()
    instead of asyncio.sleep() for backoff delays.

    NOTE: unlike the async variant, a failing on_failure callback here is
    swallowed silently before ResilientCallError is raised. Preserved as-is.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts: list[AttemptRecord] = []
            func_name = func.__qualname__
            last_failure_type = FailureType.FATAL

            for attempt in range(1, retries + 1):
                start = time.monotonic()
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    elapsed = int((time.monotonic() - start) * 1000)
                    failure_type = classify_exception(exc)
                    last_failure_type = failure_type

                    if retryable_types and not isinstance(exc, retryable_types):
                        raise

                    record = AttemptRecord(
                        attempt=attempt,
                        exception_type=type(exc).__name__,
                        exception_msg=str(exc)[:300],
                        failure_type=failure_type,
                        elapsed_ms=elapsed,
                        timestamp=time.time(),
                    )

                    if _should_stop(exc, failure_type, attempt):
                        attempts.append(record)
                        break

                    attempts.append(record)

                    logger.warning(
                        "[RESILIENCE] %s attempt %d/%d failed: %s [%s] (%dms)",
                        func_name,
                        attempt,
                        retries,
                        type(exc).__name__,
                        failure_type.value,
                        elapsed,
                    )

                    _emit_failure_event(
                        func_name, attempt, retries, failure_type, exc, elapsed
                    )

                    if attempt < retries:
                        delay = _calculate_delay(
                            attempt, backoff, base_delay, max_delay
                        )
                        if delay > 0:
                            time.sleep(delay)

            # All retries exhausted
            if on_failure is not None:
                try:
                    return on_failure(*args, **kwargs)
                except Exception:
                    pass

            raise ResilientCallError(
                message=f"All {retries} attempts failed for {func_name}",
                attempts=attempts,
                last_failure_type=last_failure_type,
                func_name=func_name,
            )

        wrapper._resilience_config = {
            "retries": retries,
            "backoff": backoff,
            "base_delay": base_delay,
            "max_delay": max_delay,
            "failure_type_hint": failure_type_hint,
        }
        return wrapper

    return decorator
