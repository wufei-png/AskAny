"""Timing utilities for metrics collection."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from prometheus_client import Histogram

from askany.metrics.registry import get_metrics

P = ParamSpec("P")
T = TypeVar("T")


class Timer:
    """Context manager for timing operations with a Prometheus histogram."""

    __slots__ = ("_histogram", "_labels", "_start_time", "_end_time")

    def __init__(self, histogram: Histogram, **labels: str) -> None:
        """Initialize timer with histogram and labels.

        Args:
            histogram: Prometheus histogram to record duration
            **labels: Label values for the histogram
        """
        self._histogram = histogram
        self._labels = labels
        self._start_time: float | None = None
        self._end_time: float | None = None

    def __enter__(self) -> Timer:
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record to histogram."""
        if self._start_time is not None:
            self._end_time = time.perf_counter()
            duration = self._end_time - self._start_time
            self._histogram.labels(**self._labels).observe(duration)

    @property
    def elapsed(self) -> float | None:
        """Get elapsed time in seconds, or None if not finished."""
        if self._start_time is None:
            return None
        if self._end_time is None:
            return time.perf_counter() - self._start_time
        return self._end_time - self._start_time


def timed_operation(
    histogram_name: str,
    **labels: str,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to time function execution and record to a Prometheus histogram.

    Works with both sync and async functions.

    Args:
        histogram_name: Name of the histogram attribute in MetricsRegistry
        **labels: Static label values to use for all recordings

    Returns:
        Decorated function

    Example:
        @timed_operation("askany_llm_request_duration_seconds", model="gpt-4", operation="complete")
        async def call_llm(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            metrics = get_metrics()
            histogram = metrics.get_histogram(histogram_name)
            if histogram is None:
                return func(*args, **kwargs)

            timer = Timer(histogram, **labels)
            with timer:
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            metrics = get_metrics()
            histogram = metrics.get_histogram(histogram_name)
            if histogram is None:
                return await func(*args, **kwargs)

            timer = Timer(histogram, **labels)
            with timer:
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def timer_context(
    histogram_name: str,
    **labels: str,
) -> Timer:
    """Context manager to time operations and record to a Prometheus histogram.

    Args:
        histogram_name: Name of the histogram attribute in MetricsRegistry
        **labels: Label values for the histogram

    Yields:
        Timer instance

    Example:
        with timer_context("askany_db_query_duration_seconds", table="faq_vectors", operation="select"):
            results = db.query("SELECT ...")
    """
    metrics = get_metrics()
    histogram = metrics.get_histogram(histogram_name)
    if histogram is None:
        # Return a dummy timer that does nothing
        yield Timer(Histogram("dummy"))  # type: ignore[arg-type]
        return

    timer = Timer(histogram, **labels)
    yield timer


def get_timer(histogram_name: str, **labels: str) -> Timer:
    """Create a Timer instance for manual timing.

    You must call __enter__ and __exit__ manually, or use as context manager.

    Args:
        histogram_name: Name of the histogram attribute in MetricsRegistry
        **labels: Label values for the histogram

    Returns:
        Timer instance

    Example:
        timer = get_timer("askany_custom_duration_seconds", operation="custom")
        timer.__enter__()
        try:
            do_work()
        finally:
            timer.__exit__(None, None, None)
    """
    metrics = get_metrics()
    histogram = metrics.get_histogram(histogram_name)
    if histogram is None:
        raise ValueError(f"Histogram '{histogram_name}' not found in registry")

    return Timer(histogram, **labels)
