
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional


logger = logging.getLogger("council.tracing")


@contextmanager
def trace_block(name: str, *, extra: Optional[dict] = None) -> Generator[None, None, None]:
    """
    Context manager that logs the start and end time of a code block.

    Example:
        with trace_block("run_debate", extra={"topic_id": topic.id}):
            result = orchestrator.run_debate(topic, council)
    """
    extra = extra or {}
    start = time.time()
    logger.debug("START %s | extra=%s", name, extra)
    try:
        yield
    finally:
        duration = time.time() - start
        logger.debug("END   %s | duration=%.3fs | extra=%s", name, duration, extra)


def traced(name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that wraps a function in a trace_block.

    Example:
        @traced("run_live_debate")
        def run_live_debate(...):
            ...

    If `name` is None, the function's __qualname__ is used.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        trace_name = name or func.__qualname__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_block(trace_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator