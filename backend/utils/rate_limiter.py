"""Async token-bucket rate limiter for per-source API calls."""

import asyncio
import time
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class AsyncRateLimiter:
    """Token-bucket rate limiter safe for use in async code.

    Limits how many requests per second can be made to a single API source.
    Call ``acquire()`` before each request; it will sleep as needed to respect
    the configured rate.
    """

    def __init__(self, requests_per_second: float, source: str = "unknown") -> None:
        """Initialise the limiter.

        Args:
            requests_per_second: Maximum number of requests allowed per second.
            source: Human-readable label used in log messages.
        """
        self._rate = requests_per_second
        self._source = source
        # Minimum gap in seconds between requests.
        self._min_interval: float = 1.0 / requests_per_second
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until the next request is permitted, then proceed."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait = self._min_interval - elapsed
            if wait > 0:
                logger.debug(
                    "rate limiter sleeping",
                    source=self._source,
                    sleep_seconds=round(wait, 3),
                )
                await asyncio.sleep(wait)
            self._last_request_time = time.monotonic()

    async def __aenter__(self) -> "AsyncRateLimiter":
        """Support use as an async context manager."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        """No teardown needed."""
