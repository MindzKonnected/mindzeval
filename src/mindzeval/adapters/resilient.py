"""Timeouts, retries with backoff and a rate limit around any adapter."""

from __future__ import annotations

import asyncio
import random
import time

from mindzeval.adapters.base import Adapter, AgentError, AgentErrorKind, AgentRequest
from mindzeval.schema.response import AgentResponse


class ResilientAdapter(Adapter):
    def __init__(
        self,
        inner: Adapter,
        *,
        timeout_s: float | None = None,
        retries: int = 0,
        requests_per_minute: float | None = None,
        backoff_s: float = 1.0,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self._timeout = timeout_s
        self._retries = retries
        self._backoff = backoff_s
        self._interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()
        self.attempts = 0
        """Total calls made, including retries."""

    async def _wait_for_slot(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def call(self, request: AgentRequest) -> AgentResponse:
        attempt = 0
        while True:
            await self._wait_for_slot()
            self.attempts += 1
            try:
                if self._timeout:
                    return await asyncio.wait_for(self.inner.call(request), self._timeout)
                return await self.inner.call(request)
            except TimeoutError:
                error = AgentError(AgentErrorKind.TIMEOUT, f"no response within {self._timeout:g}s")
            except AgentError as exc:
                error = exc
            if not error.retryable or attempt >= self._retries:
                raise error
            delay = self._backoff * (2**attempt) * (1 + random.random() * 0.25)
            if error.kind is AgentErrorKind.RATE_LIMITED:
                delay = max(delay, 5.0 * (attempt + 1))
            attempt += 1
            await asyncio.sleep(delay)

    async def aclose(self) -> None:
        await self.inner.aclose()
