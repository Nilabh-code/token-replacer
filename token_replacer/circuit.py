"""Circuit breaker for one upstream.

The gateway's job is to stop sending traffic to a provider that is currently
failing. A caller that arrives while the breaker is open should be served by
the other tier instead of waiting out a 30-second upstream timeout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    name: str = "upstream"
    failure_threshold: int = 5
    cooldown_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = field(default=0, init=False)
    _state: str = field(default=CLOSED, init=False)
    _opened_at: float = field(default=0.0, init=False)

    def allow(self) -> bool:
        """May a request be attempted right now?"""
        if self._state == OPEN:
            if self.clock() - self._opened_at >= self.cooldown_s:
                self._state = HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._state = CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = OPEN
            self._opened_at = self.clock()

    @property
    def state(self) -> str:
        # Report open->half_open transition without mutating on read only if the
        # cooldown has elapsed; allow() performs the actual transition.
        if self._state == OPEN and self.clock() - self._opened_at >= self.cooldown_s:
            return HALF_OPEN
        return self._state

    @property
    def failures(self) -> int:
        return self._failures


# Status codes that mean "this provider is unwell", not "this request is bad".
# 429 and 5xx are provider health; 400/401/404 are the caller's problem and must
# not trip a breaker, or one malformed client would black out a provider.
UNHEALTHY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


def is_unhealthy_status(status: int) -> bool:
    return status in UNHEALTHY_STATUS
