from token_replacer.circuit import CLOSED, HALF_OPEN, OPEN, CircuitBreaker, is_unhealthy_status


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_opens_after_threshold():
    clock = FakeClock()
    breaker = CircuitBreaker("up", failure_threshold=3, cooldown_s=30, clock=clock)
    for _ in range(3):
        assert breaker.allow()
        breaker.record_failure()
    assert breaker.state == OPEN
    assert not breaker.allow()


def test_half_opens_after_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker("up", failure_threshold=2, cooldown_s=30, clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == OPEN
    clock.advance(31)
    assert breaker.state == HALF_OPEN
    assert breaker.allow()


def test_success_closes():
    clock = FakeClock()
    breaker = CircuitBreaker("up", failure_threshold=2, cooldown_s=30, clock=clock)
    breaker.record_failure()
    breaker.record_failure()
    clock.advance(31)
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == CLOSED
    assert breaker.failures == 0


def test_below_threshold_stays_closed():
    breaker = CircuitBreaker("up", failure_threshold=5)
    for _ in range(4):
        breaker.record_failure()
    assert breaker.state == CLOSED
    assert breaker.allow()


def test_client_errors_are_not_provider_faults():
    """A 400 or 401 is the caller's mistake; it must not black out a provider."""
    for status in (200, 201, 400, 401, 403, 404, 422):
        assert not is_unhealthy_status(status)
    for status in (429, 500, 502, 503, 504, 529):
        assert is_unhealthy_status(status)
