from token_replacer.policy import Policy, Tier, decide
from token_replacer.tokens import estimate_request

POLICY = Policy(max_local_context=32_000, safety_margin=0.9)  # budget = 28,800


def test_small_request_goes_local():
    body = {"messages": [{"role": "user", "content": "why is my docker build so slow"}]}
    decision = decide(body, POLICY)
    assert decision.tier is Tier.LOCAL
    assert decision.reason == "fits_local_budget"


def test_large_request_goes_upstream():
    body = {"messages": [{"role": "user", "content": "x" * 200_000}]}
    decision = decide(body, POLICY)
    assert decision.tier is Tier.UPSTREAM
    assert decision.reason == "context_too_large"


def test_budget_boundary_is_respected():
    policy = Policy(max_local_context=8_000, safety_margin=1.0)
    # 3.6 chars/token -> ~7,999 tokens, just inside
    just_inside = {"messages": [{"role": "user", "content": "x" * 28_000}]}
    just_outside = {"messages": [{"role": "user", "content": "x" * 30_000}]}
    assert decide(just_inside, policy).tier is Tier.LOCAL
    assert decide(just_outside, policy).tier is Tier.UPSTREAM


def test_safety_margin_shrinks_the_budget():
    policy = Policy(max_local_context=10_000, safety_margin=0.5)
    assert policy.local_token_budget == 5_000


def test_tools_are_upstream_by_default():
    body = {
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [{"type": "function", "function": {"name": "ls", "parameters": {}}}],
    }
    assert decide(body, POLICY).reason == "tools_requested"


def test_tools_can_be_enabled_locally():
    body = {
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [{"type": "function", "function": {"name": "ls", "parameters": {}}}],
    }
    policy = Policy(allow_tools_local=True)
    assert decide(body, policy).tier is Tier.LOCAL


def test_image_input_stays_upstream():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            }
        ]
    }
    assert decide(body, POLICY).reason == "image_input"


def test_unhealthy_local_sends_traffic_upstream():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    decision = decide(body, POLICY, local_healthy=False)
    assert decision.tier is Tier.UPSTREAM
    assert decision.reason == "local_unhealthy"


def test_context_gate_beats_health():
    """A request that cannot fit locally must not try, even when local is healthy
    and upstream is down: the local model would fail on memory, not on quality."""
    body = {"messages": [{"role": "user", "content": "x" * 300_000}]}
    decision = decide(body, POLICY, local_healthy=True, upstream_healthy=False)
    assert decision.reason == "context_too_large"


def test_cjk_counts_heavier_than_ascii():
    ascii_body = {"messages": [{"role": "user", "content": "a" * 1000}]}
    cjk_body = {"messages": [{"role": "user", "content": "語" * 1000}]}
    assert estimate_request(cjk_body) > estimate_request(ascii_body) * 2


def test_estimate_is_monotonic():
    short = {"messages": [{"role": "user", "content": "a" * 100}]}
    long = {"messages": [{"role": "user", "content": "a" * 10_000}]}
    assert estimate_request(long) > estimate_request(short)
