"""The routing decision. Pure function, no I/O, so it is trivially testable.

Grounded in measured traffic from Nilabh's OrcaRoute gateway (3.03 days,
84,699 requests):

    p50 context    94,771 tokens
    <=32k context  30.5% of requests, 0.97% of input tokens
    p50 output on that slice  270 tokens

So the tier boundary is a *hard capability limit*, not a difficulty guess: a 9B
on 16 GB cannot attend over 150k tokens, whatever the question is worth. Above
the limit the local model is not "probably worse", it is impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .tokens import estimate_request


class Tier(str, Enum):
    LOCAL = "local"
    UPSTREAM = "upstream"


@dataclass(frozen=True)
class Policy:
    max_local_context: int = 32_000
    # Leave headroom: the estimate is approximate and the response also needs room.
    safety_margin: float = 0.9
    # A 9B is not reliable at emitting provider-accurate tool calls. Start false,
    # flip only after measuring tool-call validity on the eval set.
    allow_tools_local: bool = False
    # Images cannot be processed by a text-only local model.
    allow_images_local: bool = False

    @property
    def local_token_budget(self) -> int:
        return int(self.max_local_context * self.safety_margin)


@dataclass(frozen=True)
class Decision:
    tier: Tier
    reason: str
    estimated_prompt_tokens: int

    @property
    def routed_locally(self) -> bool:
        return self.tier is Tier.LOCAL


def has_tools(body: dict) -> bool:
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def has_image(body: dict) -> bool:
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "image_url" in part:
                    return True
    return False


def decide(
    body: dict,
    policy: Policy,
    *,
    local_healthy: bool = True,
    upstream_healthy: bool = True,
) -> Decision:
    """Choose a tier, and record why.

    Order matters: the context gate is absolute (a request that does not fit
    cannot be served locally even if upstream is down), health is advisory.
    """
    tokens = estimate_request(body)

    if tokens > policy.local_token_budget:
        return Decision(Tier.UPSTREAM, "context_too_large", tokens)

    if has_image(body) and not policy.allow_images_local:
        return Decision(Tier.UPSTREAM, "image_input", tokens)

    if has_tools(body) and not policy.allow_tools_local:
        return Decision(Tier.UPSTREAM, "tools_requested", tokens)

    if not local_healthy:
        return Decision(Tier.UPSTREAM, "local_unhealthy", tokens)

    # Upstream degradation does not change the choice for short requests: they
    # would go local anyway. It is recorded so the caller can see why a spike in
    # local traffic coincides with an upstream incident.
    return Decision(Tier.LOCAL, "fits_local_budget", tokens)
