"""The gateway itself.

One endpoint matters: POST /v1/chat/completions. Everything else exists to make
that endpoint observable (/metrics, /healthz) or to let a caller dry-run the
routing decision without spending a token (/tier-preview).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__
from .backends import Backend, BackendError, forward_headers
from .circuit import CircuitBreaker, is_unhealthy_status
from .config import Config, load_config
from .metrics import Metrics
from .policy import Decision, Policy, Tier, decide

log = logging.getLogger("token_replacer")

TIER_HEADER = "X-TokenReplacer-Tier"
REASON_HEADER = "X-TokenReplacer-Reason"
TOKENS_HEADER = "X-TokenReplacer-Est-Tokens"

CHAT_PATH = "/v1/chat/completions"
# Backend base_url already ends in /v1, and httpx appends relative paths, so the
# upstream path must NOT repeat the /v1 prefix.
BACKEND_CHAT_PATH = "chat/completions"
BACKEND_MODELS_PATH = "models"


def _incoming_headers(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.headers.items()}


def _decision_headers(decision: Decision) -> dict[str, str]:
    return {
        TIER_HEADER: decision.tier.value,
        REASON_HEADER: decision.reason,
        TOKENS_HEADER: str(decision.estimated_prompt_tokens),
    }


def _escalated(decision: Decision) -> Decision:
    """Relabel a request that was routed locally and had to be retried upstream,
    so the response headers and logs never claim the local tier answered it."""
    return Decision(Tier.UPSTREAM, "escalated_after_local_error", decision.estimated_prompt_tokens)


def create_app(
    config: Config | None = None,
    *,
    transport_factory=None,
) -> FastAPI:
    """Build the app.

    `transport_factory(name)` returns an httpx transport, or None for real
    network. Tests use it to stand in for both tiers.
    """
    config = config or load_config()
    factory = transport_factory or (lambda name: None)

    policy = Policy(**config.policy)
    metrics = Metrics()
    local_breaker = CircuitBreaker(
        "local", config.breaker_threshold, config.breaker_cooldown_s
    )
    upstream_breaker = CircuitBreaker(
        "upstream", config.breaker_threshold, config.breaker_cooldown_s
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.local = Backend(config.local, transport=factory("local"))
        app.state.upstream = Backend(config.upstream, transport=factory("upstream"))
        try:
            yield
        finally:
            await app.state.local.aclose()
            await app.state.upstream.aclose()

    app = FastAPI(title="token-replacer", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.policy = policy
    app.state.metrics = metrics
    app.state.local_breaker = local_breaker
    app.state.upstream_breaker = upstream_breaker

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "local": {"url": config.local.base_url, "breaker": local_breaker.state},
            "upstream": {"url": config.upstream.base_url, "breaker": upstream_breaker.state},
            "local_token_budget": policy.local_token_budget,
            "metrics": metrics.as_dict(),
        }

    @app.get("/metrics")
    async def get_metrics() -> dict[str, Any]:
        payload = metrics.as_dict()
        payload["breakers"] = {
            "local": {"state": local_breaker.state, "failures": local_breaker.failures},
            "upstream": {"state": upstream_breaker.state, "failures": upstream_breaker.failures},
        }
        return payload

    @app.post("/tier-preview")
    async def tier_preview(request: Request) -> dict[str, Any]:
        """Dry-run the router. No model is called, nothing is counted."""
        body = await request.json()
        decision = decide(
            body,
            policy,
            local_healthy=local_breaker.state != "open",
            upstream_healthy=upstream_breaker.state != "open",
        )
        return {
            "tier": decision.tier.value,
            "reason": decision.reason,
            "estimated_prompt_tokens": decision.estimated_prompt_tokens,
            "local_token_budget": policy.local_token_budget,
        }

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        """Discovery belongs to whoever serves the big contexts."""
        backend: Backend = app.state.upstream
        try:
            result = await backend.get(BACKEND_MODELS_PATH, _incoming_headers(request))
        except BackendError:
            return JSONResponse({"object": "list", "data": []}, status_code=200)
        return Response(
            content=result.body,
            status_code=result.status,
            media_type="application/json",
        )

    @app.post(CHAT_PATH)
    async def chat_completions(request: Request):
        raw = await request.body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": {"message": "body must be an object"}}, status_code=400)

        decision = decide(
            body,
            policy,
            local_healthy=local_breaker.state != "open",
            upstream_healthy=upstream_breaker.state != "open",
        )
        metrics.record_decision(decision.tier.value, decision.reason)
        if config.log_decisions:
            log.info(
                "tier=%s reason=%s est_tokens=%d model=%s",
                decision.tier.value,
                decision.reason,
                decision.estimated_prompt_tokens,
                body.get("model"),
            )

        headers = _incoming_headers(request)
        want_stream = bool(body.get("stream"))
        started = time.monotonic()

        if decision.tier is Tier.UPSTREAM:
            return await _serve(
                app.state.upstream,
                upstream_breaker,
                headers,
                decision,
                want_stream,
                metrics,
                started,
                raw,
            )

        try:
            response = await _serve(
                app.state.local,
                local_breaker,
                headers,
                decision,
                want_stream,
                metrics,
                started,
                raw,
            )
        except BackendError:
            if not config.escalate_on_error:
                raise
            metrics.record_escalation()
            log.warning("local transport failed, escalating to upstream")
            return await _serve(
                app.state.upstream,
                upstream_breaker,
                headers,
                _escalated(decision),
                want_stream,
                metrics,
                started,
                raw,
            )

        if getattr(response, "status_code", 200) >= 400 and config.escalate_on_error:
            metrics.record_escalation()
            log.warning("local returned %s, escalating", response.status_code)
            return await _serve(
                app.state.upstream,
                upstream_breaker,
                headers,
                _escalated(decision),
                want_stream,
                metrics,
                started,
                raw,
            )
        return response

    return app


async def _serve(
    backend: Backend,
    breaker: CircuitBreaker,
    headers: dict[str, str],
    decision: Decision,
    want_stream: bool,
    metrics: Metrics,
    started: float,
    raw: bytes,
):
    try:
        if want_stream:
            status, upstream_headers, iterator, holder = await backend.stream(
                BACKEND_CHAT_PATH, raw, headers
            )
            breaker.record_success() if status < 400 else breaker.record_failure()
            out_headers = forward_headers(upstream_headers)
            out_headers.update(_decision_headers(decision))
            if status >= 400:
                return Response(
                    content=getattr(holder, "body", b""),
                    status_code=status,
                    headers=out_headers,
                    media_type="application/json",
                )
            metrics.record_served(decision.tier.value)
            return StreamingResponse(
                iterator,
                status_code=status,
                headers=out_headers,
                media_type=out_headers.get("content-type", "text/event-stream"),
            )

        result = await backend.post(BACKEND_CHAT_PATH, raw, headers)
        if is_unhealthy_status(result.status):
            breaker.record_failure()
        else:
            breaker.record_success()
        if result.status < 400:
            metrics.record_served(decision.tier.value)
        out_headers = forward_headers(result.headers)
        out_headers.update(_decision_headers(decision))
        return Response(
            content=result.body,
            status_code=result.status,
            headers=out_headers,
            media_type="application/json",
        )
    except BackendError:
        breaker.record_failure()
        metrics.local_failures += 1 if decision.tier is Tier.LOCAL else 0
        if decision.tier is Tier.UPSTREAM:
            metrics.upstream_failures += 1
        raise
