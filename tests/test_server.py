from __future__ import annotations

import httpx
import pytest

from helpers import CHAT, body_of, chat_body, completion, sse_response


def test_small_request_served_locally(make_client, json_ok):
    seen = {}

    def local(request: httpx.Request) -> httpx.Response:
        seen["body"] = body_of(request)
        seen["path"] = request.url.path
        return httpx.Response(200, json=completion("local answer"))

    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream must not be called")

    with make_client(local, upstream) as client:
        response = client.post(CHAT, json=chat_body("what does strtok do"))

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "local answer"
    assert response.headers["x-tokenreplacer-tier"] == "local"
    assert response.headers["x-tokenreplacer-reason"] == "fits_local_budget"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["messages"][0]["content"] == "what does strtok do"


def test_big_request_goes_upstream(make_client):
    def local(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("local must not be called for a 200k-token prompt")

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion("big answer"))

    with make_client(local, upstream) as client:
        response = client.post(CHAT, json=chat_body("x" * 200_000))

    assert response.status_code == 200
    assert response.headers["x-tokenreplacer-tier"] == "upstream"
    assert response.headers["x-tokenreplacer-reason"] == "context_too_large"


def test_model_is_rewritten_for_the_local_tier(make_client):
    seen = {}

    def local(request: httpx.Request) -> httpx.Response:
        seen["model"] = body_of(request)["model"]
        return httpx.Response(200, json=completion())

    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream not expected")

    with make_client(local, upstream) as client:
        client.post(CHAT, json=chat_body("hi"))

    # conftest does not set model_override, so the caller's model passes through
    assert seen["model"] == "deepseek-v4.1-flash"


def test_local_503_escalates_to_upstream(make_client):
    def local(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "model overloaded"})

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion("rescued"))

    with make_client(local, upstream) as client:
        response = client.post(CHAT, json=chat_body("hi"))
        metrics = client.get("/metrics").json()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "rescued"
    assert response.headers["x-tokenreplacer-tier"] == "upstream"
    assert response.headers["x-tokenreplacer-reason"] == "escalated_after_local_error"
    assert metrics["escalated"] == 1


def test_local_transport_error_escalates(make_client):
    def local(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion("rescued"))

    with make_client(local, upstream) as client:
        response = client.post(CHAT, json=chat_body("hi"))
        metrics = client.get("/metrics").json()

    assert response.status_code == 200
    assert metrics["escalated"] == 1
    assert metrics["local_failures"] == 1


def test_escalation_can_be_disabled(make_client):
    def local(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("escalation disabled")

    with make_client(local, upstream, escalate=False) as client:
        response = client.post(CHAT, json=chat_body("hi"))

    assert response.status_code == 503


def test_breaker_opens_and_traffic_moves_upstream(make_client):
    calls = {"local": 0, "upstream": 0}

    def local(request: httpx.Request) -> httpx.Response:
        calls["local"] += 1
        return httpx.Response(503, json={"error": "down"})

    def upstream(request: httpx.Request) -> httpx.Response:
        calls["upstream"] += 1
        return httpx.Response(200, json=completion())

    # threshold 5 with escalate off, so failures are recorded without rescue
    with make_client(local, upstream, escalate=False, policy={}) as client:
        for _ in range(5):
            client.post(CHAT, json=chat_body("hi"))
        local_after_trip = calls["local"]
        response = client.post(CHAT, json=chat_body("hi"))

    assert local_after_trip == 5
    assert calls["local"] == 5  # sixth request never reached the local tier
    assert response.status_code == 200
    assert response.headers["x-tokenreplacer-reason"] == "local_unhealthy"


def test_streaming_passes_through(make_client):
    payload = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    def local(request: httpx.Request) -> httpx.Response:
        return sse_response(payload)

    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream not expected")

    with make_client(local, upstream) as client:
        with client.stream("POST", CHAT, json=chat_body("hi", stream=True)) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert b"[DONE]" in body
    assert response.headers["x-tokenreplacer-tier"] == "local"


def test_streaming_error_escalates_before_first_byte(make_client):
    def local(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    def upstream(request: httpx.Request) -> httpx.Response:
        return sse_response(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')

    with make_client(local, upstream) as client:
        with client.stream("POST", CHAT, json=chat_body("hi", stream=True)) as response:
            body = b"".join(response.iter_bytes())
        metrics = client.get("/metrics").json()

    assert response.status_code == 200
    assert b"[DONE]" in body
    assert metrics["escalated"] == 1


def test_tier_preview_spends_nothing(make_client):
    def boom(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("preview must not call any model")

    with make_client(boom, boom) as client:
        small = client.post("/tier-preview", json=chat_body("hi")).json()
        big = client.post("/tier-preview", json=chat_body("x" * 200_000)).json()

    assert small["tier"] == "local"
    assert small["local_token_budget"] == 28_800
    assert big["tier"] == "upstream"
    assert big["reason"] == "context_too_large"


def test_metrics_shape(make_client, json_ok):
    def local(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion())

    with make_client(local, local) as client:
        client.post(CHAT, json=chat_body("a"))
        client.post(CHAT, json=chat_body("b"))
        metrics = client.get("/metrics").json()

    assert metrics["requests"] == 2
    assert metrics["served_local"] == 2
    assert metrics["local_share"] == 1.0
    assert metrics["reasons"] == {"fits_local_budget": 2}
    assert metrics["breakers"]["local"]["state"] == "closed"


def test_healthz(make_client):
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion())

    with make_client(ok, ok) as client:
        payload = client.get("/healthz").json()

    assert payload["ok"] is True
    assert payload["local_token_budget"] == 28_800


def test_invalid_json_is_rejected(make_client):
    def ok(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no call expected")

    with make_client(ok, ok) as client:
        response = client.post(
            CHAT, content=b"{not json", headers={"content-type": "application/json"}
        )

    assert response.status_code == 400


def test_authorization_header_does_not_leak_upstream(make_client):
    """The caller's bearer token belongs to the gateway, not to the upstream."""
    seen = {}

    def local(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=completion())

    def upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError()

    with make_client(local, upstream) as client:
        client.post(CHAT, json=chat_body("hi"), headers={"authorization": "Bearer caller-key"})

    assert seen["auth"] is None
