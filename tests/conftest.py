from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from token_replacer.config import BackendConfig, Config
from token_replacer.server import create_app


@pytest.fixture
def make_client():
    """Build a TestClient with both tiers backed by httpx.MockTransport."""

    def _make(local_handler, upstream_handler, *, policy=None, escalate=True, local_max_conc=4):
        config = Config(
            host="127.0.0.1",
            port=0,
            policy=policy or {},
            local=BackendConfig(
                name="local", base_url="http://local.test/v1", max_concurrency=local_max_conc
            ),
            upstream=BackendConfig(name="upstream", base_url="http://upstream.test/v1"),
            escalate_on_error=escalate,
            log_decisions=False,
        )
        transports = {
            "local": httpx.MockTransport(local_handler),
            "upstream": httpx.MockTransport(upstream_handler),
        }
        app = create_app(config, transport_factory=lambda name: transports[name])
        return TestClient(app)

    return _make


@pytest.fixture
def json_ok():
    def _handler(payload):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json=payload, headers={"content-type": "application/json"}
            )

        return handler

    return _handler


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)
