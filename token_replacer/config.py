"""Configuration: one YAML file, environment overrides, no hidden defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATHS = (
    Path("token-replacer.yaml"),
    Path("config.yaml"),
    Path.home() / ".token-replacer" / "config.yaml",
)


@dataclass(frozen=True)
class BackendConfig:
    """An OpenAI-compatible endpoint."""

    name: str
    base_url: str
    api_key: str | None = None
    timeout_s: float = 120.0
    connect_timeout_s: float = 5.0
    # Concurrent requests we let through to this backend. llama.cpp's
    # `llama-server --parallel N` uses this as its slot count; going above it
    # queues inside the model server, which hides the latency instead of
    # reporting it.
    max_concurrency: int = 8
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Callers ask for "deepseek-v4.1-flash"; the local server only knows
    # "qwen3.5-9b". Rewriting here keeps the caller's request untouched.
    model_override: str | None = None


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 8470
    policy: dict[str, Any] = field(default_factory=dict)
    local: BackendConfig = field(
        default_factory=lambda: BackendConfig(name="local", base_url="http://127.0.0.1:8080/v1")
    )
    upstream: BackendConfig = field(
        default_factory=lambda: BackendConfig(
            name="upstream", base_url="http://127.0.0.1:8112/v1", timeout_s=300.0
        )
    )
    breaker_threshold: int = 5
    breaker_cooldown_s: float = 30.0
    # When the local tier fails mid-request, retry upstream inside this budget.
    escalate_on_error: bool = True
    log_decisions: bool = True


def _backend(name: str, raw: dict[str, Any] | None, default_url: str) -> BackendConfig:
    raw = raw or {}
    return BackendConfig(
        name=raw.get("name", name),
        base_url=str(raw.get("base_url", default_url)).rstrip("/"),
        api_key=raw.get("api_key") or os.environ.get("TOKEN_REPLACER_UPSTREAM_KEY"),
        timeout_s=float(raw.get("timeout_s", 120.0)),
        connect_timeout_s=float(raw.get("connect_timeout_s", 5.0)),
        max_concurrency=int(raw.get("max_concurrency", 8)),
        extra_headers=dict(raw.get("extra_headers") or {}),
        model_override=raw.get("model_override") or None,
    )


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        for candidate in DEFAULT_CONFIG_PATHS:
            if candidate.is_file():
                path = candidate
                break
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

    return Config(
        host=os.environ.get("TOKEN_REPLACER_HOST", raw.get("host", "127.0.0.1")),
        port=int(os.environ.get("TOKEN_REPLACER_PORT", raw.get("port", 8470))),
        policy=dict(raw.get("policy") or {}),
        local=_backend("local", raw.get("local"), "http://127.0.0.1:8080/v1"),
        upstream=_backend("upstream", raw.get("upstream"), "http://127.0.0.1:8112/v1"),
        breaker_threshold=int(raw.get("breaker_threshold", 5)),
        breaker_cooldown_s=float(raw.get("breaker_cooldown_s", 30.0)),
        escalate_on_error=bool(raw.get("escalate_on_error", True)),
        log_decisions=bool(raw.get("log_decisions", True)),
    )
