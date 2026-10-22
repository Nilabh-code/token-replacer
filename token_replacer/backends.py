"""HTTP clients for the two tiers.

Both tiers speak the OpenAI wire format, so one class covers llama.cpp's
`llama-server`, vLLM, LM Studio, Ollama's /v1 shim and OrcaRoute itself.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from .config import BackendConfig

# Headers that belong to one hop, not end to end.
HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
        "accept-encoding",
        "authorization",
    }
)


@dataclass
class UpstreamResult:
    status: int
    headers: dict[str, str]
    body: bytes


class BackendError(RuntimeError):
    """Transport-level failure: refused, reset, timed out."""


class Backend:
    def __init__(self, config: BackendConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_s, connect=config.connect_timeout_s),
            transport=transport,
            follow_redirects=False,
        )
        self._slots = asyncio.Semaphore(config.max_concurrency)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, incoming: dict[str, str]) -> dict[str, str]:
        headers = {k: v for k, v in incoming.items() if k.lower() not in HOP_BY_HOP}
        headers["content-type"] = "application/json"
        if self.config.api_key:
            headers["authorization"] = f"Bearer {self.config.api_key}"
        for key, value in self.config.extra_headers.items():
            headers[key] = value
        return headers

    def _body(self, raw: bytes) -> bytes:
        """Apply the model rewrite only when a backend needs it."""
        if not self.config.model_override:
            return raw
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        parsed["model"] = self.config.model_override
        return json.dumps(parsed).encode()

    async def post(self, path: str, raw: bytes, incoming: dict[str, str]) -> UpstreamResult:
        async with self._slots:
            try:
                response = await self._client.post(
                    path, content=self._body(raw), headers=self._headers(incoming)
                )
            except httpx.HTTPError as exc:
                raise BackendError(str(exc)) from exc
            return UpstreamResult(
                status=response.status_code,
                headers={k: v for k, v in response.headers.items()},
                body=response.content,
            )

    async def get(self, path: str, incoming: dict[str, str]) -> UpstreamResult:
        async with self._slots:
            try:
                response = await self._client.get(path, headers=self._headers(incoming))
            except httpx.HTTPError as exc:
                raise BackendError(str(exc)) from exc
            return UpstreamResult(
                status=response.status_code,
                headers={k: v for k, v in response.headers.items()},
                body=response.content,
            )

    async def stream(
        self, path: str, raw: bytes, incoming: dict[str, str]
    ) -> tuple[int, dict[str, str], AsyncIterator[bytes], httpx.Response]:
        """Open a streamed response. Caller owns closing `response`."""
        await self._slots.acquire()
        try:
            request = self._client.build_request(
                "POST", path, content=self._body(raw), headers=self._headers(incoming)
            )
            try:
                response = await self._client.send(request, stream=True)
            except httpx.HTTPError as exc:
                raise BackendError(str(exc)) from exc
            if response.status_code >= 400:
                # Drain and release now: an error is not something to stream.
                body = await response.aread()
                await response.aclose()
                self._slots.release()
                return (
                    response.status_code,
                    {k: v for k, v in response.headers.items()},
                    _empty(),
                    _ClosedResponse(body, response.status_code, dict(response.headers)),
                )

            async def iterator() -> AsyncIterator[bytes]:
                try:
                    async for chunk in response.aiter_raw():
                        yield chunk
                finally:
                    await response.aclose()
                    self._slots.release()

            return (
                response.status_code,
                {k: v for k, v in response.headers.items()},
                iterator(),
                response,
            )
        except BaseException:
            self._slots.release()
            raise


async def _empty() -> AsyncIterator[bytes]:  # pragma: no cover - trivial
    if False:
        yield b""


class _ClosedResponse:
    """Sentinel: an error body that has already been read and closed."""

    def __init__(self, body: bytes, status: int, headers: dict[str, str]):
        self.body = body
        self.status_code = status
        self.headers = headers

    async def aclose(self) -> None:
        return None


def forward_headers(upstream: dict[str, str]) -> dict[str, str]:
    """Response headers safe to hand back to the caller."""
    return {
        k: v
        for k, v in upstream.items()
        if k.lower()
        not in {"content-length", "transfer-encoding", "connection", "content-encoding", "keep-alive"}
    }
