"""Shared fixtures-free helpers, importable as `from helpers import ...`."""

from __future__ import annotations

import json

import httpx

CHAT = "/v1/chat/completions"


def chat_body(text: str = "hello", **extra) -> dict:
    body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": text}]}
    body.update(extra)
    return body


def completion(text: str = "ok", model: str = "served-model") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def body_of(request: httpx.Request) -> dict:
    return json.loads(request.content)


class _Stream(httpx.AsyncByteStream):
    """A response body that has not been read yet, so aiter_raw() works once.

    httpx.Response(content=b"...") marks the stream consumed at construction,
    which makes it useless for testing pass-through streaming.
    """

    def __init__(self, payload: bytes, chunk: int = 24) -> None:
        self.payload = payload
        self.chunk = chunk

    async def __aiter__(self):
        for start in range(0, len(self.payload), self.chunk):
            yield self.payload[start : start + self.chunk]


def sse_response(payload: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        stream=_Stream(payload),
        headers={"content-type": "text/event-stream"},
    )
