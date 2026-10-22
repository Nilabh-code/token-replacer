"""Token estimation.

We cannot tokenise exactly for every model family (that would mean shipping a
tokenizer per model), and the decision this feeds is a *context-length gate*,
so the estimate only has to be safely conservative: over-estimating sends a
request upstream (correct but slightly more expensive), under-estimating sends
a request the local model cannot fit (an out-of-memory failure).

English/code averages ~4.0 chars per token; we use 3.6 to buy margin. Non-ASCII
text tokenises far worse (often 1 token per character for CJK), so those are
counted per character.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

CHARS_PER_TOKEN = 3.6
MESSAGE_OVERHEAD = 4  # role/separator tokens per chat message
_ASCII_LIMIT = 0x2E80


def estimate_text(text: str) -> int:
    """Conservative token estimate for one string."""
    if not text:
        return 0
    wide = sum(1 for ch in text if ord(ch) >= _ASCII_LIMIT)
    narrow = len(text) - wide
    return wide + int(narrow / CHARS_PER_TOKEN)


def _flatten(content: Any) -> Iterable[str]:
    """Chat content may be a string or a list of typed parts."""
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                yield part
            elif isinstance(part, dict):
                for key in ("text", "content", "input_text"):
                    if isinstance(part.get(key), str):
                        yield part[key]
                if "image_url" in part:
                    # A vision request is not a text-only local job; make it look
                    # expensive so the gate keeps it upstream.
                    yield "x" * 4000


def estimate_request(body: dict[str, Any]) -> int:
    """Estimate prompt tokens for an OpenAI/Anthropic-shaped chat body."""
    total = 0
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            total += MESSAGE_OVERHEAD
            if isinstance(message, dict):
                for chunk in _flatten(message.get("content")):
                    total += estimate_text(chunk)
                # tool_calls on assistant turns are prompt tokens too
                if message.get("tool_calls"):
                    total += estimate_text(json.dumps(message["tool_calls"]))
                if isinstance(message.get("tool_call_id"), str):
                    total += 8

    for key in ("system", "prompt", "input"):
        value = body.get(key)
        if isinstance(value, str):
            total += estimate_text(value)
        elif isinstance(value, list):
            for chunk in _flatten(value):
                total += estimate_text(chunk)

    tools = body.get("tools")
    if isinstance(tools, list):
        total += estimate_text(json.dumps(tools))
    return total
