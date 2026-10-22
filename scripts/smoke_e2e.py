"""End-to-end smoke test over real HTTP (uvicorn, not the ASGI test client).

Runs three processes in-process: a stub local model, a stub upstream, and
token-replacer between them. Proves the wiring that unit tests cannot:
URL joining, streaming, header forwarding, escalation, and metrics.

    python scripts/smoke_e2e.py
"""

from __future__ import annotations

import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from token_replacer.config import BackendConfig, Config
from token_replacer.server import create_app

LOCAL_PORT = 18471
UPSTREAM_PORT = 18472
GATE_PORT = 18470


def stub(name: str, fail_first: int = 0, stream_payload: bytes | None = None) -> FastAPI:
    app = FastAPI()
    state = {"calls": 0, "fail": fail_first}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        state["calls"] += 1
        body = await request.json()
        if state["fail"] > 0:
            state["fail"] -= 1
            return JSONResponse({"error": {"message": "overloaded"}}, status_code=503)
        if body.get("stream"):
            payload = stream_payload or (
                b'data: {"choices":[{"delta":{"content":"' + name.encode() + b'"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            return StreamingResponse(iter([payload]), media_type="text/event-stream")
        return {
            "id": f"stub-{name}",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": name}}],
        }

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": f"{name}-model"}]}

    return app


def serve(app: FastAPI, port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    return server


def wait_ready(port: int, path: str = "/healthz", timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}{path}", timeout=1.0)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"port {port} never became ready")


def main() -> int:
    serve(stub("local-model", fail_first=1), LOCAL_PORT)
    serve(stub("upstream-model"), UPSTREAM_PORT)

    config = Config(
        host="127.0.0.1",
        port=GATE_PORT,
        policy={},
        local=BackendConfig(name="local", base_url=f"http://127.0.0.1:{LOCAL_PORT}/v1"),
        upstream=BackendConfig(
            name="upstream", base_url=f"http://127.0.0.1:{UPSTREAM_PORT}/v1", timeout_s=30
        ),
        breaker_threshold=2,
        breaker_cooldown_s=5,
        log_decisions=False,
    )
    serve(create_app(config), GATE_PORT)
    wait_ready(GATE_PORT)
    time.sleep(0.3)

    base = f"http://127.0.0.1:{GATE_PORT}"
    small = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}
    big = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "x" * 200_000}]}
    checks: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        checks.append((label, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {label}  {detail}")

    # 1. the local stub is primed to fail once, so this exercises real escalation
    r = httpx.post(f"{base}/v1/chat/completions", json=small, timeout=20)
    check(
        "escalation: local 503 -> upstream",
        r.status_code == 200
        and r.json()["choices"][0]["message"]["content"] == "upstream-model"
        and r.headers["x-tokenreplacer-reason"] == "escalated_after_local_error",
        f"{r.status_code} {r.headers.get('x-tokenreplacer-reason')}",
    )

    # 2. next short request is served locally
    r = httpx.post(f"{base}/v1/chat/completions", json=small, timeout=20)
    check(
        "small request served locally",
        r.json()["choices"][0]["message"]["content"] == "local-model"
        and r.headers["x-tokenreplacer-tier"] == "local",
        r.headers.get("x-tokenreplacer-tier", ""),
    )

    # 3. big context never touches the local model
    r = httpx.post(f"{base}/v1/chat/completions", json=big, timeout=20)
    check(
        "200k-token request goes upstream",
        r.json()["choices"][0]["message"]["content"] == "upstream-model"
        and r.headers["x-tokenreplacer-reason"] == "context_too_large",
        f"{r.json()['choices'][0]['message']['content']}",
    )

    # 4. streaming pass-through
    with httpx.stream(
        "POST", f"{base}/v1/chat/completions", json={**small, "stream": True}, timeout=20
    ) as s:
        body = b"".join(s.iter_bytes())
    check("streaming from local", b"[DONE]" in body and b"local-model" in body, f"{len(body)}B")

    # 5. metrics agree with what happened
    m = httpx.get(f"{base}/metrics", timeout=5).json()
    check(
        "metrics reconcile",
        m["served_local"] == 2 and m["served_upstream"] == 2 and m["escalated"] == 1,
        f"local={m['served_local']} up={m['served_upstream']} esc={m['escalated']}",
    )

    # 6. model discovery proxies to upstream
    r = httpx.get(f"{base}/v1/models", timeout=10)
    check("model discovery proxied", r.json()["data"][0]["id"] == "upstream-model-model")

    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
