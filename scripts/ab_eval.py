"""A/B a local tier against its upstream and report the parity gap.

This is the evaluation that decides whether the local model is good enough for a
slice of traffic. Plumbing tests prove requests flow; only this proves the
answers are worth having.

    python scripts/ab_eval.py prompts.jsonl \
        --local http://127.0.0.1:8080/v1 --upstream http://127.0.0.1:8112/v1 \
        --verify exact --out report.json

prompts.jsonl, one JSON object per line:

    {"id": "math-1", "messages": [{"role":"user","content":"2+2?"}], "expect": "4"}
    {"id": "code-1", "messages": [...], "files": {"solution.py": "..."}}   # for --verify cmd

Verifiers:
  exact     normalised string equality with `expect`
  contains  `expect` appears in the answer
  cmd       shell command with {answer} substituted for a temp file path;
            exit code 0 means pass. This is the one that matters for code:
            a test either runs or it doesn't.
  none      no oracle; reports agreement between tiers only. Weaker evidence —
            high agreement can mean both are wrong together.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

DEFAULT_TIMEOUT = 300.0


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_prompts(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON ({exc})") from exc
            item.setdefault("id", f"case-{number}")
            items.append(item)
    if not items:
        raise SystemExit(f"{path}: no prompts")
    return items


def chat_payload(item: dict, model: str | None) -> dict:
    body = {
        "messages": item["messages"],
        "temperature": 0,
        "stream": False,
    }
    if model:
        body["model"] = model
    if item.get("tools"):
        body["tools"] = item["tools"]
    return body


async def call(
    client: httpx.AsyncClient, url: str, item: dict, model: str | None
) -> dict:
    started = time.perf_counter()
    try:
        response = await client.post(url, json=chat_payload(item, model))
    except httpx.HTTPError as exc:
        return {"answer": "", "latency": time.perf_counter() - started, "error": str(exc)}
    latency = time.perf_counter() - started
    if response.status_code >= 400:
        return {"answer": "", "latency": latency, "error": f"HTTP {response.status_code}"}
    try:
        choice = response.json()["choices"][0]["message"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return {"answer": "", "latency": latency, "error": f"bad response: {exc}"}
    return {"answer": choice.get("content") or "", "latency": latency, "error": None}


def verify(answer: str, item: dict, mode: str, command: str | None, timeout: float = 120.0) -> bool | None:
    if mode == "none":
        return None
    if mode == "exact":
        expected = item.get("expect")
        return None if expected is None else normalise(answer) == normalise(str(expected))
    if mode == "contains":
        expected = item.get("expect")
        return None if expected is None else normalise(str(expected)) in normalise(answer)
    if mode == "cmd":
        if not command:
            raise SystemExit("--verify cmd requires --cmd")
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(answer)
            answer_path = handle.name
        try:
            completed = subprocess.run(
                command.format(answer=answer_path),
                shell=True,
                capture_output=True,
                timeout=timeout,
            )
            return completed.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        finally:
            Path(answer_path).unlink(missing_ok=True)
    raise SystemExit(f"unknown verifier: {mode}")


def agreement(a: str, b: str) -> bool:
    return normalise(a) == normalise(b)


async def run(args) -> dict:
    items = load_prompts(args.prompts)
    results: list[dict] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def one(item: dict) -> dict:
            async with semaphore:
                local, upstream = await asyncio.gather(
                    call(client, args.local + "/chat/completions", item, args.local_model),
                    call(client, args.upstream + "/chat/completions", item, args.upstream_model),
                )
            return {
                "id": item["id"],
                "local": local,
                "upstream": upstream,
                "local_pass": verify(local["answer"], item, args.verify, args.cmd, args.verify_timeout),
                "upstream_pass": verify(upstream["answer"], item, args.verify, args.cmd, args.verify_timeout),
                # Two failed calls both return "" — that is not agreement.
                "agree": agreement(local["answer"], upstream["answer"])
                and not local["error"]
                and not upstream["error"],
            }

        for coro in asyncio.as_completed([one(item) for item in items]):
            result = await coro
            results.append(result)
            mark = {True: "pass", False: "FAIL", None: "----"}[result["local_pass"]]
            print(f"  {result['id'][:40]:<40} local={mark} agree={'Y' if result['agree'] else 'n'}", flush=True)

    results.sort(key=lambda r: r["id"])
    scored = [r for r in results if r["local_pass"] is not None]
    report = {
        "prompts": len(results),
        "verifier": args.verify,
        "scored": len(scored),
        "local_pass": sum(1 for r in scored if r["local_pass"]),
        "upstream_pass": sum(1 for r in scored if r["upstream_pass"]),
        "agreement": round(100 * sum(1 for r in results if r["agree"]) / len(results), 1),
        "local_errors": sum(1 for r in results if r["local"]["error"]),
        "upstream_errors": sum(1 for r in results if r["upstream"]["error"]),
    }
    if scored:
        report["local_pass_rate"] = round(100 * report["local_pass"] / len(scored), 1)
        report["upstream_pass_rate"] = round(100 * report["upstream_pass"] / len(scored), 1)
        report["parity_gap"] = report["upstream_pass"] - report["local_pass"]
        # The interesting cases: where only one tier succeeded.
        report["local_only_pass"] = sum(
            1 for r in scored if r["local_pass"] and not r["upstream_pass"]
        )
        report["upstream_only_pass"] = sum(
            1 for r in scored if r["upstream_pass"] and not r["local_pass"]
        )
    for tier in ("local", "upstream"):
        latencies = [r[tier]["latency"] for r in results if not r[tier]["error"]]
        if latencies:
            report[f"{tier}_latency_p50_s"] = round(statistics.median(latencies), 2)
            report[f"{tier}_latency_max_s"] = round(max(latencies), 2)
    report["failures"] = [
        r["id"] for r in scored if r["local_pass"] is False and r["upstream_pass"] is True
    ]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts", help="JSONL of test cases")
    parser.add_argument("--local", required=True, help="local base url, e.g. http://127.0.0.1:8080/v1")
    parser.add_argument("--upstream", required=True, help="upstream base url")
    parser.add_argument("--verify", default="exact", choices=["exact", "contains", "cmd", "none"])
    parser.add_argument("--cmd", default=None, help="command for --verify cmd; {answer} is the file")
    parser.add_argument("--local-model", default=None)
    parser.add_argument("--upstream-model", default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--verify-timeout", type=float, default=120.0, help="per-case limit for --verify cmd"
    )
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument(
        "--max-gap",
        type=int,
        default=None,
        help="exit 1 if upstream_pass - local_pass exceeds this many cases",
    )
    args = parser.parse_args(argv)

    report = asyncio.run(run(args))

    print()
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")

    # An unreachable tier is a broken run, not a quality result: without this,
    # two dead endpoints report gap 0 and look like perfect parity.
    if report["local_errors"] == report["prompts"]:
        print("\nINVALID RUN: the local tier never answered. No quality result here.", file=sys.stderr)
        return 2
    if report["upstream_errors"] == report["prompts"]:
        print("\nINVALID RUN: the upstream tier never answered. No baseline to compare.", file=sys.stderr)
        return 2
    if report["local_errors"] or report["upstream_errors"]:
        print(
            f"\nWARNING: {report['local_errors']} local / {report['upstream_errors']} upstream "
            "requests failed — the gap below excludes them.",
            file=sys.stderr,
        )

    if args.max_gap is not None and report.get("parity_gap", 0) > args.max_gap:
        print(
            f"\nGATE FAILED: local tier loses {report['parity_gap']} cases "
            f"(limit {args.max_gap}) — do not route this slice locally yet.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
