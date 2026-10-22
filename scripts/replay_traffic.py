"""Replay real gateway token counts through the routing policy.

Input is a JSON array of [prompt_tokens, completion_tokens] measured on the
gateway. Nothing here is synthetic: the numbers come from log metadata.

    python scripts/replay_traffic.py replay_counts.json
    python scripts/replay_traffic.py replay_counts.json --budget 16000
"""

from __future__ import annotations

import argparse
import json
import sys

from token_replacer.policy import Policy


def summarise(rows: list[list[int]], policy: Policy) -> dict:
    budget = policy.local_token_budget
    total = len(rows)
    local = [r for r in rows if r[0] <= budget]
    local_out = sum(r[1] for r in local)
    all_out = sum(r[1] for r in rows)
    all_in = sum(r[0] for r in rows)
    return {
        "requests": total,
        "budget": budget,
        "local_requests": len(local),
        "local_request_share": round(100 * len(local) / total, 1) if total else 0.0,
        "local_input_token_share": round(100 * sum(r[0] for r in local) / all_in, 1) if all_in else 0.0,
        "local_output_token_share": round(100 * local_out / all_out, 1) if all_out else 0.0,
        "local_avg_prompt": round(sum(r[0] for r in local) / len(local)) if local else 0,
        "local_avg_completion": round(local_out / len(local)) if local else 0,
        "upstream_requests": total - len(local),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", help="JSON array of [prompt_tokens, completion_tokens]")
    parser.add_argument("--budget", type=int, default=None, help="override max_local_context")
    parser.add_argument("--sweep", action="store_true", help="show several budget points")
    args = parser.parse_args(argv)

    with open(args.counts, "r", encoding="utf-8") as handle:
        rows = json.load(handle)

    if args.sweep:
        print(f"{'max_ctx':>10} {'budget':>8} {'local reqs':>11} {'req %':>7} {'in %':>6} {'out %':>6}")
        for max_ctx in (4_000, 8_000, 16_000, 32_000, 64_000, 128_000):
            result = summarise(rows, Policy(max_local_context=max_ctx))
            print(
                f"{max_ctx:>10} {result['budget']:>8} {result['local_requests']:>11} "
                f"{result['local_request_share']:>6}% {result['local_input_token_share']:>5}% "
                f"{result['local_output_token_share']:>5}%"
            )
        return 0

    policy = Policy(max_local_context=args.budget) if args.budget else Policy()
    result = summarise(rows, policy)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
