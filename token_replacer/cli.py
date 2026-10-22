"""CLI: `token-replacer serve` and `token-replacer preview`."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import load_config
from .policy import Policy, decide


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="token-replacer", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the gateway")
    serve.add_argument("-c", "--config", default=None)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--log-level", default="info")

    preview = sub.add_parser("preview", help="print the routing decision for a request body")
    preview.add_argument("-c", "--config", default=None)
    preview.add_argument("file", nargs="?", default="-", help="JSON file, or - for stdin")

    args = parser.parse_args(argv)

    if args.command == "preview":
        config = load_config(args.config)
        policy = Policy(**config.policy)
        handle = sys.stdin if args.file == "-" else open(args.file, "r", encoding="utf-8")
        with handle:
            body = json.load(handle)
        decision = decide(body, policy)
        print(
            json.dumps(
                {
                    "tier": decision.tier.value,
                    "reason": decision.reason,
                    "estimated_prompt_tokens": decision.estimated_prompt_tokens,
                    "local_token_budget": policy.local_token_budget,
                },
                indent=2,
            )
        )
        return 0

    import uvicorn

    from .server import create_app

    config = load_config(args.config)
    app = create_app(config)
    uvicorn.run(
        app,
        host=args.host or config.host,
        port=args.port or config.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
