# token-replacer

A tier-0 gateway that keeps short-context LLM traffic off your upstream provider.

It sits in front of an OpenAI-compatible endpoint, measures each request, and
serves the ones that fit a small local model itself. Everything else — large
contexts, tools, images — goes upstream untouched. If the local model fails, the
request is retried upstream so the caller still gets an answer.

```
                        ┌──────────────────────────────┐
  agent ──POST /v1/chat──▶  token-replacer :8470        │
                        │  ├─ estimate prompt tokens   │
                        │  ├─ context gate  ───────────┼──▶ local 9B (:8080)
                        │  ├─ capability gate ─────────┼──▶ upstream (:8112)
                        │  └─ breaker + escalation     │
                        └──────────────────────────────┘
```

## Why this exists

Measured on a live gateway serving ~28k requests/day (3 days, 84.7k requests,
log metadata only):

| | |
|---|---|
| p50 prompt | 94,771 tokens |
| p90 prompt | 374,765 tokens |
| requests above 128k tokens | 42% |
| **requests that fit in 32k tokens** | **29.9%** |
| their share of all prompt tokens | **0.9%** |
| their share of all completion tokens | **18.6%** |
| p50 completion on that slice | 270 tokens |
| provider failures (503/429/5xx) | 6.6% of all requests |

The load is bimodal: a third of the calls are small, and they carry almost none
of the prompt volume but a fifth of the generated volume. Those are exactly the
calls a 9B can answer, and exactly the calls that don't need a 155k-token
context window.

Note what this is *not*: a cost optimisation. If the upstream is already free or
flat-rate, replacing its tokens saves nothing. The win is **latency and
availability on the small slice** — it stops rate-limited and overloaded
providers from taking down traffic they were never needed for.

## The gate is a capability limit, not a difficulty guess

The first rule is arithmetic, not opinion. A 9B on a 16 GB card needs ~144 KB of
KV cache per token; at 155k tokens that is 22 GB before any weights. So
"is this request hard?" is the wrong question — "does this request even fit?"
is the right one, and it needs no model call to answer.

Difficulty-based routing (confidence, embeddings, a classifier) belongs *inside*
the slice that fits, as a second filter. It cannot be the first one.

## Install

```bash
git clone https://github.com/Nilabh-code/token-replacer
cd token-replacer
uv venv --python 3.13
uv pip install -e . pytest pytest-asyncio
```

## Run

Start a local model (any OpenAI-compatible server; llama.cpp shown):

```bash
llama-server -m qwen3.5-9b-Q4_K_M.gguf \
  --port 8080 -c 32768 -np 8 -fa --host 127.0.0.1
```

`-np 8` gives llama.cpp eight parallel slots with continuous batching — the
gateway's `max_concurrency` should match it. Below that setting, this workload
cannot be served on one consumer card at all (see the arithmetic in
`docs/traffic-fit.md`).

Then point the gateway at it:

```bash
cp config.example.yaml config.yaml   # edit the two base_urls
token-replacer serve -c config.yaml
```

Point your agent or gateway at `http://127.0.0.1:8470/v1`.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Routed proxy. Streaming supported. |
| `GET /v1/models` | Proxied to upstream (discovery belongs to the big tier). |
| `POST /tier-preview` | Dry-run the decision. Calls no model, counts nothing. |
| `GET /metrics` | Counts, tier shares, breaker states. |
| `GET /healthz` | Liveness plus the effective token budget. |

Every response carries its routing decision:

```
X-TokenReplacer-Tier: local
X-TokenReplacer-Reason: fits_local_budget
X-TokenReplacer-Est-Tokens: 812
```

Reasons: `fits_local_budget`, `context_too_large`, `tools_requested`,
`image_input`, `local_unhealthy`, `escalated_after_local_error`.

## Configuration

See `config.example.yaml`. The parts that matter:

- `policy.max_local_context` — the local model's real context window.
- `policy.safety_margin` — the prompt-token estimate is approximate; 0.9 keeps
  10% of the window as headroom.
- `policy.allow_tools_local` — leave `false` until tool-call validity is
  measured on your traffic. A broken tool call is worse than a slow one.
- `local.max_concurrency` — match the model server's slot count.
- `local.model_override` — callers ask for their model name; this box serves
  yours.
- `escalate_on_error` — retry upstream when the local tier fails. Requires
  `stream: false`, or a failure before the first byte.

## Verify it against your own traffic

Do not trust the numbers above; they are someone else's gateway. Export token
counts and replay them through the real policy:

```bash
python scripts/replay_traffic.py counts.json --sweep
```

where `counts.json` is `[[prompt_tokens, completion_tokens], ...]` for successful
requests. The sweep shows how the routable share changes with the budget, which
is how the 32k boundary above was chosen — 64k buys another 7 points of requests
for a model that no longer fits a consumer card.

## Tests

Four layers, in increasing order of what they can prove.

**1. Logic — no network, no GPU.**
```bash
pytest -q          # 30 tests
```
Context gate, capability gate, breaker behaviour, escalation, header hygiene.

**2. Wiring — real HTTP, stub models.**
```bash
python scripts/smoke_e2e.py          # 3 uvicorn servers, 6 checks; exits 1 on failure
```
URL joining, SSE pass-through, escalation before the first byte, metric
reconciliation.

**3. Real model, real request.**
```bash
llama-server -m qwen3.5-9b-Q4_K_M.gguf --port 8080 -c 32768 -np 8 -fa &
token-replacer serve -c config.yaml &
curl -s localhost:8470/v1/chat/completions -D- -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"what does strtok return"}]}'
```
Look at `X-TokenReplacer-Tier` and `X-TokenReplacer-Reason`: that is the router
telling you what it did. This layer only proves it answers, not that the answers
are worth having.

**4. Quality — the only layer that decides whether the local tier ships.**
```bash
python scripts/ab_eval.py prompts.example.jsonl \
  --local http://127.0.0.1:8080/v1 \
  --upstream http://127.0.0.1:8112/v1 \
  --verify exact --out report.json

# use this as a release gate: fail if the local tier loses more than N cases
python scripts/ab_eval.py prompts.jsonl --local ... --upstream ... \
  --verify exact --max-gap 3
```

Same prompts through both tiers, each graded by an oracle, reported as a parity
gap with the failing cases named:

```
local_pass 18/24   upstream_pass 24/24   parity_gap 6
local_only_pass 0  upstream_only_pass 6
local_latency_p50_s 2.14   upstream_latency_p50_s 6.71
failures: ["code-4", "code-9", ...]
```

Verifiers: `exact`, `contains`, `cmd` (run a checker against the answer — use
this for code), `none` (agreement only; weak, because two models can be wrong
together). Build the suite from prompts you actually send, not from a public
benchmark: the question is whether the local model handles *your* traffic.

## Limits, stated plainly

- **No failover in the other direction.** A request routed upstream stays
  upstream. If upstream is down and the context is too large, the caller gets
  the error — there is no second option.
- **Escalation after the first streamed byte is impossible.** Once the caller
  has seen tokens, retrying would duplicate them.
- **The token estimate is ±20%.** It is conservative on purpose; a wrong guess
  in the other direction is an out-of-memory failure.
- **Text only.** Images force the upstream path.
- **No distillation in this repo yet.** It routes to a local model; training a
  better one from the logged pairs is the next layer, and the log metadata is
  already sufficient to build the corpus.

## License

MIT
