# Traffic fit — why the boundary is 32k

All numbers below were produced by replaying 79,144 successful request token
pairs from a live OpenAI-compatible gateway through `token_replacer.policy`
(`scripts/replay_traffic.py --sweep`). Only token counts were exported — no
prompts, no completions, no user identifiers.

## The distribution is bimodal

| metric | value |
|---|---|
| requests | 84,699 in 3.03 days (~28k/day) |
| succeeded | 79,101 (93.4%) |
| p50 prompt tokens | 94,771 |
| p90 prompt tokens | 374,765 |
| p95 prompt tokens | 454,275 |
| max prompt tokens | 798,806 |
| p50 completion tokens | 288 |
| requests above 128k | 42.1% |

An average of 155k prompt tokens per request with a median of 95k is not a chat
workload; it is coding agents re-sending repository context and tool output.

## How the routable share moves with the budget

| max local context | effective budget (0.9) | requests local | % of requests | % of prompt tokens | % of completion tokens |
|---|---|---|---|---|---|
| 4k | 3,600 | 16,529 | 20.9% | 0.2% | 12.5% |
| 8k | 7,200 | 20,309 | 25.7% | 0.4% | 14.9% |
| 16k | 14,400 | 21,928 | 27.7% | 0.5% | 16.8% |
| **32k** | **28,800** | **23,673** | **29.9%** | **0.9%** | **18.6%** |
| 64k | 57,600 | 28,971 | 36.6% | 2.8% | 26.0% |
| 128k | 115,200 | 41,078 | 51.9% | 12.0% | 45.8% |

At the default budget the local slice averages 4,135 prompt and 928 completion
tokens per request.

## Choosing 32k

- **32k → 64k buys 6.7 points of requests** (29.9% → 36.6%) and requires a
  context window that a 9B cannot hold on a 16 GB card. Not worth it.
- **32k → 16k loses only 2.2 points.** If a card is short on memory, 16k is a
  cheap downgrade.
- Below 8k the curve flattens: the remaining traffic is either genuinely tiny or
  genuinely huge, so the boundary stops mattering.

## Can one consumer card serve it?

Slice at 32k, ~7,960 requests/day, 4,135 prompt / 928 completion tokens:

| concurrency | decode GPU-h/day | prefill GPU-h/day | total | card left |
|---|---|---|---|---|
| 1 stream (45 tok/s) | 45.5 | 8.1 | 53.6 h/day | impossible |
| 8 streams (~400 tok/s) | 5.1 | 0.9 | 6.0 h/day | 75% |
| 24 streams (~1200 tok/s) | 1.7 | 0.3 | 2.0 h/day | 92% |

Single-stream serving is arithmetically impossible — it needs more than a day of
GPU time per day. Batching is not an optimisation here, it is the requirement,
which is why `local.max_concurrency` exists and why \(\text{llama-server}\) must
be started with a matching `-np`.

## The fourth finding: failures

6.6% of requests failed on the gateway side over the window — 2,368 upstream
`503`, 727 `429` (a 30 requests/minute ceiling), 1,671 client aborts (`499`),
648 `401`. A local tier absorbs the portion of those that fall inside the
routable slice, which is the real operational argument for this component.
