# Ticket 4 Token-Accurate ITL Design

## Goal

Make the official detailed `vllm bench serve` evidence contain exactly one
inter-token-latency sample per output-token transition when a vLLM OpenAI
completion stream coalesces multiple output tokens into one SSE delta.

This change resolves Ticket 4 failure class B without accepting console
aggregates, weakening the DS4 validator, enabling eager mode, or changing the
selected-pilot workload.

## Evidence and Root Cause

The frozen Ticket 4 run completed all 20 requests for every affected point and
reported 128 output tokens per request, but some official request-level ITL
arrays contained only 124–126 samples instead of 127.

A fresh optimized reproduction at concurrency 4 produced the sharper result
that all 20 requests reported 128 output tokens and exactly 126 ITL samples.
The run had no request errors, OOM, NIXL failure, or compatibility failure.

The OpenAI completion server supports the vLLM extension
`return_token_ids=true`. In streaming mode, each choice then includes the
delta's exact `token_ids`. A delta may contain more than one token, while the
current benchmark client records exactly one arrival timestamp for every
choice-bearing SSE message. It therefore counts messages rather than token
transitions. A terminal choice with zero delta tokens can create the inverse
error if it is treated as a generated token.

## Constraints

- Preserve `vllm-project/vllm` as read-only and mutate only `ycsxh/vllm`.
- Keep the frozen commit and all `attempt-02` artifacts immutable.
- Do not replace request-level evidence with aggregate console metrics.
- Preserve compatibility with OpenAI-compatible endpoints that do not return
  token IDs.
- Request token IDs only for the controlled DS4 completion benchmark.
- Preserve the existing TTFT definition as arrival of the first output token.
- Do not invent server-side generation timestamps that the wire protocol does
  not provide.
- Keep the main plan in optimized mode.

## Considered Approaches

### 1. Token-aware client accounting

The DS4 benchmark command requests `return_token_ids=true`. The official
completion client uses the returned delta-token count to expand each observed
SSE arrival into the corresponding number of token observations.

This is the selected approach because it repairs the evidence at the boundary
where the information is available, is backward compatible, and does not
change model scheduling. Token IDs are a mandatory DS4 profile request option:
without them, a coalesced SSE delta does not expose enough information to
produce one timing observation per output token.

### 2. Force one token per server SSE delta

Changing the serving or scheduler path to forbid multi-token deltas would make
the old client assumption true, but it would alter the runtime being measured
and has a much larger correctness and performance surface.

### 3. Reject multi-token deltas

Failing when coalescing is detected would remain strict, but would leave the
selected high-concurrency workload unmeasurable and would not resolve Gate D.

## Detailed Design

### DS4 command contract

`benchmarks.ds4_profile.run_points.build_benchmark_command` adds:

```text
--extra-body {"return_token_ids":true}
```

The flag applies to the controlled `/v1/completions` request only. The P/D
proxy forwards the request body unchanged to D after adding
`kv_transfer_params`, so no proxy protocol change is required.

### Completion-client accounting

`async_request_openai_completions` inspects
`choices[0]["token_ids"]` when it is present:

- A nonempty list supplies the number of output tokens represented by that
  delta.
- The first token-bearing delta sets TTFT.
- On the first token-bearing delta, every additional token in the same delta
  contributes an ITL of `0.0`, because those tokens have the same observable
  wire-arrival time.
- On later token-bearing deltas, the first token contributes the elapsed time
  since the previous token-bearing delta; additional tokens in the same delta
  contribute `0.0`.
- A zero-token choice, including a terminal finish-only choice, does not set
  TTFT, append ITL, or advance the most-recent-token timestamp.
- Generated text is still accumulated from every choice.

When `token_ids` is absent or `null`, the client retains its current
one-token-per-choice behavior. This preserves compatibility with generic
OpenAI endpoints, including vLLM responses that serialize an unrequested
`token_ids` field as `null`, and keeps the behavior change scoped to responses
that provide explicit token cardinality.

The DS4 command always requests token IDs. Its existing validator, rather than
a second client mode, enforces the observable measurement contract below. If
an endpoint ignores the extension and then coalesces tokens, the incomplete
ITL array is rejected. If it emits exactly one token per delta, the fallback
observation is already token-accurate.

The resulting observable contract is:

```text
len(output.itl) == output.output_tokens - 1
```

for a successful controlled DS4 request with positive output length.

### Timing interpretation

The client cannot reconstruct generation times hidden inside a coalesced SSE
delta. Recording zero between tokens observed in the same delta represents the
actual client-observed arrival boundary and preserves the total observed
latency:

```text
ttft + sum(itl) == last_token_arrival - request_start
```

The design does not interpolate, divide, or synthesize unknown server-side
latencies.

## Error Handling

- A non-null `token_ids` value that is not a list is treated as malformed
  response data and fails the request instead of silently reverting to message
  counting.
- A token-bearing response still requires a successful HTTP response and a
  valid first output token before the request is marked successful.
- The existing DS4 validator remains fail-closed and continues to reject any
  detailed array whose length differs from `output_len - 1`.
- Usage-provided `completion_tokens` remains the authoritative official output
  length; token-aware timing supplies the matching transition samples.

## Tests

### Official completion client

Extend the existing endpoint-request test area with deterministic SSE
fixtures:

1. A three-token response whose first delta contains two token IDs and whose
   second delta contains one token ID produces one zero ITL followed by one
   measured inter-delta ITL.
2. A zero-token terminal choice does not append an ITL or move the last-token
   timestamp.
3. A response without `token_ids` preserves the current one-token-per-choice
   fallback.
4. A response with `token_ids: null` preserves the same fallback used by
   ordinary vLLM completion streams.
5. A malformed non-null, non-list `token_ids` value fails the request.

The tests patch the monotonic clock with literal timestamps and assert exact
TTFT, ITL count, and latency values.

### DS4 command

Extend `tests/benchmarks/ds4_profile/test_run_points.py` to assert that the
official benchmark command contains the exact `--extra-body` value requesting
token IDs.

### Regression and target verification

The implementation is accepted only when:

- each changed-behavior regression test is observed failing before the
  implementation, while compatibility characterization tests remain green;
- the focused endpoint and DS4 suites pass after the implementation;
- the complete DS4 profile suite and relevant lint checks pass;
- independent Standards and Spec reviews approve the diff; and
- a fresh optimized concurrency-4 target result reports 128 output tokens and
  127 ITL samples for every request before the full selected-pilot rerun.

## Non-Goals

- Adding server-side per-token generation timestamps.
- Changing scheduling, batching, chunked prefill, or CUDA graph behavior.
- Generalizing token-ID requests to every benchmark backend.
- Relaxing the measurement contract or accepting incomplete historical
  evidence.
- Modifying or regenerating `attempt-02`.
