# Replacement Tickets 1–2 Session Handoff

> [!IMPORTANT]
> Continue only Ticket 2. Do not use the discarded Ticket 04 or the earlier
> 12-ticket design, do not start Ticket 3, and do not mutate
> `vllm-project/vllm`. All repository mutations belong only in the personal
> fork `ycsxh/vllm`.

## Read First

Read these files in order before making changes:

1. [`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md)
2. [`WORKFLOW.md`](WORKFLOW.md)
3. [`TICKET_02_SERVER_HANDOFF.md`](TICKET_02_SERVER_HANDOFF.md)
4. this handoff

The active worktree is:

```text
/home/lyc/vllm/.worktrees/ticket-01-02
```

The active branch is:

```text
codex/ds4-replacement-tickets-1-2-continued
```

The original replacement-ticket base is:

```text
c27a4fdf8969e2927973197257510c1666f47b64
```

The branch state immediately before this diagnostic checkpoint was:

```text
f0bc15f09  [Benchmarks] Record Ticket 2 Gate A failure
e2a2488e6  [Benchmarks] Record Ticket 1 tokenizer acceptance
faa5b9ef8  [Benchmarks] Validate real Qwen tokenizer inputs
c27a4fdf8  [Benchmarks] Harden DS4 server handoffs
```

Use `git rev-parse HEAD` to identify the diagnostic checkpoint containing this
document. The worktree was clean before its documentation changes.

## What This Session Did

This session performed a read-only diagnosis of the accepted Ticket 2 Gate A
attempt and then recorded the result. It did not run another live smoke, change
the acceptance criteria, modify runtime code, perform a formal code review,
push a branch, or operate on the upstream repository.

The accepted live attempt remains at:

```text
/home/lyc/ds4-storage/runs/ds4-ticket-02-faa5b9ef8-attempt-06-live
```

It used:

- delivery commit
  `faa5b9ef8f4a6f93f217f0d6a80035199734a8fa`;
- model and tokenizer revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- Qwen/Qwen3.5-4B BF16;
- P on GPU 0/NUMA 0 and D on GPU 1/NUMA 1;
- HND BF16 cache, Mamba/GDN `align`, configured block size 128, prefix
  caching, chunked prefill, and fail-closed NIXL loading.

The session:

- re-read the authoritative spec, workflow, Ticket 2 handoff, launcher, proxy,
  NIXL scheduler, HMA block-size logic, prefix-cache logic, and focused unit
  tests;
- checked the preserved responses, logs, metric snapshots, and concise source
  lines;
- separated the cache-page failure from the P-to-D control-plane failure;
- verified candidate prompt lengths offline with the pinned tokenizer;
- updated the Ticket 2 server handoff with the corrected diagnosis.

## Implementation Continuation

The corrected Ticket 2 implementation is now locally verified. It did not run
a live smoke, change Gate A acceptance, start Ticket 3, perform a formal code
review, push a branch, or mutate the upstream repository.

The launcher now uses a DS4-owned fixed pull proxy. Local loopback fake P/D
tests prove that P receives the remote-decode control request, valid P metadata
reaches D unchanged, and missing or malformed metadata stops before D. The
launcher also uses the 80-repetition prompt, records the 642/640/641 token
contract, and verifies the pinned server tokenizer reports exactly 642 prompt
tokens before starting the proxy or sending a smoke request.

Focused launcher/proxy tests passed with `23 passed`; Ruff check and format
checks passed for the four changed Python files. Exact commands and results are
recorded in [`TICKET_02_SERVER_HANDOFF.md`](TICKET_02_SERVER_HANDOFF.md).

## Completed State

Ticket 1 is complete at `e2a2488e6`. Do not reopen it unless new evidence
invalidates its recorded tokenizer acceptance.

Ticket 2 has a valid retained failed Gate A result:

```text
P local_compute:           0 -> 258 -> 516
P local_cache_hit:         0 ->   0 ->   0
D local_compute:           0 -> 258 -> 516
D external_kv_transfer:   0 ->   0 ->   0
NIXL successful transfer: 0 ->   0 ->   0
```

Both client-visible responses completed and were identical. Failed-transfer,
failed-notification, and expired-request deltas remained zero. Bounded cleanup
left no GPU compute process or fixed-port listener. Gate A remains
`remote_failed`; a completed deterministic request is not sufficient.

The handoff does **not** normatively require a 258-token prompt. That length was
an implementation detail in `faa5b9ef8`, indirectly frozen for that approved
run by the exact delivery commit and dry-run plan. The specification requires a
cold request, an identical repeated-prefix request, positive P local-hit
evidence, positive D external-transfer and NIXL-success evidence, and zero
failure/fallback evidence.

## Corrected Root Cause

Gate A failed for two independent reasons.

### 1. No complete HMA-aligned cache page

The launcher used:

```python
prompt = "Explain deterministic cache transfer in one sentence. " * 32
```

The pinned tokenizer produced 258 tokens. Both server logs reported that the
runtime raised the effective attention block/page size from the configured 128
tokens to 640 tokens to match the Mamba page. Prefix-cache hashes are produced
only at complete boundaries, so 258 tokens cannot create a 640-token cached
page. This fully explains why repeated P local-cache hits stayed zero.

### 2. The proxy never initiated pull transfer

`examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`:

- copied the client request for P and changed only `max_tokens` to 1;
- did not add `kv_transfer_params` with `do_remote_decode=true`;
- drained and discarded the P response body;
- sent the original request to D without P's returned
  `kv_transfer_params`, `remote_block_ids`, or NIXL coordinates.

The NIXL scheduler enters its remote-prefill path only when the request carries
the required flags and metadata. The observed full local compute on both roles,
combined with zero successful **and** zero failed transfers, is the expected
result when no transfer is scheduled.

Do not attribute D's zero transfer count solely to the short prompt. The
remote-decode lifecycle supports partial-block transfer, and NIXL's HMA path
uses N-1 prompt tokens. The missing control-plane handoff prevented even that
partial attempt.

## Minimum Safe Request Design

For the fixed revision and the observed 640-token HMA page, the smallest
original prompt that can both form one complete P page and expose that page as
a repeated P cache hit is 642 tokens:

```text
P HMA prefill truncation:       N - 1
last token reserved for logits: N - 2
required complete page:         N - 2 >= 640
minimum original prompt:        N >= 642
```

Offline checks with the pinned tokenizer produced:

```text
32 repetitions -> 258 tokens
79 repetitions -> 634 tokens
80 repetitions -> 642 tokens
95 repetitions -> 762 tokens
96 repetitions -> 770 tokens
```

The minimum request therefore uses:

```python
prompt = "Explain deterministic cache transfer in one sentence. " * 80
```

Keep the client request deterministic:

```json
{
  "max_tokens": 16,
  "temperature": 0,
  "seed": 0,
  "ignore_eos": true,
  "stream": false
}
```

The fixed pull flow must:

1. send P the same prompt with `max_tokens=1` and
   `do_remote_decode=true`;
2. parse P's non-streaming response;
3. fail closed if P's transfer metadata, remote block IDs, engine identity,
   host, or port is absent or invalid;
4. attach the validated P transfer metadata to D's request;
5. send D the original prompt and original deterministic generation settings.

For 642 original tokens, P prefill sees 641 after the HMA N-1 truncation, one
complete 640-token page plus one tail token. On the repeated P request, exactly
one 640-token page is eligible for a local cache hit. D's cold request has 641
external-token candidates in the HMA N-1 remote-prefill path.

Treat the pinned offline token count and the observed effective runtime page as
fail-closed preconditions. If either changes, stop rather than silently choosing
a different request. These checks do not replace or change Gate A acceptance.

## What Is Still Open

No external prerequisite is currently known to be missing. The only remaining
Ticket 2 action is a new Gate A execution from the clean corrected delivery
commit. It requires separate operator approval and the unchanged topology,
identity, evidence, and verdict checks in
[`TICKET_02_SERVER_HANDOFF.md`](TICKET_02_SERVER_HANDOFF.md).

Do not rerun `faa5b9ef8`, do not run Gate A without that approval, and do not
start Ticket 3 while the corrected delivery remains `remote_pending`.

## Recommended Next Session

Wait for explicit Gate A operator approval. After approval, bind
`EXPECTED_COMMIT` to the clean corrected delivery, re-run the read-only server
preflight, review the dry-run plan, and execute only the Ticket 2 live procedure
in the server handoff. Preserve all raw evidence and stop again at the Gate A
verdict; do not begin Ticket 3 in the same flow.

## Commands for Resumption

```bash
cd /home/lyc/vllm/.worktrees/ticket-01-02
git status --short --branch
git log -5 --oneline --decorate
git rev-parse HEAD
```

After implementation, run focused validation through `.venv`, never system
Python:

```bash
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py -q

.venv/bin/ruff check benchmarks/ds4_profile/run_pd.py \
  benchmarks/ds4_profile/pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py

.venv/bin/ruff format --check benchmarks/ds4_profile/run_pd.py \
  benchmarks/ds4_profile/pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py
```

## Pitfalls Already Encountered

- Source-checkout extension mismatch can fail before any smoke request.
- FlashInfer JIT requires compatible CUDA compiler visibility.
- Offline model resolution must bind the exact cached snapshot, not merely set
  offline environment flags.
- Qwen3.5 requires `VLLM_SSM_CONV_STATE_LAYOUT=DS` for this path.
- Local readiness and request traffic must bypass the host HTTP proxy.
- The configured 128-token block is not the effective HMA page; the observed
  runtime page is 640 tokens.
- A successful HTTP response and identical output do not prove P-to-D
  transfer.
- Zero failed-transfer counters do not prove success when the transfer was
  never scheduled.
- Generic partial-block transfer support means `prompt < page` cannot by
  itself explain D external-transfer staying zero.
- The exact approved delivery commit is immutable for a live attempt. A prompt
  or proxy change requires a new delivery commit and a new approval.
- Preserve every failed attempt under its own run directory and retain bounded
  cleanup evidence; never overwrite original evidence.

## Hard Boundaries

- Do not modify the authoritative acceptance criteria.
- Do not make or claim a live verification without separate approval.
- Do not begin Ticket 3.
- Do not perform a formal code review in this flow.
- Do not push or create GitHub state unless explicitly requested.
- Never mutate `vllm-project/vllm`; the upstream repository is read-only.
