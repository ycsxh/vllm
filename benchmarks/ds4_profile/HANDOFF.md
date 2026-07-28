# Replacement Tickets 1–4 Handoff

## Status

Replacement Tickets 1–3 are complete:

- Ticket 1 Gate B is `remote_verified`.
- Ticket 2 Gate A is `remote_verified`.
- Ticket 3 Gate C is `remote_verified`.
- Ticket 4 is locally implemented; target execution and Gate D are pending.

The accepted Ticket 2 runtime delivery is:

```text
4415bbe8f04c11c5beab7057effe208659f1b91f
```

The accepted Ticket 2 worktree and branch were:

```text
/home/lyc/vllm/.worktrees/ticket-01-02
codex/ds4-replacement-tickets-1-2-continued
```

[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md) remains normative. This
handoff preserves the accepted Tickets 1–3 evidence and routes the local
Ticket 4 delivery to its target-server handoff.

## Ticket 1 completion

Ticket 1 passed Gate B against delivery commit
`4415bbe8f04c11c5beab7057effe208659f1b91f`.

Frozen inputs:

```text
model/tokenizer:
  Qwen/Qwen3.5-4B

model/tokenizer revision:
  851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

dataset revision:
  4da61f3d06b48b6817a62b99e9c47035c8e59787
```

The two offline preparations were byte-identical. All 667 selected rows had
matching `input_tokens` and `prompt_ids` lengths, no row carried
`output_tokens`, and the provenance named the immutable tokenizer revision.
The final outputs were also byte-identical to the earlier accepted output.
The evidence directory is
`/home/lyc/ds4-storage/runs/ds4-ticket-01-4415bbe8f-evidence`; its
`evidence-checksums.txt` SHA-256 is
`ede34a26e8e289569fa70db61534f6d8509ade1be608df2dc437d249b54ad9aa`.
The complete paths, checksums, and validation commands remain in
[`TICKET_01_SERVER_HANDOFF.md`](TICKET_01_SERVER_HANDOFF.md).

## Ticket 2 completion

Ticket 2 passed Gate A against the clean corrected delivery commit
`4415bbe8f04c11c5beab7057effe208659f1b91f` with model and tokenizer revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

The accepted evidence directory is:

```text
/home/lyc/ds4-storage/runs/ds4-ticket-02-4415bbe8f-attempt-08
```

The sealed evidence manifest is `evidence-checksums.txt`, with SHA-256:

```text
7a1168096fcb8aeeef364dec5c20689aa92fa43d404c6ca5285cb3edbe6545e0
```

The run used:

- Qwen/Qwen3.5-4B BF16;
- P on GPU 0/NUMA 0 and D on GPU 1/NUMA 1;
- TP=1 for both roles;
- the DS4-owned fixed pull proxy;
- HND BF16 cache and Mamba/GDN `align`;
- configured block size 128 and effective HMA page size 640;
- prefix caching and chunked prefill;
- fail-closed NIXL loading;
- direct host execution with no container.

The fixed smoke contract and observed metrics were:

```text
tokenized prompt:                    642
effective HMA page:                  640
cold remote-transfer tokens:         641

P local_cache_hit:
  before 0
  cold 0
  repeated 640

D external_kv_transfer:
  before 0
  cold 641
  repeated 642

D NIXL transfer histogram count:
  before 0
  cold 1
  repeated 2

D NIXL bytes histogram count:
  before 0
  cold 1
  repeated 2
```

Both deterministic 16-token responses were identical. Failed-transfer,
failed-notification, and expired-request deltas were zero. The forbidden
failure-pattern scan was empty. Bounded cleanup left no GPU compute process,
relevant survivor, or fixed-port listener.

The exact verdict, metric review, raw metric snapshots, log source lines,
topology, provenance, responses, and cleanup evidence are retained in the
accepted evidence directory. See
[`TICKET_02_SERVER_HANDOFF.md`](TICKET_02_SERVER_HANDOFF.md) for the concise
tracked record.

## Target-host compatibility fix

The pip-provided CUDA 13 runtime contained `libcudart.so.13` but not the
unversioned `libcudart.so` linker name required by FlashInfer's `-lcudart` JIT
link. The accepted run created a compatibility symlink inside its external
`server/` evidence directory and included that directory in `LIBRARY_PATH`.

The fix was run-local. It did not modify source, the acceptance criteria, or
the repository `.venv`.

## Retained failure history

Earlier attempts remain preserved and must not be overwritten or deleted.
They record:

- the pre-correction 258-token smoke and proxy control-flow failure;
- incorrect fork remote and missing host tools;
- an operator interruption before launch;
- missing Ninja;
- missing unversioned CUDA runtime linker name;
- a dry-run artifact collision before live execution.

These failures and the earlier accepted attempt-07 are historical evidence.
Attempt-08 validates the final runtime delivery without erasing their
diagnostic value.

## Local verification

The combined focused suite passes:

```text
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py -q

79 passed
```

Ruff check and format validation pass for the adapter, launcher, proxy, point
runner, and their focused test files.

## Ticket 3 completion

Ticket 3 passed Gate C against immutable delivery commit
`163935c12db0545c12eba694bfd6316be1f4094a`. The accepted result directory is:

```text
/home/lyc/ds4-storage/runs/ds4-ticket-03-163935c12-attempt-10-full-matrix
```

The final manifest is `valid`, records that exact clean vLLM commit, and
contains six valid points with three completed repetitions each. The evidence
contains 18 official benchmark results, 18 derived results, and six point
summaries. Every repetition completed 20 requests with zero failures and an
empty official error array.

Each 75% nominal repetition observed 12,800 P local-cache-hit tokens; every 0%
repetition observed zero. D external-transfer tokens were 30,320 per
repetition, all NIXL failure-counter deltas were zero, and no summary metric
exceeded 5% CV. The independent audit is `valid`.

Four `EngineDeadError` tracebacks occurred only during bounded SIGTERM cleanup
after all measurements. They remain recorded as shutdown lifecycle errors, not
request-time failures. Final cleanup found no GPU compute process, related
survivor, or fixed-port listener. Paths, closeout-copy checksums, and the
complete verdict are recorded in
[`TICKET_03_SERVER_HANDOFF.md`](TICKET_03_SERVER_HANDOFF.md).

Hardware validation applies only to delivery commit
`163935c12db0545c12eba694bfd6316be1f4094a`. The documentation closeout commit
records that accepted state; it is not itself GPU-validated.

## Ticket 4 local delivery

The replacement Ticket 4 implementation adds:

- `config/selected-pilot-points.json`, with 30 explicit optimized-mode points;
- selected hit/chunk and concurrency/hit interactions without a generic
  Cartesian planner;
- three frozen DS4 prompt cut points and explicit 1/32/128 output lengths;
- ITL p99 alongside three-run mean/CV and noisy labeling;
- fail-closed OOM classification as `unsupported`;
- at most one optional `eager_diagnostic` point; and
- `report_results.py`, which re-audits raw official benchmark JSON and P/D
  metrics before writing `summary.csv`, `report.md`, and seven SVG plots.

Local work does not claim Gate D, does not run the model, and does not retire
the legacy path. Exact target commands, review gates, and the post-acceptance
retirement checklist are in
[`TICKET_04_SERVER_HANDOFF.md`](TICKET_04_SERVER_HANDOFF.md).

## Next scope

There is no remaining Ticket 1, Ticket 2, or Ticket 3 acceptance blocker. The
next scope is the target-server Ticket 4 dry run, live selected pilot, audited
report generation, human Gate D review, and only then legacy retirement.
