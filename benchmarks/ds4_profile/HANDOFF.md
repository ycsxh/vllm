# Replacement Tickets 1–2 Closeout Handoff

## Status

Replacement Tickets 1 and 2 are complete:

- Ticket 1 Gate B is `remote_verified`.
- Ticket 2 Gate A is `remote_verified`.
- Ticket 3 has not started.

The accepted Ticket 2 runtime delivery is:

```text
4415bbe8f04c11c5beab7057effe208659f1b91f
```

The active worktree and branch are:

```text
/home/lyc/vllm/.worktrees/ticket-01-02
codex/ds4-replacement-tickets-1-2-continued
```

[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md) remains normative. This
handoff records acceptance evidence without changing the specification or
starting the controlled serving-metric work in Ticket 3.

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
  tests/benchmarks/ds4_profile/test_pd_proxy.py -q

49 passed
```

Ruff check and format validation pass for the adapter, launcher, proxy, and
their three focused test files.

## Next scope

There is no remaining Ticket 1 or Ticket 2 acceptance blocker. Any Ticket 3
work must start as a separate scope and follow Gate C in
[`WORKFLOW.md`](WORKFLOW.md). Do not reinterpret this feasibility smoke as a
controlled performance result.
