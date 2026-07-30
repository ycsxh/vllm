# Ticket 4 Gate D measurement-fix handoff

## Status

Ticket 4 Gate D is `accepted`. The human repository operator explicitly
accepted the report, evidence review, noisy labels, and limitations at
`2026-07-30T14:56:10+08:00`.

Failure classes A and B are diagnosed, fixed, reviewed, and validated. The
fresh selected-pilot run completed with 30/30 valid points and 90/90 valid
repetitions. The acceptance record is stored separately from the immutable
run evidence. Legacy retirement remains explicitly deferred.

## Repository state

Use this worktree and branch:

```text
worktree: /home/lyc/vllm-ticket04-fix-gate-d-24e2ef4fc
branch:   codex/ticket-04-measurement-fix-gate-d
```

The frozen selected-pilot implementation and evidence commit is:

```text
24e2ef4fc1b886a88e792b0d5a833b7e1638dae5
```

The locally reviewed class-B delivery commits are:

```text
ef9da913e  docs: design token-accurate ITL accounting
ac258a7ec  docs: approve token-accurate ITL design
206499fc5  fix: account for multi-token completion deltas
6f5dca4c3  test: distinguish zero-token arrival timing
```

The class-A design, plan, and reviewed implementation commits are:

```text
5584d9718  docs: design sparse prefix retention for Ticket 4
61f58f765  docs: clarify Ticket 4 retention boundary
d914b41a6  docs: plan Ticket 4 prefix retention fix
05175de028 fix: retain DS4 Mamba cache boundaries
```

`05175de02822dde89fa7a6cdbf7f348471237608` is the clean source SHA
used by both the target validations and the successful full pilot.

Do not overwrite, regenerate, relabel, or edit `attempt-02`,
`diagnostic-a01`, `diagnostic-b01`, any `a02`-through-`a10` probe, any
validation, `attempt-03`, or `attempt-04`. This branch records human Gate D
acceptance but has not retired the legacy path.

## Class B: complete

Root cause:

- the completion benchmark counted one timing observation per choice-bearing
  SSE message;
- optimized high-concurrency serving can place multiple output token IDs in
  one SSE delta;
- official output length therefore remained 128 while request-level ITL arrays
  contained only 124–126 entries instead of 127.

The fix:

- the controlled DS4 command always requests
  `--extra-body '{"return_token_ids":true}'`;
- the shared completion client uses a returned token-ID list as the delta's
  exact token cardinality;
- tokens received in the same delta receive zero client-observed ITL;
- a zero-token choice does not establish TTFT, append ITL, or advance the
  final-token arrival time;
- absent or null token IDs retain generic one-choice/one-token compatibility;
- a non-null, non-list value fails the request;
- the existing DS4 validator still enforces
  `len(itl) == output_len - 1`.

The approved design and implementation plan are:

- `docs/superpowers/specs/2026-07-29-ticket-04-token-accurate-itl-design.md`;
- `docs/superpowers/plans/2026-07-29-ticket-04-token-accurate-itl.md`.

Local verification:

```text
completion-client tests:       5 passed
combined focused files:       37 passed
DS4 replacement suite:        88 passed
Ruff check and format:        passed
Standards review:             passed, zero findings
Spec review after correction: passed, zero findings
```

The complete DS4 suite needs permission to bind temporary local TCP ports.
An in-sandbox run produced 73 passes and 15 socket-permission setup errors;
the same command with socket access passed all 88 tests.

The fresh class-B target validation is:

```text
/home/lyc/ds4-storage/runs/
  ds4-ticket-04-05175de02-validation-b01
```

It is an optimized concurrency-4, output-length-128 point with three valid
repetitions. Every repetition completed 20/20 requests with zero failures;
every request reported exactly 128 output tokens and 127 ITL samples.

## Class A: complete

The immutable `attempt-02` failures and `diagnostic-a01` reproduction remain
the original symptom evidence. The root-cause investigation used a new result
directory for every optimized-mode probe:

```text
/home/lyc/ds4-storage/runs/
  ds4-ticket-04-3cb05de99-probe-a03-direct-p-warm-reuse-1024
  ds4-ticket-04-3cb05de99-probe-a04-full-prompt-proxy-1024
  ds4-ticket-04-3cb05de99-probe-a05-full-prompt-after-d-reset-1024
  ds4-ticket-04-3cb05de99-probe-a06-warm20-first-1024
  ds4-ticket-04-3cb05de99-probe-a07-warm20-last-1024
  ds4-ticket-04-3cb05de99-probe-a08-kv-metrics-warm20-first-1024
  ds4-ticket-04-3cb05de99-probe-a09-kv-metrics-warm20-first-4096
  ds4-ticket-04-3cb05de99-probe-a10-retention0-warm20-first-1024
```

The discriminating evidence is:

- `a03` through `a05` proved that one warm prefix is reusable through direct P
  access, proxy routing, and a D reset;
- `a06` warmed 20 unique prefixes at chunk budget 1024 and found the first
  prefix missing;
- `a07` changed only the probe to the last warmed prefix and observed the
  exact 10,240-token hit, proving warm-order LRU loss;
- `a08` added cache-lifecycle evidence and observed shared-pool reallocation;
- `a09` changed only the chunk budget to 4096 and preserved the first prefix;
- `a10` changed the failing 1024 condition only by setting the existing
  retention interval to zero and preserved the first prefix.

Qwen3.5 is a hybrid full-attention/Mamba model. With dense Mamba checkpoint
retention, the 1024- and 2048-token warm sequences consume enough of the
shared 620-block pool to evict the oldest planned prefixes. The existing
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` setting retains replay and detected
shared-prefix boundaries instead of every intermediate Mamba checkpoint.

The approved fix freezes that setting on P and D in the controlled DS4
launcher and records numeric `prefix_cache_retention_interval: 0` in the
serialized compatibility contract. The proxy, workload, reset/warm/measure
ordering, raw-evidence checks, cache-hit tolerances, vLLM core, scheduler,
cache manager, and NIXL connector are unchanged.

The design and implementation plan are:

- `docs/superpowers/specs/2026-07-29-ticket-04-prefix-cache-retention-design.md`;
- `docs/superpowers/plans/2026-07-29-ticket-04-prefix-cache-retention.md`.

The post-recovery class-A target validation is:

```text
/home/lyc/ds4-storage/runs/
  ds4-ticket-04-05175de02-validation-a04
```

`validation-a01` also passed before the external build target disappeared;
`validation-a04` is the clean confirmation after durable runtime recovery.
It passed all three official repetitions. Each completed 20/20 requests with
zero failures, and each observed 204,800 planned/P local-cache-hit tokens,
69,720 P local-compute tokens, 274,520 D external-transfer tokens, and zero
NIXL failure deltas.

## CPU, static, and independent review gates

Verification at the implementation SHA:

```text
focused launcher regression:  passed
test_run_pd.py:                21 passed
test_run_points.py:            32 passed
Mamba retention selection:     3 passed
complete DS4 suite:            140 passed, 1 skipped
pre-commit on changed files:   all hooks passed
```

The complete DS4 suite was run with the required local socket permission and
the migrated fixtures. The design review found one wording ambiguity about
the retention boundary; commit `61f58f765` corrected it and the re-review
passed. Independent implementation Standards and Spec reviews both passed
with zero findings.

## Preserved runtime incident

The first full rerun is preserved as:

```text
/home/lyc/ds4-storage/runs/
  ds4-ticket-04-05175de02-attempt-03
```

It produced seven valid points before the original external precompiled-build
target ceased to be available. The remaining 20 points failed server startup
and one point was interrupted. This was an environment failure, not a
measurement-contract regression. Two unsuccessful recovery validations are
also preserved:

```text
ds4-ticket-04-05175de02-validation-a02  missing FlashAttention extension
ds4-ticket-04-05175de02-validation-a03  stale incompatible core extension
```

The durable replacement
[precompiled runtime](/home/lyc/ds4-storage/runtime/ticket-04-precompiled-0762f2afe-a01)
and
[CUDA compatibility directory](/home/lyc/ds4-storage/runtime/ticket-04-cuda-link-compat-0762f2afe-a01)
are read-only.

It contains 11 precompiled extensions from upstream build commit
`0762f2afeb74f790e8c5ebe2b95a012eb38d499e`, variant `cu130`. The official
wheel SHA-256 is:

```text
972e5405306ad4c884511130354f1f70ae78e1f13ad4f868596fc7ae3aab76fe
```

The continuation worktree's ignored extension links resolve only to that
durable runtime. The original and failed repair states remain preserved under
`/tmp/ds4-ticket04-extension-repair-a01` through `a03`.

## Successful selected pilot

The immutable successful run and launcher log are:

```text
/home/lyc/ds4-storage/runs/
  ds4-ticket-04-05175de02-attempt-04
  ds4-ticket-04-05175de02-attempt-04.launcher.log
```

Frozen execution identity:

```text
source SHA:        05175de02822dde89fa7a6cdbf7f348471237608
source dirty:      false
model/tokenizer:   851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
hardware:          2 x NVIDIA GeForce RTX 3090, 24,576 MiB each
interconnect:      SYS
topology:          P=GPU0/NUMA0, D=GPU1/NUMA1, TP=1 per role
execution mode:    optimized
```

The final manifest is `valid`. All 30 points completed three repetitions, so
all 90 repetitions are valid. The independent fail-closed audit found:

```text
official requests:                 20 completed, 0 failed per repetition
planned/P hit mismatches:          0
P hit + P compute != D transfer:   0
nonzero NIXL failure deltas:       0
point-failure artifacts:           0
P/D retention != "0":             0
proxy retention setting present:  0
```

All 36 standard Decode repetitions reported 2,560 output tokens, output
length 128, and ITL length 127. The three output-length-32 repetitions
reported 640 output tokens, output length 32, and ITL length 31.

Two valid points are explicitly noisy because p50 TTFT CV exceeded 5%:

```text
decode-hit-0-chunk-4096-concurrency-4
decode-hit-0-chunk-4096-concurrency-8
```

Some server logs emit `EngineDeadError` only after the launcher records
`[shutdown]`, sends SIGTERM, and begins resource teardown. They do not precede
or invalidate any official request, metric snapshot, repetition, point, or
manifest. Both GPUs returned to 0 MiB with no compute process after the run.

## Report and evidence identity

The report generator independently re-derived every run from official
benchmark JSON and before/after P/D metrics before writing:

```text
/home/lyc/ds4-storage/runs/ds4-ticket-04-05175de02-attempt-04/
  summary.csv
  report.md
  report-metadata.json
  plots/*.svg
```

There are 31 CSV lines and eight SVG plots. Important SHA-256 values are:

```text
run-manifest.json  ab6ab8dd4a3d4d3d2d2cb677cf36d49d011e042f16d49a5c3d07f717759dce1c
summary.csv        d2492d2a48ebe7273438d486e8f20741ce41e4dfa049ba5f26bd571e6a6a1bb9
report.md          0f82a8956355838cff8d4d14aa678591675c4df029f91bc52550af4a62f752e7
report-metadata    afb20c99ff4a5d18aa95c294ad9571eb12a9535bad70697d098375280d7d6ae5
launcher log       4c44de70011655b626062da62dd468f56089bfbd6d1b122d4075240e0d310164
```

The SHA-256 of the sorted relative-file checksum stream for all 946 files in
the completed result directory is:

```text
d700d06974fa82a050ce851c016e8c72f5d0365a015a7eed2c1f4d1fb2832af8
```

## Gate D acceptance record

The immutable acceptance record is:

```text
/home/lyc/ds4-storage/runs/
  ds4-ticket-04-05175de02-attempt-04-gate-d-review-a01/
    gate-d-review.md
    checksums.sha256
```

It records the human verdict, plot and log review, hardware/topology
confirmation, cleanup state, accepted limitations, and explicit deferral of
legacy retirement. Its SHA-256 values are:

```text
gate-d-review.md  39ce48aba49872555439de0bf9afa1104eb74f43b9ca153b06c52a5d30dba709
checksums.sha256  334d009faa3ed9b74c3f606482914c29a36179ac9385a1e5f79ae7393b489ea9
```

The generated `attempt-04/report.md` remains unchanged, including its
pre-acceptance checklist state. This separate review artifact is the
authoritative post-run human acceptance record.

## Old checkout and migration note

`/home/lyc/vllm` remains the older
`codex/ticket-04-ds4-profile-spine` checkout. Its untracked
`TICKET_04_FIX_AND_GATE_D_HANDOFF.md` and token-accounting design were transfer
copies; their relevant content is already committed in this continuation
branch, and the old copies are superseded. No implementation change remains
stranded there. Do not clean or commit that checkout's untracked `.worktrees`
or transfer files.

## Closeout boundary

Ticket 4 Gate D is complete. The human reviewer explicitly instructed:

```text
暂不执行 legacy retirement
```

Legacy retirement therefore remains out of scope and requires a separate
future authorization and reviewed change. No push, PR, upstream mutation, or
legacy retirement has been performed here.
