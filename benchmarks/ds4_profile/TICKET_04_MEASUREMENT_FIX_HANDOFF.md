# Ticket 4 Gate D measurement-fix handoff

## Status

Ticket 4 Gate D remains blocked. The immutable selected-pilot run completed,
but 10 of 30 points failed the measurement contract:

- four small-chunk TTFT points are failure class A;
- six high-concurrency Decode points are failure class B.

Failure class B now has an approved design and a locally complete fix. Failure
class A is reproduced but not yet diagnosed or fixed. The next session should
work on class A before any new Gate D pilot.

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

Do not overwrite, regenerate, relabel, or edit the frozen `attempt-02`
artifacts. This branch has not claimed Gate D acceptance and has not retired
the legacy path.

## Class B: locally complete

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

No class-B GPU validation result directory was created. Per the approved
implementation plan, target validation waits until class A is fixed, then
starts with one optimized concurrency-4 point before a full pilot rerun.

## Class A: reproduced, root cause open

The affected immutable `attempt-02` points are:

```text
ttft-hit-75-chunk-1024-concurrency-1
ttft-hit-75-chunk-2048-concurrency-1
ttft-hit-90-chunk-1024-concurrency-1
ttft-hit-90-chunk-2048-concurrency-1
```

Every point:

- completed all 20 official requests with zero request failures;
- observed 274,520 D external-transfer tokens;
- observed zero NIXL failed-transfer, failed-notification, and expiry deltas;
- recorded all 274,520 P prompt tokens as `local_compute`;
- recorded zero P `local_cache_hit` tokens.

The 75% points planned 204,800 cached tokens per repetition. The 90% points
planned 243,200. These are measurement-contract failures, not OOM or transport
failures.

The preserved review and diagnosis records are:

```text
/home/lyc/ds4-storage/runs/
  ds4-ticket-04-24e2ef4fc-attempt-02
  ds4-ticket-04-24e2ef4fc-attempt-02-gate-d-review/gate-d-review.md
  ds4-ticket-04-24e2ef4fc-attempt-02-gate-d-review/failure-diagnosis.json
```

The independent optimized reproduction is:

```text
plan:
  /home/lyc/ds4-storage/runs/
  ds4-ticket-04-24e2ef4fc-diagnostic-a01-plan.json

result:
  /home/lyc/ds4-storage/runs/
  ds4-ticket-04-24e2ef4fc-diagnostic-a01

launcher log:
  /home/lyc/ds4-storage/runs/
  ds4-ticket-04-24e2ef4fc-diagnostic-a01.launcher.log
```

It used hit 75%, chunk budget 1024, concurrency 1, output length 1, and 20
requests. Each request had 13,727 input tokens and 10,240 planned cached
tokens. All 20 requests succeeded, but the repetition failed with:

```text
nonzero planned P cache hit was not observed
```

This confirms the symptom but does not establish the root cause. Do not assume
that cache construction, eviction, chunked-prefill scheduling, Mamba alignment,
or metric attribution is responsible until a discriminating probe proves it.

## Next-session objective

Yes: the next session should diagnose and then fix failure class A.

Start with the repository's `diagnosing-bugs` workflow. Preserve `attempt-02`,
`diagnostic-a01`, and `diagnostic-b01` as immutable evidence, and allocate a
new result directory for every probe.

Recommended order:

1. Re-read the authoritative metric and cache-hit contracts plus
   `run_repetition`'s reset/warm/measure sequence.
2. Compare one failing 1024 point with the equivalent passing 4096 condition.
3. Use the smallest new optimized-mode probe that distinguishes whether:
   - the warm request creates a reusable P prefix;
   - the prefix is present immediately after warming but later evicted;
   - chunked prefill prevents reuse at request admission;
   - the cache is reused but attributed to the wrong metric source.
4. Capture P metrics immediately around the relevant warm and probe requests;
   do not weaken `derive_run_result` or accept console aggregates.
5. Write and approve a class-A design only after the root cause is evidenced.
6. Add a failing regression test, implement the smallest fix, and rerun the
   focused and complete DS4 CPU suites plus Standards/Spec review.

Keep the main workload in optimized mode. Do not substitute eager mode, reduce
parameters, relabel the four points as unsupported, or mutate server/scheduler
behavior merely to make the result pass.

## After class A is fixed

Do not immediately rerun all 30 points. First:

1. run a fresh one-point class-A target validation;
2. run a fresh optimized concurrency-4 class-B validation and confirm every
   request reports 128 output tokens and 127 ITL samples;
3. only when both probes pass, rerun the complete selected pilot in a new
   result directory;
4. regenerate and independently audit the report;
5. request human Gate D review;
6. retire the legacy path only after explicit Gate D acceptance.

The full sequence remains fail-closed. Historical failures stay preserved even
after a later clean run passes.
