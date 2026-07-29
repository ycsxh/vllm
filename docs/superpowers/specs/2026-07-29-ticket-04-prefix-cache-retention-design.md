# Ticket 4 Prefix-Cache Retention Design

## Goal

Make the fixed DS4 Qwen3.5-4B P/D launcher retain every planned warm prefix
through the existing 20-request warm phase, including the 1024- and 2048-token
chunk-budget points, so that the official measurement phase observes the
planned P local-cache hits.

This resolves Ticket 4 failure class A without changing vLLM scheduler
behavior, the selected-pilot workload, the reset/warm/measure protocol, raw
benchmark evidence, or acceptance thresholds.

## Evidence and Root Cause

The immutable `attempt-02` run and `diagnostic-a01` reproduction established
the symptom: all 20 requests completed and D observed all expected external
transfers, but the affected 1024- and 2048-token chunk-budget points attributed
every P prompt token to local compute and none to local cache hits. Those
artifacts remain immutable.

Fresh optimized-mode probes used one new result directory each:

| Probe | Single distinction | Result |
| --- | --- | --- |
| `a03` | Warm one 10,240-token prefix, then query P directly | Exact 10,240-token P cache hit |
| `a04` | Send the corresponding 13,727-token prompt through the proxy | Exact 10,240-token P cache hit |
| `a05` | Reset D between that warm and full-prompt request | Exact 10,240-token P cache hit |
| `a06` | Warm 20 unique prefixes at chunk budget 1024, then probe the first | Zero cache hit; 13,726 P local-compute tokens |
| `a07` | Same as `a06`, but probe the last prefix | Exact 10,240-token P cache hit |
| `a08` | Repeat `a06` with official KV-cache lifecycle metrics enabled | Same failure; the 620-block pool is reallocated during the warm sequence |
| `a09` | Change only the chunk budget from 1024 to 4096 | Exact 10,240-token hit for the first prefix; no warm reallocations |
| `a10` | Change `a08` only by setting retention interval `0` on P and D | Exact 10,240-token hit for the first prefix |

Probes `a03` through `a05` exclude failed cache construction, full-request
admission, proxy routing, and the D reset as causes. The first-versus-last
comparison in `a06` and `a07` proves that the 20-prefix warm order causes LRU
loss of the oldest prefix. The 1024-versus-4096 comparison in `a08` and `a09`
ties that loss to chunk-dependent cache occupancy.

Qwen3.5 is a hybrid full-attention/Mamba model. With
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unset, `MambaManager` caches dense Mamba
state checkpoints. At the smaller chunk budgets, the warm sequence's dense
checkpoints consume enough of the shared 620-block pool to evict early planned
prefixes before measurement. The existing value `0` sparsifies Mamba
retention to proven replay/shared-prefix boundaries while full-attention
groups remain dense. Probe `a10` demonstrates that this single configuration
change preserves the first planned prefix under the otherwise failing
condition.

The KV-cache lifecycle counter is supporting occupancy evidence, not the LRU
proof by itself. The collector counts block reallocation before checking
whether the previous block had a cache hash, so `a10` can report more lifecycle
reallocations while still preserving the needed prefix. The cache-hit and
prompt-token-source deltas in `a06`, `a07`, and `a10` are the authoritative
reuse evidence.

## Constraints

- Preserve `attempt-02`, `diagnostic-a01`, `diagnostic-b01`, and every
  completed probe directory as immutable evidence.
- Allocate a new result directory for every future probe or validation.
- Keep the main workload in default optimized mode.
- Preserve the selected model, immutable revisions, 20 requests, prompt
  lengths, requested hit ratios, chunk budgets, concurrency, output lengths,
  and three-repetition protocol.
- Preserve reset P, reset D, warm P, reset D, then measure ordering.
- Continue to derive acceptance from raw official benchmark JSON and
  before/after P/D metrics.
- Do not weaken `derive_run_result`, cache-hit tolerances, or fail-closed NIXL
  checks.
- Do not modify core scheduler, cache-manager, NIXL connector, or proxy code or
  algorithms. The only allowed runtime-behavior change is the fixed Mamba
  retention setting described below.
- Do not expose prefix retention as a new experiment axis.
- Keep `vllm-project/vllm` read-only and make any eventual repository changes
  only in `ycsxh/vllm`.

## Considered Approaches

### 1. Freeze sparse Mamba retention in the DS4 launcher

Set the existing `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` environment variable
on both P and D server processes, and record the value in the serialized
server-plan compatibility contract.

This is the selected approach. It changes only which Mamba state checkpoints
the controlled launcher retains, keeps replay and shared-prefix boundaries,
leaves full-attention caching dense, and directly addresses the evidenced
hidden occupancy. It also makes the condition auditable in every dry-run and
run artifact.

### 2. Interleave warming and measurement

Measuring a prefix immediately after warming it would avoid eviction, but
would change the authoritative batch protocol, cache state seen by later
requests, and official raw evidence. It would no longer measure the selected
pilot.

### 3. Increase cache capacity or reduce the workload

Increasing GPU cache capacity, shortening prompts, reducing the 20-request
batch, or relabeling small chunks as unsupported would alter frozen experiment
conditions. Capacity also depends on target-host headroom and would conceal,
rather than control, chunk-dependent checkpoint occupancy.

## Detailed Design

### Authoritative experiment contract

Before launcher implementation, update
`benchmarks/ds4_profile/AUTHORITATIVE_SPEC.md` to add this fixed runtime
condition:

```text
Mamba prefix-cache retention:
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 on P and D
```

The specification explains that `0` keeps the Mamba replay and detected
shared-prefix boundaries needed for reuse instead of retaining every
intermediate Mamba checkpoint. The value is a fixed part of the Qwen3.5 DS4
runtime, not a workload parameter, performance tuning dimension, or host
prerequisite.

### Launcher contract

`benchmarks.ds4_profile.run_pd.build_plan` injects the exact string value `"0"`
into the shared P/D server environment:

```text
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
```

The launcher-owned value is applied after the validated host runtime
environment. The host cannot override it, it is not added to
`RUNTIME_ENVIRONMENT_NAMES`, and no CLI flag is added. This preserves the
existing separation between host prerequisites and experiment-owned settings.

Both `prefill` and `decode` receive the setting because they run the same
hybrid model and the fixed compatibility contract must be symmetric. The
proxy does not receive it because it owns no model cache.

`LaunchPlan.as_dict()["compatibility"]` adds:

```text
prefix_cache_retention_interval: 0
```

The numeric compatibility value records effective semantics, while each
server process environment records the exact child-process string. Dry runs
and `run_points` already serialize the plan, so they inherit this audit trail
without a second configuration path.

### Measurement flow

No measurement-flow code changes. `run_repetition` continues to reset P and D,
warm all 20 planned P prefixes in deterministic order, reset D, snapshot
metrics, invoke official `vllm bench serve`, snapshot metrics again, and
validate the raw request arrays and metric deltas.

### Error handling and observability

The existing vLLM retention validation remains fail closed: a negative or
misaligned positive interval is rejected, as is using the setting on a model
without a sliding-window or Mamba cache group. Qwen3.5's Mamba group makes
zero valid.

The launcher test fails if either server lacks the exact value, if the proxy
receives it, or if the serialized compatibility value differs. Runtime logs,
the server plan, raw official results, and metric snapshots continue to be
preserved on failure.

## Tests

### Test design

- Module purpose: `run_pd.build_plan` is the single source of truth for the
  fixed 1P1D child-process and compatibility contract.
- Input/output contract: a validated `LaunchConfig` produces a deterministic
  `LaunchPlan` whose two server environments and serialized compatibility
  describe the effective runtime.
- Guarded failure: a planned prefix is evicted because the launcher silently
  uses dense Mamba checkpoint retention, or the effective setting is omitted
  from provenance.
- Cheapest effective level: extend existing `test_run_pd.py` plan assertions;
  do not mock private cache-manager or scheduler internals in the DS4 suite.

### Regression sequence

1. Extend
   `test_build_plan_freezes_fail_closed_server_configuration` or the adjacent
   complete-environment test to require the exact P/D setting, its absence from
   the proxy, and the numeric compatibility field.
2. Run the focused test before implementation and preserve the expected
   failure.
3. Implement the launcher and authoritative-spec changes.
4. Run the focused launcher tests and the complete DS4 CPU suite.
5. Run the existing core retention tests that cover dense `None`, sparse `0`,
   replay-boundary retention, and shared-prefix retention.
6. Run relevant formatting and lint checks and obtain independent Standards
   and Spec review.

The existing vLLM core tests are semantic coverage for the already available
retention mechanism. The new DS4 regression covers selection, scoping, and
provenance of that mechanism; no duplicate private cache-manager test belongs
in the DS4 suite.

### Hardware validation

After CPU and review gates pass, use fresh result directories in this order:

1. Run one optimized class-A target point at chunk budget 1024 with the full
   20-request, three-repetition contract. Every repetition must observe its
   planned P local-cache hits and pass all existing raw-evidence checks.
2. Run one optimized concurrency-4 class-B target point. Every request must
   report 128 output tokens and 127 ITL samples.
3. Only if both targets pass, run the complete selected pilot in a new
   directory, regenerate the report, and request human Gate D review.

Diagnostic `a10` is root-cause evidence, not acceptance evidence: it used a
diagnostic harness and one probe request, so it cannot replace the fresh
official one-point validation.

## Non-Goals

- Changing Mamba retention defaults for general vLLM users.
- Modifying cache-manager, scheduler, NIXL connector, or proxy algorithms.
- Increasing cache memory or deriving a dynamic retention interval.
- Adding a user-selectable DS4 retention option.
- Reordering, shrinking, or splitting the warm/measure workload.
- Repairing or relabeling historical artifacts.
- Claiming Gate D acceptance before a fresh pilot and human review.
