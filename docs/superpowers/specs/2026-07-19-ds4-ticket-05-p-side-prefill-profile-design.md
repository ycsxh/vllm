# DS4 Ticket 05 P-side Prefill Profile Design

Status: approved for implementation planning

## Goal

Extend the Ticket 04 profile spine into the complete P-side chunked-prefill
matrix on GPU0. The profile must compare real local prefix-cache hits with full
accumulated-context recomputation while preserving the production
`GPUWorker.execute_model` measurement boundary, the 4096-token batch scheduling
budget, and the eight-sequence limit.

Local development remains lightweight. It may run static checks and focused CPU
contract tests, but it must not load the Qwen model, initialize a GPU worker, or
claim runtime or hardware acceptance. Full execution and acceptance belong to
the documented dual-RTX-3090 school server.

## Approved Approach

Use vLLM's real Scheduler and KV-cache allocation state to prepare each
measurement, then time only the GPU worker execution. One long-lived GPU0
executor amortizes model initialization, compilation, and CUDA Graph capture
across the matrix. Scheduler, lookup, allocation, cache reset, and prefix
priming time are recorded separately from measured model time.

This approach is preferred because it combines the required low-level timing
boundary with real prefix lookup, block allocation, and capacity behavior.

Two alternatives were rejected:

1. Extending the hand-built Ticket 04 `SchedulerOutput` path would minimize the
   diff, but it would reproduce prefix-cache and allocation semantics in the
   benchmark and could create convincing but false hit or capacity evidence.
2. Running the matrix through `LLM.generate` would provide production cache
   behavior, but it would include engine scheduling and queueing work in the
   measured boundary and violate the project specification.

## Workload Matrix

Ticket 05 consumes the pinned Ticket 02 workload plan and rendered turns
without reselecting workloads:

- 15 homogeneous points from `p_homogeneous`;
- 9 mixed points: similar-length, seeded-random, and highly skewed composition
  at batch sizes 2, 4, and 8; and
- 10 exact replays: five `new_prefill_tokens` quantiles for each of `no_think`
  and `think_high`.

Each of the 34 workloads runs a `prefix_hit` and `full_recompute` condition,
for 68 logical points. Every point has three warmups and ten steady samples.
The implementation may support four or five warmups through configuration, but
the checked-in acceptance configuration uses three.

The homogeneous matrix uses one fixed 4096-token, block-aligned prefix per
request. Prefixes are deterministic and distinct between requests in the same
batch so that the result does not accidentally measure cross-request sharing.
The configured per-request scheduled length is appended to that prefix. The hit
condition primes and reuses the 4096-token prefix; the recompute condition
processes the full 4096-token prefix plus the scheduled suffix.

Mixed and exact-replay hit conditions use the Ticket 02 block-aligned
`reusable_prefix_tokens`; their recompute conditions process the complete
accumulated prompt. Requests remain in the workload plan's deterministic order.
The planner divides each batch into steps with at most 4096 total scheduled
tokens and at most eight active sequences. Requests that finish stop consuming
the next step's budget. Exact or mixed requests longer than one step therefore
produce multiple chunk rows and one full-turn sample.

No workload is silently shortened, split into a smaller batch, or removed when
it exceeds cache capacity.

## Real Prefix-hit Invariant

A prefix hit may never be inferred from equal token IDs or declared from a
planned cached-token count. Every hit repetition performs this fail-closed
sequence on GPU0:

1. Quiesce the worker and start a fresh cache epoch so blocks left by the prior
   repetition cannot extend the intended hit.
2. Submit the exact intended prefix through the real Scheduler and execute all
   prefix chunks through `GPUWorker.execute_model` on GPU0. This setup is
   synchronized and excluded from steady timing.
3. Finish the priming requests in the scheduler so their full blocks enter the
   real local prefix cache without being discarded from GPU0.
4. Submit the measured requests through the Scheduler and inspect its actual
   cache lookup and allocation result before the first timed chunk.
5. Verify, for every request and KV-cache group, that the computed cached-token
   count equals the intended block-aligned prefix, that the returned block IDs
   are the blocks created by the prime, and that those blocks remain allocated
   in the GPU0 worker's KV-cache tensors.
6. Record the verified block IDs, cached-token count, block count, and byte
   count in setup evidence, then begin timed suffix chunks.

Any missing block, unexpected cached token, block-ID mismatch, CPU-only state,
or failed residency check fails the point before timing. The corresponding
recompute repetition also starts from a fresh cache epoch, performs no prime,
and verifies zero cached tokens before its first timed chunk.

## Architecture

### Point planning

A new `benchmarks/ds4_profile/prefill_profile.py` owns immutable P-point and
per-request plans. It converts Ticket 02 workload entries into ordered request
dimensions, expands the two cache conditions, creates deterministic chunk
plans, and enforces scheduling budgets before GPU initialization.

The planner also computes the expected row cardinality for every successful or
out-of-capacity point. This plan is persisted before execution and is the source
of truth used by the artifact validator.

### Scheduler and cache adapter

The same module contains a narrow, version-specific adapter around the current
vLLM Scheduler and KV-cache manager. It owns cache-epoch reset, request
submission, prefix lookup, block allocation, block-residency evidence, request
completion, and allocation-failure classification. It does not reimplement
hashing, block matching, LRU, or free-queue rules.

### GPU execution

The Ticket 04 executor initialization, synchronization, CUDA-event timing, and
runtime-mode observation are reused through small shared helpers. The P runner
executes Scheduler-produced work at `GPUWorker.execute_model`. It records
scheduler/setup timings separately and excludes model load, compilation,
capture, cache reset, prefix priming, and warmups from steady aggregates.

The effective compilation configuration must cover all planned batch execution
shapes through explicit sizes or supported padded capture buckets. Every warmup
and steady chunk must report `FULL` or `PIECEWISE`; a `NONE` runtime mode is an
invalid hardware result.

### Container orchestration

The Ticket 03 runtime gains a P-only command that runs one process bound to
GPU0 and NUMA node 0. It reuses the existing preflight, mount contract, source
provenance, atomic output staging, and validation command. It does not launch a
Decode worker.

## Data Flow

```text
Ticket 02 workload_plan.json + rendered_turns.parquet
  -> frozen 34-workload plan
  -> hit/recompute expansion and deterministic chunks (68 points)
  -> GPU0 cache epoch
  -> optional real prefix prime and residency verification
  -> Scheduler lookup/allocation
  -> timed GPUWorker.execute_model chunks
  -> per-turn samples
  -> aggregates and hit/miss comparisons
  -> independent schema and arithmetic validation
  -> atomically finalized result directory
```

## Artifact Contract and Schema Evolution

Ticket 05 writes schema version `2.0.0`. The validator retains an explicit v1
dispatch so accepted Ticket 04 artifacts remain valid; it never interprets a
v1 artifact as v2 or mixes schema versions within a result directory.

The effective `run-config.json` contains a canonical point manifest. Each
point's canonical JSON includes every workload dimension: workload family and
selector, ordered trajectory/turn references, reasoning modes, composition and
seed, batch size, per-request context/cached/new/scheduled-token vectors, chunk
budget, cache condition, block size, the homogeneous-prefix rule, and capacity
target. `point_id` is a deterministic hash of this complete payload. A paired
`comparison_id` hashes the same payload without cache condition. The validator
recomputes both IDs rather than trusting stored strings.

The v2 result directory contains:

- `run-config.json`: frozen runtime and complete point manifest;
- `raw_samples.parquet`: one row per executed chunk or terminal failed chunk;
- `turn_samples.parquet`: one row per repetition with summed full-turn time,
  throughput, token/block/byte accounting, and allocation state;
- `aggregates.parquet`: steady-only median, p90, mean, coefficient of variation,
  and noisy flag for full-turn time and throughput;
- `comparisons.parquet`: paired hit time, recompute time, and recompute penalty;
- `provenance.json`: source, image, hardware, topology, runtime, cache capacity,
  prime/residency evidence summary, and validation state; and
- `result.md`: concise status, capacity boundary, noisy points, and artifact
  references.

Raw and turn rows record at least phase, repetition, chunk index/count,
per-request scheduled-token vectors, total scheduled tokens, context/cached/new
and recomputed tokens, runner wall time, CUDA model time, CUDA Graph runtime
mode, requested/allocated KV blocks and bytes, hit/miss lookup time, allocation
time, status, and structured error.

All enums have explicit supported sets. The validator checks schema metadata,
nullability, schema-version equality, role, family, condition, composition,
phase, row kind, runtime mode, status, and allocation-state values. It also
recomputes sample IDs, row cardinality, chunk and turn totals, throughput,
median, inclusive p90, mean, CV, noisy flags, paired hit/miss values, and
recompute penalties from lower-level rows. Stored aggregates that disagree are
corrupt, even if their Parquet schema is valid.

## Capacity and Failure Semantics

An actual Scheduler/KV allocator refusal caused by insufficient blocks is a
valid `out_of_capacity` observation. It records requested and available blocks
and bytes, the failing request/chunk, and all earlier completed rows. The
workload is not altered, and later independent points may continue after a
successful cache reset. A run may be hardware validated when all such points
are completely and consistently classified.

An unexpected exception, CUDA allocation failure during model execution,
residency-verification failure, cache-reset failure, unexpected cache hit,
missing CUDA Graph evidence, or arithmetic/schema mismatch invalidates the
worker result. Because CUDA failures may poison the process, execution stops
instead of claiming later points are independent. Preflight failure remains
`skipped` or `invalid`, never passed.

Artifacts preserve the complete prefix setup and chunk rows available before a
failure. Output finalization still uses a staging directory; invalid but
structurally complete evidence may be finalized for diagnosis, while an
interrupted staging directory is never a successful result.

## Testing

Local tests are focused CPU contracts with fake scheduler, cache, residency,
and execution adapters. They do not import a model or initialize CUDA. Tests
cover:

- exactly 34 workloads and 68 hit/recompute points from the pinned plan;
- the fixed 4096-token homogeneous prefix and distinct per-request prefixes;
- maximum 4096 scheduled tokens and eight sequences for every chunk;
- deterministic request ordering, multi-chunk completion, and stable raw/turn
  row cardinality;
- actual-prime state transitions, block-ID/residency matching, and fail-closed
  rejection of token-only, partial, stale, or excessive hits;
- zero-cache verification for recompute conditions;
- cached/new/recomputed-token, block, byte, full-turn, and throughput
  accounting;
- valid out-of-capacity recording without workload alteration;
- point-ID sensitivity to every workload dimension and comparison-ID pairing;
- v1 compatibility plus rejection of unknown or cross-file schema versions;
- rejection of every unsupported enum value; and
- independent recomputation of aggregates and comparisons, including deliberate
  corruption of each stored statistic.

Container-plan tests assert the P command uses GPU0/NUMA0, the expected mounts,
and the same v2 validator. A hardware-gated test exists only for the school
server and asserts behavior and artifacts without latency thresholds.

## Observability and School-server Acceptance

Acceptance is smoke-first:

1. On the exact clean feature-branch commit and commit-labeled image, run
   preflight, the CPU fixture/validator, and the printed P-only container plan.
2. Run a small representative GPU0 smoke set containing homogeneous, mixed,
   exact, hit, recompute, multi-chunk, and capacity-pressure points.
3. Independently inspect prime evidence, actual cached blocks, CUDA Graph modes,
   token/block accounting, and v2 validation before starting the full matrix.
4. Run all 68 points, validate them through a fresh container invocation, run
   the hardware-gated pytest, and retain commands, logs, checksums, source SHA,
   dirty state, image ID, and result directory.

The planning estimate is three to eight hours for the full matrix, with risk of
a longer run from repeated real prefix priming, 39k-token contexts, compilation
of additional shapes, or memory-pressure recovery. No acceptance criterion
depends on a latency threshold. A failed smoke run is diagnosed and fixed
before paying for the full matrix.

## Files Owned by Ticket 05

Expected implementation scope:

- `benchmarks/ds4_profile/prefill_profile.py`;
- a Ticket 05 profile configuration under `benchmarks/ds4_profile/config/`;
- targeted v2/version-dispatch changes in `profile_spine.py`;
- small shared executor/timing changes in `gpu_profile.py` only when required;
- the P-only command in `container/runtime.py` and its runbook;
- `tests/benchmarks/ds4_profile/test_prefill_profile.py` plus targeted existing
  artifact/container tests; and
- Ticket 05 design, handoff, and README links.

`workloads.py` and Ticket 02 artifacts are inputs, not places to redesign the
selection. Ticket 07's KVCacheManager trace-replay module is outside this
branch; shared-file edits must remain narrowly scoped so the branches can be
rebased without merging two cache abstractions.

## Non-goals

- D-side decode profiling or teacher-forcing changes;
- P-to-D KV transfer;
- offline LRU/capacity curves, cache-policy simulation, or Ticket 07 replay;
- the later GPU cache-capacity sweep;
- FP8 KV results;
- HTTP, proxy, queueing, or online scheduler latency;
- a new serving or eviction policy;
- performance thresholds in tests; and
- any mutation or public interaction in `vllm-project/vllm`.

## Acceptance Criteria

Ticket 05 is complete only when:

- the frozen plan contains the specified 68 logical points with no silent
  pruning or budget violation;
- every hit repetition executes its prefix on GPU0 and passes block-level
  residency and cached-token verification before timing;
- every recompute repetition begins with a verified zero-token cache hit;
- measured work occurs at `GPUWorker.execute_model` with compile and CUDA Graphs
  enabled, and startup/capture/setup/warmup work is excluded from steady data;
- raw chunks, turn totals, hit/miss comparisons, capacity states, and v2
  artifacts pass independent identifier, enum, schema, cardinality, and
  arithmetic validation;
- expected out-of-capacity points are recorded honestly and unexpected runtime
  failures invalidate the run;
- the full P matrix runs through the GPU0/NUMA0 container wrapper on the school
  server; and
- local and server evidence remain clearly separated and tied to exact source
  and image identities.
