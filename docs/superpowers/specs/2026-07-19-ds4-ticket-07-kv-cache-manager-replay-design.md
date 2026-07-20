# DS4 Ticket 07 KV Cache Manager Replay Design

## Scope

Ticket 07 proves one thin, auditable metadata-only replay of a complete DS4
session through vLLM's real `Request`, `KVCacheManager`, and `BlockPool`
semantics. It covers batch size one, serial turn order, one pinned trajectory,
one pinned cache capacity, and one full-attention KV cache group. Ticket 08
will expand this verified seam into capacity, concurrency, batching, reasoning
mode, and interleaving sweeps.

The replay runs on CPU and does not load a model, initialize a worker, execute
Prefill or Decode, allocate KV tensors, or reserve GPU memory. Its block IDs,
hashes, capacity, occupancy, and evictions are manager metadata. In particular,
the planning pass reads prompt metadata and computes prefix hashes and block
capacity only: it never consumes completion/decode tokens and never creates HBM
KV. Ticket 07 therefore makes no claim that any reported block is resident in
GPU HBM.

Ticket 07 is isolated from the concurrently developed Ticket 05 P-side
profile. It adds a new replay module, configuration, artifact contract, and
focused tests. It does not refactor or modify `profile_spine.py`,
`gpu_profile.py`, or Ticket 05 point and validator contracts.

## Considered Approaches

### Real Request and KVCacheManager with scoped BlockPool observation

Construct real vLLM `Request` objects, let their request block hasher generate
the chained full-block hashes, and replay every turn through
`KVCacheManager.get_computed_blocks`, `allocate_slots`, `free`, and
`take_events`. A scoped observer records calls to the manager's concrete
`BlockPool.touch`, `get_new_blocks`, and `free_blocks` methods, then restores
the original methods after the replay.

This is the selected approach. It exercises the required manager interface and
the real lookup, touch-before-allocation, lazy LRU eviction, caching, and
manager-defined reverse free behavior. Observation adds timing and ordering
evidence without substituting a cache implementation.

### Scheduler-level replay

A real Scheduler could drive the same manager. This would add request queues,
scheduler state, engine configuration, connector behavior, and preemption
paths that Ticket 07 does not need. It would make the interface shallower and
the CPU contract slower without strengthening this batch-1 manager proof. It
is rejected.

### Direct BlockPool replay with injected hashes

Driving `BlockPool` directly would make event capture simple, but it would
bypass request hashing, continuous-prefix lookup, top-level allocation
semantics, and manager-defined free behavior. It would amount to a partial
cache imitation and is rejected.

## Canonical Replay Planning

The committed replay configuration must pin both `trajectory_id` and
`capacity_blocks`. They are selected once by a deterministic, CPU-only,
read-only planning pass before implementation acceptance:

1. Re-render the complete pinned pilot trajectories with logical DS4 prompt
   token IDs using the immutable snapshot, Ticket 01 normalized turns, and the
   pinned DS4 tokenizer. Validate the resulting scalar turn and prefix fields
   against Ticket 02 `rendered_turns.parquet`.
2. For each complete trajectory, construct real `Request` objects and compute
   chained prefix hashes. Completion token IDs are neither loaded nor used.
3. Define a trajectory's minimum usable capacity as the maximum, over its
   turns, of `ceil(prompt_tokens / block_size)`. Usable capacity excludes the
   manager's reserved null block.
4. Replay the trajectory at that capacity. A candidate is eligible when every
   turn is admitted. Record the native `BlockRemoved` count as an observed
   workload property; zero evictions do not invalidate metadata replay.
5. Choose the eligible candidate by the stable key
   `(capacity_blocks, reasoning_mode_rank, trajectory_id)`, with
   `no_think` before `think_high`. The planning result records every candidate,
   rejection reason, selected trajectory, selected capacity, input hashes, and
   schema version.

The smallest capacity that can admit a trajectory's largest prompt is the only
capacity tested for that candidate: increasing it cannot create additional
pressure. If no trajectory admits every turn, planning fails closed instead of
weakening admission. Planning never lowers capacity, truncates prompts, skips
turns, or invents eviction evidence to manufacture pressure. The selected
values are then written explicitly into `config/kv-cache-replay.json`; normal
replay refuses derived or automatic capacity changes. School-server acceptance
reruns the planner and verifies that its selection exactly matches the pinned
config.

The planning pass is read-only with respect to the snapshot, Ticket artifacts,
tokenizers, repository, and remotes. It may write a planning record only to an
explicit result path.

## Module and Interface

`benchmarks.ds4_profile.kv_cache_replay` is the deep module. Its public CLI has
three commands:

- `plan`: inspect all full trajectories and emit the deterministic selection
  record without changing configuration;
- `run`: replay the pinned selection and write the artifact set; and
- `validate`: independently validate an existing result directory.

The caller supplies a frozen configuration and output path. The module owns
input validation, full-session reconstruction, manager construction,
observation, miss classification, future-reuse labeling, schemas, artifact
writing, validation, and Markdown rendering.

The configuration records schema version, run ID, immutable input paths and
revisions, selected trajectory and reasoning mode, usable block capacity,
block size, reproducible hash function, maximum model length, and source
commit/dirty state. The normal container path creates and persists a run ID
before replay. No model, CUDA, GPU role, or KV precision setting is accepted
because none is exercised.

## Data Flow and Real-Manager Seam

1. Load and validate the immutable manifest, Ticket 01 normalized turns,
   Ticket 02 rendered turns and provenance, and the pinned DS4 tokenizer.
2. Reuse `workloads.render_turns(..., include_token_ids=True)` to reconstruct
   logical prompt token IDs for the selected full trajectory. Ticket 02 stores
   execution token IDs only for selected profile points, so it cannot by itself
   supply a complete session.
3. Initialize vLLM's reproducible SHA-256 block hashing and create each
   `Request` with `get_request_block_hasher`. The Request computes the actual
   chained full-block hashes; the replay never injects synthetic hashes into
   the manager.
4. Precompute exact-hash future accesses and deterministic global, task, or
   session attribution for every full block. Boundary-spanning blocks use
   Ticket 02's existing block-start attribution rule.
5. Construct `KVCacheManager` with caching and native KV events enabled, one
   `FullAttentionSpec`, no cache tensors, and
   `num_blocks = capacity_blocks + 1` to account explicitly for the null block.
6. For each turn in serial order:
   - time Request hashing;
   - time `get_computed_blocks` and record the returned continuous full-block
     prefix;
   - call `allocate_slots` for the uncached prompt suffix while the scoped
     observer records real touch and allocation calls and their ordering;
   - drain `take_events` and treat native `BlockStored` and `BlockRemoved`
     hashes as authoritative store and eviction evidence;
   - snapshot request block IDs, active blocks, cached-resident hashes, and
     free blocks; and
   - call `free` while recording the exact block order passed to
     `BlockPool.free_blocks`.
7. Classify events, attach future-reuse labels, validate invariants, aggregate
   per-turn summaries, and finalize artifacts through a staging directory.

`KVCacheManager.usage` is not treated as cached occupancy because it returns to
zero after a request is freed even while prefix hashes remain evictable. The
contract reports active allocation and cached-resident metadata separately.

## Event and Miss Contract

Every event has a version, run/session/turn/event ID, turn and operation
ordinal, status, operation enum, duration in nanoseconds, block position,
physical block ID where observable, external block hash, prefix source, token
count, before/after active and cached occupancy, and structured error.
Fields that do not apply to an operation are nullable; for example, a lookup
has no physical allocation ID and a native store or eviction has no independent
duration outside the manager call that produced it.

Operation enums are `hash`, `lookup`, `touch`, `allocate`, `store`, `evict`,
`free`, and `admission_failure`. Zero-call operations remain visible in the
turn summary with count zero. Timings for hash, lookup, touch, allocation, and
free are not collapsed into one control time.

For each requested full block:

- `hit` means that the manager returned the block inside its continuous cached
  prefix.
- `capacity` means the exact chained hash was stored during an earlier access
  but is no longer resident when requested.
- `prefix_mismatch` means the exact chained hash is new, but an earlier turn
  had already reached that block position; the prefix chain or block content
  has diverged.
- `compulsory` means this is the first access at a prefix depth that no earlier
  turn reached.

These classes are mutually exclusive. If an exact resident hash lies beyond
the manager's returned continuous prefix, validation fails because that would
contradict the chained-hash lookup contract.

Per-turn cached tokens equal the manager-reported computed-token count;
recomputed tokens equal prompt tokens minus that count. This intentionally
includes any partial tail or final token that vLLM must recompute rather than
rounding the workload down to full blocks.

Prefix source is assigned by block start: before the recorded global boundary
is `global`, before the task boundary is `task`, and the remainder is
`session`. This preserves Ticket 02's deterministic ownership of blocks that
span a source boundary.

Future-use labeling is computed from the complete selected session before the
replay. Each native eviction records `useful_later`, `never_reused`,
`next_reuse_turn`, and `turns_until_reuse` for the exact chained hash. The two
boolean labels are complements, and reuse distance is null only when the hash
is never requested again. A run with no native evictions has no future-use
labels to satisfy; validation instead requires an empty native-eviction set and
zero eviction counts throughout the artifacts.

## Artifacts

Every finalized run directory contains:

- `run-config.json`: the frozen effective configuration;
- `cache_events.parquet`: block and control-operation events;
- `turn_summaries.parquet`: per-turn hits, misses, cached/recomputed tokens,
  allocations, evictions, frees, occupancy, timings, and admission status;
- `provenance.json`: source revisions and hashes, selection-plan identity,
  invocation, image, installed versions, source state, validation status, and
  boolean `pilot_eviction_pressure_observed`; and
- `result.md`: a concise table with the native eviction count, explicit pilot
  eviction-pressure status, and explicit metadata-only validation wording.

Parquet schemas and enum values are versioned and independently validated.
Stable IDs include the run, trajectory, turn, operation, and ordinal. Hashes
are serialized canonically so a result can be compared across processes.

## Failure Handling

Manifest, revision, tokenizer, artifact-schema, scalar-turn, trajectory, or
selection-plan mismatch fails before manager replay. A pinned config that does
not equal the deterministic planner selection is invalid.

If `allocate_slots` returns `None`, the replay emits an `admission_failure`
event, preserves all completed turns, marks the current turn out of capacity,
and stops. It never increases capacity, truncates a prompt, or skips a turn.
Unexpected exceptions preserve the failed operation and turn where possible.
Empty and partial Parquet artifacts retain the same schemas, provenance is
invalid, and `result.md` cannot claim success.

Artifacts are written in a staging directory and finalized only after schema,
identifier, accounting, event-order, miss-classification, reuse-label, and
cross-file validation. Planning failure emits a structured planning record but
does not create or alter the pinned config.

## Testing

Local development runs focused CPU contract tests only. Tiny hand-checkable
prompt sequences instantiate the real `Request` and `KVCacheManager`; tests do
not load tokenizers from the full pilot, a model, CUDA, or a GPU worker.

Tests cover:

- only complete hash blocks can hit;
- lookup stops at the first divergent prefix block;
- touch calls occur before allocation calls on the real manager path;
- lazy LRU eviction selects the expected block;
- `free` passes blocks to the pool in manager-defined reverse order;
- compulsory, capacity, and prefix-mismatch classes are mutually exclusive;
- global, task, and session attribution follows the block-start rule;
- useful-later and never-reused eviction labels and turn distance;
- active, cached-resident, and free-block accounting;
- explicit admission failure without capacity or prompt mutation;
- stable IDs, schemas, enums, units, nullability, provenance, and partial
  failure artifacts; and
- deterministic planning and pinned-config mismatch rejection.

Request and manager construction follow the existing patterns in
`tests/v1/core/test_prefix_caching.py`. Tests assert observable replay artifacts
and operation order rather than private cache-map shapes.

Acceptance separates workload replay from eviction conformance. Full pilot
data proves deterministic admission, lookup, allocation, free, event,
occupancy, manifest, and artifact behavior even when it produces no eviction.
A tiny deterministic real-manager fixture separately must produce at least one
native `BlockRemoved` and cover lazy LRU selection, capacity misses,
future-reuse labels, and eviction occupancy transitions. Compulsory, capacity,
and prefix-mismatch support is mandatory in focused contract tests; the pinned
pilot run need not happen to contain every class.

## Container Integration and Acceptance

The Ticket 03 image exposes `kv-cache-replay plan`, `run`, and `validate`
through minimal registration in `container/runtime.py`. `container/run.sh`
classifies these commands as CPU-only, so Docker does not request GPUs or
`SYS_NICE`. Inputs remain read-only mounts and outputs go under
`/mnt/ds4/results/ticket-07/<run_id>`.

Full dataset planning and container/runtime acceptance happen on the school
server. The accepted sequence is:

1. run the planner and verify its selected trajectory and capacity equal the
   committed config;
2. inspect the printed replay plan;
3. run the metadata replay through the standard image and mounted result root;
4. independently run `validate` against the completed directory;
5. retain artifact checksums, exact source SHA and dirty state, image ID,
   invocation, planning record, and focused test result; and
6. confirm the result says metadata-only validated and makes no GPU/HBM claim.

Acceptance requires every selected turn to be admitted, valid event ordering,
schema-valid artifacts, exact ordered-manifest and input-provenance matches,
and a clean independent validation. Native eviction count is a zero-or-more
workload observation. When native evictions occur, every removal must have
correct event pairing, occupancy transitions, and future-use labels. When none
occur, artifacts and validation must agree on zero without claiming that the
pilot exercised eviction pressure.

The separate focused conformance gate requires a deterministic real-manager
fixture with at least one native eviction and coverage of all three miss
classes. Passing both gates permits `metadata_only_validated: true`; reporting
must set `pilot_eviction_pressure_observed` from the native eviction count and
must state `GPU/HBM validated: no`. Acceptance does not require either RTX
3090, model weights, CUDA Graphs, or a latency threshold.

## Files Owned

Ticket 07 may add or modify only its focused surface:

- `benchmarks/ds4_profile/kv_cache_replay.py`;
- `benchmarks/ds4_profile/config/kv-cache-replay.json`;
- `tests/benchmarks/ds4_profile/test_kv_cache_replay.py`;
- minimal command registration in `benchmarks/ds4_profile/container/runtime.py`
  and CPU classification in `container/run.sh`;
- the DS4 overlay `container/Dockerfile`, limited to packaging the focused
  Ticket 07 test module for the exact-image conformance gate;
- focused README, container runbook, workflow, design, and handoff updates.

It does not own Ticket 04 spine hardening or any Ticket 05 implementation,
configuration, test, schema, point-ID, or validator change.

## Non-Goals

- Capacity curves, knee selection, multiple capacities, session concurrency,
  batching, or interleavings; these belong to Ticket 08.
- GPU cache-capacity validation or recompute latency; these belong to Ticket
  09 and depend on Ticket 05.
- Actual KV tensor contents, HBM residency, model execution, Decode tokens,
  Prefill timing, or KV precision comparisons.
- Scheduler, routing, admission, retention, migration, or eviction policy
  simulation.
- P-to-D or P-to-P KV transfer.
- Refactoring the shared Ticket 04/05 GPU profile spine or unifying its artifact
  schema with the metadata replay.
