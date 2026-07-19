# Ticket 07 CPU Metadata Replay Handoff

## Status

Ticket 07 is **blocked and incomplete** on branch
`codex/ticket-07-kv-cache-manager-replay`. Keep personal-fork issue #4 open.
Do not create a PR or merge this branch.

The deterministic full-data planner failed closed because none of the 20
complete pilot trajectories produces native eviction pressure at its minimum
usable capacity. The checked-in selection remains `unselected`. Pinning a
trajectory, increasing capacity, shortening a prompt, skipping a turn, or
claiming replay acceptance would violate the approved design.

- Metadata-only validated: **no** — full replay acceptance was not reached.
- GPU/HBM validated: **no**.

Ticket 07 read prompt metadata and prompt token IDs only. It did not read
completion/decode token IDs, allocate KV tensors, load a model, execute
Prefill or Decode, use a GPU, or establish HBM residency.

## Source state

The coherent implementation checkpoint is
`06b0ebce3` (`[Benchmarks] Checkpoint DS4 replay artifacts`). The planning
diagnostic ran from base commit `e3aa14e29` with the later checkpoint changes
present as a dirty worktree; it is diagnostic failure evidence, not accepted
clean-image evidence. The documentation commit after this handoff does not
retroactively make that run clean.

Completed checkpoint work includes:

- the real `Request` → `KVCacheManager.get_computed_blocks` →
  `allocate_slots` → `take_events` → `free` seam;
- native `BlockRemoved` eviction authority, separate compulsory/capacity/
  prefix-mismatch/manager-forced-recompute outcomes, physical duplicate-hash
  occupancy, and ordered observer/native eviction pairing;
- deterministic candidate capacity and stable selection ordering;
- hashes for the manifest, Ticket 01/02 data and provenance, and every regular
  tokenizer file;
- content-addressed ordered turn manifests and pinned-selection verification;
  and
- a partial Task 5 Parquet writer/validator checkpoint with focused tamper
  tests.

Task 5 is not complete: the public `plan`/`run`/`validate` CLI, real hash
timing, complete physical occupancy reconstruction, and OOC/invalid partial
summary proof remain outstanding. Task 6 container registration was not
implemented. No Ticket 07 image was built, and no container replay or
independent container validation was run.

## Test evidence

Every Python/vLLM command used:

```bash
export PYTHONHASHSEED=0
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
```

Focused checkpoint suite:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py -q
```

Result: **33 passed**.

Real upstream manager regressions:

```bash
.venv/bin/python -m pytest \
  --confcutdir=tests/v1/core \
  tests/v1/core/test_prefix_caching.py::test_prefill \
  tests/v1/core/test_single_type_kv_cache_manager.py::test_evictable_cached_blocks_not_double_allocated \
  -v
```

Result: **3 passed, 14 warnings**. `--confcutdir` excludes the repository's
GPU cleanup fixture, which calls `torch.accelerator.empty_cache()` on this
CPU-only torch installation after otherwise passing tests.

The two focused Ruff hooks passed for the Python checkpoint:

```bash
.venv/bin/pre-commit run ruff-check --files \
  benchmarks/ds4_profile/kv_cache_replay.py \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py
.venv/bin/pre-commit run ruff-format --files \
  benchmarks/ds4_profile/kv_cache_replay.py \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py
```

The complete selected-file pre-commit command covering the replay module,
config, focused tests, Ticket 07 docs, design, and plan passed. Executed hooks
were Ruff check/format, typos, Markdown lint, mypy 3.10, SPDX, import/API and
configuration checks, plus the repository's other applicable local hooks;
non-applicable language/workflow hooks reported `Skipped`. Container tests
were not run because Task 6 was not implemented and planning had already
failed closed.

## Fail-closed planning evidence

The host-side CPU planning record is retained outside the repository at:

```text
/home/lyc/ds4-storage/results/ticket-07-selection-failed-e3aa14e29.json
```

Record facts:

- schema version: `1.0.0`;
- status: `no_selection`;
- candidates: `20`;
- eligible candidates: `0`;
- every candidate replay status: `passed`;
- every candidate native eviction count: `0`;
- canonical planning digest:
  `b14031a82a099ad48df187aa65bb7a0b3e2b112786ee94f75940fb66ae028431`;
- file SHA-256:
  `0380bbcbe9b87c103188618704990a19abc4415b395a1e0ce04fd4618215ce6a`;
  and
- input inventory: five manifest/Ticket 01/Ticket 02 data/provenance records
  plus nine regular files below the pinned DS4 tokenizer directory.

Each candidate capacity is exactly
`max(ceil(prompt_tokens / 16))` for that complete trajectory. Capacities range
from 730 to 3374 usable blocks. Every full session was admitted, but all
native eviction counts were zero. This is consistent with the pilot prompts'
prefix growth fitting within the largest prompt capacity. Lowering capacity
would reject the largest prompt; increasing it cannot create eviction
pressure.

The planning record was produced by `load_full_turns` followed by
`build_selection_plan` using the immutable school-server manifest, Ticket
01/02 artifacts and provenance, and pinned DS4 tokenizer. Triton reported zero
active drivers and disabled itself. No GPU execution occurred.

## Final branch review

The two-axis review used fixed point
`65de0de0ab4a5799284e97b823e673d5ac73ef05`.

- Standards: no hard violation. Judgement-call findings were the 1,861-line
  replay module's divergent responsibilities and the large dictionary-shaped
  event data clump.
- Spec: no-selection correctly follows the approved fail-closed rule. Missing
  work remains the public CLI, real hash timing, independently reconstructed
  physical occupancy, reliable partial invalid/OOC artifacts, CPU-only
  container registration, and all pinned container acceptance evidence.
- Scope: no vLLM core, Ticket 05, GPU profile spine, or GPU/HBM behavior was
  modified or claimed.

## Required decision before resuming

Do not resume Task 5–8 under the current approved selection rule. A human must
approve a revised design that supplies an eligible workload without weakening
admission or inventing eviction evidence. Examples of materially different
scope include selecting a workload with non-monotonic exact-hash reuse or
moving eviction-pressure exploration into Ticket 08. Any revision requires a
new design/plan review before implementation, a new planning record, and new
clean-image evidence.

The personal fork is the only writable GitHub repository. Upstream remains
read-only. No PR was created and issue #4 must remain open.
