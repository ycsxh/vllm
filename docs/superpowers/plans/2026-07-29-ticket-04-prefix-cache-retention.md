# Ticket 4 Prefix-Cache Retention Implementation Plan

**Goal:** Make every controlled DS4 Qwen3.5 P/D run use sparse Mamba
prefix-cache retention so the 20-prefix warm phase preserves planned prefixes
at the 1024- and 2048-token chunk budgets.

**Architecture:** `run_pd.build_plan` remains the single source of truth for
the fixed 1P1D runtime. It will set the existing
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` only on P and D, serialize the
effective numeric value in `LaunchPlan.compatibility`, and leave the proxy,
host-runtime input contract, measurement protocol, and vLLM core unchanged.

**Tech Stack:** Python 3.12, pytest, pre-commit, vLLM P/D serving, NIXL,
official `vllm bench serve`.

## Global Constraints

- Work only in `ycsxh/vllm`; treat `vllm-project/vllm` as read-only.
- Preserve `attempt-02`, `diagnostic-a01`, `diagnostic-b01`, and probes
  `a02` through `a10` as immutable.
- Use a new result directory for every validation or pilot.
- Keep default optimized mode, the selected workload, and the
  reset-P/reset-D/warm-P/reset-D/measure sequence unchanged.
- Preserve official detailed benchmark JSON, P/D metric deltas,
  `derive_run_result`, cache-hit tolerances, and fail-closed NIXL checks.
- Do not modify scheduler, cache-manager, NIXL connector, or proxy code or
  algorithms.
- Do not add a CLI option or a host-provided runtime-environment requirement.
- Use `.venv/bin/python` for every Python command.
- Observe the launcher contract test failing for the intended reason before
  changing `run_pd.py`.

## Confirmed Test Seam

The public seam is:

```text
run_pd.build_plan(LaunchConfig) -> LaunchPlan.as_dict()
```

The regression observes the effective P/D/proxy environments and serialized
compatibility record. It does not mock or inspect private cache-manager or
scheduler internals.

---

### Task 1: Add the Authoritative Fixed-Runtime Contract

**Files:**

- Modify: `benchmarks/ds4_profile/AUTHORITATIVE_SPEC.md`

- [ ] Add a fixed-runtime table entry requiring
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` on P and D.
- [ ] Explain immediately below the table that zero keeps Mamba replay and
  detected shared-prefix boundaries instead of dense intermediate
  checkpoints, while full-attention caching remains dense.
- [ ] State that this is a fixed Qwen3.5 DS4 condition, not an experiment axis
  or host prerequisite.
- [ ] Confirm no workload, measurement, or acceptance section changes.

### Task 2: Establish the Red Launcher Regression

**Files:**

- Modify: `tests/benchmarks/ds4_profile/test_run_pd.py`

- [ ] Extend `test_build_plan_freezes_complete_child_environments` at the
  confirmed public seam:
  - P environment has the exact string value `"0"`;
  - D environment has the exact string value `"0"`;
  - proxy environment does not contain the setting;
  - compatibility records numeric
    `prefix_cache_retention_interval == 0`.
- [ ] Use literal expected values from the approved design; do not derive the
  expected value from `run_pd`.
- [ ] Run the single test before implementation:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_run_pd.py::test_build_plan_freezes_complete_child_environments \
  -q
```

Expected: FAIL because P/D lack the fixed environment value and compatibility
lacks the field. Collection, imports, fixtures, and sockets must succeed.

### Task 3: Implement the Minimal Fixed Launcher Setting

**Files:**

- Modify: `benchmarks/ds4_profile/run_pd.py`
- Test: `tests/benchmarks/ds4_profile/test_run_pd.py`

- [ ] Add one module constant:

```text
PREFIX_CACHE_RETENTION_INTERVAL = 0
```

- [ ] In `LaunchPlan.as_dict()["compatibility"]`, serialize the constant under
  `prefix_cache_retention_interval`.
- [ ] In `build_plan`, add
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` after the validated host runtime
  environment in `common_environment`, converting the constant to `"0"`.
- [ ] Do not add the variable to `RUNTIME_ENVIRONMENT_NAMES` or the proxy
  environment.
- [ ] Rerun the single regression and require PASS.
- [ ] Run the complete launcher file:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  -q
```

### Task 4: Verify CPU and Static Contracts

- [ ] Run the existing core semantic coverage:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py \
  -k 'mamba_reachable_block_mask or mamba_shared_prefix_survives_zero_retention' \
  -q
```

- [ ] Run the focused launcher/runner files:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_run_points.py \
  -q
```

- [ ] Run the complete DS4 suite:

```bash
.venv/bin/python -m pytest tests/benchmarks/ds4_profile -q
```

If sandbox socket permissions alone fail, rerun the same command with local
socket access. Do not reinterpret a real test failure as a permission issue.

- [ ] Run checks only on changed files:

```bash
pre-commit run --files \
  benchmarks/ds4_profile/AUTHORITATIVE_SPEC.md \
  benchmarks/ds4_profile/run_pd.py \
  tests/benchmarks/ds4_profile/test_run_pd.py
```

- [ ] Inspect the diff and confirm `derive_run_result`, workload plans, proxy,
  and vLLM core are untouched.
- [ ] Commit the authoritative spec, regression, and minimal implementation
  together with attribution trailers.

### Task 5: Independent Implementation Review

- [ ] Run the two-axis `code-review` workflow from design-complete base
  `61f58f765`.
- [ ] Standards review uses `AGENTS.md` and the existing test conventions.
- [ ] Spec review uses
  `docs/superpowers/specs/2026-07-29-ticket-04-prefix-cache-retention-design.md`
  and the original Ticket 4 handoff.
- [ ] Resolve every finding and rerun affected checks before GPU execution.
- [ ] Require a clean worktree and an exact committed validation SHA.

### Task 6: Fresh Class-A Target Validation

- [ ] Copy the immutable diagnostic A plan to a new external plan path and
  change only its ID from diagnostic to validation. Preserve its 75% hit,
  chunk budget 1024, concurrency 1, output length 1, 20 prompts, three
  repetitions, and source request.
- [ ] Allocate a new result directory containing the committed validation SHA;
  prove it does not exist before launch.
- [ ] Run `benchmarks.ds4_profile.run_points` with the accepted cached
  tokenizer/model revision, FLASH_ATTN, CPUs `0,2,4,6,8,10` on NUMA 0 for P,
  CPUs `1,3,5,7,9,11` on NUMA 1 for D, and the existing 900/300/30-second
  timeouts.
- [ ] Require all three repetitions to pass the existing validator, including
  exact planned P local-cache-hit evidence, successful D external transfer,
  complete official raw results, and zero failed-transfer counters.
- [ ] Preserve the result and launcher log whether the validation passes or
  fails. Never rerun in place.

### Task 7: Fresh Class-B Target Validation

- [ ] Create a new one-point plan from the immutable selected-pilot point with
  optimized mode, concurrency 4, output length 128, 20 prompts, and three
  repetitions; do not change its selected workload parameters.
- [ ] Allocate another new result directory containing the same committed SHA.
- [ ] Run the same fixed launcher and official benchmark path.
- [ ] Require every successful request in every repetition to report exactly
  128 output tokens and 127 ITL samples, along with all existing cache-transfer
  and raw-evidence checks.
- [ ] Preserve the result and launcher log on either outcome.

### Task 8: Pilot Decision and Handoff

- [ ] If either target validation fails, stop before the full pilot, preserve
  evidence, and update the handoff with the exact failure.
- [ ] If both pass, review their raw artifacts before creating a new full-pilot
  result directory.
- [ ] Run the complete selected pilot only from a clean committed SHA, never
  overwriting `attempt-02`.
- [ ] Regenerate and independently audit the report before requesting human
  Gate D review.
- [ ] Do not retire the legacy path until explicit Gate D acceptance.
- [ ] Update `TICKET_04_MEASUREMENT_FIX_HANDOFF.md` with commits, commands,
  counts, review verdicts, immutable result paths, and the next exact gate.
