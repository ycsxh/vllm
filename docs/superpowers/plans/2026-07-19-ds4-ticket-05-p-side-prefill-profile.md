# DS4 Ticket 05 P-side Prefill Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 68-point GPU0 P-side chunked-prefill matrix with real prefix-cache priming, block-residency evidence, schema-v2 artifacts, and fail-closed validation.

**Architecture:** A pure CPU planner expands the pinned Ticket 02 inputs into canonical point and chunk plans. A narrow adapter drives the real vLLM Scheduler and KVCacheManager but times only `GPUWorker.execute_model`; shared artifact code writes and independently revalidates schema-v2 rows while retaining Ticket 04 schema-v1 validation.

**Tech Stack:** Python 3.12, vLLM V1 Scheduler/GPUWorker internals, PyTorch CUDA events, PyArrow/Parquet, pytest, argparse, Docker, numactl.

## Global Constraints

- Work only in `/private/tmp/vllm-ticket-05` on `codex/ticket-05-p-side-prefill-profile`; mutate no upstream GitHub object and push only to the personal fork when the human author later authorizes transfer.
- Never use system `python3` or bare `pip`. From this worktree, use `/Users/liuyuncong/GitHub/vllm/.venv/bin/python` for every local Python command.
- Local work is static or focused CPU-contract work only: do not load Qwen, initialize CUDA, set the hardware gate, or claim runtime acceptance.
- Full model execution, GPU timing, and acceptance run only on the school dual-RTX-3090 server through the container.
- GPU0/NUMA0 is the only Ticket 05 runtime role; never launch the Ticket 04 Decode worker.
- Enforce at most 4096 scheduled tokens across a batch and at most 8 active sequences.
- Consume the pinned Ticket 02 selections unchanged: 15 homogeneous, 9 mixed, and 10 exact workloads, each with `prefix_hit` and `full_recompute`, totaling 68 points.
- Homogeneous requests use distinct deterministic 4096-token block-aligned prefixes; checked-in configuration uses 3 warmups, 10 steady samples, and a 5% noisy-CV threshold.
- A hit repetition must persist the prime SchedulerOutput block tables sent to GPU0, execute and synchronize every prime chunk, and verify that the measured SchedulerOutput reuses those same physical block IDs in live `worker.model_runner.kv_caches` tensors before timing. Every mapped tensor must be an actual `torch.Tensor` on `cuda:0`; equal token IDs or config-only capacity are never hit evidence.
- Recompute repetitions begin in a fresh cache epoch and must verify zero cached tokens before timing.
- Time only `vllm.v1.worker.gpu_worker.Worker.execute_model`; record Scheduler, allocation, cache reset, and prime work separately.
- Main results require torch.compile and CUDA Graph runtime mode `FULL` or `PIECEWISE`; `NONE` invalidates a measured point.
- New artifacts use schema `2.0.0`; validation of accepted Ticket 04 schema `1.0.0` remains supported without cross-version coercion.
- Empty, partial, or preempted Scheduler output is never timed. The whole point is `out_of_capacity` only when the expected batch was isolated, allocator pressure is proven from required versus allocatable blocks, and cleanup returns to a verified empty reset epoch; every other mismatch invalidates the run.
- Every v2 run declares `run_kind` as `full` or `smoke` and freezes its expected manifest before execution. Full requires the canonical 68 IDs; smoke requires exactly the canonical configured selection. The validator never infers the expected set from observed rows.
- The v2 validator requires the frozen manifest ID set and exact planned phase/repetition/chunk coordinates. The only shortened coordinate sequence it accepts is a strict prefix ending in one structurally valid terminal `out_of_capacity` row.
- Emit a comparison if and only if both paired manifest points pass. Omit it if and only if valid terminal evidence proves at least one side is OOC; every missing or extra comparison invalidates the run.
- Do not add latency thresholds to correctness or hardware-gated tests.

## File Map

- Create `benchmarks/ds4_profile/prefill_profile.py`: immutable workload plans, chunk planning, Scheduler/KV adapter, GPU0 runner, v2 worker result, and CLI.
- Create `benchmarks/ds4_profile/config/p-prefill-profile.json`: frozen Ticket 05 runtime, artifact, matrix, compile/capture, and sampling settings.
- Modify `benchmarks/ds4_profile/profile_spine.py:16-63,292-325,361-464,613-684`: schema-version dispatch, chunk-aware canonical IDs, v2 timing and prefix-evidence schemas, exact-cardinality validation, independent aggregate/comparison validation, and shared validation CLI.
- Modify `benchmarks/ds4_profile/gpu_profile.py:14-89,173-212`: expose reusable executor initialization, synchronized execution, and CUDA Graph observation without changing Ticket 04 behavior.
- Modify `benchmarks/ds4_profile/container/runtime.py:734-901,902-990`: add a P-only GPU0/NUMA0 orchestration command.
- Create `tests/benchmarks/ds4_profile/test_prefill_profile.py`: CPU planner, cache-state-machine, artifact, container-plan, and hardware-gated contracts.
- Modify `tests/benchmarks/ds4_profile/test_profile_spine.py:185-348,496-716`: schema-v1 compatibility and schema-v2 fail-closed validator tests.
- Modify `benchmarks/ds4_profile/container/README.md:175-284`, `benchmarks/ds4_profile/README.md:125-170`, and `benchmarks/ds4_profile/WORKFLOW.md:95-173`: Ticket 05 local/server commands and evidence gates.
- Create `benchmarks/ds4_profile/TICKET_05_HANDOFF.md`: exact commit/image/result transfer and smoke-first acceptance checklist.

---

### Task 1: Canonical identifiers and schema-v2 validation

**Files:**
- Modify: `benchmarks/ds4_profile/profile_spine.py:16-63,292-325,361-464`
- Modify: `tests/benchmarks/ds4_profile/test_profile_spine.py:185-348,496-716`

**Interfaces:**
- Produces: `canonical_payload_json(payload: dict[str, Any]) -> str`
- Produces: `make_point_id(payload: dict[str, Any]) -> str`
- Produces: `make_comparison_id(payload: dict[str, Any]) -> str`
- Produces: `_validate_result_dir(result_dir: Path) -> None`, dispatching exact schema versions `1.0.0` and `2.0.0`
- Produces: `V2_RAW_SAMPLE_SCHEMA`, `V2_TURN_SAMPLE_SCHEMA`, `V2_AGGREGATE_SCHEMA`, `V2_COMPARISON_SCHEMA`, and `V2_PREFIX_EVIDENCE_SCHEMA`

- [ ] **Step 1: Add failing identifier and version-dispatch tests**

Add tests which construct one complete v2 result directory and retain the
existing v1 fixture. Use a canonical payload containing every dimension and
assert mutation of each nested vector changes the identifier:

```python
def test_v2_point_id_covers_every_workload_dimension() -> None:
    payload = {
        "workload_family": "homogeneous",
        "selector": "b2-t512",
        "requests": [
            {
                "request_key": "r0",
                "trajectory_id": None,
                "turn_index": None,
                "reasoning_mode": None,
                "context_tokens": 4608,
                "cached_tokens": 4096,
                "new_tokens": 512,
                "token_digest": "a" * 64,
            }
        ],
        "composition": "none",
        "seed": 20260715,
        "batch_size": 1,
        "chunk_budget": 4096,
        "cache_condition": "prefix_hit",
        "block_size": 16,
        "homogeneous_prefix_tokens": 4096,
        "capacity_target": "native",
        "planner_digest": "b" * 64,
        "planned_chunks": [
            {
                "chunk_index": 0,
                "scheduled_tokens_by_request": [["r0", 512]],
            }
        ],
    }
    original = profile_spine.make_point_id(payload)
    changed = copy.deepcopy(payload)
    changed["requests"][0]["cached_tokens"] = 4080
    assert original.startswith("p2-")
    assert profile_spine.make_point_id(changed) != original
    assert profile_spine.make_comparison_id(payload) == (
        profile_spine.make_comparison_id(
            {
                **payload,
                "cache_condition": "full_recompute",
                "planned_chunks": [
                    {
                        "chunk_index": 0,
                        "scheduled_tokens_by_request": [["r0", 4096]],
                    },
                    {
                        "chunk_index": 1,
                        "scheduled_tokens_by_request": [["r0", 512]],
                    },
                ],
            }
        )
    )
    changed_chunk = copy.deepcopy(payload)
    changed_chunk["planned_chunks"][0]["scheduled_tokens_by_request"][0][1] = 511
    assert profile_spine.make_point_id(changed_chunk) != original
    changed_planner = {**payload, "planner_digest": "c" * 64}
    assert profile_spine.make_point_id(changed_planner) != original
```

Add `test_v1_result_still_validates`,
`test_v2_validator_rejects_unknown_schema_version`,
`test_v2_validator_rejects_each_unknown_enum`, and
`test_v2_validator_recomputes_aggregate_statistics`. Add
`test_v2_validator_requires_exact_manifest_and_coordinates`, covering a
missing manifest point, an extra point, a missing warmup, duplicate chunk,
wrong per-request vector, a gap before a terminal OOC row, and any row after a
terminal OOC row. Add
`test_v2_validator_uses_frozen_manifest_for_full_and_smoke`, which requires a
`full` run to freeze exactly the canonical 68 IDs and a `smoke` run to freeze
exactly the canonical configured selection, then proves that observed-row
subsets/supersets cannot redefine either expected set. Add comparison cases
which reject a missing row when both conditions pass, an extra row when either
condition is terminal OOC, and any duplicate or unknown comparison; accept an
omission only with valid terminal OOC evidence for at least one paired point.
The arithmetic fixture
uses steady full-turn values `[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
17.0, 18.0, 19.0]`, then corrupts the stored median and expects
`ValueError("aggregate statistics do not match turn samples")`.

- [ ] **Step 2: Run the focused tests and confirm the intended failure**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_profile_spine.py \
  -k 'v1_result or v2_' -v
```

Expected: FAIL during collection or execution because `make_point_id` and the
v2 schemas do not exist.

- [ ] **Step 3: Implement exact version, enum, and identifier contracts**

Keep the existing schemas under v1 names and add these exact public constants
and helpers:

```python
V1_SCHEMA_VERSION = "1.0.0"
V2_SCHEMA_VERSION = "2.0.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({V1_SCHEMA_VERSION, V2_SCHEMA_VERSION})
V2_ENUMS = {
    "role": frozenset({"prefill"}),
    "run_kind": frozenset({"full", "smoke"}),
    "workload_family": frozenset({"homogeneous", "mixed", "exact_replay"}),
    "cache_condition": frozenset({"prefix_hit", "full_recompute"}),
    "composition": frozenset({"none", "similar", "random", "high_skew"}),
    "phase": frozenset({"warmup", "steady"}),
    "row_kind": frozenset({"chunk", "terminal"}),
    "runtime_mode": frozenset({"FULL", "PIECEWISE"}),
    "status": frozenset({"passed", "out_of_capacity", "failed"}),
    "allocation_state": frozenset({"allocated", "out_of_capacity", "failed"}),
}


def canonical_payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _identifier(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_payload_json(payload).encode()).hexdigest()
    return f"{prefix}-{digest}"


def make_point_id(payload: dict[str, Any]) -> str:
    return _identifier("p2", payload)


def make_comparison_id(payload: dict[str, Any]) -> str:
    comparison = copy.deepcopy(payload)
    comparison.pop("cache_condition")
    comparison.pop("planned_chunks")
    return _identifier("pc2", comparison)
```

`planned_chunks` is condition-specific and therefore participates in
`point_id` but is intentionally removed with `cache_condition` from the paired
`comparison_id`; planner digest and all condition-independent workload
dimensions remain in the comparison payload.

Define the five v2 Arrow schemas with the field names and nullability from the
approved design. `V2_RAW_SAMPLE_SCHEMA` contains planned/actual request-token
vectors, preempted/unrelated request IDs, allocator/reset proof, chunk timing, and token/block
fields; `V2_TURN_SAMPLE_SCHEMA` contains full-turn totals;
`V2_AGGREGATE_SCHEMA` contains timing and throughput median/p90/mean/CV/noisy;
`V2_COMPARISON_SCHEMA` contains paired hit/miss IDs and recompute penalty; and
`V2_PREFIX_EVIDENCE_SCHEMA` contains point/phase/ordinal/request/group,
prime SchedulerOutput block IDs, measured SchedulerOutput block IDs, live
worker KV tensor names, devices, shapes, block axis and block dimension,
verified physical IDs, intended/actual cached tokens, completed/synchronized
prime flags, `live_cuda_tensor_proven`, and `hardware_validated`.
Every schema carries `metadata={b"schema_version": b"2.0.0"}`.

Refactor `_validate_result_dir` to read `run-config.json` first and dispatch to
`_validate_v1_result_dir` or `_validate_v2_result_dir`. The v2 path must check
all required files, exact schemas and metadata, all enums, run/point/comparison
references, and deterministic IDs from `run-config.json["points"]`. Require
`run_kind`, `canonical_full_manifest`, and `expected_manifest` to be frozen
before execution. Recompute the canonical planner output from immutable config:
`canonical_full_manifest` must be exactly its 68 distinct IDs; for `full`,
`expected_manifest` must equal all 68, while for `smoke` it must equal exactly
the canonical IDs resolved from the configured smoke selectors. Never derive
either manifest from
raw, turn, aggregate, comparison, or evidence rows. Require the union of
successfully measured and terminal-OOC point IDs to equal the frozen expected
set. For a feasible point, require all
planned chunks for warmup ordinals `0..2` and steady ordinals `0..9`. For OOC,
accept only a lexicographic coordinate prefix followed by exactly one terminal
row at the next planned coordinate, with no later raw/turn/aggregate rows.
Also validate sample IDs, exact stored per-request vectors, planner digest,
prefix-evidence cardinality for every completed hit repetition, and recomputed
statistics using
`statistics.quantiles(values, n=10, method="inclusive")[8]`.
For every hardware-validated prefix row, require completed/synchronized/live
proof flags, only `cuda:0` device strings, valid block axes for the stored
shapes, a positive common block dimension, and all verified physical IDs in
that dimension. Preserve `hardware_validated=False` for CPU fixture evidence;
serialized metadata alone is not a hardware acceptance claim.

Partition the frozen expected manifest by `comparison_id`, requiring exactly
one hit and one recompute point per pair. If both outcomes are passed, require
exactly one comparison row whose point IDs and statistics recompute from their
aggregates. If either outcome has a valid terminal OOC row, require zero
comparison rows. Reject all missing, extra, duplicate, unknown, or otherwise
unproven omissions.

- [ ] **Step 4: Run focused and full Ticket 04 artifact tests**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_profile_spine.py -v
```

Expected: all existing v1 tests and new v2 hardening tests PASS.

- [ ] **Step 5: Commit the validator boundary**

```bash
git add benchmarks/ds4_profile/profile_spine.py \
  tests/benchmarks/ds4_profile/test_profile_spine.py
git commit -m "[Benchmarks] Harden DS4 profile artifact contracts" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

### Task 2: Expand the pinned workload plan into 68 points

**Files:**
- Create: `benchmarks/ds4_profile/prefill_profile.py`
- Create: `tests/benchmarks/ds4_profile/test_prefill_profile.py`

**Interfaces:**
- Consumes: `make_point_id(payload)` and `make_comparison_id(payload)` from Task 1
- Produces: `PRequestPlan`, `PChunkPlan`, and `PPointPlan` frozen dataclasses
- Produces: `make_planner_digest(workload_plan: dict[str, Any], rendered_turns: list[dict[str, Any]], *, block_size: int, token_budget: int, homogeneous_prefix_tokens: int, seed: int) -> str`
- Produces: `build_prefill_points(workload_plan: dict[str, Any], rendered_turns: list[dict[str, Any]], *, block_size: int, token_budget: int, homogeneous_prefix_tokens: int, seed: int) -> tuple[PPointPlan, ...]`
- Produces: `plan_chunks(point: PPointPlan, token_budget: int) -> tuple[PChunkPlan, ...]`

- [ ] **Step 1: Write failing matrix and chunk-planning tests**

Create tests that load the pinned plan and rendered-turn fixture through a
small helper, then assert exact family counts, condition pairing, fixed
homogeneous prefixes, and chunk budgets:

```python
def test_planner_expands_the_pinned_matrix_to_68_points() -> None:
    points = _build_pinned_points()
    assert len(points) == 68
    assert Counter(point.workload_family for point in points) == {
        "homogeneous": 30,
        "mixed": 18,
        "exact_replay": 20,
    }
    assert all(point.batch_size <= 8 for point in points)
    assert all(
        sum(chunk.scheduled_tokens_by_request.values()) <= 4096
        for point in points
        for chunk in point.chunks
    )
    assert len({point.planner_digest for point in points}) == 1
    assert all(
        point.canonical_payload["planned_chunks"]
        == [
            {
                "chunk_index": chunk.chunk_index,
                "scheduled_tokens_by_request": sorted(
                    chunk.scheduled_tokens_by_request.items()
                ),
            }
            for chunk in point.chunks
        ]
        for point in points
    )


def test_homogeneous_prefixes_are_4096_tokens_and_request_distinct() -> None:
    point = next(
        point
        for point in _build_pinned_points()
        if point.selector == "b8-t512" and point.cache_condition == "prefix_hit"
    )
    assert {request.cached_tokens for request in point.requests} == {4096}
    assert len({request.token_digest for request in point.requests}) == 8
    assert point.chunks[0].scheduled_tokens_by_request == {
        request.request_key: 512 for request in point.requests
    }
```

Also assert deterministic IDs, mixed request order, exact replay selection,
`full_recompute` context lengths, active-request removal, and rejection of a
plan whose workload exceeds eight sequences or whose block-aligned prefix is
not divisible by 16. Mutate one planned chunk vector and the planner algorithm
version independently; each mutation must change `point_id`.

- [ ] **Step 2: Run planner tests and verify failure**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  -k 'planner or homogeneous or chunk' -v
```

Expected: FAIL with `ModuleNotFoundError` for
`benchmarks.ds4_profile.prefill_profile`.

- [ ] **Step 3: Implement immutable plans and deterministic chunking**

Add these exact types:

```python
CacheCondition = Literal["prefix_hit", "full_recompute"]
WorkloadFamily = Literal["homogeneous", "mixed", "exact_replay"]


@dataclass(frozen=True)
class PRequestPlan:
    request_key: str
    trajectory_id: str | None
    turn_index: int | None
    reasoning_mode: str | None
    prompt_token_ids: tuple[int, ...]
    context_tokens: int
    cached_tokens: int
    new_tokens: int
    token_digest: str


@dataclass(frozen=True)
class PChunkPlan:
    chunk_index: int
    scheduled_tokens_by_request: dict[str, int]


@dataclass(frozen=True)
class PPointPlan:
    point_id: str
    comparison_id: str
    workload_family: WorkloadFamily
    selector: str
    composition: str
    seed: int
    batch_size: int
    cache_condition: CacheCondition
    planner_digest: str
    requests: tuple[PRequestPlan, ...]
    chunks: tuple[PChunkPlan, ...]
    canonical_payload: dict[str, Any]
```

Generate homogeneous token IDs deterministically from the sorted set of legal
Ticket 02 execution token IDs. Hash `(seed, selector, request_index, position)`
to select each token, and reject any pair of 16-token blocks shared by requests
in the same batch. Mixed and exact requests use the exact rendered execution
prompt IDs and slice the approved block-aligned prefix.

For each chunk, set `cap = token_budget // active_request_count` and schedule
`min(remaining, cap)` in request order. Remove completed requests and continue
until all intended timed tokens are consumed. Hit remaining lengths equal
`new_tokens`; recompute remaining lengths equal `context_tokens`.

Compute one planner digest before expanding points:

```python
def make_planner_digest(
    workload_plan: dict[str, Any],
    rendered_turns: list[dict[str, Any]],
    *,
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
    seed: int,
) -> str:
    payload = {
        "planner_schema": "ds4-p-prefill-plan-v1",
        "workload_plan": workload_plan,
        "rendered_turns": rendered_turns,
        "block_size": block_size,
        "token_budget": token_budget,
        "homogeneous_prefix_tokens": homogeneous_prefix_tokens,
        "seed": seed,
        "chunk_algorithm": "equal-active-cap-v1",
        "homogeneous_token_algorithm": "sha256-legal-pool-v1",
    }
    return hashlib.sha256(canonical_payload_json(payload).encode()).hexdigest()
```

Plan chunks before computing identifiers. Store the planner digest and the
ordered list of every `chunk_index` plus sorted per-request scheduled-token
vector in the canonical point payload. `make_point_id` therefore changes if
the inputs, planner algorithm, chunk count, request order, or any planned token
allocation changes.

- [ ] **Step 4: Run planner tests**

Run the command from Step 2.

Expected: all selected planner tests PASS with 68 points and no chunk above
4096 tokens.

- [ ] **Step 5: Commit the pure planner**

```bash
git add benchmarks/ds4_profile/prefill_profile.py \
  tests/benchmarks/ds4_profile/test_prefill_profile.py
git commit -m "[Benchmarks] Plan the DS4 P-side matrix" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

### Task 3: Build v2 chunk, turn, aggregate, and comparison artifacts

**Files:**
- Modify: `benchmarks/ds4_profile/profile_spine.py:292-464`
- Modify: `benchmarks/ds4_profile/prefill_profile.py`
- Modify: `tests/benchmarks/ds4_profile/test_prefill_profile.py`
- Modify: `tests/benchmarks/ds4_profile/test_profile_spine.py`

**Interfaces:**
- Consumes: `PPointPlan` and v2 schemas from Tasks 1-2
- Produces: `make_prefill_chunk_row(*, run_id: str, point: PPointPlan, phase: Literal["warmup", "steady"], ordinal: int, chunk: PChunkPlan, runner_wall_time_ms: float | None, cuda_model_time_ms: float | None, allocation: dict[str, Any], status: str, error: str | None) -> dict[str, Any]`
- Produces: `summarize_turn_samples(raw_rows: list[dict[str, Any]], points: tuple[PPointPlan, ...]) -> list[dict[str, Any]]`
- Produces: `aggregate_turn_samples(turn_rows: list[dict[str, Any]], noisy_cv_threshold: float) -> list[dict[str, Any]]`
- Produces: `compare_conditions(aggregate_rows: list[dict[str, Any]], points: tuple[PPointPlan, ...], terminal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]`
- Produces: `write_v2_result_artifacts(config: dict[str, Any], raw_rows: list[dict[str, Any]], prefix_evidence_rows: list[dict[str, Any]], provenance: dict[str, Any], output_dir: Path) -> None`

- [ ] **Step 1: Add failing accounting and expected-OOC tests**

Use two chunks with wall times 4 ms and 6 ms and assert the turn is 10 ms,
throughput is `1000 * new_tokens / 10`, and only ten steady turn rows enter the
aggregate. Pair a 10 ms hit with a 16 ms recompute and assert a 6 ms penalty.

```python
def test_turn_and_comparison_statistics_are_recomputed_from_chunks() -> None:
    point_pair, raw_rows = _two_condition_rows()
    turns = profile_spine.summarize_turn_samples(raw_rows, point_pair)
    aggregates = profile_spine.aggregate_turn_samples(turns, 0.05)
    comparisons = profile_spine.compare_conditions(aggregates, point_pair, [])
    assert turns[0]["runner_full_turn_time_ms"] == 10.0
    assert comparisons[0]["recompute_penalty_ms"] == 6.0
```

Add an OOC fixture whose rows are an exact coordinate prefix followed by one
terminal row. Assert the whole point status is `out_of_capacity`,
requested/allocatable block counts are retained, no aggregate is emitted for
that point, and v2 validation succeeds. Add rejection cases for a partial
nonterminal batch, a coordinate gap, multiple terminal rows, a terminal row
without proven pressure/reset evidence, and any timing on a partial batch.
Add paired-outcome fixtures: two passed conditions emit exactly one comparison;
one or two terminal-OOC conditions emit none; a missing comparison for two
passed conditions, an extra comparison for an OOC pair, or a comparison that
does not reference the frozen pair fails validation.

- [ ] **Step 2: Run artifact tests and verify failure**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  tests/benchmarks/ds4_profile/test_profile_spine.py \
  -k 'turn_and_comparison or out_of_capacity or v2_validator_recomputes' -v
```

Expected: FAIL because the v2 summarizers and writer do not exist.

- [ ] **Step 3: Implement one-way derivation and atomic writing**

Use raw chunks as the sole arithmetic source. Group by
`(run_id, point_id, phase, ordinal)`, require chunk indices
`0..chunk_count-1`, sum wall/CUDA/token/block fields, then derive turns.
Aggregate only `phase == "steady" and status == "passed"`. Pair conditions by
`comparison_id`; require exactly one hit and one recompute manifest point.
`compare_conditions` emits exactly one row only when both have passed
aggregates. It emits no row only when a validated terminal row proves one or
both conditions OOC; absence of an aggregate without that terminal proof is an
error, not permission to omit the comparison.

Build the expected coordinate sequence directly from each manifest point's
`planned_chunks`: warmup ordinals 0 through 2, then steady ordinals 0 through
9, each with every planned chunk in order. A passed point must equal the full
sequence. An OOC point may equal only a strict prefix plus one terminal row at
the next coordinate; terminal rows have null wall/CUDA timings and include
`allocator_pressure_proven=True`, `clean_reset_proven=True`, exact planned and
actual request vectors, required blocks, and allocatable blocks. Every point in
the frozen expected manifest must resolve to one of those two states; full has
exactly the canonical 68 and smoke has exactly the configured canonical subset.

```python
def _distribution(values: list[float], threshold: float) -> dict[str, Any]:
    mean = statistics.fmean(values)
    cv = statistics.pstdev(values) / mean if mean else 0.0
    return {
        "median": statistics.median(values),
        "p90": statistics.quantiles(values, n=10, method="inclusive")[8],
        "mean": mean,
        "cv": cv,
        "noisy": cv > threshold,
    }
```

Write `prefix_evidence.parquet` and all other v2 files into a sibling temporary directory, call the same v2
validator against staging, create the final directory only after validation,
and move each file into it. Invalid but structurally complete runs use the same
schemas and explicit status; an interrupted staging directory is never passed.

- [ ] **Step 4: Run artifact tests and the full v1/v2 validator suite**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_profile_spine.py \
  tests/benchmarks/ds4_profile/test_prefill_profile.py -v
```

Expected: all CPU artifact, planner, and Ticket 04 compatibility tests PASS.

- [ ] **Step 5: Commit v2 artifact production**

```bash
git add benchmarks/ds4_profile/profile_spine.py \
  benchmarks/ds4_profile/prefill_profile.py \
  tests/benchmarks/ds4_profile/test_profile_spine.py \
  tests/benchmarks/ds4_profile/test_prefill_profile.py
git commit -m "[Benchmarks] Add P-side profile artifacts" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

### Task 4: Adapt the real Scheduler and KV cache state

**Files:**
- Modify: `benchmarks/ds4_profile/prefill_profile.py`
- Modify: `tests/benchmarks/ds4_profile/test_prefill_profile.py`

**Interfaces:**
- Consumes: vLLM `Scheduler.add_request`, `schedule`, `update_from_output`, `finish_requests`, `reset_prefix_cache`; `KVCacheManager.get_block_ids` and `get_computed_blocks`
- Produces: `SchedulerBlockTableEvidence`, `WorkerKvTensorGroupEvidence`, `PrefixPrimeEvidence`, `AllocationEvidence`, and `ScheduledChunk` frozen dataclasses
- Produces: `VllmSchedulerCacheAdapter.reset_epoch() -> None`
- Produces: `VllmSchedulerCacheAdapter.prime(point: PPointPlan, phase: str, ordinal: int) -> tuple[PrefixPrimeEvidence, ...]`
- Produces: `VllmSchedulerCacheAdapter.schedule_chunk(point: PPointPlan, chunk: PChunkPlan) -> ScheduledChunk`
- Produces: `VllmSchedulerCacheAdapter.classify_out_of_capacity(point: PPointPlan, chunk: PChunkPlan, scheduled: ScheduledChunk) -> AllocationEvidence | None`
- Produces: `VllmSchedulerCacheAdapter.verify_recompute_miss(point, scheduler_output) -> None`

- [ ] **Step 1: Write failing fake-cache state-machine tests**

Build CPU fakes with the same methods as the current APIs. The hit test must
prove that equal tokens without an executed prime fail, and that mismatched,
partial, stale, excessive, or out-of-range block IDs fail before the timed
callback is called. CPU fakes may model the expected device/shape/block-axis
evidence contract for unit tests, but must always produce
`hardware_validated=False`; no fake tensor or CPU tensor may satisfy the
hardware-smoke predicate.

```python
def test_hit_requires_executed_gpu_prime_and_matching_resident_blocks() -> None:
    fake = FakeSchedulerCache(
        actual_cached_tokens=4096,
        prime_block_ids=((10, 11),),
        hit_block_ids=((10, 11),),
        gpu_block_capacity=128,
        prime_executed=False,
    )
    adapter = VllmSchedulerCacheAdapter(fake.scheduler, fake.executor, fake.worker)
    with pytest.raises(RuntimeError, match="prefix was not executed on GPU0"):
        adapter.verify_hit(_hit_point(), fake.scheduler_output)
    assert fake.timed_execute_calls == 0
```

Add reset failure, nonempty running queue, zero-hit recompute, and exact
planned-versus-actual per-request scheduled-vector tests. Add separate cases
for an empty SchedulerOutput, a partial request set, a partial token vector,
preempted expected requests, and an unrelated request in Scheduler state or
output. Every case asserts the GPU execute callback remains uncalled.

Add live-tensor contract cases which reject a non-`torch.Tensor`, a CPU tensor,
`cuda:1`, a missing configured layer tensor, an invalid block axis, and a
physical block ID equal to or larger than the live tensor's block dimension.
The CPU form tests the fail-closed inspection helper without claiming hardware
validation; only the hardware-gated test may exercise the successful live
`cuda:0` branch.

For OOC classification, prove all of the following in tests: the Scheduler was
isolated to the exact expected active request set, the empty/partial/preempted
output contains no unrelated request, `required_blocks > allocatable_blocks`,
no GPU timing occurred, cleanup flushed every request, and the subsequent
cache reset reported zero live requests and only the null block in use. If any
predicate is false, expect an invalid-result exception rather than OOC.

- [ ] **Step 2: Run cache-adapter tests and verify failure**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  -k 'prime or resident or reset_epoch or allocator' -v
```

Expected: FAIL because `VllmSchedulerCacheAdapter` is not defined.

- [ ] **Step 3: Implement the version-specific adapter with lazy vLLM imports**

Define evidence types and keep all vLLM imports inside the real adapter factory
so planner/artifact tests stay lightweight:

```python
@dataclass(frozen=True)
class SchedulerBlockTableEvidence:
    chunk_index: int
    request_key: str
    block_ids_by_group: tuple[tuple[int, ...], ...]
    execution_completed: bool
    gpu_synchronized: bool


@dataclass(frozen=True)
class WorkerKvTensorGroupEvidence:
    group_index: int
    tensor_names: tuple[str, ...]
    tensor_devices: tuple[str, ...]
    tensor_shapes: tuple[tuple[int, ...], ...]
    block_axis: int
    block_dimension: int
    verified_block_ids: tuple[int, ...]
    live_cuda_tensor_proven: bool


@dataclass(frozen=True)
class PrefixPrimeEvidence:
    request_key: str
    intended_cached_tokens: int
    actual_cached_tokens: int
    prime_scheduler_outputs: tuple[SchedulerBlockTableEvidence, ...]
    measured_block_ids_by_group: tuple[tuple[int, ...], ...]
    worker_groups: tuple[WorkerKvTensorGroupEvidence, ...]
    prime_completed: bool
    prime_synchronized: bool
    hardware_validated: bool
    kv_bytes: int


@dataclass(frozen=True)
class AllocationEvidence:
    state: Literal["allocated", "out_of_capacity", "failed"]
    requested_blocks: int
    allocated_blocks: int
    allocatable_blocks: int
    requested_bytes: int
    allocated_bytes: int
    lookup_time_ms: float
    allocation_time_ms: float
    allocator_pressure_proven: bool
    clean_reset_proven: bool


@dataclass(frozen=True)
class ScheduledChunk:
    scheduler_output: Any
    expected_request_ids: tuple[str, ...]
    actual_request_ids: tuple[str, ...]
    expected_tokens_by_request: dict[str, int]
    actual_tokens_by_request: dict[str, int]
    preempted_request_ids: tuple[str, ...]
    unrelated_request_ids: tuple[str, ...]
    allocation: AllocationEvidence
```

Construct real requests as
`Request(request_id, list(tokens), SamplingParams(max_tokens=1,
temperature=0.0), None)`. Before each scheduler call, set
`scheduler.max_num_scheduled_tokens` and
`scheduler.scheduler_config.long_prefill_token_threshold` from the planned
chunk, then assert `SchedulerOutput.num_scheduled_tokens` exactly equals the
planned vector, its request-ID set exactly equals the planned active request
set, and its total is at most 4096. Before calling `schedule`, require
`scheduler.requests`, running, and waiting state to contain no unrelated
request. Empty, partial, preempted, or unrelated output returns a
`ScheduledChunk` for fail-closed classification but may never reach
`execute_model`.

For reset, finish all known requests with
`RequestStatus.FINISHED_ABORTED`, execute one zero-token SchedulerOutput to
flush `finished_req_ids` from the worker, assert no running/waiting requests,
then require `scheduler.reset_prefix_cache()` to return `True`.

For every prime chunk, persist the exact block tables in
`SchedulerOutput.scheduled_new_reqs[*].block_ids` and
`scheduled_cached_reqs.new_block_ids` that are sent to the worker. Execute the
prime output untimed, call Scheduler `update_from_output`, and synchronize GPU0
before marking that chunk complete. Capture final
`scheduler.kv_cache_manager.get_block_ids(request_id)` before finish/free.

For the measured request, inspect `NewRequestData.num_computed_tokens` and its
SchedulerOutput block tables after real lookup. The prefix slice must equal the
same physical IDs persisted from the completed prime for every cache group.
Map every cache-group entry and configured layer name to the corresponding live
entry in the ordered `worker.model_runner.kv_caches` list using the same layer
ordering as vLLM's `bind_kv_cache` after the synchronized prime. Require
every mapped value to satisfy `isinstance(tensor, torch.Tensor)`,
`tensor.is_cuda`, and `tensor.device == torch.device("cuda:0")`. Resolve and
record the cache group's physical-block axis from its backend/live layout,
record each
tensor's exact device string and shape, require all tensors in the group to
agree on the block dimension, and require every measured physical block ID to
satisfy `0 <= block_id < tensor.shape[block_axis]` for every mapped live
tensor. Configured capacity alone is not evidence of residency.

Persist this live mapping, plus `prime_completed=True`,
`prime_synchronized=True`, `live_cuda_tensor_proven=True`, and
`hardware_validated=True`, in `PrefixPrimeEvidence`. The CPU fake adapter may
construct structurally identical rows only with both proof fields false; the
artifact writer and validator must not promote them. Compute bytes from
`kv_cache_group.kv_cache_spec.page_size_bytes`.

Classify the whole point OOC only when an empty/partial/preempted output occurs
with the exact isolated expected batch and an allocator snapshot proves
`required_blocks > allocatable_blocks`. Do not time any fraction of that
batch. Finish and flush all point requests, require a successful prefix-cache
reset, and verify the clean epoch before setting
`allocator_pressure_proven=True` and `clean_reset_proven=True`. A partial
output without proven pressure, any unrelated request, or failed cleanup/reset
is `invalid` and stops the run.

- [ ] **Step 4: Run all adapter CPU contracts**

Run the command from Step 2.

Expected: all selected state-machine tests PASS without importing a model or
initializing CUDA.

- [ ] **Step 5: Commit the real-cache boundary**

```bash
git add benchmarks/ds4_profile/prefill_profile.py \
  tests/benchmarks/ds4_profile/test_prefill_profile.py
git commit -m "[Benchmarks] Add the P-side scheduler cache adapter" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

### Task 5: Execute prefix primes and timed GPU0 chunks

**Files:**
- Modify: `benchmarks/ds4_profile/gpu_profile.py:14-89,173-212`
- Modify: `benchmarks/ds4_profile/prefill_profile.py`
- Modify: `tests/benchmarks/ds4_profile/test_prefill_profile.py`

**Interfaces:**
- Produces: `GpuRuntime` dataclass carrying executor, worker, vLLM config, and KV config
- Produces: `initialize_gpu_runtime(config: dict[str, Any]) -> GpuRuntime`
- Produces: `execute_worker_step(runtime: GpuRuntime, scheduler_output: Any, *, timed: bool) -> tuple[Any, float | None, float | None]`
- Produces: `run_prefill_matrix(config: dict[str, Any], points: tuple[PPointPlan, ...]) -> dict[str, Any]`

- [ ] **Step 1: Add failing orchestration and hardware-gate tests**

Use fake runtime/adapter callbacks to verify exact ordering for every
repetition: reset, optional prime, verify hit/miss, warmup or steady timed
chunks, turn record. Assert an OOC point continues only after a successful
reset; a residency or CUDA-mode error stops the worker.

```python
def test_hit_repetition_primes_and_verifies_before_any_timed_chunk() -> None:
    events: list[str] = []
    result = run_point_repetition(
        _hit_point(),
        phase="steady",
        ordinal=0,
        adapter=FakeAdapter(events),
        execute_timed=FakeTimedExecutor(events),
    )
    assert events[:4] == ["reset", "prime_gpu0", "verify_resident", "timed:0"]
    assert result["status"] == "passed"
```

Add `test_empty_partial_preempted_or_unrelated_scheduler_output_is_never_timed`
with four parametrized SchedulerOutput shapes. Add
`test_allocator_pressure_marks_the_whole_point_ooc_after_clean_reset`, which
asserts the terminal row is emitted at the failed planned coordinate, all
later repetitions for that point are skipped, and the next point begins only
after a verified empty reset epoch. Add the inverse cases: insufficient proof
or failed reset invalidates the worker.

Add a prefix-evidence test that compares the exact prime SchedulerOutput block
tables with the measured SchedulerOutput prefix block IDs and each worker KV
tensor group's live tensor names, `cuda:0` devices, shapes, physical block axis,
block dimension, and verified IDs. Flip every proof bit or physical ID
independently and require failure before `timed:0`. Assert the fake runtime's
otherwise valid evidence remains `hardware_validated=False`.

Add a hardware-gated test decorated with
`pytest.mark.skipif(os.environ.get("DS4_P_PREFILL_GPU_SMOKE") != "1",
reason="school-server GPU acceptance only")`.
When enabled on the server, it runs the smoke config and asserts real prime
evidence from `prefix_evidence.parquet`: every mapped value was a live
`torch.Tensor` on `cuda:0`, every recorded block axis/dimension matches its
runtime shape, and all physical IDs are in bounds. It also asserts
`hardware_validated=True`, GPU0 role, low-level boundary, CUDA Graph modes,
`run_kind == "smoke"`, the exact frozen configured smoke manifest and
coordinates, comparison completeness, and valid v2 output. A fake or CPU
tensor must fail this test even if its serialized metadata looks plausible.

- [ ] **Step 2: Run local orchestration tests and confirm failure/skip**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  -k 'repetition or gpu_smoke' -v
```

Expected: fake orchestration test FAIL because `run_point_repetition` is
missing; the real hardware test is SKIPPED.

- [ ] **Step 3: Extract shared GPU helpers without changing Ticket 04**

Replace the tuple returned by `_initialize_executor` with a typed helper while
keeping a compatibility wrapper for Ticket 04:

```python
@dataclass(frozen=True)
class GpuRuntime:
    executor: Any
    worker: Any
    vllm_config: Any
    kv_cache_config: Any
    startup_ms: float
    capture_ms: float
```

`execute_worker_step(runtime, scheduler_output, timed=False)` synchronizes after execution but
returns `None` timings. `timed=True` creates CUDA events, calls only
`executor.execute_model(scheduler_output)`, samples only when the current
runner contract returns `None`, synchronizes, and returns wall/CUDA times.
Ticket 04 `_execute_timed` delegates to this helper and retains all existing
tests and artifact values.

- [ ] **Step 4: Implement the matrix runner and fail-closed evidence**

Initialize one runtime and one real Scheduler using the same constructor
arguments as `vllm/v1/engine/core.py:150-158`, including
`StructuredOutputManager(vllm_config)` and resolved scheduler/hash block sizes.
For each point and repetition, execute the ordered state machine from Step 1.
Before every GPU call, require exact equality between the planned and actual
request-ID/token vector and require no unrelated request in Scheduler state or
output. Populate raw chunk rows only after the cache invariant has passed; no
partial batch is timed or written as a passed chunk. Require every
warmup/steady output's `cudagraph_stats.runtime_mode` to be `FULL` or
`PIECEWISE`.

Return a worker result containing schema/run/role/boundary, point manifest,
full per-repetition prime SchedulerOutput/block-to-worker-tensor evidence,
including live tensor devices, shapes, block axes/dimensions, and hardware
proof state, plus rows, capacity, compile/CUDA flags, and status. Set
`hardware_validated=True` only in the real GPU path after all synchronized
`worker.model_runner.kv_caches` checks pass; fake and CPU paths keep it false.
When proven allocator pressure
occurs, emit one terminal row, mark the whole point OOC, perform and record a
clean reset, and continue with the next point only after the reset invariant
passes. Convert every unproven partial/empty/preempted/unrelated output and all
other exceptions to a structured failed row and stop the worker. Always shut
down the executor in `finally`.

- [ ] **Step 5: Run lightweight tests and Ticket 04 regression tests**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  tests/benchmarks/ds4_profile/test_profile_spine.py -v
```

Expected: CPU tests PASS and the Ticket 05 real GPU test is SKIPPED. No model
loads and no CUDA process starts.

- [ ] **Step 6: Commit GPU execution wiring**

```bash
git add benchmarks/ds4_profile/gpu_profile.py \
  benchmarks/ds4_profile/prefill_profile.py \
  tests/benchmarks/ds4_profile/test_prefill_profile.py
git commit -m "[Benchmarks] Execute real P-side prefix profiles" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

### Task 6: Add the frozen config and P-only container command

**Files:**
- Create: `benchmarks/ds4_profile/config/p-prefill-profile.json`
- Modify: `benchmarks/ds4_profile/prefill_profile.py`
- Modify: `benchmarks/ds4_profile/container/runtime.py:734-901,902-990`
- Modify: `tests/benchmarks/ds4_profile/test_prefill_profile.py`

**Interfaces:**
- Produces: `python -m benchmarks.ds4_profile.prefill_profile plan|fixture|gpu-worker|assemble|validate`
- Produces: `python -m benchmarks.ds4_profile.container.runtime p-profile [--smoke] [--print-plan] [--output-dir PATH]`
- Produces: `freeze_expected_manifest(config: dict[str, Any], points: tuple[PPointPlan, ...], run_kind: Literal["full", "smoke"]) -> dict[str, Any]`

- [ ] **Step 1: Add failing config and container-plan tests**

Assert schema 2.0.0, 4096 prefix/budget, max sequences 8, 3/10 repetitions,
FP16 weights/KV, TP1, prefix caching/chunked prefill enabled, eager disabled,
and compile/capture buckets covering 128, 256, 512, 1024, 2048, and 4096.

```python
def test_p_profile_container_plan_is_gpu0_numa0_only(tmp_path: Path) -> None:
    command = runtime._p_profile_worker_command(Path("run.json"), tmp_path, False)
    assert "--membind=0" in command
    assert "CUDA_VISIBLE_DEVICES=0" in command
    assert "--role" not in command
    assert "decode" not in command
```

Assert `--smoke` selects one homogeneous pair, one mixed pair, one exact pair,
one pair containing a multi-chunk condition, and one capacity-pressure pair
without changing their
canonical IDs, planned chunk vectors, or planner digest. Assert the full
printed plan contains exactly the same 68 point IDs that the v2 validator later
requires. Assert the effective config is frozen before the worker starts with
`run_kind`, the complete 68-ID `canonical_full_manifest`, and an
`expected_manifest`: `full` equals all 68, while `smoke` equals exactly the
configured selected IDs. Delete or add an observed row and prove neither
manifest changes and validation fails.

- [ ] **Step 2: Run container/config tests and verify failure**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  -k 'config or container_plan or smoke_selection' -v
```

Expected: FAIL because the config and P-only runtime command do not exist.

- [ ] **Step 3: Add config, CLI, preflight, and atomic assembly**

Create the config with Ticket 04 artifact/model paths and these exact profile
values:

```json
{
  "schema_version": "2.0.0",
  "profile": {
    "block_size": 16,
    "homogeneous_prefix_tokens": 4096,
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 8,
    "warmup_repetitions": 3,
    "measured_repetitions": 10,
    "noisy_cv_threshold": 0.05,
    "seed": 20260715
  }
}
```

Retain the approved Qwen revision and full runtime fields from
`profile-spine.json`. Configure compile and capture buckets
`[128, 256, 512, 1024, 2048, 4096]` and validate observed runtime mode per row.
Store the representative smoke selectors in the checked-in config; resolve
them against the canonical 68-point plan before execution and fail if any
selector is missing, ambiguous, or breaks a hit/recompute pair.

The container command creates the run ID, runs existing preflight, launches
one numactl-bound worker, assembles v2 artifacts, and returns nonzero for
invalid output. A preflight failure writes a structurally valid skipped/invalid
result and never calls the worker. `--print-plan` performs no model or GPU work.
Before launch, write an immutable effective `run-config.json` containing
`run_kind`, the planner-recomputed `canonical_full_manifest`, and the selected
`expected_manifest`. Assembly must require `prefix_evidence.parquet`; it may
not reduce either manifest to observed rows. A full worker result must match
the exact frozen 68-ID set, and a smoke worker result must match exactly the
frozen configured selected set.

- [ ] **Step 4: Run container/config tests and print-plan locally**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py -v
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m \
  benchmarks.ds4_profile.container.runtime p-profile --print-plan
```

Expected: tests PASS; print-plan emits preflight, one GPU0/NUMA0 worker, and
assemble/validate commands without loading the model.

- [ ] **Step 5: Commit container integration**

```bash
git add benchmarks/ds4_profile/config/p-prefill-profile.json \
  benchmarks/ds4_profile/prefill_profile.py \
  benchmarks/ds4_profile/container/runtime.py \
  tests/benchmarks/ds4_profile/test_prefill_profile.py
git commit -m "[Benchmarks] Add the P-side container workflow" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

### Task 7: Document handoff and complete local/server gates

**Files:**
- Modify: `benchmarks/ds4_profile/container/README.md:175-284`
- Modify: `benchmarks/ds4_profile/README.md:125-170`
- Modify: `benchmarks/ds4_profile/WORKFLOW.md:95-173`
- Create: `benchmarks/ds4_profile/TICKET_05_HANDOFF.md`

**Interfaces:**
- Consumes: all prior CLIs and artifacts
- Produces: exact lightweight-local checklist and smoke-first school-server acceptance procedure

- [ ] **Step 1: Add concise local and school-server runbooks**

Document that local validation runs only the focused CPU suites, schema
validator, and print-plan. The handoff must name the exact feature commit,
clean/dirty state, config checksum, expected 68-point plan checksum, container
image ID field, result root, and the 3-8 hour estimate with long-context risk.

Document these school-server commands using the existing `DS4_RUN` array:

```bash
"${DS4_RUN[@]}" preflight
"${DS4_RUN[@]}" p-profile --print-plan
"${DS4_RUN[@]}" p-profile --smoke \
  --output-dir /mnt/ds4/results/ticket-05/smoke
"${DS4_RUN[@]}" exec --output /mnt/ds4/results/ticket-05-smoke-validation.json \
  -- /opt/ds4-profile/bin/python -m benchmarks.ds4_profile.prefill_profile \
  validate --result-dir /mnt/ds4/results/ticket-05/smoke
"${DS4_RUN[@]}" p-profile \
  --output-dir /mnt/ds4/results/ticket-05/full
```

The runbook requires human inspection of `prefix_evidence.parquet`: prime
SchedulerOutput block tables actually sent to GPU0, completed and synchronized
prime flags, measured SchedulerOutput reuse of the same physical IDs, and each
ID's mapping into live `worker.model_runner.kv_caches` `torch.Tensor` values on
`cuda:0`, including recorded shape, physical block axis, and block dimension.
It rejects fake/config-only evidence and requires `hardware_validated=True`.
It also checks cached-token counts,
zero-hit recompute evidence, CUDA Graph modes, OOC pressure/reset proof, the
declared `run_kind`, exact frozen smoke/full manifest and coordinate sets,
paired comparison completeness, and independent v2 arithmetic validation
before the full run. It retains
failed smoke/full artifacts and never relabels them as accepted.

- [ ] **Step 2: Run the complete lightweight local gate**

Run:

```bash
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_profile_spine.py \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  tests/benchmarks/ds4_profile/test_container_workflow.py \
  tests/benchmarks/ds4_profile/test_workloads.py -v
/Users/liuyuncong/GitHub/vllm/.venv/bin/python -m pre_commit run --files \
  benchmarks/ds4_profile/profile_spine.py \
  benchmarks/ds4_profile/prefill_profile.py \
  benchmarks/ds4_profile/gpu_profile.py \
  benchmarks/ds4_profile/container/runtime.py \
  benchmarks/ds4_profile/config/p-prefill-profile.json \
  tests/benchmarks/ds4_profile/test_profile_spine.py \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  benchmarks/ds4_profile/container/README.md \
  benchmarks/ds4_profile/README.md \
  benchmarks/ds4_profile/WORKFLOW.md \
  benchmarks/ds4_profile/TICKET_05_HANDOFF.md
```

Expected: all CPU tests and hooks PASS; the hardware-gated test reports SKIPPED.
If hook-environment download fails, preserve the exact infrastructure error and
rerun after network access is available before server handoff.

- [ ] **Step 3: Commit documentation and local evidence**

```bash
git add benchmarks/ds4_profile/container/README.md \
  benchmarks/ds4_profile/README.md \
  benchmarks/ds4_profile/WORKFLOW.md \
  benchmarks/ds4_profile/TICKET_05_HANDOFF.md
git commit -m "[Docs] Hand off the DS4 P-side prefill profile" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

- [ ] **Step 4: Run smoke-first hardware acceptance on the school server**

After the human pushes the exact reviewed commit to `ycsxh/vllm`, build a
commit-labeled image and run the commands from Step 1. Then enable the hardware
test only on the school server:

```bash
DS4_P_PREFILL_GPU_SMOKE=1 /opt/ds4-profile/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py \
  -k gpu_smoke -v
```

Expected smoke result: all selected points either pass or are honestly
`out_of_capacity`; no empty, partial, preempted, or unrelated SchedulerOutput
was timed; every hit has persisted prime SchedulerOutput tables, synchronized
completion, identical measured physical IDs, and live `cuda:0` torch-tensor
device/shape/block-axis evidence with in-bounds physical IDs and
`hardware_validated=True`; `run_kind` is `smoke` and the expected manifest is
exactly the configured canonical smoke set; every fully passed pair has exactly
one comparison and every OOC pair has none;
every recompute has zero cached tokens; every OOC point has allocator-pressure
and clean-reset proof; all measured rows use `FULL` or `PIECEWISE`; independent
validation returns zero.

Run the full matrix only after smoke acceptance. Expected full result: all 68
planned point IDs exactly match the frozen full manifest and planner digest,
`run_kind` is `full`, each
feasible point has every planned chunk for 3 warmups and 10 steady turns, OOC
points contain only a valid coordinate prefix plus one terminal evidence row,
every passed pair has exactly one recomputed comparison and every OOC pair has
none, all hit evidence comes from live `cuda:0` tensors, and provenance reports
a clean exact source SHA and immutable image ID.

- [ ] **Step 5: Record immutable acceptance evidence in a final docs-only commit**

Update only `TICKET_05_HANDOFF.md` with exact run IDs, image ID, source SHA,
dirty state, commands, test counts, validation result, checksums, failures, and
runtime. Do not alter runtime code after hardware acceptance.

```bash
git add benchmarks/ds4_profile/TICKET_05_HANDOFF.md
git commit -m "[Docs] Record Ticket 05 hardware acceptance" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```
