# DS4 Ticket 07 KV Cache Manager Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one auditable, metadata-only replay of a pinned full DS4 trajectory through vLLM's real `Request` and `KVCacheManager`, with deterministic selection, cache-event artifacts, miss attribution, future-reuse labels, and container acceptance.

**Architecture:** Add one deep public module, `benchmarks.ds4_profile.kv_cache_replay`, whose CLI owns full-session reconstruction, deterministic planning, real-manager replay, scoped `BlockPool` observation, artifact writing, and independent validation. It calls `Request` → `KVCacheManager.get_computed_blocks` → `allocate_slots` → `take_events` → `free`; it observes but never replaces `BlockPool.touch`, `get_new_blocks`, or `free_blocks`. Ticket 07 stays isolated from the Ticket 04/05 GPU profile spine.

**Tech Stack:** Python 3.12, vLLM V1 `Request`/`KVCacheManager`/`BlockPool`, PyTorch CPU metadata types, PyArrow/Parquet, Hugging Face tokenizer files in offline mode, pytest, Docker Ticket 03 runtime.

## Global Constraints

- Run every Python command through `uv` and `.venv/bin/python`; never use system `python3`, bare `pip`, or `pip install`.
- Local work is limited to focused CPU contract tests and deterministic fixture artifacts; do not load Qwen, initialize CUDA, run a GPU worker, or opt into Ticket 04's GPU test.
- Full pilot planning and complete container/runtime acceptance run on the school server in the existing Ticket 03 image.
- The planner and replay consume prompt metadata and prompt token IDs only; they never read completion/decode token IDs, allocate KV tensors, reserve HBM, or make a GPU-residency claim.
- Use batch size one, serial turn order, block size `16`, SHA-256 chained
  hashes, `PYTHONHASHSEED=0`, one `FullAttentionSpec`, one pinned full
  trajectory, and one pinned usable capacity. Initialize vLLM's process-global
  null hash once at replay/process entry, never once per request. Usable
  capacity excludes the reserved null block.
- Set `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` before process startup and before
  any vLLM import in local tests, direct CLI use, and Docker. Native event
  hashes must remain byte-valued SHA-256 hashes identical to the request
  prepass; never convert or compare lossy 64-bit integer event hashes.
- Exercise the real `Request` and `KVCacheManager` interface. Do not implement an LRU, prefix lookup, allocation, or free-order imitation.
- Do not modify `benchmarks/ds4_profile/profile_spine.py`, `gpu_profile.py`, `config/profile-spine.json`, or Ticket 05 tests, schemas, point IDs, validators, and runtime behavior.
- Keep mounted snapshot, Ticket 01/02 artifacts, and tokenizers read-only. Write planning and replay outputs only below the explicit result directory.
- Upstream `vllm-project/vllm` remains read-only. Any later GitHub mutation or push must target only `ycsxh/vllm`; this plan itself creates no remote state.
- Preserve partial, invalid, and out-of-capacity artifacts with the same schemas. Never enlarge capacity, shorten a prompt, skip a turn, or turn a failed/skipped run into success.
- Use Google-style docstrings, Python line length `88`, minimal comments, and existing DS4 benchmark conventions.

---

## File Map

- Create `benchmarks/ds4_profile/kv_cache_replay.py`: the only public replay module; owns types, reconstruction, planning, real-manager adapter, scoped observation, classification, schemas, artifacts, validation, and CLI.
- Create `benchmarks/ds4_profile/config/kv-cache-replay.json`: frozen input/replay configuration. It begins in the explicit `unselected` state, then a server planning result pins the canonical trajectory, reasoning mode, capacity, and planning-record SHA-256.
- Create `tests/benchmarks/ds4_profile/test_kv_cache_replay.py`: focused CPU contracts using tiny prompts and the real manager, plus CLI/container plan checks.
- Modify `benchmarks/ds4_profile/container/runtime.py`: register CPU-only `kv-cache-replay plan|run|validate` commands and generate the effective run ID/source state.
- Modify `benchmarks/ds4_profile/container/run.sh`: classify `kv-cache-replay` as CPU-only so Docker does not request GPUs or `SYS_NICE`.
- Modify `benchmarks/ds4_profile/README.md`: document metadata-only semantics and local validation.
- Modify `benchmarks/ds4_profile/container/README.md`: document the school-server planning, pinning, run, validation, and evidence sequence.
- Modify `benchmarks/ds4_profile/WORKFLOW.md`: add Ticket 07's local/server gate without copying the runbook.
- Create `benchmarks/ds4_profile/TICKET_07_HANDOFF.md`: record the selected inputs, exact accepted SHA/image/result, commands, checksums, limitations, and restart instructions.

### Shared Interfaces Defined by This Plan

Use these names and types consistently in every task:

```python
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ReasoningMode = Literal["no_think", "think_high"]
PrefixSource = Literal["global", "task", "session"]
Operation = Literal[
    "hash",
    "lookup",
    "touch",
    "allocate",
    "store",
    "evict",
    "free",
    "admission_failure",
]
MissClass = Literal["compulsory", "capacity", "prefix_mismatch"]
CacheOutcome = Literal["hit", "miss", "manager_forced_recompute"]
EventSource = Literal["lookup", "observer", "native", "replay"]


@dataclass(frozen=True)
class ReplayTurn:
    trajectory_id: str
    task_id: str
    reasoning_mode: ReasoningMode
    turn_index: int
    prompt_token_ids: tuple[int, ...]
    prompt_tokens: int
    exact_lcp_tokens: int
    reusable_prefix_tokens: int
    global_prefix_tokens: int
    task_prefix_tokens: int


@dataclass(frozen=True)
class PoolCall:
    operation: Literal["touch", "allocate", "evict", "free"]
    call_ordinal: int
    block_ids: tuple[int, ...]
    duration_ns: int
    evicted: bool | None


@dataclass(frozen=True)
class InputRecord:
    logical_name: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ReplayResult:
    status: Literal["passed", "invalid", "out_of_capacity"]
    event_rows: tuple[dict[str, Any], ...]
    turn_rows: tuple[dict[str, Any], ...]
    eviction_count: int
    error: str | None
```

The module-level functions used across tasks have these exact signatures:

- `load_full_turns(config: dict[str, Any]) -> list[ReplayTurn]`
- `_initialize_hashing() -> None`
- `make_request(turn: ReplayTurn, block_size: int, request_id: str) -> Request`
- `make_manager(capacity_blocks: int, block_size: int, max_model_len: int) -> KVCacheManager`
- `observe_block_pool(manager: KVCacheManager) -> Iterator[list[PoolCall]]`
- `replay_session(*, run_id: str, turns: Sequence[ReplayTurn], capacity_blocks: int, block_size: int, max_model_len: int) -> ReplayResult`
- `build_selection_plan(config: dict[str, Any], turns: Sequence[ReplayTurn]) -> dict[str, Any]`
- `verify_pinned_selection(config: dict[str, Any], plan: dict[str, Any]) -> None`
- `write_result(config: dict[str, Any], result: ReplayResult, output_dir: Path) -> None`
- `validate_result_dir(result_dir: Path) -> None`
- `collect_input_records(config: dict[str, Any]) -> list[InputRecord]`
- `verify_input_records(records: Sequence[InputRecord]) -> None`
- `_selected_turn_manifest(turns: Sequence[ReplayTurn], block_size: int) -> list[dict[str, Any]]`
- `_out_of_capacity_result(*, run_id: str, turn: ReplayTurn, event_rows: list[dict[str, Any]], turn_rows: list[dict[str, Any]], manager: KVCacheManager, error: str) -> ReplayResult`

---

### Task 1: Full-Session Reconstruction and Input Contract

**Files:**
- Create: `benchmarks/ds4_profile/kv_cache_replay.py`
- Test: `tests/benchmarks/ds4_profile/test_kv_cache_replay.py`

**Interfaces:**
- Consumes: `benchmarks.ds4_profile.workloads.render_turns(manifest_path, normalized_turns_path, tokenizer_path, block_size=16, include_token_ids=True) -> list[dict[str, Any]]` and Ticket 02 `rendered_turns.parquet`.
- Produces: `ReplayTurn`, `load_full_turns(config) -> list[ReplayTurn]`, `_sha256(path: Path) -> str`, and `_canonical_hash(value: bytes | int) -> str`.

- [ ] **Step 1: Write the failing reconstruction tests**

Add imports and a row factory that contains prompt fields only; deliberately omit execution completion IDs:

```python
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

assert os.environ["PYTHONHASHSEED"] == "0"
assert os.environ["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] == "0"


def _turn_row(
    trajectory_id: str,
    turn_index: int,
    prompt_token_ids: list[int],
) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "task_id": trajectory_id.split(":", 1)[0],
        "reasoning_mode": trajectory_id.rsplit(":", 1)[1],
        "turn_index": turn_index,
        "prompt_tokens": len(prompt_token_ids),
        "exact_lcp_tokens": 0 if turn_index == 0 else 4,
        "reusable_prefix_tokens": 0 if turn_index == 0 else 4,
        "global_prefix_tokens": 2,
        "task_prefix_tokens": 4,
        "_prompt_token_ids": prompt_token_ids,
    }


def test_load_full_turns_rejects_ticket_02_scalar_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.ds4_profile import workloads
    from benchmarks.ds4_profile.kv_cache_replay import load_full_turns

    row = _turn_row("task:no_think", 0, [1, 2, 3, 4])
    monkeypatch.setattr(workloads, "render_turns", lambda **_: [row])
    ticket_02 = dict(row)
    ticket_02.pop("_prompt_token_ids")
    ticket_02["prompt_tokens"] = 5
    rendered_path = tmp_path / "rendered_turns.parquet"
    pq.write_table(pa.Table.from_pylist([ticket_02]), rendered_path)
    config = {
        "artifacts": {
            "manifest": str(tmp_path / "manifest.json"),
            "normalized_turns": str(tmp_path / "turns.parquet"),
            "rendered_turns": str(rendered_path),
        },
        "tokenizer": {"path": str(tmp_path / "tokenizer")},
        "replay": {"block_size": 4},
    }

    with pytest.raises(ValueError, match="Ticket 02 scalar mismatch"):
        load_full_turns(config)


def test_load_full_turns_keeps_prompt_ids_and_never_requires_decode_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.ds4_profile import workloads
    from benchmarks.ds4_profile.kv_cache_replay import load_full_turns

    rows = [
        _turn_row("task:no_think", 0, [1, 2, 3, 4]),
        _turn_row("task:no_think", 1, [1, 2, 3, 4, 5, 6]),
    ]
    monkeypatch.setattr(workloads, "render_turns", lambda **_: rows)
    ticket_02_rows = []
    for row in rows:
        ticket_02_row = dict(row)
        ticket_02_row.pop("_prompt_token_ids")
        ticket_02_rows.append(ticket_02_row)
    rendered_path = tmp_path / "rendered_turns.parquet"
    pq.write_table(pa.Table.from_pylist(ticket_02_rows), rendered_path)
    config = {
        "artifacts": {
            "manifest": str(tmp_path / "manifest.json"),
            "normalized_turns": str(tmp_path / "turns.parquet"),
            "rendered_turns": str(rendered_path),
        },
        "tokenizer": {"path": str(tmp_path / "tokenizer")},
        "replay": {"block_size": 4},
    }

    turns = load_full_turns(config)

    assert [turn.turn_index for turn in turns] == [0, 1]
    assert turns[1].prompt_token_ids == (1, 2, 3, 4, 5, 6)
```

- [ ] **Step 2: Run the focused tests and verify the red state**

Run:

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  -k 'load_full_turns' -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'benchmarks.ds4_profile.kv_cache_replay'`.

- [ ] **Step 3: Implement the immutable turn type and scalar validation**

Create the module with `SCHEMA_VERSION = "1.0.0"`, the shared types, and these exact validation fields:

```python
SCALAR_TURN_FIELDS = (
    "trajectory_id",
    "task_id",
    "reasoning_mode",
    "turn_index",
    "prompt_tokens",
    "exact_lcp_tokens",
    "reusable_prefix_tokens",
    "global_prefix_tokens",
    "task_prefix_tokens",
)


def _canonical_hash(value: bytes | int) -> str:
    if isinstance(value, bytes):
        return f"sha256:{value.hex()}"
    raise ValueError("Ticket 07 requires byte-valued SHA-256 KV event hashes")


def load_full_turns(config: dict[str, Any]) -> list[ReplayTurn]:
    from benchmarks.ds4_profile import workloads

    artifacts = config["artifacts"]
    rendered = workloads.render_turns(
        manifest_path=Path(artifacts["manifest"]),
        normalized_turns_path=Path(artifacts["normalized_turns"]),
        tokenizer_path=Path(config["tokenizer"]["path"]),
        block_size=config["replay"]["block_size"],
        include_token_ids=True,
    )
    ticket_02 = pq.read_table(artifacts["rendered_turns"]).to_pylist()
    expected = {
        (row["trajectory_id"], row["turn_index"]): row for row in ticket_02
    }
    turns: list[ReplayTurn] = []
    for row in rendered:
        key = (row["trajectory_id"], row["turn_index"])
        if key not in expected or any(
            row[field] != expected[key][field] for field in SCALAR_TURN_FIELDS
        ):
            raise ValueError(f"Ticket 02 scalar mismatch for {key}")
        prompt_token_ids = tuple(row["_prompt_token_ids"])
        if len(prompt_token_ids) != row["prompt_tokens"]:
            raise ValueError(f"prompt token mismatch for {key}")
        turns.append(
            ReplayTurn(
                trajectory_id=row["trajectory_id"],
                task_id=row["task_id"],
                reasoning_mode=row["reasoning_mode"],
                turn_index=row["turn_index"],
                prompt_token_ids=prompt_token_ids,
                prompt_tokens=row["prompt_tokens"],
                exact_lcp_tokens=row["exact_lcp_tokens"],
                reusable_prefix_tokens=row["reusable_prefix_tokens"],
                global_prefix_tokens=row["global_prefix_tokens"],
                task_prefix_tokens=row["task_prefix_tokens"],
            )
        )
    return sorted(turns, key=lambda turn: (turn.trajectory_id, turn.turn_index))
```

Do not read `execution_completion_token_ids` or `execution_prompt_token_ids` anywhere in this module.

- [ ] **Step 4: Run reconstruction tests and verify green**

Run the Step 2 command again.

Expected: `2 passed` and no tokenizer/model initialization because `render_turns` is replaced by the test seam.

- [ ] **Step 5: Commit the reconstruction contract**

```bash
git add \
  benchmarks/ds4_profile/kv_cache_replay.py \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py
git commit -m "[Benchmarks] Add DS4 cache replay inputs" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

---

### Task 2: Real Manager Factory and Scoped BlockPool Observation

**Files:**
- Modify: `benchmarks/ds4_profile/kv_cache_replay.py`
- Modify: `tests/benchmarks/ds4_profile/test_kv_cache_replay.py`

**Interfaces:**
- Consumes: `ReplayTurn` from Task 1; vLLM `Request`, `KVCacheManager`, `KVCacheConfig`, `KVCacheGroupSpec`, and `FullAttentionSpec`.
- Produces: `PoolCall`, `_initialize_hashing`, `make_request`, `make_manager`,
  `observe_block_pool`, `_resident_hashes(manager) -> set[str]`, and
  `_active_block_count(manager) -> int`.

- [ ] **Step 1: Write failing real-manager and observer tests**

```python
def _replay_turn(
    index: int, tokens: list[int], trajectory_id: str = "task:no_think"
):
    from benchmarks.ds4_profile.kv_cache_replay import ReplayTurn

    return ReplayTurn(
        trajectory_id=trajectory_id,
        task_id="task",
        reasoning_mode="no_think",
        turn_index=index,
        prompt_token_ids=tuple(tokens),
        prompt_tokens=len(tokens),
        exact_lcp_tokens=0,
        reusable_prefix_tokens=0,
        global_prefix_tokens=2,
        task_prefix_tokens=4,
    )


@pytest.fixture(autouse=True)
def _stable_hash_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks.ds4_profile.kv_cache_replay import _initialize_hashing

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0")
    _initialize_hashing()


def test_hash_initialization_is_once_and_equal_prefixes_hash_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.ds4_profile import kv_cache_replay

    original = kv_cache_replay.init_none_hash
    calls = []

    def counting_init(hash_fn) -> None:
        calls.append(hash_fn)
        original(hash_fn)

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(kv_cache_replay, "_HASHING_INITIALIZED", False)
    monkeypatch.setattr(kv_cache_replay, "init_none_hash", counting_init)
    kv_cache_replay._initialize_hashing()
    kv_cache_replay._initialize_hashing()
    first = kv_cache_replay.make_request(
        _replay_turn(0, [1, 1, 2, 2]), 2, "first"
    )
    second = kv_cache_replay.make_request(
        _replay_turn(1, [1, 1, 9, 9]), 2, "second"
    )

    assert calls == [kv_cache_replay.sha256]
    assert first.block_hashes[0] == second.block_hashes[0]


def test_native_events_preserve_byte_sha_hashes() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import make_manager, make_request
    from vllm.distributed.kv_events import BlockStored

    manager = make_manager(capacity_blocks=2, block_size=2, max_model_len=32)
    request = make_request(_replay_turn(0, [1, 1, 2, 2]), 2, "request")
    hit, hit_tokens, _ = manager.get_computed_blocks(request)
    assert manager.allocate_slots(
        request, request.num_tokens, hit_tokens, hit
    ) is not None

    hashes = [
        value
        for event in manager.take_events()
        if isinstance(event, BlockStored)
        for value in event.block_hashes or []
    ]
    assert hashes
    assert all(isinstance(value, bytes) and len(value) == 32 for value in hashes)


def test_real_manager_hashes_only_full_blocks_and_reserves_null_block() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import make_manager, make_request

    request = make_request(_replay_turn(0, [1, 1, 2, 2, 3]), 2, "request")
    manager = make_manager(capacity_blocks=3, block_size=2, max_model_len=32)

    assert len(request.block_hashes) == 2
    assert manager.block_pool.num_gpu_blocks == 4
    assert manager.block_pool.null_block.is_null


def test_observer_records_touch_before_allocate_and_reverse_free() -> None:
    from vllm.distributed.kv_events import BlockRemoved
    from benchmarks.ds4_profile.kv_cache_replay import (
        make_manager,
        make_request,
        observe_block_pool,
    )

    manager = make_manager(capacity_blocks=3, block_size=2, max_model_len=32)
    first = make_request(_replay_turn(0, [1, 1, 2, 2, 3]), 2, "first")
    hit, hit_tokens, _ = manager.get_computed_blocks(first)
    assert manager.allocate_slots(first, first.num_tokens, hit_tokens, hit) is not None
    first_ids = manager.get_block_ids("first")[0]
    manager.free(first)
    manager.take_events()

    second = make_request(_replay_turn(1, [1, 1, 9, 9, 8]), 2, "second")
    hit, hit_tokens, _ = manager.get_computed_blocks(second)
    with observe_block_pool(manager) as calls:
        assert manager.allocate_slots(
            second, second.num_tokens - hit_tokens, hit_tokens, hit
        ) is not None
        second_ids = manager.get_block_ids("second")[0]
        manager.free(second)
    native_events = manager.take_events()

    operations = [call.operation for call in calls]
    assert operations[0] == "touch"
    assert operations.index("allocate") > max(
        index for index, operation in enumerate(operations) if operation == "evict"
    )
    assert operations[-1] == "free"
    eviction_calls = [call for call in calls if call.operation == "evict"]
    removed = [event for event in native_events if isinstance(event, BlockRemoved)]
    assert all(call.duration_ns >= 0 for call in eviction_calls)
    assert sum(call.evicted is True for call in eviction_calls) == sum(
        len(event.block_hashes) for event in removed
    )
    free = next(call for call in calls if call.operation == "free")
    assert free.block_ids == tuple(reversed(second_ids))
    assert first_ids[-1] not in second_ids or len(set(first_ids)) < len(first_ids)
```

- [ ] **Step 2: Run tests and verify missing interfaces**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  -k 'hash_initialization or native_events or real_manager or observer' -v
```

Expected: FAIL with `ImportError` for `_initialize_hashing`, `make_manager`,
`make_request`, or `observe_block_pool`.

- [ ] **Step 3: Implement the exact vLLM factory**

```python
import os
from contextlib import contextmanager
from time import perf_counter_ns

if os.environ.get("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") != "0":
    raise RuntimeError("Ticket 07 requires byte-valued KV event hashes")

import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request


_HASHING_INITIALIZED = False


def _initialize_hashing() -> None:
    global _HASHING_INITIALIZED
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Ticket 07 requires PYTHONHASHSEED=0")
    if os.environ.get("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") != "0":
        raise RuntimeError("Ticket 07 requires byte-valued KV event hashes")
    if not _HASHING_INITIALIZED:
        init_none_hash(sha256)
        _HASHING_INITIALIZED = True


def make_request(turn: ReplayTurn, block_size: int, request_id: str) -> Request:
    if not _HASHING_INITIALIZED:
        raise RuntimeError("initialize Ticket 07 hashing before creating requests")
    sampling = SamplingParams(max_tokens=1)
    sampling.update_from_generation_config({}, eos_token_id=0)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(turn.prompt_token_ids),
        mm_features=None,
        sampling_params=sampling,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def make_manager(
    capacity_blocks: int, block_size: int, max_model_len: int
) -> KVCacheManager:
    if capacity_blocks <= 0:
        raise ValueError("capacity_blocks must be positive")
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    config = KVCacheConfig(
        num_blocks=capacity_blocks + 1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["metadata-only"], spec)],
    )
    return KVCacheManager(
        config,
        max_model_len=max_model_len,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
        enable_kv_cache_events=True,
        log_stats=False,
    )
```

`replay_session` and each CLI command call `_initialize_hashing()` before making
requests. The Boolean guard guarantees exactly one `init_none_hash(sha256)` call
per process even when a planner evaluates multiple trajectories. Never call
`init_none_hash` from `make_request`.

- [ ] **Step 4: Implement scoped observation without changing semantics**

Materialize the free iterable once so observation and the original method see the same order. Restore all methods in `finally`:

```python
@contextmanager
def observe_block_pool(
    manager: KVCacheManager,
) -> Iterator[list[PoolCall]]:
    pool = manager.block_pool
    calls: list[PoolCall] = []
    originals = (
        pool.touch,
        pool.get_new_blocks,
        pool._maybe_evict_cached_block,
        pool.free_blocks,
    )

    def record(
        operation: Literal["touch", "allocate", "evict", "free"],
        block_ids: tuple[int, ...],
        duration_ns: int,
        evicted: bool | None = None,
    ) -> None:
        calls.append(
            PoolCall(operation, len(calls), block_ids, duration_ns, evicted)
        )

    def touch(blocks):
        block_list = list(blocks)
        started = perf_counter_ns()
        try:
            return originals[0](block_list)
        finally:
            record(
                "touch",
                tuple(block.block_id for block in block_list),
                perf_counter_ns() - started,
            )

    def get_new_blocks(num_blocks: int):
        nested_before = sum(
            call.duration_ns for call in calls if call.operation == "evict"
        )
        started = perf_counter_ns()
        blocks = originals[1](num_blocks)
        elapsed = perf_counter_ns() - started
        nested_after = sum(
            call.duration_ns for call in calls if call.operation == "evict"
        )
        record(
            "allocate",
            tuple(block.block_id for block in blocks),
            elapsed - (nested_after - nested_before),
        )
        return blocks

    def maybe_evict_cached_block(block):
        started = perf_counter_ns()
        evicted = originals[2](block)
        record(
            "evict",
            (block.block_id,),
            perf_counter_ns() - started,
            evicted,
        )
        return evicted

    def free_blocks(blocks):
        block_list = list(blocks)
        started = perf_counter_ns()
        try:
            return originals[3](block_list)
        finally:
            record(
                "free",
                tuple(block.block_id for block in block_list),
                perf_counter_ns() - started,
            )

    pool.touch = touch
    pool.get_new_blocks = get_new_blocks
    pool._maybe_evict_cached_block = maybe_evict_cached_block
    pool.free_blocks = free_blocks
    try:
        yield calls
    finally:
        (
            pool.touch,
            pool.get_new_blocks,
            pool._maybe_evict_cached_block,
            pool.free_blocks,
        ) = originals
```

Use narrow `# type: ignore[method-assign]` annotations on these eight assignments
if mypy requires them. The wrapper must invoke the bound original exactly once,
return its exact Boolean, and restore it in `finally`; it observes real eviction
semantics and never reimplements hash removal or event emission. Do not suppress
any other type error.

- [ ] **Step 5: Run tests and verify green**

Run the Step 2 command.

Expected: both tests PASS; no CUDA device is queried and no KV tensor is allocated.

- [ ] **Step 6: Commit the real-manager seam**

```bash
git add \
  benchmarks/ds4_profile/kv_cache_replay.py \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py
git commit -m "[Benchmarks] Add the real KV cache replay seam" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

---

### Task 3: Serial Replay, Miss Attribution, and Future Reuse

**Files:**
- Modify: `benchmarks/ds4_profile/kv_cache_replay.py`
- Modify: `tests/benchmarks/ds4_profile/test_kv_cache_replay.py`

**Interfaces:**
- Consumes: Task 2 manager and observer functions; native `BlockStored` and `BlockRemoved` returned by `manager.take_events()`.
- Produces: `ReplayResult`, `replay_session`, `_prefix_source(turn, block_position, block_size) -> PrefixSource`, `_future_accesses(turns, block_size) -> dict[str, tuple[int, ...]]`, `_lookup_outcome`, and `_classify_miss`.

- [ ] **Step 1: Write the failing hand-checkable replay test**

This trace creates all three miss classes at usable capacity three: turn 0 is compulsory; turn 1 diverges at an existing position and reaches a new depth; allocation evicts a hash reused by turn 2.

```python
def test_replay_classifies_misses_eviction_and_future_reuse() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import replay_session

    turns = [
        _replay_turn(0, [1, 1, 2, 2, 3]),
        _replay_turn(1, [1, 1, 9, 9, 8, 8]),
        _replay_turn(2, [1, 1, 2, 2, 7]),
    ]

    result = replay_session(
        run_id="fixture-run",
        turns=turns,
        capacity_blocks=3,
        block_size=2,
        max_model_len=32,
    )

    misses = {
        row["miss_class"]
        for row in result.event_rows
        if row["cache_outcome"] == "miss"
    }
    evictions = [row for row in result.event_rows if row["operation"] == "evict"]
    assert result.status == "passed"
    assert misses == {"compulsory", "prefix_mismatch", "capacity"}
    assert any(
        row["useful_later"] is True
        and row["turns_until_reuse"] == 1
        and row["never_reused"] is False
        for row in evictions
    )
    assert all(
        row["prefix_source"] in {"global", "task", "session"}
        for row in result.event_rows
        if row["block_position"] is not None
    )
```

Add an admission-failure contract:

```python
def test_replay_stops_without_mutating_an_out_of_capacity_prompt() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import replay_session

    turn = _replay_turn(0, [1, 1, 2, 2, 3, 3])
    result = replay_session(
        run_id="fixture-run",
        turns=[turn],
        capacity_blocks=2,
        block_size=2,
        max_model_len=32,
    )

    assert result.status == "out_of_capacity"
    assert result.turn_rows[0]["prompt_tokens"] == 6
    assert result.turn_rows[0]["status"] == "out_of_capacity"
    assert result.event_rows[-1]["operation"] == "admission_failure"


def test_replay_separates_manager_forced_recompute_from_capacity_miss() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import replay_session

    result = replay_session(
        run_id="fixture-run",
        turns=[
            _replay_turn(0, [1, 1, 2, 2]),
            _replay_turn(1, [1, 1, 2, 2]),
        ],
        capacity_blocks=2,
        block_size=2,
        max_model_len=32,
    )

    forced = [
        row
        for row in result.event_rows
        if row["turn_index"] == 1
        and row["cache_outcome"] == "manager_forced_recompute"
    ]
    assert len(forced) == 1
    assert forced[0]["block_position"] == 1
    assert forced[0]["miss_class"] is None
    assert result.turn_rows[1]["manager_forced_recompute_blocks"] == 1
    assert result.turn_rows[1]["capacity_miss_blocks"] == 0
```

- [ ] **Step 2: Run replay tests and verify red**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  -k 'replay_classifies or out_of_capacity or manager_forced' -v
```

Expected: FAIL with `ImportError: cannot import name 'replay_session'`.

- [ ] **Step 3: Implement deterministic hash prepass and attribution**

```python
def _prefix_source(
    turn: ReplayTurn, block_position: int, block_size: int
) -> PrefixSource:
    block_start = block_position * block_size
    if block_start < turn.global_prefix_tokens:
        return "global"
    if block_start < turn.task_prefix_tokens:
        return "task"
    return "session"


def _turn_hashes(turn: ReplayTurn, block_size: int) -> tuple[str, ...]:
    request = make_request(turn, block_size, f"prepass:{turn.turn_index}")
    return tuple(_canonical_hash(value) for value in request.block_hashes)


def _future_accesses(
    turns: Sequence[ReplayTurn], block_size: int
) -> dict[str, tuple[int, ...]]:
    accesses: dict[str, list[int]] = {}
    for turn in turns:
        for block_hash in _turn_hashes(turn, block_size):
            accesses.setdefault(block_hash, []).append(turn.turn_index)
    return {key: tuple(values) for key, values in accesses.items()}


def _classify_miss(
    *,
    block_hash: str,
    block_position: int,
    ever_stored: set[str],
    max_seen_depth: int,
) -> MissClass:
    if block_hash in ever_stored:
        return "capacity"
    if block_position < max_seen_depth:
        return "prefix_mismatch"
    return "compulsory"


def _lookup_outcome(
    *,
    block_hash: str,
    block_position: int,
    full_block_count: int,
    hit_blocks: int,
    prompt_tokens: int,
    block_size: int,
    resident_before: set[str],
) -> CacheOutcome:
    if block_position < hit_blocks:
        return "hit"
    if block_hash not in resident_before:
        return "miss"
    if (
        prompt_tokens % block_size == 0
        and block_position == full_block_count - 1
        and block_position == hit_blocks
    ):
        return "manager_forced_recompute"
    raise RuntimeError("resident hash exists beyond the legal manager hit")
```

- [ ] **Step 4: Implement `replay_session` around the real manager calls**

For each turn, snapshot resident hashes before lookup, time `get_computed_blocks`, invoke `allocate_slots` with exactly `request.num_tokens - hit_tokens`, drain native events, then free. Use these invariants in the implementation:

```python
computed, hit_tokens, _ = manager.get_computed_blocks(request)
hit_blocks = hit_tokens // block_size
resident_before = _resident_hashes(manager)
request_hashes = tuple(_canonical_hash(value) for value in request.block_hashes)
outcomes = tuple(
    _lookup_outcome(
        block_hash=value,
        block_position=position,
        full_block_count=len(request_hashes),
        hit_blocks=hit_blocks,
        prompt_tokens=request.num_tokens,
        block_size=block_size,
        resident_before=resident_before,
    )
    for position, value in enumerate(request_hashes)
)

with observe_block_pool(manager) as pool_calls:
    allocated = manager.allocate_slots(
        request,
        request.num_tokens - hit_tokens,
        hit_tokens,
        computed,
    )
    if allocated is None:
        return _out_of_capacity_result(
            run_id=run_id,
            turn=turn,
            event_rows=event_rows,
            turn_rows=turn_rows,
            manager=manager,
            error="KV cache admission failed at pinned capacity",
        )
    native_events = manager.take_events()
    request_block_ids = manager.get_block_ids(request.request_id)[0]
    manager.free(request)
```

`_out_of_capacity_result` appends exactly one `admission_failure` event with
`status="out_of_capacity"`, null block/hash/outcome/miss/reuse fields, unchanged
before/after occupancy, and the supplied error. It then appends exactly one turn
row with the original `prompt_tokens`, zero hit/allocation/eviction/free counts,
`recomputed_tokens=prompt_tokens`, and the same error, and returns all previously
completed rows plus these two rows. No exception should escape for ordinary
admission failure. For successful turns:

- `cached_tokens = hit_tokens`;
- `recomputed_tokens = request.num_tokens - hit_tokens`;
- one lookup row per full request hash, with `cache_outcome` `hit`, `miss`, or
  `manager_forced_recompute`;
- `manager_forced_recompute` is legal only for the resident final full block of
  a block-aligned prompt immediately beyond the returned hit, because
  `get_computed_blocks` caps the hit at `prompt_len - 1`; it has null
  `miss_class` and is never counted as capacity loss;
- one `event_source="observer"` control row per `PoolCall`, including every
  true and false call to the real `_maybe_evict_cached_block`;
- one `event_source="native"` `store` or `evict` row per native block hash;
- `BlockRemoved` is the only source of eviction truth;
- pair observer calls with `evicted=True` to native removals in order for this
  single-group manager; never infer a removal from allocation or free;
- sum observer eviction-call durations into `eviction_time_ns`, and subtract
  those nested durations from `allocation_time_ns` so the two timings do not
  overlap;
- future reuse is the first access turn greater than the eviction turn;
- native evictions with no later exact-hash access set `never_reused=True`, `useful_later=False`, and null reuse-distance fields; and
- call ordering must show every turn's touches before allocations and free last.

Use a `try/except` around each turn. On unexpected error, append an `invalid`
turn with `error=f"{type(error).__name__}: {error}"`, preserve completed event
and turn rows, set `eviction_count` to the number of completed native removals,
and return an invalid `ReplayResult` with that same error.

- [ ] **Step 5: Run replay tests and verify green**

Run the Step 2 command.

Expected: `2 passed`; the first test observes all three miss classes and a useful-later eviction, while the second preserves all six prompt tokens.

- [ ] **Step 6: Run the real-manager regression tests used as prior art**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py::test_prefill \
  tests/v1/core/test_single_type_kv_cache_manager.py::test_evictable_cached_blocks_not_double_allocated \
  -v
```

Expected: both selected upstream tests PASS; Ticket 07 does not modify vLLM core behavior.

- [ ] **Step 7: Commit replay semantics**

```bash
git add \
  benchmarks/ds4_profile/kv_cache_replay.py \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py
git commit -m "[Benchmarks] Replay DS4 cache metadata" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

---

### Task 4: Deterministic Selection Planner and Frozen Config

**Files:**
- Modify: `benchmarks/ds4_profile/kv_cache_replay.py`
- Create: `benchmarks/ds4_profile/config/kv-cache-replay.json`
- Modify: `tests/benchmarks/ds4_profile/test_kv_cache_replay.py`

**Interfaces:**
- Consumes: `load_full_turns`, `replay_session`, and `ReplayResult`.
- Produces: `collect_input_records`, `verify_input_records`,
  `build_selection_plan`, `verify_pinned_selection`,
  `_select_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]`,
  and a config whose `selection.status` is either `unselected` for planning or
  `pinned` for normal replay.

- [ ] **Step 1: Write failing deterministic-selection tests**

```python
def test_selection_uses_capacity_mode_and_trajectory_stable_key() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import _select_candidate

    selected = _select_candidate(
        [
            {
                "trajectory_id": "z:think_high",
                "reasoning_mode": "think_high",
                "capacity_blocks": 12,
                "status": "eligible",
            },
            {
                "trajectory_id": "b:no_think",
                "reasoning_mode": "no_think",
                "capacity_blocks": 12,
                "status": "eligible",
            },
            {
                "trajectory_id": "a:no_think",
                "reasoning_mode": "no_think",
                "capacity_blocks": 13,
                "status": "eligible",
            },
        ]
    )

    assert selected["trajectory_id"] == "b:no_think"


def test_verify_pinned_selection_rejects_drift() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import (
        _canonical_json_sha256,
        _selected_turn_manifest,
        _with_canonical_sha256,
        verify_pinned_selection,
    )

    input_set_sha256 = _canonical_json_sha256([])
    turn_manifest_sha256 = _canonical_json_sha256([])
    plan = _with_canonical_sha256(
        {
            "schema_version": "1.0.0",
            "status": "selected",
            "inputs": [],
            "candidates": [],
            "selected": {
                "trajectory_id": "task:no_think",
                "reasoning_mode": "no_think",
                "capacity_blocks": 11,
                "input_set_sha256": input_set_sha256,
                "turns": [],
                "turn_manifest_sha256": turn_manifest_sha256,
            },
        }
    )

    config = {
        "selection": {
            "status": "pinned",
            "trajectory_id": "task:no_think",
            "reasoning_mode": "no_think",
            "capacity_blocks": 10,
            "input_set_sha256": input_set_sha256,
            "planning_sha256": plan["sha256"],
            "turn_manifest_sha256": turn_manifest_sha256,
        }
    }

    with pytest.raises(ValueError, match="pinned selection does not match"):
        verify_pinned_selection(config, plan)


def test_input_records_cover_data_provenance_and_every_tokenizer_file(
    tmp_path: Path,
) -> None:
    from benchmarks.ds4_profile.kv_cache_replay import (
        collect_input_records,
        verify_input_records,
    )

    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    files = {
        "manifest": tmp_path / "manifest.json",
        "normalized_turns": tmp_path / "turns.parquet",
        "normalized_provenance": tmp_path / "ticket-01-provenance.json",
        "rendered_turns": tmp_path / "rendered.parquet",
        "workload_provenance": tmp_path / "ticket-02-provenance.json",
    }
    for index, path in enumerate(files.values()):
        path.write_bytes(f"input-{index}".encode())
    (tokenizer / "tokenizer.json").write_text("tokenizer")
    (tokenizer / "tokenizer_config.json").write_text("config")
    config = {
        "artifacts": {name: str(path) for name, path in files.items()},
        "tokenizer": {"path": str(tokenizer)},
    }

    records = collect_input_records(config)
    verify_input_records(records)

    assert {record.logical_name for record in records} == {
        "manifest",
        "ticket_01_data",
        "ticket_01_provenance",
        "ticket_02_data",
        "ticket_02_provenance",
        "tokenizer:tokenizer.json",
        "tokenizer:tokenizer_config.json",
    }
    (tokenizer / "tokenizer.json").write_text("tampered")
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        verify_input_records(records)
```

- [ ] **Step 2: Run selection tests and verify red**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  -k 'selection_uses or pinned_selection or input_records' -v
```

Expected: FAIL with missing planner interfaces.

- [ ] **Step 3: Implement candidate evaluation and stable selection**

Implement `collect_input_records` over exactly five fixed inputs—manifest,
Ticket 01 data/provenance, and Ticket 02 data/provenance—plus every regular file
under the tokenizer directory in sorted relative-path order. Each record stores
logical name, mounted path, byte length, and a streaming SHA-256. Follow
symlinks only to read their file content, retain the mounted symlink path in the
record, reject duplicate logical names and an empty tokenizer directory, and
make `verify_input_records` recompute size and digest from disk. Canonicalize the
record list as sorted compact JSON and expose its SHA-256 as
`input_set_sha256`.

Implement `_selected_turn_manifest` by sorting one trajectory on `turn_index`
and rejecting duplicates or gaps. Each ordered row contains exactly
`trajectory_id`, `turn_index`, `prompt_tokens`,
`prompt_token_ids_sha256` (SHA-256 of compact JSON token IDs), and
`block_hashes_sha256` (SHA-256 of compact JSON canonical chained block hashes).
Hash the complete ordered row list as `turn_manifest_sha256`.

Group turns by trajectory. For each group, set capacity to the maximum `ceil(prompt_tokens / block_size)`, run the real metadata replay, and mark it eligible only when every turn passed and at least one native eviction occurred:

```python
def _select_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reasoning_rank = {"no_think": 0, "think_high": 1}
    eligible = [item for item in candidates if item["status"] == "eligible"]
    if not eligible:
        raise ValueError("no full trajectory admits all turns with eviction pressure")
    return min(
        eligible,
        key=lambda item: (
            item["capacity_blocks"],
            reasoning_rank[item["reasoning_mode"]],
            item["trajectory_id"],
        ),
    )


def build_selection_plan(
    config: dict[str, Any], turns: Sequence[ReplayTurn]
) -> dict[str, Any]:
    _initialize_hashing()
    block_size = config["replay"]["block_size"]
    inputs = collect_input_records(config)
    verify_input_records(inputs)
    input_rows = [asdict(record) for record in inputs]
    input_set_sha256 = _canonical_json_sha256(input_rows)
    candidates = []
    for trajectory_id in sorted({turn.trajectory_id for turn in turns}):
        session = sorted(
            (turn for turn in turns if turn.trajectory_id == trajectory_id),
            key=lambda turn: turn.turn_index,
        )
        selected_turns = _selected_turn_manifest(session, block_size)
        capacity = max(
            (turn.prompt_tokens + block_size - 1) // block_size for turn in session
        )
        result = replay_session(
            run_id=f"planning:{trajectory_id}",
            turns=session,
            capacity_blocks=capacity,
            block_size=block_size,
            max_model_len=config["replay"]["max_model_len"],
        )
        candidates.append(
            {
                "trajectory_id": trajectory_id,
                "reasoning_mode": session[0].reasoning_mode,
                "turn_count": len(session),
                "capacity_blocks": capacity,
                "eviction_count": result.eviction_count,
                "turns": selected_turns,
                "turn_manifest_sha256": _canonical_json_sha256(selected_turns),
                "status": (
                    "eligible"
                    if result.status == "passed" and result.eviction_count > 0
                    else "rejected"
                ),
                "reason": result.error,
            }
        )
    selected = {
        **_select_candidate(candidates),
        "input_set_sha256": input_set_sha256,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "selected",
        "selected": selected,
        "candidates": candidates,
        "inputs": input_rows,
    }
    return _with_canonical_sha256(payload)
```

`_with_canonical_sha256` serializes the plan without its digest using sorted
compact JSON, computes SHA-256, and adds it as top-level `sha256`.
`verify_pinned_selection` verifies that digest, recomputes every input record,
recomputes `input_set_sha256`, and compares trajectory, reasoning mode,
capacity, input-set digest, ordered-turn-manifest digest, and planning digest
exactly. `run` reconstructs turns from the hash-verified pinned inputs, requires
their ordered manifest to equal `plan["selected"]["turns"]`, and copies that
exact list to top-level `selected_turns` in `run-config.json`.

- [ ] **Step 4: Add the explicit pre-selection config state**

Create this exact JSON before the school-server planning pass:

```json
{
  "artifacts": {
    "manifest": "/mnt/ds4/raw/manifest.json",
    "normalized_turns": "/mnt/ds4/ticket-01/turns.parquet",
    "normalized_provenance": "/mnt/ds4/ticket-01/provenance.json",
    "rendered_turns": "/mnt/ds4/ticket-02/rendered_turns.parquet",
    "workload_provenance": "/mnt/ds4/ticket-02/provenance.json"
  },
  "environment": {
    "PYTHONHASHSEED": "0",
    "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": "0"
  },
  "replay": {
    "batch_size": 1,
    "block_size": 16,
    "hash_function": "sha256",
    "kv_event_hash_format": "bytes",
    "max_model_len": 65536,
    "order": "serial"
  },
  "run_id": null,
  "schema_version": "1.0.0",
  "selection": {
    "capacity_blocks": null,
    "input_set_sha256": null,
    "planning_sha256": null,
    "reasoning_mode": null,
    "status": "unselected",
    "turn_manifest_sha256": null,
    "trajectory_id": null
  },
  "source": {
    "commit": "unknown",
    "dirty": true
  },
  "tokenizer": {
    "repo_id": "deepseek-ai/DeepSeek-V4-Flash",
    "revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
    "path": "/mnt/ds4/tokenizers/deepseek-v4-flash/60d8d70770c6776ff598c94bb586a859a38244f1"
  }
}
```

`run` must reject `status != "pinned"`; only `plan` accepts `unselected`. A run
also fails before manager construction if any planning-record input path, size,
or SHA differs from current mounted content.

- [ ] **Step 5: Run selection tests and full focused tests**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py -v
```

Expected: all Ticket 07 tests created through Task 4 PASS. The local suite still uses tiny prompts; it does not run the 20-trajectory planner.

- [ ] **Step 6: Commit the planner and pre-selection config**

```bash
git add \
  benchmarks/ds4_profile/kv_cache_replay.py \
  benchmarks/ds4_profile/config/kv-cache-replay.json \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py
git commit -m "[Benchmarks] Select the DS4 cache replay" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

---

### Task 5: Versioned Artifacts, Independent Validator, and CLI

**Files:**
- Modify: `benchmarks/ds4_profile/kv_cache_replay.py`
- Modify: `tests/benchmarks/ds4_profile/test_kv_cache_replay.py`

**Interfaces:**
- Consumes: `ReplayResult`, frozen config, and selection plan.
- Produces: `CACHE_EVENT_SCHEMA`, `TURN_SUMMARY_SCHEMA`, `write_result`, `validate_result_dir`, `_result_markdown`, and CLI commands `plan`, `run`, `validate`.

- [ ] **Step 1: Write failing artifact and tamper-rejection tests**

```python
def test_artifacts_are_versioned_metadata_only_and_cross_validated(
    tmp_path: Path,
) -> None:
    from benchmarks.ds4_profile.kv_cache_replay import (
        _canonical_json_sha256,
        _with_canonical_sha256,
        replay_session,
        validate_result_dir,
        write_result,
    )

    input_path = tmp_path / "manifest.json"
    input_path.write_bytes(b"fixture-input")
    inputs = [
        {
            "logical_name": "manifest",
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        }
    ]
    input_set_sha256 = _canonical_json_sha256(inputs)
    replay_turns = [
        _replay_turn(0, [1, 1, 2, 2, 3]),
        _replay_turn(1, [1, 1, 9, 9, 8, 8]),
    ]
    selected_turns = _selected_turn_manifest(replay_turns, block_size=2)
    turn_manifest_sha256 = _canonical_json_sha256(selected_turns)
    planning_record = _with_canonical_sha256(
        {
            "schema_version": "1.0.0",
            "status": "selected",
            "inputs": inputs,
            "candidates": [],
            "selected": {
                "trajectory_id": "task:no_think",
                "reasoning_mode": "no_think",
                "capacity_blocks": 3,
                "input_set_sha256": input_set_sha256,
                "turns": selected_turns,
                "turn_manifest_sha256": turn_manifest_sha256,
            },
        }
    )
    config = {
        "schema_version": "1.0.0",
        "run_id": "fixture-run",
        "environment": {
            "PYTHONHASHSEED": "0",
            "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": "0",
        },
        "inputs": inputs,
        "planning_record": planning_record,
        "selected_turns": selected_turns,
        "replay": {
            "block_size": 2,
            "kv_event_hash_format": "bytes",
            "max_model_len": 32,
        },
        "selection": {
            "status": "pinned",
            "trajectory_id": "task:no_think",
            "reasoning_mode": "no_think",
            "capacity_blocks": 3,
            "input_set_sha256": input_set_sha256,
            "planning_sha256": planning_record["sha256"],
            "turn_manifest_sha256": turn_manifest_sha256,
        },
        "source": {"commit": "abc123", "dirty": False},
    }
    result = replay_session(
        run_id="fixture-run",
        turns=replay_turns,
        capacity_blocks=3,
        block_size=2,
        max_model_len=32,
    )
    output = tmp_path / "result"

    write_result(config, result, output)
    validate_result_dir(output)

    assert {path.name for path in output.iterdir()} == {
        "cache_events.parquet",
        "provenance.json",
        "result.md",
        "run-config.json",
        "turn_summaries.parquet",
    }
    assert "Metadata only: yes" in (output / "result.md").read_text()
    assert "GPU/HBM validated: no" in (output / "result.md").read_text()


def test_validator_recomputes_counts_and_rejects_unknown_enums(tmp_path: Path) -> None:
    # Build a valid fixture through the helper above, then replace the first
    # operation value without changing the Arrow schema.
    output = _write_valid_result(tmp_path)
    path = output / "cache_events.parquet"
    table = pq.read_table(path)
    operations = table.column("operation").to_pylist()
    operations[0] = "not-an-operation"
    pq.write_table(
        table.set_column(
            table.schema.get_field_index("operation"),
            "operation",
            pa.array(operations, type=pa.string()),
        ),
        path,
    )

    from benchmarks.ds4_profile.kv_cache_replay import validate_result_dir

    with pytest.raises(ValueError, match="unknown operation"):
        validate_result_dir(output)


def _rewrite_event_rows(output: Path, mutate) -> None:
    from benchmarks.ds4_profile.kv_cache_replay import CACHE_EVENT_SCHEMA

    path = output / "cache_events.parquet"
    rows = pq.read_table(path).to_pylist()
    mutate(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=CACHE_EVENT_SCHEMA), path)


def _validate_fixture(output: Path) -> None:
    from benchmarks.ds4_profile.kv_cache_replay import validate_result_dir

    validate_result_dir(output)


def test_validator_recomputes_miss_attribution(tmp_path: Path) -> None:
    output = _write_valid_result(tmp_path)

    def corrupt(rows) -> None:
        row = next(item for item in rows if item["cache_outcome"] == "miss")
        row["miss_class"] = (
            "capacity" if row["miss_class"] != "capacity" else "compulsory"
        )

    _rewrite_event_rows(output, corrupt)
    with pytest.raises(ValueError, match="miss attribution mismatch"):
        _validate_fixture(output)


def test_validator_recomputes_future_reuse(tmp_path: Path) -> None:
    output = _write_valid_result(tmp_path)

    def corrupt(rows) -> None:
        row = next(
            item
            for item in rows
            if item["event_source"] == "native" and item["operation"] == "evict"
        )
        row["useful_later"] = True
        row["never_reused"] = False
        row["next_reuse_turn"] = row["turn_index"] + 1
        row["turns_until_reuse"] = 1

    _rewrite_event_rows(output, corrupt)
    with pytest.raises(ValueError, match="future reuse mismatch"):
        _validate_fixture(output)


def test_validator_reconstructs_call_order(tmp_path: Path) -> None:
    output = _write_valid_result(tmp_path)

    def corrupt(rows) -> None:
        touch = next(item for item in rows if item["operation"] == "touch")
        allocate = next(
            item
            for item in rows
            if item["turn_index"] == touch["turn_index"]
            and item["operation"] == "allocate"
        )
        touch["operation_ordinal"], allocate["operation_ordinal"] = (
            allocate["operation_ordinal"],
            touch["operation_ordinal"],
        )
        for row in (touch, allocate):
            row["event_id"] = (
                f'{row["run_id"]}:{row["trajectory_id"]}:{row["turn_index"]}:'
                f'{row["operation"]}:{row["operation_ordinal"]}'
            )

    _rewrite_event_rows(output, corrupt)
    with pytest.raises(ValueError, match="operation ordering mismatch"):
        _validate_fixture(output)


def test_validator_replays_occupancy_transitions(tmp_path: Path) -> None:
    output = _write_valid_result(tmp_path)

    def corrupt(rows) -> None:
        rows[1]["active_blocks_before"] += 1

    _rewrite_event_rows(output, corrupt)
    with pytest.raises(ValueError, match="occupancy transition mismatch"):
        _validate_fixture(output)


@pytest.mark.parametrize("corruption", ["omit", "reorder", "tamper"])
def test_validator_reconstructs_the_exact_ordered_selected_turns(
    tmp_path: Path, corruption: str
) -> None:
    output = _write_valid_result(tmp_path)
    path = output / "run-config.json"
    config = json.loads(path.read_text())
    turns = config["selected_turns"]
    if corruption == "omit":
        config["selected_turns"] = turns[:-1]
    elif corruption == "reorder":
        config["selected_turns"] = list(reversed(turns))
    else:
        config["selected_turns"][0]["prompt_token_ids_sha256"] = "0" * 64
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="selected turn manifest mismatch"):
        _validate_fixture(output)
```

- [ ] **Step 2: Run artifact tests and verify red**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  -k 'artifacts_are or validator' -v
```

Expected: FAIL because `write_result` and `validate_result_dir` are absent.

- [ ] **Step 3: Define complete Arrow schemas**

Define non-null identity, enum, count, and occupancy fields; nullable fields apply only when an operation has no block/timing/reuse value:

```python
CACHE_EVENT_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("trajectory_id", pa.string(), nullable=False),
        pa.field("turn_index", pa.int32(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_source", pa.string(), nullable=False),
        pa.field("operation", pa.string(), nullable=False),
        pa.field("operation_ordinal", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("cache_outcome", pa.string()),
        pa.field("miss_class", pa.string()),
        pa.field("duration_ns", pa.int64()),
        pa.field("eviction_time_ns", pa.int64()),
        pa.field("evicted", pa.bool_()),
        pa.field("block_position", pa.int32()),
        pa.field("block_id", pa.int64()),
        pa.field("block_hash", pa.string()),
        pa.field("prefix_source", pa.string()),
        pa.field("token_count", pa.int32()),
        pa.field("active_blocks_before", pa.int32(), nullable=False),
        pa.field("active_blocks_after", pa.int32(), nullable=False),
        pa.field("cached_blocks_before", pa.int32(), nullable=False),
        pa.field("cached_blocks_after", pa.int32(), nullable=False),
        pa.field("free_blocks_before", pa.int32(), nullable=False),
        pa.field("free_blocks_after", pa.int32(), nullable=False),
        pa.field("useful_later", pa.bool_()),
        pa.field("never_reused", pa.bool_()),
        pa.field("next_reuse_turn", pa.int32()),
        pa.field("turns_until_reuse", pa.int32()),
        pa.field("error", pa.string()),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode()},
)

TURN_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("trajectory_id", pa.string(), nullable=False),
        pa.field("turn_index", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("prompt_tokens", pa.int32(), nullable=False),
        pa.field("full_blocks", pa.int32(), nullable=False),
        pa.field("cached_tokens", pa.int32(), nullable=False),
        pa.field("recomputed_tokens", pa.int32(), nullable=False),
        pa.field("hit_blocks", pa.int32(), nullable=False),
        pa.field("manager_forced_recompute_blocks", pa.int32(), nullable=False),
        pa.field("compulsory_miss_blocks", pa.int32(), nullable=False),
        pa.field("capacity_miss_blocks", pa.int32(), nullable=False),
        pa.field("prefix_mismatch_blocks", pa.int32(), nullable=False),
        pa.field("allocated_blocks", pa.int32(), nullable=False),
        pa.field("evicted_blocks", pa.int32(), nullable=False),
        pa.field("freed_blocks", pa.int32(), nullable=False),
        pa.field("cached_resident_blocks_after_free", pa.int32(), nullable=False),
        pa.field("hash_time_ns", pa.int64(), nullable=False),
        pa.field("lookup_time_ns", pa.int64(), nullable=False),
        pa.field("touch_time_ns", pa.int64(), nullable=False),
        pa.field("allocation_time_ns", pa.int64(), nullable=False),
        pa.field("eviction_time_ns", pa.int64(), nullable=False),
        pa.field("free_time_ns", pa.int64(), nullable=False),
        pa.field("error", pa.string()),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode()},
)
```

- [ ] **Step 4: Implement staged writing and fail-closed validation**

`write_result` writes the five required files into a temporary sibling directory, calls `validate_result_dir`, then moves files into a newly created output directory. `provenance.json` contains:

```python
{
    "artifact_schema_version": SCHEMA_VERSION,
    "hardware_validated": False,
    "environment": config["environment"],
    "inputs": config["inputs"],
    "metadata_only_validated": result.status == "passed",
    "planning_record": config["planning_record"],
    "planning_sha256": config["selection"]["planning_sha256"],
    "run_id": config["run_id"],
    "selected_turns": config["selected_turns"],
    "selection": config["selection"],
    "source": config["source"],
    "status": result.status,
}
```

The validator must never trust derived labels or summaries. It sorts by
trajectory, turn, and operation ordinal, then reconstructs the resident-hash,
ever-stored-hash, active-block, cached-block, and free-block state from lookup,
observer, and native rows. It must:

- require all five files and exact Arrow schemas;
- require every row's `schema_version == "1.0.0"`;
- require one run ID across config, provenance, events, and turns;
- require unique `event_id == f"{run_id}:{trajectory_id}:{turn_index}:{operation}:{operation_ordinal}"`;
- validate event-source, operation, status, outcome (including
  `manager_forced_recompute`), miss-class, and prefix-source enums;
- require miss class only when `cache_outcome == "miss"`;
- independently recompute compulsory/capacity/prefix-mismatch from the
  reconstructed resident and ever-stored sets, and require capacity only when
  the exact hash was stored previously but is not resident;
- independently recognize the legal final-block manager-forced-recompute case
  and reject that outcome anywhere else;
- independently scan future ordered lookup hashes after every native removal to
  recompute next reuse turn, turn distance, and
  `useful_later`/`never_reused` plus their nullability;
- require lookup rows first, touch before eviction checks, each true observer
  eviction paired one-to-one with the following native removal, allocation
  after its nested eviction checks, native stores after allocation, and free
  last for each turn;
- replay every active/cached/free before/after transition, require each row's
  before state equals the preceding after state, and reject impossible deltas;
- require `eviction_time_ns` only on observer eviction calls, equal their
  duration, sum it independently per turn, and ensure allocation timing is
  exclusive of nested eviction timing;
- recompute per-turn hit/miss/allocation/eviction/free counts from events;
- recompute manager-forced-recompute counts separately from all miss classes;
- require `cached_tokens + recomputed_tokens == prompt_tokens`;
- require all successful configured turns in increasing serial order;
- require at least one eviction for a passed result; and
- recompute and verify every manifest, Ticket 01/02 data/provenance, and
  tokenizer input size/SHA from provenance, then verify the canonical input-set
  and planning-record digests; and
- reconstruct the selected trajectory from those verified inputs, recompute
  every prompt-token and chained-block-hash digest, and require exact ordered
  equality among reconstructed turns, planning-record selected turns,
  run-config selected turns, provenance selected turns, and artifact turn
  summaries; omitted, extra, reordered, or mutated turns all fail closed; and
- require config, provenance, and the validator process environment all contain
  `PYTHONHASHSEED="0"` and `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES="0"`, require
  `kv_event_hash_format="bytes"`, and reject any integer native event hash; and
- require `hardware_validated is False` and `metadata_only_validated` only for passed results.

- [ ] **Step 5: Implement the three-command CLI**

Use exact argument shapes:

```python
plan = subparsers.add_parser("plan")
plan.add_argument("--config", type=Path, required=True)
plan.add_argument("--output", type=Path, required=True)

run = subparsers.add_parser("run")
run.add_argument("--config", type=Path, required=True)
run.add_argument("--planning-record", type=Path, required=True)
run.add_argument("--output-dir", type=Path, required=True)

validate = subparsers.add_parser("validate")
validate.add_argument("--result-dir", type=Path, required=True)
```

`plan` loads full turns, builds the plan, and writes sorted JSON. `run` requires
`selection.status == "pinned"`, checks the planning record SHA and selection,
recomputes every input file record, copies the verified planning record and
input records into the effective config, filters exactly one complete
trajectory, calls `replay_session`, writes artifacts, and exits `0` only for
passed status. `validate` exits `0` only after
independent validation. Validation errors print
`validation failed: {validation message}` to stderr and exit `2`.

- [ ] **Step 6: Run artifact tests and the whole Ticket 07 suite**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py -v
```

Expected: all tests PASS, including tamper rejection.

- [ ] **Step 7: Commit artifact and CLI contracts**

```bash
git add \
  benchmarks/ds4_profile/kv_cache_replay.py \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py
git commit -m "[Benchmarks] Persist DS4 cache replay events" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

---

### Task 6: CPU-Only Container Registration

**Files:**
- Modify: `benchmarks/ds4_profile/container/runtime.py`
- Modify: `benchmarks/ds4_profile/container/run.sh`
- Modify: `tests/benchmarks/ds4_profile/test_kv_cache_replay.py`

**Interfaces:**
- Consumes: Task 5 module CLI.
- Produces: container runtime command `kv-cache-replay plan|run|validate`, `_effective_kv_cache_replay_config`, default paths under `/mnt/ds4/config` and `/mnt/ds4/results/ticket-07`, and print-plan output.

- [ ] **Step 1: Write failing runtime print-plan and Docker classification tests**

```python
def test_container_runtime_prints_cpu_only_cache_replay_plan(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "kv-cache-replay",
            "plan",
            "--config",
            "/mnt/ds4/config/kv-cache-replay.json",
            "--output",
            str(tmp_path / "selection.json"),
            "--print-plan",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "benchmarks.ds4_profile.kv_cache_replay plan" in result.stdout
    assert "--config /mnt/ds4/config/kv-cache-replay.json" in result.stdout


def test_container_wrapper_marks_cache_replay_as_cpu_only() -> None:
    result = subprocess.run(
        [
            "bash",
            "benchmarks/ds4_profile/container/run.sh",
            "--dry-run",
            "kv-cache-replay",
            "plan",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "ai.vllm.ds4.runtime=cpu" in result.stdout
    assert "--gpus" not in result.stdout
    assert "SYS_NICE" not in result.stdout
    assert "PYTHONHASHSEED=0" in result.stdout
    assert "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0" in result.stdout
```

- [ ] **Step 2: Run container tests and verify red**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  -k 'container_runtime or container_wrapper' -v
```

Expected: first test fails because `kv-cache-replay` is not a recognized runtime command; second fails because the wrapper adds `--gpus all`.

- [ ] **Step 3: Register nested runtime commands**

Add `_kv_cache_replay_command` with the exact signature below; it maps nested
commands without invoking a shell:

```python
def _kv_cache_replay_command(
    replay_command: str,
    config_path: Path,
    output: Path | None,
    planning_record: Path | None,
    result_dir: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.ds4_profile.kv_cache_replay",
        replay_command,
    ]
    if replay_command in {"plan", "run"}:
        command.extend(["--config", str(config_path)])
    if replay_command == "plan":
        assert output is not None
        command.extend(["--output", str(output)])
    elif replay_command == "run":
        assert planning_record is not None and result_dir is not None
        command.extend(
            [
                "--planning-record",
                str(planning_record),
                "--output-dir",
                str(result_dir),
            ]
        )
    else:
        assert result_dir is not None
        command.extend(["--result-dir", str(result_dir)])
    return command
```

Add parser arguments:

```python
cache_replay = subparsers.add_parser("kv-cache-replay")
cache_replay.add_argument("replay_command", choices=("plan", "run", "validate"))
cache_replay.add_argument(
    "--config",
    type=Path,
    default=Path("/mnt/ds4/config/kv-cache-replay.json"),
)
cache_replay.add_argument(
    "--output",
    type=Path,
    default=Path("/mnt/ds4/results/ticket-07-selection.json"),
)
cache_replay.add_argument(
    "--planning-record",
    type=Path,
    default=Path("/mnt/ds4/results/ticket-07-selection.json"),
)
cache_replay.add_argument("--result-dir", type=Path)
cache_replay.add_argument("--print-plan", action="store_true")
```

Add `_effective_kv_cache_replay_config(config_path: Path) -> dict[str, Any]` by
following `_effective_profile_config`: copy the checked-in JSON, generate
`ds4-kv-replay-{UTC timestamp}-{8 hex characters}` only when `run_id` is null,
and replace `source` from `DS4_VLLM_COMMIT` and `DS4_VLLM_DIRTY` with the same
boolean-or-`"unknown"` parsing. Read the two hash environment variables from
the already-started process, require both equal the checked-in values, and
persist those actual values under `environment`. For `run`, choose
`/mnt/ds4/results/ticket-07/{run_id}` when `--result-dir` is absent, write the
effective JSON to the sibling staging path
`/mnt/ds4/results/ticket-07/.{run_id}.work/run-config.json`, and pass that path
to the module CLI. For `validate`, reject a missing `--result-dir` with
`ValueError("kv-cache-replay validate requires --result-dir")`. The
`--print-plan` branch prints the generated command without creating directories
or files; tests normalize the dynamic run ID with a monkeypatched UUID and UTC
clock.

The runtime branch prints `shlex.join(command)` and returns `0` for
`--print-plan`; otherwise it creates the run staging directory, writes the
effective config for `run`, and executes the list with
`subprocess.run(command, check=False).returncode`. Do not call Ticket 03 GPU
preflight for this CPU-only metadata path.

- [ ] **Step 4: Make the wrapper CPU allowlist explicit**

Replace the two-command negative condition with a positive case:

```bash
case "$1" in
    cache-model|cpu-dry-run|kv-cache-replay)
        gpu_args=(--label ai.vllm.ds4.runtime=cpu)
        ;;
    *)
        gpu_args=(--gpus all --cpuset-cpus=0-11 --cap-add SYS_NICE)
        ;;
esac
```

Keep `cache-model` as the only command with `HF_HUB_OFFLINE=0`; cache replay stays offline.
Add `--env PYTHONHASHSEED=0` and
`--env VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` to the common Docker invocation
so planning, replay, validation, and exact-image pytest share the same chained
hash seed and byte-valued native event format.

- [ ] **Step 5: Run container tests and regression plan checks**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  tests/benchmarks/ds4_profile/test_container_workflow.py \
  -v
```

Expected: all selected tests PASS; existing Ticket 03/04 printed plans remain unchanged.

- [ ] **Step 6: Commit container registration**

```bash
git add \
  benchmarks/ds4_profile/container/runtime.py \
  benchmarks/ds4_profile/container/run.sh \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py
git commit -m "[Benchmarks] Run cache replay in the DS4 container" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

---

### Task 7: Local Quality Gate and Operator Documentation

**Files:**
- Modify: `benchmarks/ds4_profile/README.md`
- Modify: `benchmarks/ds4_profile/container/README.md`
- Modify: `benchmarks/ds4_profile/WORKFLOW.md`
- Create: `benchmarks/ds4_profile/TICKET_07_HANDOFF.md`

**Interfaces:**
- Consumes: final local commands and artifact contract from Tasks 1-6.
- Produces: one non-duplicated operator workflow and an explicit unaccepted local handoff ready for school-server continuation.

- [ ] **Step 1: Add focused README documentation**

Add one Ticket 07 section to the main README that states:

```text
Ticket 07 replays one complete DS4 prompt sequence through the real CPU-side
KVCacheManager metadata path. It hashes prompt token IDs only. It does not read
decode tokens, allocate KV tensors, use a GPU, establish HBM residency, or
measure Prefill/Decode latency. Local development runs focused CPU contracts;
the full planner and container artifact acceptance run on the school server.
```

Link to the container runbook and `TICKET_07_HANDOFF.md`; do not duplicate server commands.

- [ ] **Step 2: Document exact school-server commands in the container runbook**

Use the existing `DS4_RUN` array and these exact paths. Resolve the newest host
directory first, then translate only its basename into the container mount:

```bash
"${DS4_RUN[@]}" kv-cache-replay plan \
  --output /mnt/ds4/results/ticket-07-selection.json

"${DS4_RUN[@]}" kv-cache-replay run \
  --planning-record /mnt/ds4/results/ticket-07-selection.json

RESULT_DIR="$(find "$STORAGE/results/ticket-07" \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
"${DS4_RUN[@]}" kv-cache-replay validate \
  --result-dir "/mnt/ds4/results/ticket-07/$(basename "$RESULT_DIR")"
```

Explain that pinning the selection requires a new clean commit/image before the `run` command can be accepted. State that successful output contains five artifacts and says `Metadata only: yes` and `GPU/HBM validated: no`.

- [ ] **Step 3: Add workflow gates and initial handoff**

`WORKFLOW.md` adds only the local/server split and links to the runbook. The initial handoff records:

- design and implementation branch name;
- local source SHA and dirty state;
- exact focused test and pre-commit commands/results;
- planner not yet accepted and selection config still `unselected`;
- no model/GPU/HBM claim;
- exact next server command; and
- requirement to update the handoff only after a new exact SHA and image pass.

- [ ] **Step 4: Run the complete lightweight local gate**

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  tests/benchmarks/ds4_profile/test_container_workflow.py \
  -v

pre-commit run --files \
  benchmarks/ds4_profile/kv_cache_replay.py \
  benchmarks/ds4_profile/config/kv-cache-replay.json \
  benchmarks/ds4_profile/container/runtime.py \
  benchmarks/ds4_profile/container/run.sh \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py \
  benchmarks/ds4_profile/README.md \
  benchmarks/ds4_profile/container/README.md \
  benchmarks/ds4_profile/WORKFLOW.md \
  benchmarks/ds4_profile/TICKET_07_HANDOFF.md
```

Expected: focused pytest PASS; all selected pre-commit hooks PASS. Record exact counts and hook names in the handoff. A proxy/bootstrap failure is infrastructure evidence, not a passing hook result.

- [ ] **Step 5: Commit local documentation and handoff**

```bash
git add \
  benchmarks/ds4_profile/README.md \
  benchmarks/ds4_profile/container/README.md \
  benchmarks/ds4_profile/WORKFLOW.md \
  benchmarks/ds4_profile/TICKET_07_HANDOFF.md
git commit -m "[Docs] Hand off the DS4 cache replay" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

---

### Task 8: School-Server Selection, Pinning, and Container Acceptance

**Files:**
- Modify: `benchmarks/ds4_profile/config/kv-cache-replay.json`
- Modify: `benchmarks/ds4_profile/TICKET_07_HANDOFF.md`

**Interfaces:**
- Consumes: clean local implementation SHA, school-server Ticket 03 image workflow, mounted immutable inputs, and planner CLI.
- Produces: pinned selection config, accepted exact SHA/image/result, independent validation, and final handoff evidence. This task still performs no GPU work.

- [ ] **Step 1: Build a clean planning image on the school server**

```bash
cd "$HOME/vllm"
git rev-parse HEAD
git status --short
bash benchmarks/ds4_profile/container/build.sh \
  --image local/vllm-ds4-profile:ticket-07-plan \
  --metadata-out "$HOME/ds4-storage/results/ticket-07-plan-image.json"
```

Expected: `git status --short` is empty; the metadata file records the exact planning SHA and immutable image ID.

- [ ] **Step 2: Run the CPU-only planner and inspect the deterministic result**

With `DS4_RUN` defined as in the runbook, using image `local/vllm-ds4-profile:ticket-07-plan`:

```bash
"${DS4_RUN[@]}" kv-cache-replay plan \
  --output /mnt/ds4/results/ticket-07-selection.json

PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -c '
import json
from pathlib import Path
p = Path.home() / "ds4-storage/results/ticket-07-selection.json"
value = json.loads(p.read_text())
assert value["status"] == "selected"
assert value["selected"]["eviction_count"] > 0
assert {
    "manifest",
    "ticket_01_data",
    "ticket_01_provenance",
    "ticket_02_data",
    "ticket_02_provenance",
}.issubset({item["logical_name"] for item in value["inputs"]})
assert any(item["logical_name"].startswith("tokenizer:") for item in value["inputs"])
print(json.dumps({
    "capacity_blocks": value["selected"]["capacity_blocks"],
    "input_set_sha256": value["selected"]["input_set_sha256"],
    "planning_sha256": value["sha256"],
    "reasoning_mode": value["selected"]["reasoning_mode"],
    "status": "pinned",
    "turn_manifest_sha256": value["selected"]["turn_manifest_sha256"],
    "trajectory_id": value["selected"]["trajectory_id"],
}, indent=2, sort_keys=True))
'
```

Expected: exit `0` and one complete JSON `selection` object. The planner
candidate list shows every rejected/eligible trajectory, every admitted
selected turn, and at least one native eviction. Its hashed input inventory
covers both ticket data/provenance pairs and every tokenizer file, and it
contains no completion/decode token field. The selected ordered turn manifest
contains every selected turn exactly once with prompt-token and block-hash
digests.

- [ ] **Step 3: Pin exactly the planner output and commit it**

Use `apply_patch` to replace the seven fields under `selection` with the exact
JSON object printed in Step 2. Do not edit `replay`, input paths, or tokenizer
revision. Then verify equality and all mounted input hashes using the module:

```bash
PYTHONHASHSEED=0 VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0 .venv/bin/python -c '
import json
from pathlib import Path
from benchmarks.ds4_profile.kv_cache_replay import verify_pinned_selection
config = json.loads(Path("benchmarks/ds4_profile/config/kv-cache-replay.json").read_text())
plan = json.loads((Path.home() / "ds4-storage/results/ticket-07-selection.json").read_text())
verify_pinned_selection(config, plan)
print("pinned selection matches planning record")
'

git add benchmarks/ds4_profile/config/kv-cache-replay.json
git commit -m "[Benchmarks] Pin the DS4 cache replay selection" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

Expected: verification prints exactly `pinned selection matches planning record`; commit changes only the config.

- [ ] **Step 4: Build the exact pinned image and print the runtime plan**

```bash
bash benchmarks/ds4_profile/container/build.sh \
  --image local/vllm-ds4-profile:ticket-07 \
  --metadata-out "$HOME/ds4-storage/results/ticket-07-image.json"

"${DS4_RUN[@]}" kv-cache-replay run \
  --planning-record /mnt/ds4/results/ticket-07-selection.json \
  --print-plan
```

Expected: image metadata source SHA equals the pinned config commit; printed command uses `/opt/ds4-profile/bin/python -m benchmarks.ds4_profile.kv_cache_replay run`, contains no `CUDA_VISIBLE_DEVICES`, and targets `/mnt/ds4/results/ticket-07/`.

- [ ] **Step 5: Run and independently validate the full metadata replay**

```bash
"${DS4_RUN[@]}" kv-cache-replay run \
  --planning-record /mnt/ds4/results/ticket-07-selection.json

RESULT_DIR="$(find "$HOME/ds4-storage/results/ticket-07" \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
"${DS4_RUN[@]}" kv-cache-replay validate \
  --result-dir "/mnt/ds4/results/ticket-07/$(basename "$RESULT_DIR")"

sha256sum \
  "$RESULT_DIR/cache_events.parquet" \
  "$RESULT_DIR/turn_summaries.parquet" \
  "$RESULT_DIR/run-config.json" \
  "$RESULT_DIR/provenance.json" \
  "$RESULT_DIR/result.md"
```

Expected: run and validator exit `0`; all configured turns pass; at least one
eviction exists; manager-forced recomputes are separate from misses; observer
eviction timing reconciles with native removals; every pinned input SHA is
recomputed; result text contains `Metadata only: yes` and
`GPU/HBM validated: no`; five checksums are printed.

- [ ] **Step 6: Run the focused tests in the exact image**

```bash
"${DS4_RUN[@]}" exec \
  --output /mnt/ds4/results/ticket-07-pytest.json \
  -- /opt/ds4-profile/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_kv_cache_replay.py -v
```

Expected: all Ticket 07 tests PASS with no skips and the exec record exits `0`.

- [ ] **Step 7: Update and commit the accepted handoff**

Record the exact pinned commit, image ID, selection values and planning SHA, result directory, test count, independent validator result, five checksums, source clean state, and the explicit no-HBM/no-decode limitation in `TICKET_07_HANDOFF.md`.

```bash
git add benchmarks/ds4_profile/TICKET_07_HANDOFF.md
git commit -m "[Docs] Record Ticket 07 metadata acceptance" \
  -m "Co-authored-by: OpenAI Codex <codex@openai.com>" \
  -m "Signed-off-by: ycsxh <1002533186@qq.com>"
```

Expected: the evidence commit changes only the handoff. Acceptance remains bound to the earlier pinned implementation SHA and immutable image ID, not to the later documentation commit.

---

## Final Review Gate

- [ ] Confirm `git diff 65de0de0ab4a5799284e97b823e673d5ac73ef05 HEAD --name-only` contains only the Ticket 07 files listed in the File Map plus this approved design/plan.
- [ ] Confirm no diff exists in `profile_spine.py`, `gpu_profile.py`, Ticket 05 files, or vLLM core manager/pool implementations.
- [ ] Confirm the planner's selected capacity is exactly the maximum live prompt block count for the selected trajectory and that the planner observed at least one native `BlockRemoved` event.
- [ ] Confirm every event/turn artifact uses schema version `1.0.0`, stable IDs, validated enum values, and cross-file run/trajectory/turn integrity.
- [ ] Confirm all native evictions have complementary `useful_later`/`never_reused` labels and correct nullability/reuse distance.
- [ ] Confirm admission failure is preserved rather than changing capacity or prompt length.
- [ ] Confirm `PYTHONHASHSEED=0` is present in local commands and Docker, null-hash initialization occurs once per process, and equal prefixes in separate requests produce equal hashes.
- [ ] Confirm `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` is set before every test/CLI/container process imports vLLM, is recorded in config/provenance, and native/prepass hashes remain the same 32-byte SHA values.
- [ ] Confirm scoped observation calls and restores the real `_maybe_evict_cached_block`, records exclusive eviction timing, and pairs true calls with native removals without implementing eviction semantics.
- [ ] Confirm resident final blocks withheld by the manager's `prompt_len - 1` cap are labeled `manager_forced_recompute`, never capacity misses.
- [ ] Confirm the validator independently reconstructs miss classes, future reuse, ordering, occupancy, and exclusive eviction timing, with corruption tests for each derived contract.
- [ ] Confirm planning, run, provenance, and validation recompute manifest, Ticket 01/02 data and provenance, and every tokenizer file SHA rather than trusting configured paths.
- [ ] Confirm the canonical selected-turn list and its prompt/block-hash digests are identical and ordered across the planning record, run config, provenance, reconstructed pinned inputs, and turn artifacts; omission, reorder, and tamper tests fail.
- [ ] Confirm local evidence contains only focused CPU contracts and server evidence contains the full planning/container run.
- [ ] Confirm docs and artifacts state that Ticket 07 used prompt metadata only and did not read Decode tokens, allocate KV tensors, establish HBM residency, or validate GPU behavior.
- [ ] Confirm no remote mutation or push occurred during implementation without a separate user instruction.
