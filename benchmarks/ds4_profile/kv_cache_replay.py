# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

if os.environ.get("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") != "0":
    raise RuntimeError("Ticket 07 requires byte-valued KV event hashes")

import torch

from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

SCHEMA_VERSION = "1.0.0"

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

ReasoningMode = Literal["no_think", "think_high"]
PrefixSource = Literal["global", "task", "session"]
CacheOutcome = Literal["hit", "miss", "manager_forced_recompute"]
MissClass = Literal["compulsory", "prefix_mismatch", "capacity"]
ReplayStatus = Literal["passed", "out_of_capacity", "invalid"]

_HASHING_INITIALIZED = False

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
    index: int
    block_ids: tuple[int, ...]
    duration_ns: int
    occupancy_before: "PoolOccupancy"
    occupancy_after: "PoolOccupancy"
    evicted: bool | None = None


@dataclass(frozen=True)
class PoolOccupancy:
    active_blocks: int
    cached_resident_blocks: int


@dataclass(frozen=True)
class ReplayResult:
    status: ReplayStatus
    event_rows: list[dict[str, Any]]
    turn_rows: list[dict[str, Any]]
    eviction_count: int
    error: str | None = None


@dataclass(frozen=True)
class InputRecord:
    logical_name: str
    path: str
    size_bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_input_records(config: dict[str, Any]) -> list[InputRecord]:
    artifact_names = (
        ("manifest", "manifest"),
        ("ticket_01_data", "normalized_turns"),
        ("ticket_01_provenance", "normalized_provenance"),
        ("ticket_02_data", "rendered_turns"),
        ("ticket_02_provenance", "workload_provenance"),
    )
    paths = [
        (logical_name, Path(config["artifacts"][config_name]))
        for logical_name, config_name in artifact_names
    ]
    tokenizer_dir = Path(config["tokenizer"]["path"])
    tokenizer_files = sorted(
        (path for path in tokenizer_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(tokenizer_dir).as_posix(),
    )
    if not tokenizer_files:
        raise ValueError("tokenizer directory contains no regular files")
    paths.extend(
        (
            f"tokenizer:{path.relative_to(tokenizer_dir).as_posix()}",
            path,
        )
        for path in tokenizer_files
    )
    logical_names = [logical_name for logical_name, _ in paths]
    if len(logical_names) != len(set(logical_names)):
        raise ValueError("duplicate input logical name")
    return [
        InputRecord(
            logical_name=logical_name,
            path=str(path),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
        )
        for logical_name, path in paths
    ]


def verify_input_records(records: Sequence[InputRecord]) -> None:
    logical_names = [record.logical_name for record in records]
    if len(logical_names) != len(set(logical_names)):
        raise ValueError("duplicate input logical name")
    for record in records:
        path = Path(record.path)
        if path.stat().st_size != record.size_bytes or _sha256(path) != record.sha256:
            raise ValueError(f"input SHA-256 mismatch for {record.logical_name}")


def _canonical_hash(value: bytes | int) -> str:
    if isinstance(value, bytes):
        return f"sha256:{value.hex()}"
    raise ValueError("Ticket 07 requires byte-valued SHA-256 KV event hashes")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _with_canonical_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    without_digest = {key: value for key, value in payload.items() if key != "sha256"}
    return without_digest | {"sha256": _canonical_json_sha256(without_digest)}


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


def _resident_hashes(manager: KVCacheManager) -> set[str]:
    return {
        _canonical_hash(get_block_hash(block_hash))
        for block_hash in manager.block_pool.cached_block_hash_to_block._cache
    }


def _pool_occupancy(manager: KVCacheManager) -> PoolOccupancy:
    pool = manager.block_pool
    return PoolOccupancy(
        active_blocks=sum(
            not block.is_null and block.ref_cnt > 0 for block in pool.blocks
        ),
        cached_resident_blocks=sum(
            block.block_hash is not None
            or block.block_id in pool.cached_block_hashes_by_block
            for block in pool.blocks
            if not block.is_null
        ),
    )


@contextmanager
def observe_block_pool(manager: KVCacheManager) -> Iterator[list[PoolCall]]:
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
        occupancy_before: PoolOccupancy,
        occupancy_after: PoolOccupancy,
        evicted: bool | None = None,
    ) -> None:
        calls.append(
            PoolCall(
                operation,
                len(calls),
                block_ids,
                duration_ns,
                occupancy_before,
                occupancy_after,
                evicted,
            )
        )

    def touch(blocks):
        block_list = list(blocks)
        occupancy_before = _pool_occupancy(manager)
        started = perf_counter_ns()
        try:
            return originals[0](block_list)
        finally:
            record(
                "touch",
                tuple(block.block_id for block in block_list),
                perf_counter_ns() - started,
                occupancy_before,
                _pool_occupancy(manager),
            )

    def get_new_blocks(num_blocks: int):
        occupancy_before = _pool_occupancy(manager)
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
            max(0, elapsed - (nested_after - nested_before)),
            occupancy_before,
            _pool_occupancy(manager),
        )
        return blocks

    def maybe_evict_cached_block(block):
        occupancy_before = _pool_occupancy(manager)
        started = perf_counter_ns()
        evicted = originals[2](block)
        record(
            "evict",
            (block.block_id,),
            perf_counter_ns() - started,
            occupancy_before,
            _pool_occupancy(manager),
            evicted,
        )
        return evicted

    def free_blocks(blocks):
        block_list = list(blocks)
        occupancy_before = _pool_occupancy(manager)
        started = perf_counter_ns()
        try:
            return originals[3](block_list)
        finally:
            record(
                "free",
                tuple(block.block_id for block in block_list),
                perf_counter_ns() - started,
                occupancy_before,
                _pool_occupancy(manager),
            )

    pool.touch = touch  # type: ignore[method-assign]
    pool.get_new_blocks = get_new_blocks  # type: ignore[method-assign]
    pool._maybe_evict_cached_block = (  # type: ignore[method-assign]
        maybe_evict_cached_block
    )
    pool.free_blocks = free_blocks  # type: ignore[method-assign]
    try:
        yield calls
    finally:
        pool.touch = originals[0]  # type: ignore[method-assign]
        pool.get_new_blocks = originals[1]  # type: ignore[method-assign]
        pool._maybe_evict_cached_block = originals[2]  # type: ignore[method-assign]
        pool.free_blocks = originals[3]  # type: ignore[method-assign]


def load_full_turns(config: dict[str, Any]) -> list[ReplayTurn]:
    """Reconstruct full prompt-only turns and validate Ticket 02 scalars."""
    from benchmarks.ds4_profile import workloads

    artifacts = config["artifacts"]
    rendered = workloads.render_turns(
        manifest_path=Path(artifacts["manifest"]),
        normalized_turns_path=Path(artifacts["normalized_turns"]),
        tokenizer_path=Path(config["tokenizer"]["path"]),
        block_size=config["replay"]["block_size"],
        include_token_ids=True,
    )
    ticket_02 = pq.read_table(
        artifacts["rendered_turns"], columns=list(SCALAR_TURN_FIELDS)
    ).to_pylist()
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for row in ticket_02:
        key = (row["trajectory_id"], row["turn_index"])
        if key in expected:
            raise ValueError(f"Ticket 02 duplicate key for {key}")
        expected[key] = row

    rendered_keys = [(row["trajectory_id"], row["turn_index"]) for row in rendered]
    if len(rendered_keys) != len(set(rendered_keys)):
        raise ValueError("reconstructed turns contain duplicate keys")
    if set(rendered_keys) != set(expected):
        raise ValueError("Ticket 02 key set mismatch")
    if len(rendered_keys) != len(expected):
        raise ValueError("Ticket 02 row cardinality mismatch")

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


def _selected_turn_manifest(
    turns: Sequence[ReplayTurn], block_size: int
) -> list[dict[str, Any]]:
    ordered = sorted(turns, key=lambda turn: turn.turn_index)
    if not ordered:
        return []
    trajectory_ids = {turn.trajectory_id for turn in ordered}
    if len(trajectory_ids) != 1:
        raise ValueError("selected turns contain multiple trajectories")
    turn_indices = [turn.turn_index for turn in ordered]
    if turn_indices != list(range(len(ordered))):
        raise ValueError("selected turn indices contain duplicates or gaps")
    return [
        {
            "trajectory_id": turn.trajectory_id,
            "turn_index": turn.turn_index,
            "prompt_tokens": turn.prompt_tokens,
            "prompt_token_ids_sha256": _canonical_json_sha256(
                list(turn.prompt_token_ids)
            ),
            "block_hashes_sha256": _canonical_json_sha256(
                list(_turn_hashes(turn, block_size))
            ),
        }
        for turn in ordered
    ]


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


def _row_base(run_id: str, turn: ReplayTurn) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "trajectory_id": turn.trajectory_id,
        "task_id": turn.task_id,
        "reasoning_mode": turn.reasoning_mode,
        "turn_index": turn.turn_index,
    }


def _event_row(
    run_id: str,
    turn: ReplayTurn,
    *,
    status: ReplayStatus = "passed",
    operation: str,
    event_source: str,
    occupancy_before: PoolOccupancy,
    occupancy_after: PoolOccupancy,
    block_position: int | None = None,
    block_id: int | None = None,
    block_hash: str | None = None,
    cache_outcome: CacheOutcome | None = None,
    miss_class: MissClass | None = None,
    prefix_source: PrefixSource | None = None,
    duration_ns: int | None = None,
    observer_call_index: int | None = None,
    observer_evicted: bool | None = None,
    reuse_turn_index: int | None = None,
    turns_until_reuse: int | None = None,
    useful_later: bool | None = None,
    never_reused: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return _row_base(run_id, turn) | {
        "status": status,
        "operation": operation,
        "event_source": event_source,
        "active_blocks_before": occupancy_before.active_blocks,
        "active_blocks_after": occupancy_after.active_blocks,
        "cached_resident_blocks_before": occupancy_before.cached_resident_blocks,
        "cached_resident_blocks_after": occupancy_after.cached_resident_blocks,
        # Retain the Task 3 field names while exposing the two distinct values
        # required by the artifact contract.
        "occupancy_before": occupancy_before.active_blocks,
        "occupancy_after": occupancy_after.active_blocks,
        "block_position": block_position,
        "block_id": block_id,
        "block_hash": block_hash,
        "cache_outcome": cache_outcome,
        "miss_class": miss_class,
        "prefix_source": prefix_source,
        "duration_ns": duration_ns,
        "observer_call_index": observer_call_index,
        "observer_evicted": observer_evicted,
        "reuse_turn_index": reuse_turn_index,
        "turns_until_reuse": turns_until_reuse,
        "useful_later": useful_later,
        "never_reused": never_reused,
        "error": error,
    }


def _turn_row(
    run_id: str,
    turn: ReplayTurn,
    *,
    status: ReplayStatus,
    error: str | None,
    cached_tokens: int,
    recomputed_tokens: int,
    hit_blocks: int,
    miss_blocks: int,
    manager_forced_recompute_blocks: int,
    capacity_miss_blocks: int,
    allocation_count: int,
    eviction_count: int,
    free_count: int,
    occupancy_before: PoolOccupancy,
    occupancy_after: PoolOccupancy,
    lookup_time_ns: int = 0,
    allocation_time_ns: int = 0,
    eviction_time_ns: int = 0,
) -> dict[str, Any]:
    return _row_base(run_id, turn) | {
        "status": status,
        "error": error,
        "prompt_tokens": turn.prompt_tokens,
        "cached_tokens": cached_tokens,
        "recomputed_tokens": recomputed_tokens,
        "hit_blocks": hit_blocks,
        "miss_blocks": miss_blocks,
        "manager_forced_recompute_blocks": manager_forced_recompute_blocks,
        "capacity_miss_blocks": capacity_miss_blocks,
        "allocation_count": allocation_count,
        "eviction_count": eviction_count,
        "free_count": free_count,
        "active_blocks_before": occupancy_before.active_blocks,
        "active_blocks_after": occupancy_after.active_blocks,
        "cached_resident_blocks_before": occupancy_before.cached_resident_blocks,
        "cached_resident_blocks_after": occupancy_after.cached_resident_blocks,
        "cached_resident_blocks_after_free": occupancy_after.cached_resident_blocks,
        "occupancy_before": occupancy_before.active_blocks,
        "occupancy_after": occupancy_after.active_blocks,
        "lookup_time_ns": lookup_time_ns,
        "allocation_time_ns": allocation_time_ns,
        "eviction_time_ns": eviction_time_ns,
    }


def _out_of_capacity_result(
    *,
    run_id: str,
    turn: ReplayTurn,
    event_rows: list[dict[str, Any]],
    turn_rows: list[dict[str, Any]],
    manager: KVCacheManager,
    error: str,
    occupancy_before: PoolOccupancy,
    lookup_time_ns: int,
) -> ReplayResult:
    occupancy = _pool_occupancy(manager)
    event_rows.append(
        _event_row(
            run_id,
            turn,
            status="out_of_capacity",
            operation="admission_failure",
            event_source="replay",
            occupancy_before=occupancy_before,
            occupancy_after=occupancy,
            error=error,
        )
    )
    turn_rows.append(
        _turn_row(
            run_id,
            turn,
            status="out_of_capacity",
            error=error,
            cached_tokens=0,
            recomputed_tokens=turn.prompt_tokens,
            hit_blocks=0,
            miss_blocks=0,
            manager_forced_recompute_blocks=0,
            capacity_miss_blocks=0,
            allocation_count=0,
            eviction_count=0,
            free_count=0,
            occupancy_before=occupancy_before,
            occupancy_after=occupancy,
            lookup_time_ns=lookup_time_ns,
        )
    )
    return ReplayResult(
        "out_of_capacity",
        event_rows,
        turn_rows,
        _native_eviction_count(event_rows),
        error,
    )


def _distributed_durations(total_duration_ns: int, count: int) -> tuple[int, ...]:
    if count == 0:
        return ()
    duration_ns, remainder = divmod(total_duration_ns, count)
    return tuple(duration_ns + (position < remainder) for position in range(count))


def _native_eviction_count(event_rows: Sequence[dict[str, Any]]) -> int:
    return sum(
        row["operation"] == "evict" and row["event_source"] == "native"
        for row in event_rows
    )


def _future_reuse(
    block_hash: str,
    turn_index: int,
    future_accesses: dict[str, tuple[int, ...]],
) -> tuple[int | None, int | None, bool, bool]:
    reuse_turn = next(
        (
            access_turn
            for access_turn in future_accesses.get(block_hash, ())
            if access_turn > turn_index
        ),
        None,
    )
    if reuse_turn is None:
        return None, None, False, True
    return reuse_turn, reuse_turn - turn_index, True, False


def replay_session(
    *,
    run_id: str,
    turns: Sequence[ReplayTurn],
    capacity_blocks: int,
    block_size: int,
    max_model_len: int,
) -> ReplayResult:
    _initialize_hashing()
    manager = make_manager(capacity_blocks, block_size, max_model_len)
    event_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    future_accesses = _future_accesses(turns, block_size)
    ever_stored: set[str] = set()
    max_seen_depth = 0

    for turn in turns:
        occupancy_before = _pool_occupancy(manager)
        try:
            request = make_request(turn, block_size, f"{run_id}:{turn.turn_index}")
            resident_before = _resident_hashes(manager)
            lookup_started = perf_counter_ns()
            computed, hit_tokens, _ = manager.get_computed_blocks(request)
            lookup_time_ns = perf_counter_ns() - lookup_started
            occupancy_after_lookup = _pool_occupancy(manager)
            hit_blocks = hit_tokens // block_size
            request_hashes = tuple(
                _canonical_hash(value) for value in request.block_hashes
            )
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
            miss_classes = tuple(
                _classify_miss(
                    block_hash=block_hash,
                    block_position=position,
                    ever_stored=ever_stored,
                    max_seen_depth=max_seen_depth,
                )
                if outcome == "miss"
                else None
                for position, (block_hash, outcome) in enumerate(
                    zip(request_hashes, outcomes, strict=True)
                )
            )

            allocation_before = _pool_occupancy(manager)
            with observe_block_pool(manager) as pool_calls:
                allocation_started = perf_counter_ns()
                allocated = manager.allocate_slots(
                    request,
                    request.num_tokens - hit_tokens,
                    hit_tokens,
                    computed,
                )
                allocation_elapsed = perf_counter_ns() - allocation_started
                native_events = manager.take_events()
                occupancy_after_allocation = _pool_occupancy(manager)
                request_block_ids = ()
                if allocated is not None:
                    request_block_ids = tuple(
                        manager.get_block_ids(request.request_id)[0]
                    )
                    manager.free(request)

            eviction_time_ns = sum(
                call.duration_ns for call in pool_calls if call.operation == "evict"
            )
            allocation_time_ns = max(0, allocation_elapsed - eviction_time_ns)
            occupancy_after_free = _pool_occupancy(manager)
            block_ids = tuple(request_block_ids[: len(request_hashes)])
            status: ReplayStatus = (
                "passed" if allocated is not None else "out_of_capacity"
            )

            for position, (block_hash, outcome, miss_class, duration_ns) in enumerate(
                zip(
                    request_hashes,
                    outcomes,
                    miss_classes,
                    _distributed_durations(lookup_time_ns, len(request_hashes)),
                    strict=True,
                )
            ):
                event_rows.append(
                    _event_row(
                        run_id,
                        turn,
                        status=status,
                        operation="lookup",
                        event_source="replay",
                        occupancy_before=occupancy_before,
                        occupancy_after=occupancy_after_lookup,
                        block_position=position,
                        block_id=(
                            block_ids[position] if position < len(block_ids) else None
                        ),
                        block_hash=block_hash,
                        cache_outcome=outcome,
                        miss_class=miss_class,
                        prefix_source=_prefix_source(turn, position, block_size),
                        duration_ns=duration_ns,
                    )
                )

            removed_hashes = [
                _canonical_hash(block_hash)
                for event in native_events
                if isinstance(event, BlockRemoved)
                for block_hash in event.block_hashes
            ]
            evict_calls = [
                call
                for call in pool_calls
                if call.operation == "evict" and call.evicted is True
            ]
            if len(evict_calls) != len(removed_hashes):
                raise RuntimeError("observer/native eviction count mismatch")
            evict_call_iter = iter(evict_calls)

            for call in pool_calls:
                if call.operation == "free":
                    continue
                physical_ids = call.block_ids or (None,)
                for occurrence, block_id in enumerate(physical_ids):
                    event_rows.append(
                        _event_row(
                            run_id,
                            turn,
                            status=status,
                            operation=call.operation,
                            event_source="observer",
                            occupancy_before=call.occupancy_before,
                            occupancy_after=call.occupancy_after,
                            block_id=block_id,
                            duration_ns=(call.duration_ns if occurrence == 0 else 0),
                            observer_call_index=call.index,
                            observer_evicted=call.evicted,
                        )
                    )

            for event in native_events:
                if isinstance(event, BlockStored):
                    for block_hash in event.block_hashes or []:
                        canonical_hash = _canonical_hash(block_hash)
                        position = request_hashes.index(canonical_hash)
                        event_rows.append(
                            _event_row(
                                run_id,
                                turn,
                                status=status,
                                operation="store",
                                event_source="native",
                                occupancy_before=allocation_before,
                                occupancy_after=occupancy_after_allocation,
                                block_position=position,
                                block_id=(
                                    block_ids[position]
                                    if position < len(block_ids)
                                    else None
                                ),
                                block_hash=canonical_hash,
                                prefix_source=_prefix_source(
                                    turn, position, block_size
                                ),
                            )
                        )
                        ever_stored.add(canonical_hash)
                elif isinstance(event, BlockRemoved):
                    for block_hash in event.block_hashes:
                        canonical_hash = _canonical_hash(block_hash)
                        evict_call = next(evict_call_iter)
                        reuse_turn, reuse_distance, useful_later, never_reused = (
                            _future_reuse(
                                canonical_hash, turn.turn_index, future_accesses
                            )
                        )
                        event_rows.append(
                            _event_row(
                                run_id,
                                turn,
                                status=status,
                                operation="evict",
                                event_source="native",
                                occupancy_before=evict_call.occupancy_before,
                                occupancy_after=evict_call.occupancy_after,
                                block_hash=canonical_hash,
                                observer_call_index=evict_call.index,
                                reuse_turn_index=reuse_turn,
                                turns_until_reuse=reuse_distance,
                                useful_later=useful_later,
                                never_reused=never_reused,
                            )
                        )

            for call in pool_calls:
                if call.operation != "free":
                    continue
                for occurrence, block_id in enumerate(call.block_ids or (None,)):
                    event_rows.append(
                        _event_row(
                            run_id,
                            turn,
                            status=status,
                            operation=call.operation,
                            event_source="observer",
                            occupancy_before=call.occupancy_before,
                            occupancy_after=call.occupancy_after,
                            block_id=block_id,
                            duration_ns=(call.duration_ns if occurrence == 0 else 0),
                            observer_call_index=call.index,
                            observer_evicted=call.evicted,
                        )
                    )

            if allocated is None and (
                pool_calls
                or native_events
                or occupancy_after_allocation != allocation_before
            ):
                for row in event_rows:
                    if (
                        row["trajectory_id"] == turn.trajectory_id
                        and row["turn_index"] == turn.turn_index
                    ):
                        row["status"] = "invalid"
                raise RuntimeError(
                    "KVCacheManager returned a non-atomic admission failure"
                )

            if allocated is None:
                return _out_of_capacity_result(
                    run_id=run_id,
                    turn=turn,
                    event_rows=event_rows,
                    turn_rows=turn_rows,
                    manager=manager,
                    error="KV cache admission failed at pinned capacity",
                    occupancy_before=occupancy_before,
                    lookup_time_ns=lookup_time_ns,
                )

            miss_blocks = sum(outcome == "miss" for outcome in outcomes)
            capacity_miss_blocks = sum(
                miss_class == "capacity" for miss_class in miss_classes
            )
            turn_rows.append(
                _turn_row(
                    run_id,
                    turn,
                    status="passed",
                    error=None,
                    cached_tokens=hit_tokens,
                    recomputed_tokens=request.num_tokens - hit_tokens,
                    hit_blocks=sum(outcome == "hit" for outcome in outcomes),
                    miss_blocks=miss_blocks,
                    manager_forced_recompute_blocks=sum(
                        outcome == "manager_forced_recompute" for outcome in outcomes
                    ),
                    capacity_miss_blocks=capacity_miss_blocks,
                    allocation_count=sum(
                        call.operation == "allocate" for call in pool_calls
                    ),
                    eviction_count=len(removed_hashes),
                    free_count=sum(call.operation == "free" for call in pool_calls),
                    occupancy_before=occupancy_before,
                    occupancy_after=occupancy_after_free,
                    lookup_time_ns=lookup_time_ns,
                    allocation_time_ns=allocation_time_ns,
                    eviction_time_ns=eviction_time_ns,
                )
            )
            max_seen_depth = max(max_seen_depth, len(request_hashes))
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            occupancy_after = _pool_occupancy(manager)
            completed_evictions = sum(
                row["operation"] == "evict" and row["event_source"] == "native"
                for row in event_rows
                if row["trajectory_id"] == turn.trajectory_id
                and row["turn_index"] == turn.turn_index
            )
            turn_rows.append(
                _turn_row(
                    run_id,
                    turn,
                    status="invalid",
                    error=error_text,
                    cached_tokens=0,
                    recomputed_tokens=turn.prompt_tokens,
                    hit_blocks=0,
                    miss_blocks=0,
                    manager_forced_recompute_blocks=0,
                    capacity_miss_blocks=0,
                    allocation_count=0,
                    eviction_count=completed_evictions,
                    free_count=0,
                    occupancy_before=occupancy_before,
                    occupancy_after=occupancy_after,
                )
            )
            return ReplayResult(
                "invalid",
                event_rows,
                turn_rows,
                _native_eviction_count(event_rows),
                error_text,
            )

    return ReplayResult(
        "passed", event_rows, turn_rows, _native_eviction_count(event_rows)
    )


def _select_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reasoning_rank = {"no_think": 0, "think_high": 1}
    eligible = [item for item in candidates if item["status"] == "eligible"]
    if not eligible:
        raise ValueError("no full trajectory admits all turns")
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
    input_rows = sorted(
        (asdict(record) for record in inputs),
        key=lambda row: row["logical_name"],
    )
    input_set_sha256 = _canonical_json_sha256(input_rows)
    candidates: list[dict[str, Any]] = []
    for trajectory_id in sorted({turn.trajectory_id for turn in turns}):
        session = sorted(
            (turn for turn in turns if turn.trajectory_id == trajectory_id),
            key=lambda turn: turn.turn_index,
        )
        selected_turns = _selected_turn_manifest(session, block_size)
        reasoning_modes = {turn.reasoning_mode for turn in session}
        if len(reasoning_modes) != 1:
            raise ValueError(f"trajectory {trajectory_id} mixes reasoning modes")
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
        eligible = result.status == "passed"
        reason = result.error
        candidates.append(
            {
                "trajectory_id": trajectory_id,
                "reasoning_mode": session[0].reasoning_mode,
                "turn_count": len(session),
                "capacity_blocks": capacity,
                "eviction_count": result.eviction_count,
                "turns": selected_turns,
                "turn_manifest_sha256": _canonical_json_sha256(selected_turns),
                "status": "eligible" if eligible else "rejected",
                "reason": reason,
            }
        )
    try:
        selected = {
            **_select_candidate(candidates),
            "input_set_sha256": input_set_sha256,
        }
    except ValueError as error:
        return _with_canonical_sha256(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "no_selection",
                "selected": None,
                "candidates": candidates,
                "inputs": input_rows,
                "error": str(error),
            }
        )
    return _with_canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "selected",
            "selected": selected,
            "candidates": candidates,
            "inputs": input_rows,
        }
    )


def verify_pinned_selection(config: dict[str, Any], plan: dict[str, Any]) -> None:
    selection = config["selection"]
    if selection.get("status") != "pinned":
        raise ValueError("replay requires a pinned selection")
    expected_plan_sha256 = _canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    if plan.get("sha256") != expected_plan_sha256:
        raise ValueError("planning SHA-256 mismatch")
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("status") != (
        "selected"
    ):
        raise ValueError("invalid selection planning record")

    try:
        records = [InputRecord(**row) for row in plan["inputs"]]
        selected = plan["selected"]
        selected_turns = selected["turns"]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid selection planning record") from error
    verify_input_records(records)
    if "artifacts" in config and "tokenizer" in config:
        current_records = collect_input_records(config)
        if current_records != records:
            raise ValueError("planning input paths do not match pinned config")
    input_rows = [asdict(record) for record in records]
    input_set_sha256 = _canonical_json_sha256(input_rows)
    turn_manifest_sha256 = _canonical_json_sha256(selected_turns)
    pinned_fields = (
        "trajectory_id",
        "reasoning_mode",
        "capacity_blocks",
        "input_set_sha256",
        "turn_manifest_sha256",
    )
    if (
        selected.get("input_set_sha256") != input_set_sha256
        or selected.get("turn_manifest_sha256") != turn_manifest_sha256
        or selection.get("planning_sha256") != plan["sha256"]
        or any(selection.get(field) != selected.get(field) for field in pinned_fields)
    ):
        raise ValueError("pinned selection does not match planning record")


def _normalized_event_rows(
    result: ReplayResult, *, capacity_blocks: int, block_size: int
) -> list[dict[str, Any]]:
    """Convert the raw manager trace into the chronological artifact ledger."""
    normalized: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in result.event_rows:
        grouped.setdefault((row["trajectory_id"], row["turn_index"]), []).append(row)
    summaries = {
        (row["trajectory_id"], row["turn_index"]): row for row in result.turn_rows
    }
    state = (0, 0)

    def append(raw: dict[str, Any], *, transition: bool = False) -> None:
        nonlocal state
        before = state
        after = before
        if transition:
            after = (
                raw["active_blocks_after"],
                raw["cached_resident_blocks_after"],
            )
        ordinal = len(
            [
                row
                for row in normalized
                if (row["trajectory_id"], row["turn_index"])
                == (raw["trajectory_id"], raw["turn_index"])
            ]
        )
        operation = raw["operation"]
        normalized.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": raw["run_id"],
                "trajectory_id": raw["trajectory_id"],
                "turn_index": raw["turn_index"],
                "event_id": (
                    f"{raw['run_id']}:{raw['trajectory_id']}:"
                    f"{raw['turn_index']}:{operation}:{ordinal}"
                ),
                "event_source": (
                    "lookup" if operation == "lookup" else raw["event_source"]
                ),
                "operation": operation,
                "operation_ordinal": ordinal,
                "status": raw["status"],
                "cache_outcome": raw.get("cache_outcome"),
                "miss_class": raw.get("miss_class"),
                "duration_ns": raw.get("duration_ns"),
                "eviction_time_ns": (
                    raw.get("duration_ns")
                    if operation == "evict" and raw["event_source"] == "observer"
                    else None
                ),
                "evicted": raw.get("observer_evicted"),
                "block_position": raw.get("block_position"),
                "block_id": raw.get("block_id"),
                "block_hash": raw.get("block_hash"),
                "prefix_source": raw.get("prefix_source"),
                "token_count": block_size if operation == "lookup" else None,
                "active_blocks_before": before[0],
                "active_blocks_after": after[0],
                "cached_blocks_before": before[1],
                "cached_blocks_after": after[1],
                "free_blocks_before": capacity_blocks - before[0],
                "free_blocks_after": capacity_blocks - after[0],
                "useful_later": raw.get("useful_later"),
                "never_reused": raw.get("never_reused"),
                "next_reuse_turn": raw.get("reuse_turn_index"),
                "turns_until_reuse": raw.get("turns_until_reuse"),
                "error": raw.get("error"),
            }
        )
        state = after

    for key in sorted(grouped):
        rows = grouped[key]
        summary = summaries[key]
        seed = rows[0]
        append(
            seed
            | {
                "operation": "hash",
                "event_source": "replay",
                "duration_ns": summary.get("hash_time_ns", 0),
                "block_position": None,
                "block_id": None,
                "block_hash": None,
                "cache_outcome": None,
                "miss_class": None,
                "prefix_source": None,
            }
        )
        for row in rows:
            if row["operation"] == "lookup":
                append(row)

        observers = [row for row in rows if row["event_source"] == "observer"]
        native_evictions = {
            row["observer_call_index"]: row
            for row in rows
            if row["event_source"] == "native" and row["operation"] == "evict"
        }
        native_stores = [
            row
            for row in rows
            if row["event_source"] == "native" and row["operation"] == "store"
        ]
        calls: dict[int, list[dict[str, Any]]] = {}
        for row in observers:
            calls.setdefault(row["observer_call_index"], []).append(row)
        deferred_free: list[list[dict[str, Any]]] = []
        for call_index in sorted(calls):
            call_rows = calls[call_index]
            operation = call_rows[0]["operation"]
            if operation == "free":
                deferred_free.append(call_rows)
                continue
            for occurrence, row in enumerate(call_rows):
                append(row, transition=occurrence == 0)
            if operation == "evict" and call_rows[0]["observer_evicted"]:
                native = native_evictions[call_index]
                append(native | {"block_id": call_rows[0]["block_id"]})
            if operation == "allocate":
                for native in native_stores:
                    append(native)
        for call_rows in deferred_free:
            for occurrence, row in enumerate(call_rows):
                append(row, transition=occurrence == 0)
        for row in rows:
            if row["operation"] == "admission_failure":
                append(row, transition=True)
    return normalized


def _normalized_turn_rows(
    result: ReplayResult, events: Sequence[dict[str, Any]], block_size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in result.turn_rows:
        turn_events = [
            row
            for row in events
            if row["trajectory_id"] == source["trajectory_id"]
            and row["turn_index"] == source["turn_index"]
        ]
        lookup = [row for row in turn_events if row["operation"] == "lookup"]
        observer = [row for row in turn_events if row["event_source"] == "observer"]
        misses = [row["miss_class"] for row in lookup if row["miss_class"]]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": source["run_id"],
                "trajectory_id": source["trajectory_id"],
                "turn_index": source["turn_index"],
                "status": source["status"],
                "prompt_tokens": source["prompt_tokens"],
                "full_blocks": len(lookup),
                "cached_tokens": source["cached_tokens"],
                "recomputed_tokens": source["recomputed_tokens"],
                "hit_blocks": sum(row["cache_outcome"] == "hit" for row in lookup),
                "manager_forced_recompute_blocks": sum(
                    row["cache_outcome"] == "manager_forced_recompute" for row in lookup
                ),
                "compulsory_miss_blocks": misses.count("compulsory"),
                "capacity_miss_blocks": misses.count("capacity"),
                "prefix_mismatch_blocks": misses.count("prefix_mismatch"),
                "allocated_blocks": sum(
                    row["operation"] == "allocate" for row in observer
                ),
                "evicted_blocks": sum(
                    row["operation"] == "evict" and row["event_source"] == "native"
                    for row in turn_events
                ),
                "freed_blocks": sum(row["operation"] == "free" for row in observer),
                "cached_resident_blocks_after_free": source[
                    "cached_resident_blocks_after_free"
                ],
                "hash_time_ns": sum(
                    row["duration_ns"] or 0
                    for row in turn_events
                    if row["operation"] == "hash"
                ),
                "lookup_time_ns": sum(row["duration_ns"] or 0 for row in lookup),
                "touch_time_ns": sum(
                    row["duration_ns"] or 0
                    for row in observer
                    if row["operation"] == "touch"
                ),
                "allocation_time_ns": sum(
                    row["duration_ns"] or 0
                    for row in observer
                    if row["operation"] == "allocate"
                ),
                "eviction_time_ns": sum(
                    row["eviction_time_ns"] or 0 for row in observer
                ),
                "free_time_ns": sum(
                    row["duration_ns"] or 0
                    for row in observer
                    if row["operation"] == "free"
                ),
                "error": source["error"],
            }
        )
    return rows


def _result_markdown(result: ReplayResult) -> str:
    pilot_eviction_pressure_observed = result.eviction_count > 0
    return (
        "# DS4 Ticket 07 KV Cache Replay\n\n"
        f"Status: {result.status}\n\n"
        "Metadata only: yes\n\n"
        "Pilot eviction pressure observed: "
        f"{'yes' if pilot_eviction_pressure_observed else 'no'}\n\n"
        f"Native eviction count: {result.eviction_count}\n\n"
        "GPU/HBM validated: no\n"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_result(
    config: dict[str, Any], result: ReplayResult, output_dir: Path
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"result directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    events = _normalized_event_rows(
        result,
        capacity_blocks=config["selection"]["capacity_blocks"],
        block_size=config["replay"]["block_size"],
    )
    turns = _normalized_turn_rows(result, events, config["replay"]["block_size"])
    provenance = {
        "artifact_schema_version": SCHEMA_VERSION,
        "hardware_validated": False,
        "environment": config["environment"],
        "inputs": config["inputs"],
        "metadata_only_validated": result.status == "passed",
        "pilot_eviction_pressure_observed": result.eviction_count > 0,
        "planning_record": config["planning_record"],
        "planning_sha256": config["selection"]["planning_sha256"],
        "run_id": config["run_id"],
        "selected_turns": config["selected_turns"],
        "selection": config["selection"],
        "source": config["source"],
        "status": result.status,
    }
    pq.write_table(
        pa.Table.from_pylist(events, schema=CACHE_EVENT_SCHEMA),
        stage / "cache_events.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(turns, schema=TURN_SUMMARY_SCHEMA),
        stage / "turn_summaries.parquet",
    )
    _write_json(stage / "run-config.json", config)
    _write_json(stage / "provenance.json", provenance)
    (stage / "result.md").write_text(_result_markdown(result))
    validate_result_dir(stage)
    stage.rename(output_dir)


def _require_exact_schema(table: pa.Table, expected: pa.Schema, name: str) -> None:
    if table.schema != expected:
        raise ValueError(f"{name} schema mismatch")


def _validate_manifests(
    config: dict[str, Any], provenance: dict[str, Any], turns: list[dict[str, Any]]
) -> None:
    plan = config.get("planning_record")
    selected = config.get("selected_turns")
    if not isinstance(plan, dict) or plan != provenance.get("planning_record"):
        raise ValueError("selected turn manifest mismatch: planning record")
    if plan.get("sha256") != _canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "sha256"}
    ):
        raise ValueError("planning SHA-256 mismatch")
    plan_turns = plan.get("selected", {}).get("turns")
    if selected != plan_turns or selected != provenance.get("selected_turns"):
        raise ValueError("selected turn manifest mismatch")
    if _canonical_json_sha256(selected) != config["selection"].get(
        "turn_manifest_sha256"
    ):
        raise ValueError("selected turn manifest mismatch: digest")
    artifact_manifest = [
        {
            "trajectory_id": row["trajectory_id"],
            "turn_index": row["turn_index"],
            "prompt_tokens": row["prompt_tokens"],
        }
        for row in turns
    ]
    selected_projection = [
        {
            "trajectory_id": row["trajectory_id"],
            "turn_index": row["turn_index"],
            "prompt_tokens": row["prompt_tokens"],
        }
        for row in selected
    ]
    if artifact_manifest != selected_projection:
        raise ValueError("selected turn manifest mismatch: artifact turns")

    records = [InputRecord(**row) for row in provenance.get("inputs", [])]
    verify_input_records(records)
    input_rows = [asdict(record) for record in records]
    if _canonical_json_sha256(input_rows) != config["selection"].get(
        "input_set_sha256"
    ):
        raise ValueError("input set SHA-256 mismatch")
    if input_rows != config.get("inputs") or input_rows != plan.get("inputs"):
        raise ValueError("input records mismatch")

    if "artifacts" in config and "tokenizer" in config:
        reconstructed = [
            turn
            for turn in load_full_turns(config)
            if turn.trajectory_id == config["selection"]["trajectory_id"]
        ]
        expected = _selected_turn_manifest(
            reconstructed, config["replay"]["block_size"]
        )
        if expected != selected:
            raise ValueError("selected turn manifest mismatch: reconstructed inputs")


def _validate_event_rows(
    events: list[dict[str, Any]], turns: list[dict[str, Any]], capacity: int
) -> None:
    operations = {
        "hash",
        "lookup",
        "touch",
        "allocate",
        "store",
        "evict",
        "free",
        "admission_failure",
    }
    sources = {"lookup", "observer", "native", "replay"}
    statuses = {"passed", "out_of_capacity", "invalid"}
    outcomes = {None, "hit", "miss", "manager_forced_recompute"}
    miss_classes = {None, "compulsory", "capacity", "prefix_mismatch"}
    prefixes = {None, "global", "task", "session"}
    seen_ids: set[str] = set()
    prior_after: tuple[int, int, int] | None = None
    for row in events:
        if row["operation"] not in operations:
            raise ValueError(f"unknown operation: {row['operation']}")
        if row["event_source"] not in sources:
            raise ValueError("unknown event source")
        if row["status"] not in statuses:
            raise ValueError("unknown status")
        if row["cache_outcome"] not in outcomes:
            raise ValueError("unknown cache outcome")
        if row["miss_class"] not in miss_classes:
            raise ValueError("unknown miss class")
        if row["prefix_source"] not in prefixes:
            raise ValueError("unknown prefix source")
        expected_id = (
            f"{row['run_id']}:{row['trajectory_id']}:{row['turn_index']}:"
            f"{row['operation']}:{row['operation_ordinal']}"
        )
        if row["event_id"] != expected_id or expected_id in seen_ids:
            raise ValueError("event ID mismatch or duplicate")
        seen_ids.add(expected_id)
        before = (
            row["active_blocks_before"],
            row["cached_blocks_before"],
            row["free_blocks_before"],
        )
        after = (
            row["active_blocks_after"],
            row["cached_blocks_after"],
            row["free_blocks_after"],
        )
        if prior_after is not None and before != prior_after:
            raise ValueError("occupancy transition mismatch")
        if (
            min(*before, *after) < 0
            or before[0] + before[2] != capacity
            or after[0] + after[2] != capacity
            or before[1] > capacity
            or after[1] > capacity
        ):
            raise ValueError("occupancy transition mismatch")
        prior_after = after
        if (row["cache_outcome"] == "miss") != (row["miss_class"] is not None):
            raise ValueError("miss attribution mismatch")
        is_observer_evict = (
            row["event_source"] == "observer" and row["operation"] == "evict"
        )
        if is_observer_evict:
            if row["eviction_time_ns"] != row["duration_ns"]:
                raise ValueError("eviction timing mismatch")
        elif row["eviction_time_ns"] is not None:
            raise ValueError("eviction timing mismatch")

    summary_by_key = {(row["trajectory_id"], row["turn_index"]): row for row in turns}
    resident: dict[int, str] = {}
    ever_stored: set[str] = set()
    max_seen_depth = 0
    for key in sorted(summary_by_key):
        rows = sorted(
            (row for row in events if (row["trajectory_id"], row["turn_index"]) == key),
            key=lambda row: row["operation_ordinal"],
        )
        if [row["operation_ordinal"] for row in rows] != list(range(len(rows))):
            raise ValueError("operation ordering mismatch: ordinals")
        operations_in_turn = [row["operation"] for row in rows]
        if not operations_in_turn or operations_in_turn[0] != "hash":
            raise ValueError("operation ordering mismatch: hash")
        lookup_indices = [
            i for i, row in enumerate(rows) if row["operation"] == "lookup"
        ]
        if lookup_indices and lookup_indices != list(range(1, 1 + len(lookup_indices))):
            raise ValueError("operation ordering mismatch: lookup")
        if any(
            row["operation"] == "touch"
            for row in rows[
                next(
                    (i for i, row in enumerate(rows) if row["operation"] == "allocate"),
                    len(rows),
                ) :
            ]
        ):
            raise ValueError("operation ordering mismatch: touch")
        free_at = [i for i, row in enumerate(rows) if row["operation"] == "free"]
        if free_at and free_at != list(range(free_at[0], len(rows))):
            raise ValueError("operation ordering mismatch: free")
        for index, row in enumerate(rows):
            if (
                row["event_source"] == "observer"
                and row["operation"] == "evict"
                and row["evicted"] is True
                and (
                    index + 1 >= len(rows)
                    or not (
                        rows[index + 1]["event_source"] == "native"
                        and rows[index + 1]["operation"] == "evict"
                        and rows[index + 1]["block_id"] == row["block_id"]
                    )
                )
            ):
                raise ValueError("operation ordering mismatch: eviction adjacency")

        lookups = [row for row in rows if row["operation"] == "lookup"]
        hit_count = 0
        for position, row in enumerate(lookups):
            block_hash = row["block_hash"]
            resident_hashes = set(resident.values())
            expected_miss = None
            if block_hash not in resident_hashes:
                expected_miss = (
                    "capacity"
                    if block_hash in ever_stored
                    else "prefix_mismatch"
                    if position < max_seen_depth
                    else "compulsory"
                )
            if row["cache_outcome"] == "hit":
                if block_hash not in resident_hashes or position != hit_count:
                    raise ValueError("miss attribution mismatch")
                hit_count += 1
            elif row["cache_outcome"] == "manager_forced_recompute":
                summary = summary_by_key[key]
                if not (
                    block_hash in resident_hashes
                    and position == len(lookups) - 1
                    and summary["prompt_tokens"] % row["token_count"] == 0
                    and position == hit_count
                ):
                    raise ValueError("miss attribution mismatch")
            elif row["cache_outcome"] != "miss" or row["miss_class"] != expected_miss:
                raise ValueError("miss attribution mismatch")

        for row in rows:
            if row["event_source"] == "native" and row["operation"] == "store":
                if row["block_id"] is None or row["block_hash"] is None:
                    raise ValueError("native store lacks physical identity")
                resident[row["block_id"]] = row["block_hash"]
                ever_stored.add(row["block_hash"])
            elif row["event_source"] == "native" and row["operation"] == "evict":
                if resident.get(row["block_id"]) != row["block_hash"]:
                    raise ValueError("native eviction physical accounting mismatch")
                del resident[row["block_id"]]
        max_seen_depth = max(max_seen_depth, len(lookups))

        summary = summary_by_key[key]
        misses = [row["miss_class"] for row in lookups if row["miss_class"]]
        expected_counts = {
            "full_blocks": len(lookups),
            "hit_blocks": sum(row["cache_outcome"] == "hit" for row in lookups),
            "manager_forced_recompute_blocks": sum(
                row["cache_outcome"] == "manager_forced_recompute" for row in lookups
            ),
            "compulsory_miss_blocks": misses.count("compulsory"),
            "capacity_miss_blocks": misses.count("capacity"),
            "prefix_mismatch_blocks": misses.count("prefix_mismatch"),
            "allocated_blocks": sum(
                row["event_source"] == "observer" and row["operation"] == "allocate"
                for row in rows
            ),
            "evicted_blocks": sum(
                row["event_source"] == "native" and row["operation"] == "evict"
                for row in rows
            ),
            "freed_blocks": sum(
                row["event_source"] == "observer" and row["operation"] == "free"
                for row in rows
            ),
        }
        if any(summary[field] != value for field, value in expected_counts.items()):
            raise ValueError("turn summary count mismatch")
        if (
            summary["cached_tokens"] + summary["recomputed_tokens"]
            != summary["prompt_tokens"]
        ):
            raise ValueError("turn token accounting mismatch")

    for index, row in enumerate(events):
        if row["event_source"] != "native" or row["operation"] != "evict":
            if any(
                row[field] is not None
                for field in (
                    "useful_later",
                    "never_reused",
                    "next_reuse_turn",
                    "turns_until_reuse",
                )
            ):
                raise ValueError("future reuse mismatch: nullability")
            continue
        future_turn = next(
            (
                candidate["turn_index"]
                for candidate in events[index + 1 :]
                if candidate["operation"] == "lookup"
                and candidate["block_hash"] == row["block_hash"]
                and candidate["turn_index"] > row["turn_index"]
            ),
            None,
        )
        expected = (
            future_turn is not None,
            future_turn is None,
            future_turn,
            None if future_turn is None else future_turn - row["turn_index"],
        )
        actual = tuple(
            row[field]
            for field in (
                "useful_later",
                "never_reused",
                "next_reuse_turn",
                "turns_until_reuse",
            )
        )
        if actual != expected:
            raise ValueError("future reuse mismatch")


def validate_result_dir(result_dir: Path) -> None:
    required = {
        "cache_events.parquet",
        "turn_summaries.parquet",
        "run-config.json",
        "provenance.json",
        "result.md",
    }
    present = (
        {path.name for path in result_dir.iterdir()} if result_dir.is_dir() else set()
    )
    if present != required:
        raise ValueError("result file set mismatch")
    event_table = pq.read_table(result_dir / "cache_events.parquet")
    turn_table = pq.read_table(result_dir / "turn_summaries.parquet")
    _require_exact_schema(event_table, CACHE_EVENT_SCHEMA, "cache event")
    _require_exact_schema(turn_table, TURN_SUMMARY_SCHEMA, "turn summary")
    events = event_table.to_pylist()
    turns = turn_table.to_pylist()
    config = json.loads((result_dir / "run-config.json").read_text())
    provenance = json.loads((result_dir / "provenance.json").read_text())
    if any(row["schema_version"] != SCHEMA_VERSION for row in events + turns):
        raise ValueError("artifact row schema version mismatch")
    run_ids = {row["run_id"] for row in events + turns}
    run_ids.update((config.get("run_id"), provenance.get("run_id")))
    if len(run_ids) != 1:
        raise ValueError("run ID mismatch")
    required_environment = {
        "PYTHONHASHSEED": "0",
        "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": "0",
    }
    if (
        config.get("environment") != required_environment
        or provenance.get("environment") != required_environment
        or any(
            os.environ.get(key) != value for key, value in required_environment.items()
        )
    ):
        raise ValueError("deterministic environment mismatch")
    if config.get("replay", {}).get("kv_event_hash_format") != "bytes":
        raise ValueError("KV event hash format mismatch")
    if provenance.get("hardware_validated") is not False:
        raise ValueError("hardware validation claim is forbidden")
    status = provenance.get("status")
    if provenance.get("metadata_only_validated") != (status == "passed"):
        raise ValueError("metadata-only validation status mismatch")
    _validate_manifests(config, provenance, turns)
    _validate_event_rows(events, turns, config["selection"]["capacity_blocks"])
    pilot_eviction_pressure_observed = any(
        row["event_source"] == "native" and row["operation"] == "evict"
        for row in events
    )
    if (
        provenance.get("pilot_eviction_pressure_observed")
        is not pilot_eviction_pressure_observed
    ):
        raise ValueError("pilot eviction pressure status mismatch")


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _plan_cli(config_path: Path, output: Path) -> int:
    config = _load_json_object(config_path)
    plan = build_selection_plan(config, load_full_turns(config))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, plan)
    return 0 if plan["status"] == "selected" else 2


def _run_cli(config_path: Path, planning_record_path: Path, output_dir: Path) -> int:
    _initialize_hashing()
    config = _load_json_object(config_path)
    plan = _load_json_object(planning_record_path)
    verify_pinned_selection(config, plan)
    all_turns = load_full_turns(config)
    trajectory_id = config["selection"]["trajectory_id"]
    turns = [turn for turn in all_turns if turn.trajectory_id == trajectory_id]
    selected_turns = _selected_turn_manifest(turns, config["replay"]["block_size"])
    if selected_turns != plan["selected"]["turns"]:
        raise ValueError("selected turn manifest mismatch")
    effective_config = {
        **config,
        "inputs": plan["inputs"],
        "planning_record": plan,
        "selected_turns": selected_turns,
    }
    result = replay_session(
        run_id=effective_config["run_id"],
        turns=turns,
        capacity_blocks=effective_config["selection"]["capacity_blocks"],
        block_size=effective_config["replay"]["block_size"],
        max_model_len=effective_config["replay"]["max_model_len"],
    )
    write_result(effective_config, result, output_dir)
    return 0 if result.status == "passed" else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="DS4 Ticket 07 metadata replay")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--planning-record", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            returncode = _plan_cli(args.config, args.output)
        elif args.command == "run":
            returncode = _run_cli(args.config, args.planning_record, args.output_dir)
        else:
            validate_result_dir(args.result_dir)
            returncode = 0
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        failure_name = "validation" if args.command == "validate" else args.command
        print(f"{failure_name} failed: {error}", file=sys.stderr)
        returncode = 2
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
