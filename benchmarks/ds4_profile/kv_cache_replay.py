# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

import pyarrow.parquet as pq

if os.environ.get("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES") != "0":
    raise RuntimeError("Ticket 07 requires byte-valued KV event hashes")

import torch

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

ReasoningMode = Literal["no_think", "think_high"]

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
    evicted: bool | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: bytes | int) -> str:
    if isinstance(value, bytes):
        return f"sha256:{value.hex()}"
    raise ValueError("Ticket 07 requires byte-valued SHA-256 KV event hashes")


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


def _active_block_count(manager: KVCacheManager) -> int:
    pool = manager.block_pool
    return pool.num_gpu_blocks - pool.get_num_free_blocks() - 1


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
        evicted: bool | None = None,
    ) -> None:
        calls.append(PoolCall(operation, len(calls), block_ids, duration_ns, evicted))

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
