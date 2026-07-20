# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

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
V2_VALIDATION_STATES = frozenset({"remote_pending", "remote_failed", "remote_verified"})

# Task 2 consumes this contract to generate the executable planner output.  Task
# 1 deliberately keeps it small and data-only so validation can independently
# recompute the frozen manifest without trusting emitted rows or point records.
CANONICAL_V2_PLANNER_INPUTS = {
    "schema_version": V2_SCHEMA_VERSION,
    "workload_selectors": tuple(f"canonical-{index:02d}" for index in range(34)),
    "kv_cache_groups": ("0",),
    "seed": 20260715,
    "block_size": 16,
    "chunk_budget": 4096,
}

# Kept as an alias for the Ticket 04 writer and its public helper functions.
SCHEMA_VERSION = V1_SCHEMA_VERSION

RAW_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("point_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("ordinal", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("runner_boundary", pa.string(), nullable=False),
        pa.field("batch_size", pa.int32(), nullable=False),
        pa.field("num_scheduled_tokens", pa.int32(), nullable=False),
        pa.field("prompt_tokens", pa.int32(), nullable=False),
        pa.field("context_tokens", pa.int32(), nullable=False),
        pa.field("cached_tokens", pa.int32(), nullable=False),
        pa.field("new_tokens", pa.int32(), nullable=False),
        pa.field("sampled_token_id", pa.int64()),
        pa.field("injected_token_id", pa.int64()),
        pa.field("sampled_token_discarded", pa.bool_(), nullable=False),
        pa.field("runner_wall_time_ms", pa.float64()),
        pa.field("cuda_model_time_ms", pa.float64()),
        pa.field("derived_time_ms", pa.float64()),
        pa.field("compile_enabled", pa.bool_(), nullable=False),
        pa.field("cudagraph_enabled", pa.bool_(), nullable=False),
        pa.field("cudagraph_runtime_mode", pa.string()),
        pa.field("error", pa.string()),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode()},
)

V1_RAW_SAMPLE_SCHEMA = RAW_SAMPLE_SCHEMA

AGGREGATE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("point_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("sample_count", pa.int32(), nullable=False),
        pa.field("runner_wall_time_median_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_p90_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_mean_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_cv", pa.float64(), nullable=False),
        pa.field("noisy", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode()},
)

V1_AGGREGATE_SCHEMA = AGGREGATE_SCHEMA


def canonical_payload_json(payload: dict[str, Any]) -> str:
    """Return the canonical JSON representation used by v2 identifiers."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _identifier(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_payload_json(payload).encode()).hexdigest()
    return f"{prefix}-{digest}"


def make_point_id(payload: dict[str, Any]) -> str:
    """Return the deterministic identifier for one condition-specific point."""
    return _identifier("p2", payload)


def make_comparison_id(payload: dict[str, Any]) -> str:
    """Return the deterministic identifier shared by a hit/recompute pair."""
    comparison = copy.deepcopy(payload)
    comparison.pop("cache_condition")
    comparison.pop("planned_chunks")
    return _identifier("pc2", comparison)


def canonical_v2_planner_inputs() -> dict[str, Any]:
    """Return the immutable inputs used to derive the v2 68-point manifest."""
    return copy.deepcopy(CANONICAL_V2_PLANNER_INPUTS)


def canonical_v2_points() -> list[dict[str, Any]]:
    """Return the Task 2 planner contract's deterministic v2 point records."""
    points = []
    inputs = canonical_v2_planner_inputs()
    for selector in inputs["workload_selectors"]:
        for cache_condition in ("prefix_hit", "full_recompute"):
            planned_chunks = [
                {
                    "chunk_index": 0,
                    "scheduled_tokens_by_request": [["r0", 512]],
                }
            ]
            payload = {
                "workload_family": "homogeneous",
                "selector": selector,
                "requests": [
                    {
                        "request_key": "r0",
                        "trajectory_id": None,
                        "turn_index": None,
                        "reasoning_mode": None,
                        "context_tokens": 4608,
                        "cached_tokens": 4096,
                        "new_tokens": 512,
                        "token_digest": hashlib.sha256(selector.encode()).hexdigest(),
                    }
                ],
                "composition": "none",
                "seed": inputs["seed"],
                "batch_size": 1,
                "chunk_budget": inputs["chunk_budget"],
                "cache_condition": cache_condition,
                "block_size": inputs["block_size"],
                "homogeneous_prefix_tokens": 4096,
                "capacity_target": "native",
                "planner_digest": hashlib.sha256(
                    canonical_payload_json(inputs).encode()
                ).hexdigest(),
                "planned_chunks": planned_chunks,
            }
            points.append(
                {
                    "point_id": make_point_id(payload),
                    "comparison_id": make_comparison_id(payload),
                    "canonical_payload": payload,
                }
            )
    return points


_V2_METADATA = {b"schema_version": V2_SCHEMA_VERSION.encode()}
_REQUEST_TOKEN_VECTOR = pa.list_(
    pa.struct(
        [
            pa.field("request_key", pa.string(), nullable=False),
            pa.field("scheduled_tokens", pa.int32(), nullable=False),
        ]
    )
)
_INT_LIST = pa.list_(pa.int32())
_STRING_LIST = pa.list_(pa.string())
_SHAPE = pa.list_(pa.int64())


def _v2_schema(fields: list[pa.Field]) -> pa.Schema:
    return pa.schema(fields, metadata=_V2_METADATA)


V2_RAW_SAMPLE_SCHEMA = _v2_schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("point_id", pa.string(), nullable=False),
        pa.field("comparison_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("workload_family", pa.string(), nullable=False),
        pa.field("selector", pa.string(), nullable=False),
        pa.field("composition", pa.string(), nullable=False),
        pa.field("cache_condition", pa.string(), nullable=False),
        pa.field("planner_digest", pa.string(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("ordinal", pa.int32(), nullable=False),
        pa.field("chunk_index", pa.int32(), nullable=False),
        pa.field("chunk_count", pa.int32(), nullable=False),
        pa.field("row_kind", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("allocation_state", pa.string(), nullable=False),
        pa.field(
            "planned_scheduled_tokens_by_request",
            _REQUEST_TOKEN_VECTOR,
            nullable=False,
        ),
        pa.field(
            "actual_scheduled_tokens_by_request",
            _REQUEST_TOKEN_VECTOR,
            nullable=False,
        ),
        pa.field("preempted_request_ids", _STRING_LIST, nullable=False),
        pa.field("unrelated_request_ids", _STRING_LIST, nullable=False),
        pa.field("cache_epoch", pa.int64(), nullable=False),
        pa.field("cache_reset_completed", pa.bool_(), nullable=False),
        pa.field("cache_reset_empty", pa.bool_(), nullable=False),
        pa.field("allocator_pressure_proven", pa.bool_(), nullable=False),
        pa.field("clean_reset_proven", pa.bool_(), nullable=False),
        pa.field("requested_kv_blocks", pa.int32(), nullable=False),
        pa.field("allocatable_kv_blocks", pa.int32(), nullable=False),
        pa.field("allocated_kv_blocks", pa.int32(), nullable=False),
        pa.field("kv_block_bytes", pa.int64(), nullable=False),
        pa.field("requested_kv_bytes", pa.int64(), nullable=False),
        pa.field("allocated_kv_bytes", pa.int64(), nullable=False),
        pa.field("scheduled_tokens", pa.int32(), nullable=False),
        pa.field("context_tokens", pa.int32(), nullable=False),
        pa.field("cached_tokens", pa.int32(), nullable=False),
        pa.field("new_tokens", pa.int32(), nullable=False),
        pa.field("recomputed_tokens", pa.int32(), nullable=False),
        pa.field("lookup_time_ms", pa.float64()),
        pa.field("allocation_time_ms", pa.float64()),
        pa.field("scheduler_time_ms", pa.float64()),
        pa.field("cache_reset_time_ms", pa.float64()),
        pa.field("prefix_prime_time_ms", pa.float64()),
        pa.field("runner_wall_time_ms", pa.float64()),
        pa.field("cuda_model_time_ms", pa.float64()),
        pa.field("runtime_mode", pa.string()),
        pa.field("error", pa.string()),
    ]
)

V2_TURN_SAMPLE_SCHEMA = _v2_schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("point_id", pa.string(), nullable=False),
        pa.field("comparison_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("workload_family", pa.string(), nullable=False),
        pa.field("selector", pa.string(), nullable=False),
        pa.field("composition", pa.string(), nullable=False),
        pa.field("cache_condition", pa.string(), nullable=False),
        pa.field("planner_digest", pa.string(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("ordinal", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("allocation_state", pa.string(), nullable=False),
        pa.field("chunk_count", pa.int32(), nullable=False),
        pa.field("scheduled_tokens", pa.int32(), nullable=False),
        pa.field("context_tokens", pa.int32(), nullable=False),
        pa.field("cached_tokens", pa.int32(), nullable=False),
        pa.field("new_tokens", pa.int32(), nullable=False),
        pa.field("recomputed_tokens", pa.int32(), nullable=False),
        pa.field("requested_kv_blocks", pa.int32(), nullable=False),
        pa.field("allocated_kv_blocks", pa.int32(), nullable=False),
        pa.field("requested_kv_bytes", pa.int64(), nullable=False),
        pa.field("allocated_kv_bytes", pa.int64(), nullable=False),
        pa.field("lookup_time_ms", pa.float64(), nullable=False),
        pa.field("allocation_time_ms", pa.float64(), nullable=False),
        pa.field("scheduler_time_ms", pa.float64(), nullable=False),
        pa.field("cache_reset_time_ms", pa.float64(), nullable=False),
        pa.field("prefix_prime_time_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_ms", pa.float64(), nullable=False),
        pa.field("cuda_model_time_ms", pa.float64(), nullable=False),
        pa.field("throughput_tokens_per_s", pa.float64(), nullable=False),
        pa.field("runtime_mode", pa.string(), nullable=False),
    ]
)

V2_AGGREGATE_SCHEMA = _v2_schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("point_id", pa.string(), nullable=False),
        pa.field("comparison_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("sample_count", pa.int32(), nullable=False),
        pa.field("runner_wall_time_median_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_p90_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_mean_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_cv", pa.float64(), nullable=False),
        pa.field("throughput_median_tokens_per_s", pa.float64(), nullable=False),
        pa.field("throughput_p90_tokens_per_s", pa.float64(), nullable=False),
        pa.field("throughput_mean_tokens_per_s", pa.float64(), nullable=False),
        pa.field("throughput_cv", pa.float64(), nullable=False),
        pa.field("noisy", pa.bool_(), nullable=False),
    ]
)

V2_COMPARISON_SCHEMA = _v2_schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("comparison_id", pa.string(), nullable=False),
        pa.field("prefix_hit_point_id", pa.string(), nullable=False),
        pa.field("full_recompute_point_id", pa.string(), nullable=False),
        pa.field("prefix_hit_median_ms", pa.float64(), nullable=False),
        pa.field("full_recompute_median_ms", pa.float64(), nullable=False),
        pa.field("recompute_penalty_ms", pa.float64(), nullable=False),
        pa.field("recompute_penalty_ratio", pa.float64(), nullable=False),
    ]
)

V2_PREFIX_EVIDENCE_SCHEMA = _v2_schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("point_id", pa.string(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("ordinal", pa.int32(), nullable=False),
        pa.field("request_key", pa.string(), nullable=False),
        pa.field("kv_cache_group", pa.string(), nullable=False),
        pa.field("prime_scheduler_block_ids", _INT_LIST, nullable=False),
        pa.field("measured_scheduler_block_ids", _INT_LIST, nullable=False),
        pa.field("live_kv_tensor_names", _STRING_LIST, nullable=False),
        pa.field("live_kv_tensor_devices", _STRING_LIST, nullable=False),
        pa.field("live_kv_tensor_shapes", pa.list_(_SHAPE), nullable=False),
        pa.field("block_axis", pa.int32(), nullable=False),
        pa.field("block_dimension", pa.int32(), nullable=False),
        pa.field("verified_physical_block_ids", _INT_LIST, nullable=False),
        pa.field("intended_cached_tokens", pa.int32(), nullable=False),
        pa.field("actual_cached_tokens", pa.int32(), nullable=False),
        pa.field("prime_completed", pa.bool_(), nullable=False),
        pa.field("prime_synchronized", pa.bool_(), nullable=False),
        pa.field("live_cuda_tensor_proven", pa.bool_(), nullable=False),
        pa.field("hardware_validated", pa.bool_(), nullable=False),
    ]
)


class PChunkView(Protocol):
    chunk_index: int
    scheduled_tokens_by_request: dict[str, int]


class PPointView(Protocol):
    point_id: str
    comparison_id: str
    workload_family: str
    selector: str
    composition: str
    cache_condition: str
    planner_digest: str
    chunks: tuple[PChunkView, ...]


@dataclass(frozen=True)
class _ManifestChunk:
    chunk_index: int
    scheduled_tokens_by_request: dict[str, int]


@dataclass(frozen=True)
class _ManifestPoint:
    point_id: str
    comparison_id: str
    workload_family: str
    selector: str
    composition: str
    cache_condition: str
    planner_digest: str
    chunks: tuple[_ManifestChunk, ...]


def summarize_turn_samples(
    raw_rows: list[dict[str, Any]], points: tuple[PPointView, ...]
) -> list[dict[str, Any]]:
    """Derive complete turn samples from passed schema-v2 chunk rows."""
    points_by_id = {point.point_id: point for point in points}
    terminal_point_ids = {
        row["point_id"] for row in raw_rows if row["row_kind"] == "terminal"
    }
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        if row["point_id"] in terminal_point_ids:
            continue
        if row["row_kind"] != "chunk" or row["status"] != "passed":
            raise ValueError("turn samples require passed chunk rows")
        key = (row["run_id"], row["point_id"], row["phase"], row["ordinal"])
        grouped.setdefault(key, []).append(row)

    turns = []
    for (run_id, point_id, phase, ordinal), chunks in sorted(grouped.items()):
        point = points_by_id.get(point_id)
        if point is None:
            raise ValueError("raw chunk references an unknown point")
        chunks.sort(key=lambda row: row["chunk_index"])
        if [row["chunk_index"] for row in chunks] != list(
            range(len(point.chunks))
        ) or any(row["chunk_count"] != len(point.chunks) for row in chunks):
            raise ValueError("turn samples require every planned chunk")
        first = chunks[0]
        totals = {
            field: sum(row[field] or 0.0 for row in chunks)
            for field in _TURN_TOTAL_FIELDS
        }
        wall_time = totals["runner_wall_time_ms"]
        runtime_modes = {row["runtime_mode"] for row in chunks}
        if not runtime_modes <= V2_ENUMS["runtime_mode"]:
            raise ValueError("turn samples require CUDA Graph runtime evidence")
        turns.append(
            {
                "schema_version": V2_SCHEMA_VERSION,
                "run_id": run_id,
                "point_id": point_id,
                "comparison_id": point.comparison_id,
                "sample_id": f"{run_id}:{point_id}:{phase}:{ordinal}",
                "role": "prefill",
                "workload_family": point.workload_family,
                "selector": point.selector,
                "composition": point.composition,
                "cache_condition": point.cache_condition,
                "planner_digest": point.planner_digest,
                "phase": phase,
                "ordinal": ordinal,
                "status": "passed",
                "allocation_state": "allocated",
                "chunk_count": len(chunks),
                **totals,
                "throughput_tokens_per_s": (
                    totals["new_tokens"] * 1000 / wall_time if wall_time else 0.0
                ),
                "runtime_mode": (
                    "PIECEWISE" if "PIECEWISE" in runtime_modes else "FULL"
                ),
            }
        )
        if any(
            row["comparison_id"] != first["comparison_id"]
            or row["planner_digest"] != first["planner_digest"]
            for row in chunks
        ):
            raise ValueError("turn chunks do not share point metadata")
    return turns


def _distribution(values: list[float], threshold: float) -> dict[str, Any]:
    if len(values) < 2:
        raise ValueError("distribution requires at least two samples")
    mean = statistics.fmean(values)
    cv = statistics.pstdev(values) / mean if mean else 0.0
    return {
        "median": statistics.median(values),
        "p90": statistics.quantiles(values, n=10, method="inclusive")[8],
        "mean": mean,
        "cv": cv,
        "noisy": cv > threshold,
    }


def aggregate_turn_samples(
    turn_rows: list[dict[str, Any]], noisy_cv_threshold: float
) -> list[dict[str, Any]]:
    """Aggregate exactly the ten steady passed turns for each point."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in turn_rows:
        if row["phase"] == "steady" and row["status"] == "passed":
            grouped.setdefault((row["run_id"], row["point_id"]), []).append(row)
    aggregates = []
    for (run_id, point_id), rows in sorted(grouped.items()):
        if len(rows) != 10 or {row["ordinal"] for row in rows} != set(range(10)):
            raise ValueError("aggregate requires ten steady turn samples")
        wall = _distribution(
            [row["runner_wall_time_ms"] for row in rows], noisy_cv_threshold
        )
        throughput = _distribution(
            [row["throughput_tokens_per_s"] for row in rows],
            noisy_cv_threshold,
        )
        first = rows[0]
        aggregates.append(
            {
                "schema_version": V2_SCHEMA_VERSION,
                "run_id": run_id,
                "point_id": point_id,
                "comparison_id": first["comparison_id"],
                "role": "prefill",
                "sample_count": len(rows),
                "runner_wall_time_median_ms": wall["median"],
                "runner_wall_time_p90_ms": wall["p90"],
                "runner_wall_time_mean_ms": wall["mean"],
                "runner_wall_time_cv": wall["cv"],
                "throughput_median_tokens_per_s": throughput["median"],
                "throughput_p90_tokens_per_s": throughput["p90"],
                "throughput_mean_tokens_per_s": throughput["mean"],
                "throughput_cv": throughput["cv"],
                "noisy": wall["noisy"] or throughput["noisy"],
            }
        )
    return aggregates


def _terminal_ooc_proven(row: dict[str, Any], point: PPointView) -> bool:
    chunk_index = row["chunk_index"]
    phase = row["phase"]
    ordinal_limit = 3 if phase == "warmup" else 10 if phase == "steady" else 0
    if (
        not isinstance(chunk_index, int)
        or chunk_index < 0
        or chunk_index >= len(point.chunks)
        or row["chunk_count"] != len(point.chunks)
        or ordinal_limit == 0
        or row["ordinal"] < 0
        or row["ordinal"] >= ordinal_limit
        or row["sample_id"] != _raw_sample_id(row)
    ):
        return False
    planned = point.chunks[chunk_index].scheduled_tokens_by_request
    expected_vector = [
        {"request_key": key, "scheduled_tokens": tokens}
        for key, tokens in planned.items()
    ]
    if row["planned_scheduled_tokens_by_request"] != expected_vector:
        return False
    actual_items = row["actual_scheduled_tokens_by_request"]
    actual = {item["request_key"]: item["scheduled_tokens"] for item in actual_items}
    if len(actual) != len(actual_items) or any(
        key not in planned or tokens <= 0 or tokens > planned[key]
        for key, tokens in actual.items()
    ):
        return False
    expected_request_ids = set(planned)
    if (
        not set(row["preempted_request_ids"]) <= expected_request_ids
        or row["unrelated_request_ids"]
    ):
        return False
    block_bytes = row["kv_block_bytes"]
    return (
        row["row_kind"] == "terminal"
        and row["status"] == "out_of_capacity"
        and row["allocation_state"] == "out_of_capacity"
        and row["scheduled_tokens"] == sum(actual.values())
        and row["requested_kv_blocks"] > row["allocatable_kv_blocks"]
        and row["allocated_kv_blocks"] <= row["allocatable_kv_blocks"]
        and block_bytes > 0
        and row["requested_kv_bytes"] == row["requested_kv_blocks"] * block_bytes
        and row["allocated_kv_bytes"] == row["allocated_kv_blocks"] * block_bytes
        and row["allocator_pressure_proven"]
        and row["clean_reset_proven"]
        and row["cache_reset_completed"]
        and row["cache_reset_empty"]
        and row["runner_wall_time_ms"] is None
        and row["cuda_model_time_ms"] is None
        and row["runtime_mode"] is None
    )


def compare_conditions(
    aggregate_rows: list[dict[str, Any]],
    points: tuple[PPointView, ...],
    terminal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair hit/recompute aggregates or require terminal OOC proof."""
    aggregates = {row["point_id"]: row for row in aggregate_rows}
    pairs: dict[str, dict[str, Any]] = {}
    points_by_id = {point.point_id: point for point in points}
    terminals = set()
    for row in terminal_rows:
        point = points_by_id.get(row["point_id"])
        if point is None or not _terminal_ooc_proven(row, point):
            raise ValueError("comparison received unvalidated terminal OOC evidence")
        if row["point_id"] in terminals:
            raise ValueError("comparison received duplicate terminal OOC evidence")
        terminals.add(row["point_id"])
    for point in points:
        pair = pairs.setdefault(point.comparison_id, {})
        if point.cache_condition in pair:
            raise ValueError("comparison pair contains a duplicate condition")
        pair[point.cache_condition] = point
    comparisons = []
    for comparison_id, pair in sorted(pairs.items()):
        if set(pair) != {"prefix_hit", "full_recompute"}:
            raise ValueError("comparison pair is incomplete")
        hit = pair["prefix_hit"]
        recompute = pair["full_recompute"]
        hit_row = aggregates.get(hit.point_id)
        recompute_row = aggregates.get(recompute.point_id)
        if hit_row is None or recompute_row is None:
            if hit.point_id in terminals or recompute.point_id in terminals:
                continue
            raise ValueError("comparison aggregate is missing without OOC proof")
        if hit_row["run_id"] != recompute_row["run_id"]:
            raise ValueError("comparison aggregates have different run IDs")
        hit_median = hit_row["runner_wall_time_median_ms"]
        recompute_median = recompute_row["runner_wall_time_median_ms"]
        comparisons.append(
            {
                "schema_version": V2_SCHEMA_VERSION,
                "run_id": hit_row["run_id"],
                "comparison_id": comparison_id,
                "prefix_hit_point_id": hit.point_id,
                "full_recompute_point_id": recompute.point_id,
                "prefix_hit_median_ms": hit_median,
                "full_recompute_median_ms": recompute_median,
                "recompute_penalty_ms": recompute_median - hit_median,
                "recompute_penalty_ratio": recompute_median / hit_median,
            }
        )
    return comparisons


def _point_plans_from_config(config: dict[str, Any]) -> tuple[_ManifestPoint, ...]:
    expected_manifest = set(config["expected_manifest"])
    points = []
    for record in config["points"]:
        if record["point_id"] not in expected_manifest:
            continue
        payload = record.get("canonical_payload", record.get("payload"))
        if not isinstance(payload, dict):
            raise ValueError("config point is missing its canonical payload")
        chunks = tuple(
            _ManifestChunk(
                chunk_index=index,
                scheduled_tokens_by_request={
                    item["request_key"]: item["scheduled_tokens"] for item in vector
                },
            )
            for index, vector in enumerate(_planned_chunks(payload))
        )
        points.append(
            _ManifestPoint(
                point_id=record["point_id"],
                comparison_id=record["comparison_id"],
                workload_family=payload["workload_family"],
                selector=payload["selector"],
                composition=payload["composition"],
                cache_condition=payload["cache_condition"],
                planner_digest=payload["planner_digest"],
                chunks=chunks,
            )
        )
    if {point.point_id for point in points} != expected_manifest:
        raise ValueError("expected_manifest references an unknown config point")
    return tuple(points)


def write_v2_result_artifacts(
    config: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    prefix_evidence_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    output_dir: Path,
) -> None:
    """Derive, validate, and atomically publish schema-v2 artifacts."""
    if output_dir.exists():
        raise FileExistsError(f"result directory already exists: {output_dir}")
    points = _point_plans_from_config(config)
    terminal_rows = [row for row in raw_rows if row["row_kind"] == "terminal"]
    derivation_error = None
    try:
        turn_rows = summarize_turn_samples(raw_rows, points)
        aggregate_rows = aggregate_turn_samples(
            turn_rows, config.get("profile", {}).get("noisy_cv_threshold", 0.05)
        )
        comparison_rows = compare_conditions(aggregate_rows, points, terminal_rows)
    except ValueError as error:
        if provenance.get("validation_state") != "remote_failed":
            raise
        derivation_error = str(error)
        turn_rows = []
        aggregate_rows = []
        comparison_rows = []
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging-"
    ) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        _write_json(staging / "run-config.json", config)
        staged_provenance = copy.deepcopy(provenance)
        staged_provenance.setdefault("validation_state", "remote_pending")
        if derivation_error is not None:
            staged_provenance["validation_error"] = derivation_error
        _write_json(staging / "provenance.json", staged_provenance)
        for name, rows, schema in (
            ("raw_samples.parquet", raw_rows, V2_RAW_SAMPLE_SCHEMA),
            ("turn_samples.parquet", turn_rows, V2_TURN_SAMPLE_SCHEMA),
            ("aggregates.parquet", aggregate_rows, V2_AGGREGATE_SCHEMA),
            ("comparisons.parquet", comparison_rows, V2_COMPARISON_SCHEMA),
            (
                "prefix_evidence.parquet",
                prefix_evidence_rows,
                V2_PREFIX_EVIDENCE_SCHEMA,
            ),
        ):
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), staging / name)
        noisy_points = [row["point_id"] for row in aggregate_rows if row["noisy"]]
        artifact_names = sorted(_v2_required_files() - {"result.md"})
        result_lines = [
            "# DS4 P-side profile result",
            "",
            "- Validation state: see `provenance.json`",
            f"- Run: `{config['run_id']}`",
            f"- Points: {len(config['expected_manifest'])}",
            f"- Capacity boundary: {len(terminal_rows)} out-of-capacity points",
            f"- Noisy points: {', '.join(noisy_points) if noisy_points else 'none'}",
            f"- Artifacts: {', '.join(artifact_names)}",
        ]
        (staging / "result.md").write_text("\n".join(result_lines) + "\n")
        try:
            _validate_result_dir(staging)
        except ValueError as error:
            if staged_provenance.get("validation_state") != "remote_failed":
                raise
            staged_provenance["validation_error"] = str(error)
            _write_json(staging / "provenance.json", staged_provenance)
        staging.replace(output_dir)


def _load_replay(config: dict[str, Any]) -> dict[str, Any]:
    plan = json.loads(Path(config["artifacts"]["workload_plan"]).read_text())
    replay_ref = min(
        plan["exact_replays"], key=lambda item: item.get("prompt_tokens", 0)
    )
    rows = pq.read_table(
        config["artifacts"]["rendered_turns"],
        columns=[
            "trajectory_id",
            "turn_index",
            "execution_prompt_token_ids",
            "execution_completion_token_ids",
        ],
    ).to_pylist()
    replay = next(
        (
            row
            for row in rows
            if row["trajectory_id"] == replay_ref["trajectory_id"]
            and row["turn_index"] == replay_ref["turn_index"]
        ),
        None,
    )
    if replay is None:
        raise ValueError("the selected exact replay is absent from rendered turns")
    if not replay["execution_prompt_token_ids"]:
        raise ValueError("the selected exact replay has no execution prompt tokens")
    if not replay["execution_completion_token_ids"]:
        raise ValueError("the selected exact replay has no execution completion tokens")
    return replay


def _resolve_config(config: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    runtime = resolved["runtime"]
    runtime["effective_max_model_len"] = max(
        8192,
        len(replay["execution_prompt_token_ids"])
        + len(replay["execution_completion_token_ids"])
        + 1,
    )
    compilation = runtime["compilation"]
    required_execution_sizes = {
        1,
        resolved["profile"]["prefill_chunk_tokens"],
    }
    if not required_execution_sizes.issubset(compilation["compile_sizes"]):
        raise ValueError(
            "compile_sizes must cover decode and the configured prefill chunk"
        )
    if not required_execution_sizes.issubset(compilation["capture_sizes"]):
        raise ValueError(
            "capture_sizes must cover decode and the configured prefill chunk"
        )
    return resolved


def make_sample_row(
    *,
    run_id: str,
    point_id: str,
    role: str,
    phase: str,
    ordinal: int,
    scheduled_tokens: int,
    prompt_tokens: int,
    elapsed_ms: float | None,
    sampled_token_id: int | None = None,
    injected_token_id: int | None = None,
    runner_boundary: str = "deterministic-fixture",
    cuda_model_time_ms: float | None = None,
    cached_tokens: int = 0,
    context_tokens: int = 0,
    new_tokens: int | None = None,
    sampled_token_discarded: bool = False,
    cudagraph_runtime_mode: str | None = None,
    compile_enabled: bool = True,
    cudagraph_enabled: bool = True,
    status: str = "passed",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "point_id": point_id,
        "sample_id": f"{run_id}:{point_id}:{phase}:{ordinal}",
        "role": role,
        "phase": phase,
        "ordinal": ordinal,
        "status": status,
        "runner_boundary": runner_boundary,
        "batch_size": 1,
        "num_scheduled_tokens": scheduled_tokens,
        "prompt_tokens": prompt_tokens,
        "context_tokens": context_tokens,
        "cached_tokens": cached_tokens,
        "new_tokens": scheduled_tokens if new_tokens is None else new_tokens,
        "sampled_token_id": sampled_token_id,
        "injected_token_id": injected_token_id,
        "sampled_token_discarded": sampled_token_discarded,
        "runner_wall_time_ms": elapsed_ms,
        "cuda_model_time_ms": cuda_model_time_ms,
        "derived_time_ms": None,
        "compile_enabled": compile_enabled,
        "cudagraph_enabled": cudagraph_enabled,
        "cudagraph_runtime_mode": cudagraph_runtime_mode,
        "error": error,
    }


def _fixture_samples(config: dict[str, Any], replay: dict[str, Any]) -> list[dict]:
    profile = config["profile"]
    run_id = config["run_id"]
    points = (
        (
            "prefill",
            f"prefill-b1-t{profile['prefill_chunk_tokens']}",
            profile["prefill_chunk_tokens"],
        ),
        ("decode", "decode-b1-t1", 1),
    )
    rows = []
    prompt_tokens = len(replay["execution_prompt_token_ids"])
    completion_token_ids = replay["execution_completion_token_ids"]
    for point_offset, (role, point_id, scheduled_tokens) in enumerate(points):
        decode_step_index = 0
        for phase, count, base_ms in (
            ("startup", 1, 100.0),
            ("capture", 1, 50.0),
            ("warmup", profile["warmup_repetitions"], 5.0),
            ("steady", profile["measured_repetitions"], 10.0),
        ):
            for ordinal in range(count):
                is_decode_step = role == "decode" and phase in {"warmup", "steady"}
                sampled_token_id = (
                    100_000 + decode_step_index if is_decode_step else None
                )
                injected_token_id = (
                    completion_token_ids[decode_step_index + 1]
                    if is_decode_step
                    else None
                )
                rows.append(
                    make_sample_row(
                        run_id=run_id,
                        point_id=point_id,
                        role=role,
                        phase=phase,
                        ordinal=ordinal,
                        scheduled_tokens=scheduled_tokens,
                        prompt_tokens=prompt_tokens,
                        elapsed_ms=base_ms + point_offset + ordinal / 10,
                        sampled_token_id=sampled_token_id,
                        injected_token_id=injected_token_id,
                        sampled_token_discarded=is_decode_step,
                        context_tokens=(
                            prompt_tokens + decode_step_index + 1
                            if is_decode_step
                            else scheduled_tokens
                        ),
                        cached_tokens=(
                            prompt_tokens + decode_step_index if is_decode_step else 0
                        ),
                        cudagraph_runtime_mode=("FULL" if is_decode_step else None),
                    )
                )
                if is_decode_step:
                    decode_step_index += 1
    return rows


def _gpu_worker_plan(
    config: dict[str, Any], replay: dict[str, Any], role: str
) -> dict[str, Any]:
    profile = config["profile"]
    runtime = config["runtime"]
    prompt_token_count = len(replay["execution_prompt_token_ids"])
    max_batched_tokens = profile["max_num_batched_tokens"]
    setup_prefill_chunks = []
    remaining = prompt_token_count
    while remaining:
        chunk_tokens = min(remaining, max_batched_tokens)
        setup_prefill_chunks.append(chunk_tokens)
        remaining -= chunk_tokens
    step_count = profile["warmup_repetitions"] + profile["measured_repetitions"]
    completion_token_ids = replay["execution_completion_token_ids"]
    required_decode_tokens = step_count + 1
    if role == "decode" and len(completion_token_ids) < required_decode_tokens:
        raise ValueError(
            "the selected exact replay does not contain enough decode tokens "
            f"for setup plus {step_count} profile steps"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["run_id"],
        "role": role,
        "runner_boundary": "vllm.v1.worker.gpu_worker.Worker.execute_model",
        "engine": {
            "block_size": profile["block_size"],
            "compilation_mode": runtime["compilation"]["mode"],
            "cudagraph_enabled": (runtime["compilation"]["cudagraph_mode"] != "NONE"),
            "enforce_eager": runtime["enforce_eager"],
            "max_num_batched_tokens": max_batched_tokens,
            "max_num_seqs": runtime["max_num_seqs"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
            "dtype": runtime["dtype"],
            "kv_cache_dtype": runtime["kv_cache_dtype"],
            "effective_max_model_len": runtime["effective_max_model_len"],
        },
        "sample_phases": {
            "steady": profile["measured_repetitions"],
            "warmup": profile["warmup_repetitions"],
        },
        "scheduled_tokens_per_step": (
            1 if role == "decode" else profile["prefill_chunk_tokens"]
        ),
        "setup_prefill_chunks": setup_prefill_chunks,
        "initial_teacher_forced_token_id": (
            completion_token_ids[0] if role == "decode" else None
        ),
        "teacher_forced_token_ids": (
            completion_token_ids[1:required_decode_tokens] if role == "decode" else []
        ),
    }


def _aggregate_rows(
    samples: list[dict[str, Any]], noisy_cv_threshold: float
) -> list[dict[str, Any]]:
    point_ids = sorted({row["point_id"] for row in samples})
    aggregates = []
    for point_id in point_ids:
        point_samples = [
            row
            for row in samples
            if row["point_id"] == point_id
            and row["phase"] == "steady"
            and row["status"] == "passed"
        ]
        values = [row["runner_wall_time_ms"] for row in point_samples]
        if not values or any(value is None for value in values):
            continue
        mean = statistics.fmean(values)
        cv = statistics.pstdev(values) / mean if mean else 0.0
        p90 = statistics.quantiles(values, n=10, method="inclusive")[8]
        aggregates.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": point_samples[0]["run_id"],
                "point_id": point_id,
                "role": point_samples[0]["role"],
                "sample_count": len(values),
                "runner_wall_time_median_ms": statistics.median(values),
                "runner_wall_time_p90_ms": p90,
                "runner_wall_time_mean_ms": mean,
                "runner_wall_time_cv": cv,
                "noisy": cv > noisy_cv_threshold,
            }
        )
    return aggregates


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _result_markdown(
    *, status_label: str, hardware_validated: bool, aggregates: list[dict[str, Any]]
) -> str:
    lines = [
        "# DS4 profile spine result",
        "",
        f"Status: {status_label}",
        "",
        f"Hardware validated: {'yes' if hardware_validated else 'no'}",
        "",
        "| Point | Samples | Median runner wall time (ms) | Noisy |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in aggregates:
        lines.append(
            f"| `{row['point_id']}` | {row['sample_count']} | "
            f"{row['runner_wall_time_median_ms']:.3f} | "
            f"{'yes' if row['noisy'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Artifacts: `raw_samples.parquet`, `aggregates.parquet`, "
            "`provenance.json`, and `run-config.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_result_artifacts(
    *,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    provenance: dict[str, Any],
    status_label: str,
    output_dir: Path,
) -> None:
    aggregates = _aggregate_rows(samples, config["profile"]["noisy_cv_threshold"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging-"
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        pq.write_table(
            pa.Table.from_pylist(samples, schema=RAW_SAMPLE_SCHEMA),
            staging_dir / "raw_samples.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(aggregates, schema=AGGREGATE_SCHEMA),
            staging_dir / "aggregates.parquet",
        )
        _write_json(staging_dir / "run-config.json", config)
        _write_json(staging_dir / "provenance.json", provenance)
        (staging_dir / "result.md").write_text(
            _result_markdown(
                status_label=status_label,
                hardware_validated=provenance["hardware_validated"],
                aggregates=aggregates,
            )
        )
        _validate_result_dir(staging_dir)
        output_dir.mkdir()
        for path in staging_dir.iterdir():
            path.replace(output_dir / path.name)


def _validate_v1_result_dir(result_dir: Path, config: dict[str, Any]) -> None:
    required_files = {
        "aggregates.parquet",
        "provenance.json",
        "raw_samples.parquet",
        "result.md",
        "run-config.json",
    }
    missing = sorted(
        name for name in required_files if not (result_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"missing result artifacts: {', '.join(missing)}")

    raw = pq.read_table(result_dir / "raw_samples.parquet")
    aggregates = pq.read_table(result_dir / "aggregates.parquet")
    if raw.schema != RAW_SAMPLE_SCHEMA:
        raise ValueError("raw_samples.parquet does not match the versioned schema")
    if aggregates.schema != AGGREGATE_SCHEMA:
        raise ValueError("aggregates.parquet does not match the versioned schema")

    raw_rows = raw.to_pylist()
    aggregate_rows = aggregates.to_pylist()
    provenance = json.loads((result_dir / "provenance.json").read_text())
    run_id = config["run_id"]
    observed_run_ids = {row["run_id"] for row in raw_rows + aggregate_rows} | {
        provenance.get("run_id")
    }
    if observed_run_ids != {run_id}:
        raise ValueError(
            f"run_id mismatch: expected {run_id}, found {sorted(observed_run_ids)}"
        )

    sample_ids = [row["sample_id"] for row in raw_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    if any(
        row["sample_id"]
        != f"{row['run_id']}:{row['point_id']}:{row['phase']}:{row['ordinal']}"
        for row in raw_rows
    ):
        raise ValueError("sample_id values must match their sample coordinates")
    phases = {row["phase"] for row in raw_rows}
    if not phases.issubset({"startup", "capture", "setup", "warmup", "steady"}):
        raise ValueError(f"unknown sample phase: {sorted(phases)}")

    aggregate_point_ids = {row["point_id"] for row in aggregate_rows}
    passed_steady_point_ids = {
        row["point_id"]
        for row in raw_rows
        if row["phase"] == "steady" and row["status"] == "passed"
    }
    if aggregate_point_ids != passed_steady_point_ids:
        raise ValueError("aggregate point_id values do not match passed steady samples")
    for aggregate in aggregate_rows:
        steady_count = sum(
            row["point_id"] == aggregate["point_id"]
            and row["phase"] == "steady"
            and row["status"] == "passed"
            for row in raw_rows
        )
        if aggregate["sample_count"] != steady_count:
            raise ValueError(
                f"sample_count mismatch for point_id {aggregate['point_id']}"
            )


def _v2_required_files() -> set[str]:
    return {
        "aggregates.parquet",
        "comparisons.parquet",
        "prefix_evidence.parquet",
        "provenance.json",
        "raw_samples.parquet",
        "result.md",
        "run-config.json",
        "turn_samples.parquet",
    }


def _validate_v2_schema(path: Path, expected: pa.Schema) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if table.schema != expected:
        raise ValueError(f"{path.name} does not match the versioned schema")
    rows = table.to_pylist()
    if any(row.get("schema_version") != V2_SCHEMA_VERSION for row in rows):
        raise ValueError(f"{path.name} contains a non-v2 row")
    return rows


def _payload_for_manifest_point(point: dict[str, Any]) -> dict[str, Any]:
    payload = point.get("canonical_payload", point.get("payload", point))
    if not isinstance(payload, dict):
        raise ValueError("manifest point payload must be an object")
    return payload


def _manifest_points(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    points = config.get("points")
    if not isinstance(points, list):
        raise ValueError("v2 run-config.json requires points")
    resolved: dict[str, dict[str, Any]] = {}
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("manifest points must be objects")
        payload = _payload_for_manifest_point(point)
        point_id = point.get("point_id", payload.get("point_id"))
        comparison_id = point.get("comparison_id", payload.get("comparison_id"))
        if not isinstance(point_id, str) or point_id != make_point_id(payload):
            raise ValueError("manifest point_id does not match canonical payload")
        if not isinstance(comparison_id, str) or comparison_id != make_comparison_id(
            payload
        ):
            raise ValueError("manifest comparison_id does not match canonical payload")
        if point_id in resolved:
            raise ValueError("manifest point_id values must be unique")
        resolved[point_id] = {
            "payload": payload,
            "comparison_id": comparison_id,
        }
    return resolved


def _canonical_manifest_points(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate the frozen point records against immutable planner inputs."""
    planner_inputs = config.get("canonical_planner_inputs")
    if not isinstance(planner_inputs, dict):
        raise ValueError("v2 run-config has noncanonical planner inputs")
    points = _manifest_points(config)
    if canonical_payload_json(planner_inputs) == canonical_payload_json(
        canonical_v2_planner_inputs()
    ):
        canonical = canonical_v2_points()
        canonical_by_id = {
            point["point_id"]: {
                "payload": point["canonical_payload"],
                "comparison_id": point["comparison_id"],
            }
            for point in canonical
        }
        if points != canonical_by_id:
            raise ValueError("v2 points do not match canonical planner inputs")
        return canonical_by_id

    from benchmarks.ds4_profile.prefill_profile import load_prefill_points

    recomputed = load_prefill_points({**config, "points": None})
    recomputed_by_id = {
        point.point_id: {
            "payload": json.loads(canonical_payload_json(point.canonical_payload)),
            "comparison_id": point.comparison_id,
        }
        for point in recomputed
    }
    if points != recomputed_by_id:
        raise ValueError("v2 points do not match the pinned planner artifacts")

    expected_inputs = {
        "schema_version": V2_SCHEMA_VERSION,
        "kv_cache_groups": ["0"],
        "seed": 20260715,
        "block_size": 16,
        "chunk_budget": 4096,
    }
    if any(planner_inputs.get(key) != value for key, value in expected_inputs.items()):
        raise ValueError("v2 run-config has noncanonical planner inputs")
    selectors = planner_inputs.get("workload_selectors")
    planner_digest = planner_inputs.get("planner_digest")
    if (
        not isinstance(selectors, list)
        or len(selectors) != 34
        or len(set(selectors)) != 34
        or not isinstance(planner_digest, str)
        or len(planner_digest) != 64
    ):
        raise ValueError("v2 run-config has noncanonical planner inputs")
    by_selector: dict[str, list[dict[str, Any]]] = {}
    for point in points.values():
        payload = point["payload"]
        if (
            payload.get("planner_digest") != planner_digest
            or payload.get("seed") != 20260715
            or payload.get("block_size") != 16
            or payload.get("chunk_budget") != 4096
            or payload.get("homogeneous_prefix_tokens") != 4096
            or not 1 <= payload.get("batch_size", 0) <= 8
        ):
            raise ValueError("v2 points do not match canonical planner inputs")
        by_selector.setdefault(payload.get("selector"), []).append(point)
    if set(by_selector) != set(selectors) or any(
        len(pair) != 2
        or {item["payload"].get("cache_condition") for item in pair}
        != {"prefix_hit", "full_recompute"}
        or len({item["comparison_id"] for item in pair}) != 1
        for pair in by_selector.values()
    ):
        raise ValueError("v2 points do not form 34 canonical condition pairs")
    return points


def _manifest_id_set(value: Any, name: str) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of point IDs")
    identifiers = set(value)
    if len(identifiers) != len(value):
        raise ValueError(f"{name} must not contain duplicate point IDs")
    return identifiers


def _smoke_selectors(config: dict[str, Any]) -> set[str]:
    selectors = config.get("smoke_selectors")
    if selectors is None and isinstance(config.get("smoke"), dict):
        selectors = config["smoke"].get("selectors")
    if not isinstance(selectors, list) or not all(
        isinstance(selector, str) for selector in selectors
    ):
        raise ValueError("smoke runs require configured smoke_selectors")
    return set(selectors)


def _planned_chunks(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    chunks = payload.get("planned_chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("manifest point requires planned_chunks")
    expected = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or chunk.get("chunk_index") != index:
            raise ValueError("planned chunks must have contiguous chunk_index values")
        vector = chunk.get("scheduled_tokens_by_request")
        if isinstance(vector, dict):
            vector = list(vector.items())
        if not isinstance(vector, list):
            raise ValueError("planned chunk requires a request-token vector")
        normalized = []
        for item in vector:
            if isinstance(item, dict):
                request_key = item.get("request_key")
                scheduled_tokens = item.get("scheduled_tokens")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                request_key, scheduled_tokens = item
            else:
                raise ValueError("invalid planned request-token vector")
            if not isinstance(request_key, str) or not isinstance(
                scheduled_tokens, int
            ):
                raise ValueError("invalid planned request-token value")
            normalized.append(
                {
                    "request_key": request_key,
                    "scheduled_tokens": scheduled_tokens,
                }
            )
        expected.append(normalized)
    return expected


def _raw_sample_id(row: dict[str, Any]) -> str:
    return (
        f"{row['run_id']}:{row['point_id']}:{row['phase']}:"
        f"{row['ordinal']}:{row['chunk_index']}"
    )


def _turn_sample_id(row: dict[str, Any]) -> str:
    return f"{row['run_id']}:{row['point_id']}:{row['phase']}:{row['ordinal']}"


def _require_v2_enums(rows: list[dict[str, Any]], path_name: str) -> None:
    for row in rows:
        for field, allowed in V2_ENUMS.items():
            if field not in row or row[field] is None:
                continue
            if row[field] not in allowed:
                raise ValueError(f"unknown {field} in {path_name}: {row[field]}")


def _same_float(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def _validate_v2_evidence(
    rows: list[dict[str, Any]],
    expected_points: dict[str, dict[str, Any]],
    outcomes: dict[str, str],
    expected_kv_cache_groups: frozenset[str],
    raw_rows: list[dict[str, Any]],
) -> None:
    observed: set[tuple[str, str, int, str, str]] = set()
    for row in rows:
        point_id = row["point_id"]
        if point_id not in expected_points:
            raise ValueError("prefix evidence references an unknown point")
        if row["phase"] not in V2_ENUMS["phase"]:
            raise ValueError("prefix evidence has an unknown phase")
        key = (
            point_id,
            row["phase"],
            row["ordinal"],
            row["request_key"],
            row["kv_cache_group"],
        )
        if key in observed:
            raise ValueError("duplicate prefix evidence for a request repetition")
        observed.add(key)
        if row["hardware_validated"]:
            if not (
                row["prime_completed"]
                and row["prime_synchronized"]
                and row["live_cuda_tensor_proven"]
            ):
                raise ValueError("hardware-validated prefix evidence lacks live proof")
            shapes = row["live_kv_tensor_shapes"]
            devices = row["live_kv_tensor_devices"]
            if not shapes or len(shapes) != len(devices) or set(devices) != {"cuda:0"}:
                raise ValueError("hardware-validated prefix evidence is not on cuda:0")
            dimension = row["block_dimension"]
            axis = row["block_axis"]
            if dimension <= 0 or any(
                axis < 0 or axis >= len(shape) for shape in shapes
            ):
                raise ValueError("prefix evidence has an invalid block axis")
            if any(shape[axis] != dimension for shape in shapes):
                raise ValueError("prefix evidence has inconsistent block dimensions")
            if any(
                block_id < 0 or block_id >= dimension
                for block_id in row["verified_physical_block_ids"]
            ):
                raise ValueError("prefix evidence has an invalid physical block ID")

    expected: set[tuple[str, str, int, str, str]] = set()
    terminal_repetition = {
        row["point_id"]: (row["phase"], row["ordinal"])
        for row in raw_rows
        if row["status"] == "out_of_capacity"
    }
    for point_id, point in expected_points.items():
        payload = point["payload"]
        outcome = outcomes.get(point_id)
        if payload.get("cache_condition") != "prefix_hit" or outcome not in {
            "passed",
            "out_of_capacity",
        }:
            continue
        request_keys = [
            request["request_key"]
            for request in payload.get("requests", [])
            if request["cached_tokens"] > 0
        ]
        for phase in ("warmup", "steady"):
            for ordinal in range(3 if phase == "warmup" else 10):
                keys = {
                    (point_id, phase, ordinal, request_key, kv_cache_group)
                    for request_key in request_keys
                    for kv_cache_group in expected_kv_cache_groups
                }
                if outcome == "passed":
                    expected.update(keys)
                else:
                    terminal_phase, terminal_ordinal = terminal_repetition[point_id]
                    coordinate = (0 if phase == "warmup" else 1, ordinal)
                    terminal_coordinate = (
                        0 if terminal_phase == "warmup" else 1,
                        terminal_ordinal,
                    )
                    if coordinate <= terminal_coordinate:
                        expected.update(keys)
    if observed != expected:
        raise ValueError("prefix evidence does not match completed hit repetitions")


_TURN_TOTAL_FIELDS = (
    "scheduled_tokens",
    "context_tokens",
    "cached_tokens",
    "new_tokens",
    "recomputed_tokens",
    "requested_kv_blocks",
    "allocated_kv_blocks",
    "requested_kv_bytes",
    "allocated_kv_bytes",
    "lookup_time_ms",
    "allocation_time_ms",
    "scheduler_time_ms",
    "cache_reset_time_ms",
    "prefix_prime_time_ms",
    "runner_wall_time_ms",
    "cuda_model_time_ms",
)


def _validate_v2_turn_totals(
    turns: list[dict[str, Any]], raw: list[dict[str, Any]]
) -> None:
    """Reconcile each full-turn record with its exact successful chunks."""
    raw_by_turn: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_turn.setdefault((row["phase"], row["ordinal"]), []).append(row)
    for turn in turns:
        chunks = raw_by_turn.get((turn["phase"], turn["ordinal"]), [])
        if not chunks or turn["chunk_count"] != len(chunks):
            raise ValueError("turn sample does not match raw chunk totals")
        for field in _TURN_TOTAL_FIELDS:
            values = [row[field] for row in chunks]
            if not _same_float(turn[field], sum(value or 0.0 for value in values)):
                raise ValueError("turn sample does not match raw chunk totals")
        expected_throughput = (
            turn["new_tokens"] * 1000 / turn["runner_wall_time_ms"]
            if turn["runner_wall_time_ms"]
            else 0.0
        )
        if not _same_float(turn["throughput_tokens_per_s"], expected_throughput):
            raise ValueError("turn sample does not match raw chunk totals")


def _validate_v2_statistics(
    turn_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    expected_points: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    aggregate_by_point = {row["point_id"]: row for row in aggregate_rows}
    if len(aggregate_by_point) != len(aggregate_rows):
        raise ValueError("aggregate point_id values must be unique")
    threshold = config.get("profile", {}).get("noisy_cv_threshold", 0.05)
    for point_id in expected_points:
        steady = [
            row
            for row in turn_rows
            if row["point_id"] == point_id and row["phase"] == "steady"
        ]
        aggregate = aggregate_by_point.get(point_id)
        if not steady:
            if aggregate is not None:
                raise ValueError("out-of-capacity point must not have an aggregate")
            continue
        if aggregate is None:
            raise ValueError("passed point is missing an aggregate")
        values = [row["runner_wall_time_ms"] for row in steady]
        throughput = [row["throughput_tokens_per_s"] for row in steady]
        mean = statistics.fmean(values)
        throughput_mean = statistics.fmean(throughput)
        expected = {
            "sample_count": len(values),
            "runner_wall_time_median_ms": statistics.median(values),
            "runner_wall_time_p90_ms": statistics.quantiles(
                values, n=10, method="inclusive"
            )[8],
            "runner_wall_time_mean_ms": mean,
            "runner_wall_time_cv": statistics.pstdev(values) / mean if mean else 0.0,
            "throughput_median_tokens_per_s": statistics.median(throughput),
            "throughput_p90_tokens_per_s": statistics.quantiles(
                throughput, n=10, method="inclusive"
            )[8],
            "throughput_mean_tokens_per_s": throughput_mean,
            "throughput_cv": (
                statistics.pstdev(throughput) / throughput_mean
                if throughput_mean
                else 0.0
            ),
        }
        expected_noisy = (
            expected["runner_wall_time_cv"] > threshold
            or expected["throughput_cv"] > threshold
        )
        if (
            aggregate["sample_count"] != expected["sample_count"]
            or any(
                not _same_float(aggregate[field], value)
                for field, value in expected.items()
                if field != "sample_count"
            )
            or aggregate["noisy"] != expected_noisy
        ):
            raise ValueError("aggregate statistics do not match turn samples")
    if set(aggregate_by_point) - set(expected_points):
        raise ValueError("aggregate references an unknown point")


def _validate_v2_comparisons(
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    outcomes: dict[str, str],
    expected_points: dict[str, dict[str, Any]],
) -> None:
    aggregate_by_point = {row["point_id"]: row for row in aggregate_rows}
    pairs: dict[str, dict[str, str]] = {}
    for point_id, point in expected_points.items():
        condition = point["payload"].get("cache_condition")
        pair = pairs.setdefault(point["comparison_id"], {})
        if condition in pair:
            raise ValueError(
                "manifest comparison pairs must contain one hit and one recompute"
            )
        pair[condition] = point_id
    if any(set(pair) != {"prefix_hit", "full_recompute"} for pair in pairs.values()):
        raise ValueError(
            "manifest comparison pairs must contain one hit and one recompute"
        )
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        comparison_id = row["comparison_id"]
        if comparison_id in observed or comparison_id not in pairs:
            raise ValueError("duplicate or unknown comparison")
        observed[comparison_id] = row
    for comparison_id, pair in pairs.items():
        hit = pair["prefix_hit"]
        recompute = pair["full_recompute"]
        row = observed.get(comparison_id)
        if outcomes[hit] == outcomes[recompute] == "passed":
            if row is None:
                raise ValueError("passed comparison pair is missing a comparison")
            if (
                row["prefix_hit_point_id"] != hit
                or row["full_recompute_point_id"] != recompute
            ):
                raise ValueError("comparison point IDs do not match the manifest")
            hit_median = aggregate_by_point[hit]["runner_wall_time_median_ms"]
            recompute_median = aggregate_by_point[recompute][
                "runner_wall_time_median_ms"
            ]
            if not (
                _same_float(row["prefix_hit_median_ms"], hit_median)
                and _same_float(row["full_recompute_median_ms"], recompute_median)
                and _same_float(
                    row["recompute_penalty_ms"], recompute_median - hit_median
                )
                and _same_float(
                    row["recompute_penalty_ratio"], recompute_median / hit_median
                )
            ):
                raise ValueError("comparison statistics do not match aggregates")
        elif row is not None:
            raise ValueError(
                "out-of-capacity comparison pair must not have a comparison"
            )


def _validate_v2_result_dir(result_dir: Path, config: dict[str, Any]) -> None:
    missing = sorted(
        name for name in _v2_required_files() if not (result_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"missing result artifacts: {', '.join(missing)}")
    raw_rows = _validate_v2_schema(
        result_dir / "raw_samples.parquet", V2_RAW_SAMPLE_SCHEMA
    )
    turn_rows = _validate_v2_schema(
        result_dir / "turn_samples.parquet", V2_TURN_SAMPLE_SCHEMA
    )
    aggregate_rows = _validate_v2_schema(
        result_dir / "aggregates.parquet", V2_AGGREGATE_SCHEMA
    )
    comparison_rows = _validate_v2_schema(
        result_dir / "comparisons.parquet", V2_COMPARISON_SCHEMA
    )
    evidence_rows = _validate_v2_schema(
        result_dir / "prefix_evidence.parquet", V2_PREFIX_EVIDENCE_SCHEMA
    )
    for rows, name in (
        (raw_rows, "raw_samples.parquet"),
        (turn_rows, "turn_samples.parquet"),
        (aggregate_rows, "aggregates.parquet"),
        (comparison_rows, "comparisons.parquet"),
        (evidence_rows, "prefix_evidence.parquet"),
    ):
        _require_v2_enums(rows, name)
    required_setup_timing_fields = (
        "lookup_time_ms",
        "allocation_time_ms",
        "scheduler_time_ms",
    )
    if any(
        not math.isfinite(row[field]) or row[field] < 0
        for row in raw_rows
        for field in required_setup_timing_fields
    ) or any(
        value is not None and (not math.isfinite(value) or value < 0)
        for row in raw_rows
        for value in (
            row["cache_reset_time_ms"],
            row["prefix_prime_time_ms"],
        )
    ):
        raise ValueError("raw sample has invalid setup timing")
    if any(
        ((row["chunk_index"] == 0) != (row["cache_reset_time_ms"] is not None))
        or (
            (row["chunk_index"] == 0 and row["cache_condition"] == "prefix_hit")
            != (row["prefix_prime_time_ms"] is not None)
        )
        for row in raw_rows
    ):
        raise ValueError("raw sample has inconsistent setup timing")
    if config.get("bootstrap_failure") is True:
        provenance = json.loads((result_dir / "provenance.json").read_text())
        if any((raw_rows, turn_rows, aggregate_rows, comparison_rows, evidence_rows)):
            raise ValueError("bootstrap failure artifacts must have empty data tables")
        if (
            config.get("run_kind") not in V2_ENUMS["run_kind"]
            or config.get("points") != []
            or config.get("canonical_full_manifest") != []
            or config.get("expected_manifest") != []
            or provenance.get("run_id") != config.get("run_id")
            or provenance.get("model") != config.get("model")
            or provenance.get("runtime") != config.get("runtime")
            or provenance.get("validation_state") != "remote_failed"
            or provenance.get("hardware_validated") is not False
            or not isinstance(provenance.get("validation_error"), str)
        ):
            raise ValueError("invalid bootstrap failure artifact state")
        return
    expected_points = _canonical_manifest_points(config)
    expected_kv_cache_groups = frozenset(
        config["canonical_planner_inputs"]["kv_cache_groups"]
    )
    canonical_full = _manifest_id_set(
        config.get("canonical_full_manifest"), "canonical_full_manifest"
    )
    if len(canonical_full) != 68 or canonical_full != set(expected_points):
        raise ValueError(
            "canonical_full_manifest must be the canonical 68-point manifest"
        )
    expected_manifest = _manifest_id_set(
        config.get("expected_manifest"), "expected_manifest"
    )
    run_kind = config.get("run_kind")
    if run_kind not in V2_ENUMS["run_kind"]:
        raise ValueError("unknown run_kind")
    if run_kind == "full" and expected_manifest != canonical_full:
        raise ValueError("full expected_manifest must equal canonical_full_manifest")
    if run_kind == "smoke":
        selectors = _smoke_selectors(config)
        configured = {
            point_id
            for point_id, point in expected_points.items()
            if point["payload"].get("selector") in selectors
        }
        if expected_manifest != configured:
            raise ValueError("smoke expected_manifest does not match smoke selectors")
    expected_points = {
        point_id: expected_points[point_id] for point_id in expected_manifest
    }
    point_views = {point.point_id: point for point in _point_plans_from_config(config)}
    provenance = json.loads((result_dir / "provenance.json").read_text())
    if provenance.get("model") != config.get("model") or provenance.get(
        "runtime"
    ) != config.get("runtime"):
        raise ValueError("provenance model/runtime differs from run config")
    validation_state = provenance.get("validation_state")
    if validation_state is not None and validation_state not in V2_VALIDATION_STATES:
        raise ValueError("unknown validation_state")
    run_id = config.get("run_id")
    all_rows = raw_rows + turn_rows + aggregate_rows + comparison_rows + evidence_rows
    observed_run_ids = {row["run_id"] for row in all_rows} | {provenance.get("run_id")}
    if not isinstance(run_id, str) or observed_run_ids != {run_id}:
        raise ValueError(
            f"run_id mismatch: expected {run_id}, found {sorted(observed_run_ids)}"
        )
    outcomes: dict[str, str] = {}
    raw_by_point: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        point_id = row["point_id"]
        if point_id not in expected_points:
            raise ValueError("raw sample references an unknown point")
        point = expected_points[point_id]
        payload = point["payload"]
        expected_values = {
            "role": "prefill",
            "workload_family": payload.get("workload_family"),
            "selector": payload.get("selector"),
            "composition": payload.get("composition"),
            "cache_condition": payload.get("cache_condition"),
            "planner_digest": payload.get("planner_digest"),
        }
        if row["comparison_id"] != point["comparison_id"] or any(
            row[field] != value for field, value in expected_values.items()
        ):
            raise ValueError("raw sample does not match its manifest point")
        if row["sample_id"] != _raw_sample_id(row):
            raise ValueError("raw sample_id does not match its coordinates")
        raw_by_point.setdefault(point_id, []).append(row)
    if len({row["sample_id"] for row in raw_rows}) != len(raw_rows):
        raise ValueError("raw sample_id values must be unique")
    turn_by_point: dict[str, list[dict[str, Any]]] = {}
    for row in turn_rows:
        point_id = row["point_id"]
        if point_id not in expected_points or row["sample_id"] != _turn_sample_id(row):
            raise ValueError("turn sample has an unknown point or invalid sample_id")
        point = expected_points[point_id]
        if row["comparison_id"] != point["comparison_id"]:
            raise ValueError("turn sample comparison_id does not match its point")
        payload = point["payload"]
        if any(
            row[field] != value
            for field, value in {
                "role": "prefill",
                "workload_family": payload.get("workload_family"),
                "selector": payload.get("selector"),
                "composition": payload.get("composition"),
                "cache_condition": payload.get("cache_condition"),
                "planner_digest": payload.get("planner_digest"),
            }.items()
        ):
            raise ValueError("turn sample does not match its manifest point")
        turn_by_point.setdefault(point_id, []).append(row)
    if len({row["sample_id"] for row in turn_rows}) != len(turn_rows):
        raise ValueError("turn sample_id values must be unique")
    for point_id, point in expected_points.items():
        chunks = _planned_chunks(point["payload"])
        raw = sorted(
            raw_by_point.get(point_id, []),
            key=lambda row: (
                0 if row["phase"] == "warmup" else 1,
                row["ordinal"],
                row["chunk_index"],
            ),
        )
        expected_coordinates = [
            (phase, ordinal, chunk_index)
            for phase, count in (("warmup", 3), ("steady", 10))
            for ordinal in range(count)
            for chunk_index in range(len(chunks))
        ]
        actual_coordinates = [
            (row["phase"], row["ordinal"], row["chunk_index"]) for row in raw
        ]
        terminal_rows = [row for row in raw if row["row_kind"] == "terminal"]
        if terminal_rows:
            if len(terminal_rows) != 1:
                raise ValueError(
                    "out-of-capacity point requires exactly one terminal row"
                )
            terminal = terminal_rows[0]
            terminal_coordinate = (
                terminal["phase"],
                terminal["ordinal"],
                terminal["chunk_index"],
            )
            terminal_position = (
                expected_coordinates.index(terminal_coordinate)
                if terminal_coordinate in expected_coordinates
                else -1
            )
            if (
                terminal_position < 0
                or actual_coordinates != expected_coordinates[: terminal_position + 1]
                or terminal["status"] != "out_of_capacity"
                or terminal["allocation_state"] != "out_of_capacity"
            ):
                raise ValueError(
                    "terminal out-of-capacity row is not a coordinate prefix"
                )
            if any(
                row["row_kind"] != "chunk" or row["status"] != "passed"
                for row in raw[:-1]
            ):
                raise ValueError("out-of-capacity prefix contains a non-passed chunk")
            if any(
                row["planned_scheduled_tokens_by_request"] != chunks[row["chunk_index"]]
                for row in raw
            ):
                raise ValueError("raw chunk does not match the planned vector")
            if not _terminal_ooc_proven(terminal, point_views[point_id]):
                expected_vector = chunks[terminal["chunk_index"]]
                actual_items = terminal["actual_scheduled_tokens_by_request"]
                actual = {
                    item["request_key"]: item["scheduled_tokens"]
                    for item in actual_items
                }
                planned = {
                    item["request_key"]: item["scheduled_tokens"]
                    for item in expected_vector
                }
                if (
                    len(actual) != len(actual_items)
                    or any(
                        key not in planned or tokens <= 0 or tokens > planned[key]
                        for key, tokens in actual.items()
                    )
                    or terminal["scheduled_tokens"] != sum(actual.values())
                ):
                    raise ValueError("terminal row does not match the planned vector")
            if not (
                terminal["allocator_pressure_proven"]
                and terminal["requested_kv_blocks"] > terminal["allocatable_kv_blocks"]
                and terminal["allocated_kv_blocks"] <= terminal["allocatable_kv_blocks"]
            ):
                raise ValueError("terminal row does not prove allocator pressure")
            if not (
                terminal["clean_reset_proven"]
                and terminal["cache_reset_completed"]
                and terminal["cache_reset_empty"]
            ):
                raise ValueError("terminal row does not prove a clean reset")
            if (
                terminal["runner_wall_time_ms"] is not None
                or terminal["cuda_model_time_ms"] is not None
                or terminal["runtime_mode"] is not None
            ):
                raise ValueError("terminal row must not contain GPU timing")
            expected_request_ids = {
                item["request_key"] for item in chunks[terminal["chunk_index"]]
            }
            if (
                not set(terminal["preempted_request_ids"]) <= expected_request_ids
                or terminal["unrelated_request_ids"]
            ):
                raise ValueError("terminal row is not an isolated planned batch")
            block_bytes = terminal["kv_block_bytes"]
            if (
                block_bytes <= 0
                or terminal["requested_kv_bytes"]
                != terminal["requested_kv_blocks"] * block_bytes
                or terminal["allocated_kv_bytes"]
                != terminal["allocated_kv_blocks"] * block_bytes
            ):
                raise ValueError("terminal row has inconsistent KV byte accounting")
            if not _terminal_ooc_proven(terminal, point_views[point_id]):
                raise ValueError("terminal row does not match the planned vector")
            if turn_by_point.get(point_id) or any(
                row["point_id"] == point_id for row in aggregate_rows
            ):
                raise ValueError("out-of-capacity point has turn or aggregate rows")
            outcomes[point_id] = "out_of_capacity"
            continue
        if actual_coordinates != expected_coordinates or any(
            row["row_kind"] != "chunk"
            or row["status"] != "passed"
            or row["allocation_state"] != "allocated"
            for row in raw
        ):
            raise ValueError(
                "passed point does not have exact planned chunk coordinates"
            )
        for row in raw:
            expected_vector = chunks[row["chunk_index"]]
            if (
                row["chunk_count"] != len(chunks)
                or row["planned_scheduled_tokens_by_request"] != expected_vector
                or row["actual_scheduled_tokens_by_request"] != expected_vector
                or row["runtime_mode"] not in V2_ENUMS["runtime_mode"]
            ):
                raise ValueError("raw chunk does not match the planned vector")
        turns = turn_by_point.get(point_id, [])
        if {(row["phase"], row["ordinal"]) for row in turns} != {
            (phase, ordinal)
            for phase, count in (("warmup", 3), ("steady", 10))
            for ordinal in range(count)
        } or len(turns) != 13:
            raise ValueError("passed point does not have exact turn coordinates")
        if any(
            row["status"] != "passed"
            or row["allocation_state"] != "allocated"
            or row["runtime_mode"] not in V2_ENUMS["runtime_mode"]
            for row in turns
        ):
            raise ValueError("passed point has an invalid turn sample")
        _validate_v2_turn_totals(turns, raw)
        outcomes[point_id] = "passed"
    for row in aggregate_rows:
        point_id = row["point_id"]
        if point_id not in expected_points:
            raise ValueError("aggregate references an unknown point")
        point = expected_points[point_id]
        if row["comparison_id"] != point["comparison_id"] or row["role"] != "prefill":
            raise ValueError("aggregate does not match its manifest point")
    if any(
        row["point_id"] not in expected_points
        or outcomes.get(row["point_id"]) not in {"passed", "out_of_capacity"}
        for row in evidence_rows
    ):
        raise ValueError("prefix evidence must describe a completed point")
    _validate_v2_evidence(
        evidence_rows,
        expected_points,
        outcomes,
        expected_kv_cache_groups,
        raw_rows,
    )
    _validate_v2_statistics(turn_rows, aggregate_rows, expected_points, config)
    _validate_v2_comparisons(comparison_rows, aggregate_rows, outcomes, expected_points)


def _validate_result_dir(result_dir: Path) -> None:
    config_path = result_dir / "run-config.json"
    if not config_path.is_file():
        raise ValueError("missing result artifacts: run-config.json")
    config = json.loads(config_path.read_text())
    version = config.get("schema_version")
    if version == V1_SCHEMA_VERSION:
        _validate_v1_result_dir(result_dir, config)
    elif version == V2_SCHEMA_VERSION:
        _validate_v2_result_dir(result_dir, config)
    else:
        raise ValueError(f"unsupported schema_version: {version}")


def _write_fixture_result(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version {SCHEMA_VERSION}")
    replay = _load_replay(config)
    config = _resolve_config(config, replay)
    samples = _fixture_samples(config, replay)
    _write_result_artifacts(
        config=config,
        samples=samples,
        provenance={
            "artifact_schema_version": SCHEMA_VERSION,
            "hardware_validated": False,
            "run_id": config["run_id"],
            "source": config["source"],
            "status": "fixture-only",
        },
        status_label="Fixture only",
        output_dir=output_dir,
    )
    print(output_dir)


def _worker_samples_valid(
    worker: dict[str, Any], config: dict[str, Any], replay: dict[str, Any]
) -> bool:
    role = worker.get("role")
    if role not in {"prefill", "decode"}:
        return False
    plan = _gpu_worker_plan(config, replay, role)
    samples = worker.get("samples", [])
    expected_point_id = (
        "decode-b1-t1"
        if role == "decode"
        else f"prefill-b1-t{config['profile']['prefill_chunk_tokens']}"
    )
    expected_counts = {
        "startup": 1,
        "capture": 1,
        "warmup": config["profile"]["warmup_repetitions"],
        "steady": config["profile"]["measured_repetitions"],
    }
    if any(
        sum(sample.get("phase") == phase for sample in samples) != expected_count
        for phase, expected_count in expected_counts.items()
    ):
        return False
    observations = worker.get("cudagraph_observations", [])
    if not observations or any(
        not isinstance(observation, dict)
        or observation.get("runtime_mode") not in {"FULL", "PIECEWISE"}
        for observation in observations
    ):
        return False
    measured_samples = [
        sample for sample in samples if sample["phase"] in {"warmup", "steady"}
    ]
    if any(
        sample.get("cudagraph_runtime_mode") not in {"FULL", "PIECEWISE"}
        for sample in measured_samples
    ):
        return False
    if any(
        sample.get("schema_version") != SCHEMA_VERSION
        or sample.get("run_id") != config["run_id"]
        or sample.get("role") != role
        or sample.get("point_id") != expected_point_id
        or sample.get("runner_boundary") != plan["runner_boundary"]
        or sample.get("status") != "passed"
        or sample.get("compile_enabled") is not True
        or sample.get("cudagraph_enabled") is not True
        for sample in samples
    ):
        return False
    if role == "decode":
        decode_samples = [
            sample for sample in samples if sample["phase"] in {"warmup", "steady"}
        ]
        if [sample["injected_token_id"] for sample in decode_samples] != plan[
            "teacher_forced_token_ids"
        ]:
            return False
        if any(
            sample["sampled_token_id"] is None
            or sample.get("sampled_token_discarded") is not True
            for sample in decode_samples
        ):
            return False
    return True


def _assemble_gpu_result(
    *,
    config_path: Path,
    preflight_path: Path,
    worker_paths: list[Path],
    output_dir: Path,
) -> bool:
    config = json.loads(config_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    workers = [json.loads(path.read_text()) for path in worker_paths]
    replay = _load_replay(config)
    config = _resolve_config(config, replay)
    roles = {worker.get("role") for worker in workers}
    workers_valid = (
        roles == {"prefill", "decode"}
        and all(worker.get("schema_version") == SCHEMA_VERSION for worker in workers)
        and all(worker.get("run_id") == config["run_id"] for worker in workers)
        and all(worker.get("status") == "passed" for worker in workers)
        and all(worker.get("hardware_validated") is True for worker in workers)
        and all(worker.get("compile_enabled") is True for worker in workers)
        and all(worker.get("cudagraph_enabled") is True for worker in workers)
        and all(_worker_samples_valid(worker, config, replay) for worker in workers)
    )
    hardware_validated = preflight.get("status") == "ready" and workers_valid
    samples = [sample for worker in workers for sample in worker.get("samples", [])]
    worker_summaries = [
        {key: value for key, value in worker.items() if key != "samples"}
        for worker in workers
    ]
    provenance = {
        "artifact_schema_version": SCHEMA_VERSION,
        "hardware_validated": hardware_validated,
        "invocation": [sys.executable, *sys.argv],
        "model": config["model"],
        "preflight": preflight,
        "roles": config["roles"],
        "run_id": config["run_id"],
        "run_parameters": {
            "profile": config["profile"],
            "runtime": config["runtime"],
        },
        "source": config["source"],
        "status": "passed" if hardware_validated else "invalid",
        "workers": worker_summaries,
    }
    _write_result_artifacts(
        config=config,
        samples=samples,
        provenance=provenance,
        status_label="Passed" if hardware_validated else "Invalid",
        output_dir=output_dir,
    )
    print(output_dir)
    return hardware_validated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DS4 profile spine.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--config", type=Path, required=True)
    fixture.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--result-dir", type=Path, required=True)
    gpu_worker = subparsers.add_parser("gpu-worker")
    gpu_worker.add_argument("--config", type=Path, required=True)
    gpu_worker.add_argument("--role", choices=("prefill", "decode"), required=True)
    gpu_worker.add_argument("--output", type=Path, required=True)
    gpu_worker.add_argument("--inspect-plan", action="store_true")
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--config", type=Path, required=True)
    assemble.add_argument("--preflight", type=Path, required=True)
    assemble.add_argument("--worker-result", type=Path, action="append", required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fixture":
            _write_fixture_result(args.config, args.output_dir)
        elif args.command == "validate":
            _validate_result_dir(args.result_dir)
            print(args.result_dir)
        elif args.command == "gpu-worker":
            config = json.loads(args.config.read_text())
            replay = _load_replay(config)
            config = _resolve_config(config, replay)
            plan = _gpu_worker_plan(config, replay, args.role)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.inspect_plan:
                _write_json(args.output, plan)
                print(args.output)
                return
            try:
                from benchmarks.ds4_profile.gpu_profile import run_gpu_worker

                worker_result = run_gpu_worker(config, replay, plan)
                returncode = 0 if worker_result["status"] == "passed" else 2
            except Exception as error:
                worker_result = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": config["run_id"],
                    "hardware_validated": False,
                    "role": args.role,
                    "runner_boundary": plan["runner_boundary"],
                    "samples": [],
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                returncode = 2
            _write_json(args.output, worker_result)
            print(args.output)
            if returncode:
                raise SystemExit(returncode)
        elif args.command == "assemble":
            passed = _assemble_gpu_result(
                config_path=args.config,
                preflight_path=args.preflight,
                worker_paths=args.worker_result,
                output_dir=args.output_dir,
            )
            if not passed:
                raise SystemExit(2)
    except (KeyError, OSError, ValueError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
