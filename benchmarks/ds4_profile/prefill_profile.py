# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic P-side prefill workload planning."""

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from benchmarks.ds4_profile.profile_spine import (
    V2_SCHEMA_VERSION,
    canonical_payload_json,
    make_comparison_id,
    make_point_id,
)

CacheCondition = Literal["prefix_hit", "full_recompute"]
WorkloadFamily = Literal["homogeneous", "mixed", "exact_replay"]

CHUNK_ALGORITHM = "equal-active-cap-v1"
HOMOGENEOUS_TOKEN_ALGORITHM = "sha256-legal-pool-v1"
CANONICAL_BLOCK_SIZE = 16
CANONICAL_HOMOGENEOUS_PREFIX_TOKENS = 4096


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


def _request_chunk_accounting(
    point: PPointPlan, chunk: PChunkPlan
) -> tuple[int, int, int, int]:
    context_tokens = 0
    cached_tokens = 0
    new_tokens = 0
    recomputed_tokens = 0
    requests = {request.request_key: request for request in point.requests}
    for request_key, scheduled_tokens in chunk.scheduled_tokens_by_request.items():
        request = requests[request_key]
        prior_tokens = sum(
            prior.scheduled_tokens_by_request.get(request_key, 0)
            for prior in point.chunks[: chunk.chunk_index]
        )
        if point.cache_condition == "prefix_hit":
            cached = request.cached_tokens if prior_tokens == 0 else 0
            context_tokens += cached + scheduled_tokens
            cached_tokens += cached
            new_tokens += scheduled_tokens
            continue
        prefix_tokens = request.context_tokens - request.new_tokens
        recomputed = max(
            0,
            min(prior_tokens + scheduled_tokens, prefix_tokens) - prior_tokens,
        )
        context_tokens += scheduled_tokens
        recomputed_tokens += recomputed
        new_tokens += scheduled_tokens - recomputed
    return context_tokens, cached_tokens, new_tokens, recomputed_tokens


def _request_token_vector(tokens_by_request: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"request_key": request_key, "scheduled_tokens": scheduled_tokens}
        for request_key, scheduled_tokens in tokens_by_request.items()
    ]


def make_prefill_chunk_row(
    *,
    run_id: str,
    point: PPointPlan,
    phase: Literal["warmup", "steady"],
    ordinal: int,
    chunk: PChunkPlan,
    runner_wall_time_ms: float | None,
    cuda_model_time_ms: float | None,
    allocation: dict[str, Any],
    status: str,
    error: str | None,
) -> dict[str, Any]:
    """Build one schema-v2 row from a planned Scheduler chunk."""
    if (
        chunk.chunk_index >= len(point.chunks)
        or point.chunks[chunk.chunk_index] != chunk
    ):
        raise ValueError("chunk does not belong to the point plan")
    actual_tokens = allocation["actual_scheduled_tokens_by_request"]
    context, cached, new, recomputed = _request_chunk_accounting(point, chunk)
    kv_block_bytes = allocation["kv_block_bytes"]
    requested_blocks = allocation["requested_blocks"]
    allocated_blocks = allocation["allocated_blocks"]
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "run_id": run_id,
        "point_id": point.point_id,
        "comparison_id": point.comparison_id,
        "sample_id": (
            f"{run_id}:{point.point_id}:{phase}:{ordinal}:{chunk.chunk_index}"
        ),
        "role": "prefill",
        "workload_family": point.workload_family,
        "selector": point.selector,
        "composition": point.composition,
        "cache_condition": point.cache_condition,
        "planner_digest": point.planner_digest,
        "phase": phase,
        "ordinal": ordinal,
        "chunk_index": chunk.chunk_index,
        "chunk_count": len(point.chunks),
        "row_kind": "chunk" if status == "passed" else "terminal",
        "status": status,
        "allocation_state": allocation["state"],
        "planned_scheduled_tokens_by_request": _request_token_vector(
            chunk.scheduled_tokens_by_request
        ),
        "actual_scheduled_tokens_by_request": _request_token_vector(actual_tokens),
        "preempted_request_ids": list(allocation["preempted_request_ids"]),
        "unrelated_request_ids": list(allocation["unrelated_request_ids"]),
        "cache_epoch": allocation["cache_epoch"],
        "cache_reset_completed": allocation["cache_reset_completed"],
        "cache_reset_empty": allocation["cache_reset_empty"],
        "requested_kv_blocks": requested_blocks,
        "allocatable_kv_blocks": allocation["allocatable_blocks"],
        "allocated_kv_blocks": allocated_blocks,
        "kv_block_bytes": kv_block_bytes,
        "requested_kv_bytes": requested_blocks * kv_block_bytes,
        "allocated_kv_bytes": allocated_blocks * kv_block_bytes,
        "scheduled_tokens": sum(actual_tokens.values()),
        "context_tokens": context,
        "cached_tokens": cached,
        "new_tokens": new,
        "recomputed_tokens": recomputed,
        "lookup_time_ms": allocation["lookup_time_ms"],
        "allocation_time_ms": allocation["allocation_time_ms"],
        "runner_wall_time_ms": runner_wall_time_ms,
        "cuda_model_time_ms": cuda_model_time_ms,
        "runtime_mode": allocation["runtime_mode"],
        "error": error,
    }


def make_planner_digest(
    workload_plan: dict[str, Any],
    rendered_turns: list[dict[str, Any]],
    *,
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
    seed: int,
) -> str:
    """Return the digest binding all inputs to a planner expansion."""
    payload = {
        "planner_schema": "ds4-p-prefill-plan-v1",
        "workload_plan": workload_plan,
        "rendered_turns": rendered_turns,
        "block_size": block_size,
        "token_budget": token_budget,
        "homogeneous_prefix_tokens": homogeneous_prefix_tokens,
        "seed": seed,
        "chunk_algorithm": CHUNK_ALGORITHM,
        "homogeneous_token_algorithm": HOMOGENEOUS_TOKEN_ALGORITHM,
    }
    return hashlib.sha256(canonical_payload_json(payload).encode()).hexdigest()


def _token_digest(token_ids: tuple[int, ...]) -> str:
    payload = canonical_payload_json({"tokens": token_ids})
    return hashlib.sha256(payload.encode()).hexdigest()


def _request_payload(request: PRequestPlan) -> dict[str, Any]:
    return {
        "request_key": request.request_key,
        "trajectory_id": request.trajectory_id,
        "turn_index": request.turn_index,
        "reasoning_mode": request.reasoning_mode,
        "context_tokens": request.context_tokens,
        "cached_tokens": request.cached_tokens,
        "new_tokens": request.new_tokens,
        "token_digest": request.token_digest,
    }


def _planned_chunks_payload(chunks: tuple[PChunkPlan, ...]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_index": chunk.chunk_index,
            "scheduled_tokens_by_request": sorted(
                chunk.scheduled_tokens_by_request.items()
            ),
        }
        for chunk in chunks
    ]


def _point_payload(
    *,
    workload_family: WorkloadFamily,
    selector: str,
    composition: str,
    seed: int,
    cache_condition: CacheCondition,
    planner_digest: str,
    requests: tuple[PRequestPlan, ...],
    chunks: tuple[PChunkPlan, ...],
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
) -> dict[str, Any]:
    return {
        "workload_family": workload_family,
        "selector": selector,
        "requests": [_request_payload(request) for request in requests],
        "composition": composition,
        "seed": seed,
        "batch_size": len(requests),
        "chunk_budget": token_budget,
        "cache_condition": cache_condition,
        "block_size": block_size,
        "homogeneous_prefix_tokens": homogeneous_prefix_tokens,
        "capacity_target": "native",
        "planner_digest": planner_digest,
        "planned_chunks": _planned_chunks_payload(chunks),
    }


def plan_chunks(point: PPointPlan, token_budget: int) -> tuple[PChunkPlan, ...]:
    """Split one point into equal-active-cap chunks in request order."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if not point.requests:
        raise ValueError("point must contain at least one request")
    if len({request.request_key for request in point.requests}) != len(point.requests):
        raise ValueError("point request keys must be unique")

    remaining = {
        request.request_key: (
            request.new_tokens
            if point.cache_condition == "prefix_hit"
            else request.context_tokens
        )
        for request in point.requests
    }
    if any(tokens <= 0 for tokens in remaining.values()):
        raise ValueError("scheduled request tokens must be positive")

    chunks = []
    chunk_index = 0
    while remaining:
        active = [
            request.request_key
            for request in point.requests
            if request.request_key in remaining
        ]
        cap = token_budget // len(active)
        if cap <= 0:
            raise ValueError("token_budget cannot schedule all active requests")
        scheduled = {key: min(remaining[key], cap) for key in active}
        chunks.append(PChunkPlan(chunk_index, scheduled))
        for key, tokens in scheduled.items():
            remaining[key] -= tokens
            if remaining[key] == 0:
                del remaining[key]
        chunk_index += 1
    return tuple(chunks)


def _legal_execution_tokens(rendered_turns: list[dict[str, Any]]) -> tuple[int, ...]:
    legal_tokens = {
        token_id
        for turn in rendered_turns
        for key in ("execution_prompt_token_ids", "execution_completion_token_ids")
        for token_id in (turn.get(key) or [])
    }
    if not legal_tokens:
        raise ValueError("rendered turns contain no execution token IDs")
    return tuple(sorted(legal_tokens))


def _homogeneous_requests(
    *,
    selector: str,
    batch_size: int,
    new_tokens: int,
    prefix_tokens: int,
    legal_tokens: tuple[int, ...],
    seed: int,
    block_size: int,
) -> tuple[PRequestPlan, ...]:
    request_length = prefix_tokens + new_tokens
    token_blocks: set[tuple[int, ...]] = set()
    requests = []
    for request_index in range(batch_size):
        token_ids = tuple(
            legal_tokens[
                int.from_bytes(
                    hashlib.sha256(
                        f"{seed}:{selector}:{request_index}:{position}".encode()
                    ).digest(),
                    "big",
                )
                % len(legal_tokens)
            ]
            for position in range(request_length)
        )
        blocks = [
            token_ids[position : position + block_size]
            for position in range(0, len(token_ids), block_size)
        ]
        if any(block in token_blocks for block in blocks):
            raise ValueError("homogeneous requests share a block-aligned token block")
        token_blocks.update(blocks)
        requests.append(
            PRequestPlan(
                request_key=f"r{request_index}",
                trajectory_id=None,
                turn_index=None,
                reasoning_mode=None,
                prompt_token_ids=token_ids,
                context_tokens=request_length,
                cached_tokens=prefix_tokens,
                new_tokens=new_tokens,
                token_digest=_token_digest(token_ids),
            )
        )
    return tuple(requests)


def _rendered_turns_by_key(
    rendered_turns: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {}
    for turn in rendered_turns:
        key = (turn["trajectory_id"], turn["turn_index"])
        if key in indexed:
            raise ValueError("rendered turns contain duplicate trajectory turns")
        indexed[key] = turn
    return indexed


def _trace_requests(
    references: list[dict[str, Any]],
    rendered_by_key: dict[tuple[str, int], dict[str, Any]],
    block_size: int,
) -> tuple[PRequestPlan, ...]:
    requests = []
    for request_index, reference in enumerate(references):
        key = (reference["trajectory_id"], reference["turn_index"])
        turn = rendered_by_key.get(key)
        if turn is None:
            raise ValueError(f"missing rendered execution prompt for {key}")
        prompt_token_ids = turn.get("execution_prompt_token_ids")
        if prompt_token_ids is None:
            raise ValueError(f"missing rendered execution prompt IDs for {key}")
        token_ids = tuple(prompt_token_ids)
        context_tokens = reference["prompt_tokens"]
        cached_tokens = reference["reusable_prefix_tokens"]
        new_tokens = reference["new_prefill_tokens"]
        if len(token_ids) != context_tokens:
            raise ValueError(f"execution prompt length does not match {key}")
        if cached_tokens % block_size:
            raise ValueError(f"cached prefix for {key} is not block aligned")
        if context_tokens - cached_tokens != new_tokens:
            raise ValueError(f"prefill token accounting does not match {key}")
        requests.append(
            PRequestPlan(
                request_key=f"r{request_index}",
                trajectory_id=reference["trajectory_id"],
                turn_index=reference["turn_index"],
                reasoning_mode=reference["reasoning_mode"],
                prompt_token_ids=token_ids,
                context_tokens=context_tokens,
                cached_tokens=cached_tokens,
                new_tokens=new_tokens,
                token_digest=_token_digest(token_ids),
            )
        )
    return tuple(requests)


def _expand_condition_pair(
    *,
    workload_family: WorkloadFamily,
    selector: str,
    composition: str,
    requests: tuple[PRequestPlan, ...],
    planner_digest: str,
    seed: int,
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
) -> tuple[PPointPlan, PPointPlan]:
    points = []
    for cache_condition in ("prefix_hit", "full_recompute"):
        draft = PPointPlan(
            point_id="",
            comparison_id="",
            workload_family=workload_family,
            selector=selector,
            composition=composition,
            seed=seed,
            batch_size=len(requests),
            cache_condition=cache_condition,
            planner_digest=planner_digest,
            requests=requests,
            chunks=(),
            canonical_payload={},
        )
        chunks = plan_chunks(draft, token_budget)
        payload = _point_payload(
            workload_family=workload_family,
            selector=selector,
            composition=composition,
            seed=seed,
            cache_condition=cache_condition,
            planner_digest=planner_digest,
            requests=requests,
            chunks=chunks,
            block_size=block_size,
            token_budget=token_budget,
            homogeneous_prefix_tokens=homogeneous_prefix_tokens,
        )
        points.append(
            PPointPlan(
                point_id=make_point_id(payload),
                comparison_id=make_comparison_id(payload),
                workload_family=workload_family,
                selector=selector,
                composition=composition,
                seed=seed,
                batch_size=len(requests),
                cache_condition=cache_condition,
                planner_digest=planner_digest,
                requests=requests,
                chunks=chunks,
                canonical_payload=payload,
            )
        )
    return points[0], points[1]


def _validate_inputs(
    workload_plan: dict[str, Any],
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
) -> None:
    if block_size != CANONICAL_BLOCK_SIZE:
        raise ValueError(f"block_size must be {CANONICAL_BLOCK_SIZE}")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if homogeneous_prefix_tokens != CANONICAL_HOMOGENEOUS_PREFIX_TOKENS:
        raise ValueError(
            f"homogeneous prefix must be {CANONICAL_HOMOGENEOUS_PREFIX_TOKENS} tokens"
        )
    if homogeneous_prefix_tokens % CANONICAL_BLOCK_SIZE:
        raise ValueError("homogeneous prefix must be divisible by 16")
    if workload_plan.get("token_budget") != token_budget:
        raise ValueError("workload plan token_budget does not match planner")


def build_prefill_points(
    workload_plan: dict[str, Any],
    rendered_turns: list[dict[str, Any]],
    *,
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
    seed: int,
) -> tuple[PPointPlan, ...]:
    """Expand pinned workload artifacts into paired executable prefill points."""
    _validate_inputs(workload_plan, block_size, token_budget, homogeneous_prefix_tokens)
    planner_digest = make_planner_digest(
        workload_plan,
        rendered_turns,
        block_size=block_size,
        token_budget=token_budget,
        homogeneous_prefix_tokens=homogeneous_prefix_tokens,
        seed=seed,
    )
    legal_tokens = _legal_execution_tokens(rendered_turns)
    rendered_by_key = _rendered_turns_by_key(rendered_turns)
    points: list[PPointPlan] = []

    for item in workload_plan["p_homogeneous"]:
        batch_size = item["batch_size"]
        new_tokens = item["per_request_scheduled_tokens"]
        if batch_size > 8:
            raise ValueError("workload exceeds eight sequences")
        if batch_size <= 0 or new_tokens <= 0:
            raise ValueError("homogeneous workloads must be positive")
        if batch_size * new_tokens > token_budget:
            raise ValueError("homogeneous workload exceeds token budget")
        if item["total_scheduled_tokens"] != batch_size * new_tokens:
            raise ValueError("homogeneous workload has inconsistent token total")
        selector = f"b{batch_size}-t{new_tokens}"
        requests = _homogeneous_requests(
            selector=selector,
            batch_size=batch_size,
            new_tokens=new_tokens,
            prefix_tokens=homogeneous_prefix_tokens,
            legal_tokens=legal_tokens,
            seed=seed,
            block_size=block_size,
        )
        points.extend(
            _expand_condition_pair(
                workload_family="homogeneous",
                selector=selector,
                composition="none",
                requests=requests,
                planner_digest=planner_digest,
                seed=seed,
                block_size=block_size,
                token_budget=token_budget,
                homogeneous_prefix_tokens=homogeneous_prefix_tokens,
            )
        )

    for batch in workload_plan["mixed_batches"]:
        batch_size = batch["batch_size"]
        references = batch["turns"]
        if batch_size > 8:
            raise ValueError("workload exceeds eight sequences")
        if len(references) != batch_size:
            raise ValueError("mixed workload batch size does not match references")
        requests = _trace_requests(references, rendered_by_key, block_size)
        if sum(request.new_tokens for request in requests) > token_budget:
            raise ValueError("mixed workload exceeds token budget")
        selector = f"{batch['composition']}-b{batch_size}"
        points.extend(
            _expand_condition_pair(
                workload_family="mixed",
                selector=selector,
                composition=batch["composition"],
                requests=requests,
                planner_digest=planner_digest,
                seed=seed,
                block_size=block_size,
                token_budget=token_budget,
                homogeneous_prefix_tokens=homogeneous_prefix_tokens,
            )
        )

    for replay in workload_plan["exact_replays"]:
        quantile = int(replay["selection_quantile"] * 100)
        selector = f"{replay['reasoning_mode']}-q{quantile:02d}"
        requests = _trace_requests([replay], rendered_by_key, block_size)
        points.extend(
            _expand_condition_pair(
                workload_family="exact_replay",
                selector=selector,
                composition="quantile",
                requests=requests,
                planner_digest=planner_digest,
                seed=seed,
                block_size=block_size,
                token_budget=token_budget,
                homogeneous_prefix_tokens=homogeneous_prefix_tokens,
            )
        )

    return tuple(points)
