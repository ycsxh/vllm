# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections import Counter

import pytest

import benchmarks.ds4_profile.prefill_profile as prefill_profile
from benchmarks.ds4_profile import profile_spine
from benchmarks.ds4_profile.prefill_profile import build_prefill_points
from benchmarks.ds4_profile.profile_spine import make_comparison_id, make_point_id

HOMOGENEOUS_CASES = (
    (1, 128),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (2, 128),
    (2, 512),
    (2, 1024),
    (2, 2048),
    (4, 128),
    (4, 512),
    (4, 1024),
    (8, 128),
    (8, 256),
    (8, 512),
)


def _fixture_turn(index: int, new_tokens: int, reasoning_mode: str) -> dict:
    cached_tokens = 16
    prompt_tokens = cached_tokens + new_tokens
    token_ids = [1 + (index * 37 + position) % 512 for position in range(prompt_tokens)]
    return {
        "trajectory_id": f"trajectory-{index}",
        "turn_index": index,
        "reasoning_mode": reasoning_mode,
        "prompt_tokens": prompt_tokens,
        "reusable_prefix_tokens": cached_tokens,
        "new_prefill_tokens": new_tokens,
        "execution_prompt_token_ids": token_ids,
        "execution_completion_token_ids": [1 + index],
    }


def _pinned_inputs() -> tuple[dict, list[dict]]:
    """Return a compact, deterministic Ticket 02-shaped planner fixture."""
    lengths = (16, 32, 48, 64, 80, 96, 112, 128, 256, 512)
    turns = [
        _fixture_turn(
            index,
            length,
            "no_think" if index % 2 == 0 else "think_high",
        )
        for index, length in enumerate(lengths)
    ]

    def reference(index: int) -> dict:
        turn = turns[index]
        return {
            key: turn[key]
            for key in (
                "trajectory_id",
                "turn_index",
                "reasoning_mode",
                "prompt_tokens",
                "reusable_prefix_tokens",
                "new_prefill_tokens",
            )
        }

    mixed_batches = []
    batch_indexes = {
        "similar": {2: (0, 1), 4: (0, 1, 2, 3), 8: tuple(range(8))},
        "random": {2: (8, 9), 4: (8, 3, 7, 1), 8: tuple(range(8))},
        "high_skew": {
            2: (0, 9),
            4: (0, 1, 2, 9),
            8: (0, 1, 2, 3, 4, 5, 6, 9),
        },
    }
    for composition, by_batch_size in batch_indexes.items():
        for batch_size, indexes in by_batch_size.items():
            references = [reference(index) for index in indexes]
            mixed_batches.append(
                {
                    "composition": composition,
                    "batch_size": batch_size,
                    "turns": references,
                    "total_scheduled_tokens": sum(
                        item["new_prefill_tokens"] for item in references
                    ),
                }
            )

    exact_replays = []
    for offset, reasoning_mode in enumerate(("no_think", "think_high")):
        for quantile, index in zip((0.0, 0.25, 0.5, 0.75, 1.0), range(offset, 10, 2)):
            exact_replays.append(
                {
                    **reference(index),
                    "selection_quantile": quantile,
                }
            )
    return {
        "token_budget": 4096,
        "p_homogeneous": [
            {
                "batch_size": batch_size,
                "per_request_scheduled_tokens": new_tokens,
                "total_scheduled_tokens": batch_size * new_tokens,
            }
            for batch_size, new_tokens in HOMOGENEOUS_CASES
        ],
        "mixed_batches": mixed_batches,
        "exact_replays": exact_replays,
    }, turns


def _build_pinned_points():
    plan, turns = _pinned_inputs()
    return build_prefill_points(
        plan,
        turns,
        block_size=16,
        token_budget=4096,
        homogeneous_prefix_tokens=4096,
        seed=20260715,
    )


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


def test_planner_preserves_requests_and_pairs_conditions() -> None:
    plan, _ = _pinned_inputs()
    points = _build_pinned_points()

    assert len({point.point_id for point in points}) == len(points)
    assert [point.point_id for point in points] == [
        point.point_id for point in _build_pinned_points()
    ]
    assert set(Counter(point.comparison_id for point in points).values()) == {2}

    mixed = next(
        point
        for point in points
        if point.workload_family == "mixed"
        and point.selector == "random-b4"
        and point.cache_condition == "prefix_hit"
    )
    expected_mixed = next(
        batch
        for batch in plan["mixed_batches"]
        if batch["composition"] == "random" and batch["batch_size"] == 4
    )
    assert [request.trajectory_id for request in mixed.requests] == [
        turn["trajectory_id"] for turn in expected_mixed["turns"]
    ]
    assert [request.turn_index for request in mixed.requests] == [
        turn["turn_index"] for turn in expected_mixed["turns"]
    ]

    replay = next(
        point
        for point in points
        if point.workload_family == "exact_replay"
        and point.selector == "no_think-q00"
        and point.cache_condition == "prefix_hit"
    )
    selected = next(
        item
        for item in plan["exact_replays"]
        if item["reasoning_mode"] == "no_think" and item["selection_quantile"] == 0.0
    )
    assert (replay.requests[0].trajectory_id, replay.requests[0].turn_index) == (
        selected["trajectory_id"],
        selected["turn_index"],
    )


def test_full_recompute_chunks_context_and_removes_completed_requests() -> None:
    points = _build_pinned_points()
    point = next(
        point
        for point in points
        if point.selector == "high_skew-b8"
        and point.cache_condition == "full_recompute"
    )

    scheduled: Counter[str] = Counter()
    active_sizes = []
    for chunk in point.chunks:
        active_sizes.append(len(chunk.scheduled_tokens_by_request))
        scheduled.update(chunk.scheduled_tokens_by_request)
    assert active_sizes == sorted(active_sizes, reverse=True)
    assert active_sizes[0] == 8
    assert active_sizes[-1] < 8
    assert scheduled == {
        request.request_key: request.context_tokens for request in point.requests
    }


def test_planner_rejects_invalid_capacity_or_prefix_alignment() -> None:
    plan, turns = _pinned_inputs()
    too_many = copy.deepcopy(plan)
    too_many["p_homogeneous"].append(
        {
            "batch_size": 9,
            "per_request_scheduled_tokens": 1,
            "total_scheduled_tokens": 9,
        }
    )

    with pytest.raises(ValueError, match="eight sequences"):
        build_prefill_points(
            too_many,
            turns,
            block_size=16,
            token_budget=4096,
            homogeneous_prefix_tokens=4096,
            seed=20260715,
        )
    with pytest.raises(ValueError, match="block_size must be 16"):
        build_prefill_points(
            plan,
            turns,
            block_size=8,
            token_budget=4096,
            homogeneous_prefix_tokens=4096,
            seed=20260715,
        )
    with pytest.raises(ValueError, match="homogeneous prefix must be 4096"):
        build_prefill_points(
            plan,
            turns,
            block_size=16,
            token_budget=4096,
            homogeneous_prefix_tokens=4080,
            seed=20260715,
        )


def test_point_id_covers_planned_chunks_and_planner_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = _build_pinned_points()[0]
    changed_chunk = copy.deepcopy(point.canonical_payload)
    changed_chunk["planned_chunks"][0]["scheduled_tokens_by_request"][0] = (
        "r0",
        changed_chunk["planned_chunks"][0]["scheduled_tokens_by_request"][0][1] - 1,
    )
    assert make_point_id(changed_chunk) != point.point_id

    monkeypatch.setattr(prefill_profile, "CHUNK_ALGORITHM", "changed-chunk-v2")
    changed_chunk_algorithm = _build_pinned_points()[0]
    assert changed_chunk_algorithm.planner_digest != point.planner_digest
    assert changed_chunk_algorithm.point_id != point.point_id

    monkeypatch.setattr(
        prefill_profile,
        "HOMOGENEOUS_TOKEN_ALGORITHM",
        "changed-homogeneous-token-v2",
    )
    changed_algorithms = _build_pinned_points()[0]
    assert changed_algorithms.planner_digest != changed_chunk_algorithm.planner_digest
    assert changed_algorithms.point_id != changed_chunk_algorithm.point_id


def _artifact_point(cache_condition: str) -> prefill_profile.PPointPlan:
    request = prefill_profile.PRequestPlan(
        request_key="r0",
        trajectory_id=None,
        turn_index=None,
        reasoning_mode=None,
        prompt_token_ids=tuple(range(200)),
        context_tokens=200,
        cached_tokens=100,
        new_tokens=100,
        token_digest="a" * 64,
    )
    chunks = (
        prefill_profile.PChunkPlan(0, {"r0": 40}),
        prefill_profile.PChunkPlan(1, {"r0": 60}),
    )
    payload = {
        "workload_family": "homogeneous",
        "selector": "artifact-b1-t100",
        "requests": [
            {
                "request_key": request.request_key,
                "trajectory_id": None,
                "turn_index": None,
                "reasoning_mode": None,
                "context_tokens": request.context_tokens,
                "cached_tokens": request.cached_tokens,
                "new_tokens": request.new_tokens,
                "token_digest": request.token_digest,
            }
        ],
        "composition": "none",
        "seed": 20260715,
        "batch_size": 1,
        "chunk_budget": 4096,
        "cache_condition": cache_condition,
        "block_size": 16,
        "homogeneous_prefix_tokens": 4096,
        "capacity_target": "native",
        "planner_digest": "b" * 64,
        "planned_chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "scheduled_tokens_by_request": sorted(
                    chunk.scheduled_tokens_by_request.items()
                ),
            }
            for chunk in chunks
        ],
    }
    return prefill_profile.PPointPlan(
        point_id=make_point_id(payload),
        comparison_id=make_comparison_id(payload),
        workload_family="homogeneous",
        selector=payload["selector"],
        composition="none",
        seed=payload["seed"],
        batch_size=1,
        cache_condition=cache_condition,
        planner_digest=payload["planner_digest"],
        requests=(request,),
        chunks=chunks,
        canonical_payload=payload,
    )


def _passed_chunk_row(
    point: prefill_profile.PPointPlan,
    ordinal: int,
    chunk: prefill_profile.PChunkPlan,
    wall_time_ms: float,
) -> dict:
    return prefill_profile.make_prefill_chunk_row(
        run_id="run-task-3",
        point=point,
        phase="steady",
        ordinal=ordinal,
        chunk=chunk,
        runner_wall_time_ms=wall_time_ms,
        cuda_model_time_ms=wall_time_ms - 0.5,
        allocation={
            "state": "allocated",
            "actual_scheduled_tokens_by_request": (chunk.scheduled_tokens_by_request),
            "preempted_request_ids": (),
            "unrelated_request_ids": (),
            "cache_epoch": ordinal,
            "cache_reset_completed": True,
            "cache_reset_empty": True,
            "requested_blocks": 8,
            "allocatable_blocks": 128,
            "allocated_blocks": 8,
            "kv_block_bytes": 1024,
            "lookup_time_ms": 0.2,
            "allocation_time_ms": 0.3,
            "runtime_mode": "FULL",
        },
        status="passed",
        error=None,
    )


def test_turn_and_comparison_statistics_are_recomputed_from_chunks() -> None:
    hit = _artifact_point("prefix_hit")
    recompute = _artifact_point("full_recompute")
    raw_rows: list[dict] = []
    for ordinal in range(10):
        raw_rows.extend(
            _passed_chunk_row(hit, ordinal, chunk, wall_time)
            for chunk, wall_time in zip(hit.chunks, (4.0, 6.0), strict=True)
        )
        raw_rows.extend(
            _passed_chunk_row(recompute, ordinal, chunk, wall_time)
            for chunk, wall_time in zip(recompute.chunks, (6.0, 10.0), strict=True)
        )

    turns = profile_spine.summarize_turn_samples(raw_rows, (hit, recompute))
    aggregates = profile_spine.aggregate_turn_samples(turns, 0.05)
    comparisons = profile_spine.compare_conditions(aggregates, (hit, recompute), [])

    hit_turn = next(row for row in turns if row["cache_condition"] == "prefix_hit")
    assert hit_turn["runner_wall_time_ms"] == 10.0
    assert hit_turn["throughput_tokens_per_s"] == 10_000.0
    assert {row["sample_count"] for row in aggregates} == {10}
    assert comparisons[0]["recompute_penalty_ms"] == 6.0
