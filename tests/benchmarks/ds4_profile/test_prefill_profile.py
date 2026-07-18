# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from benchmarks.ds4_profile.prefill_profile import build_prefill_points
from benchmarks.ds4_profile.profile_spine import make_point_id

PROJECT_DIR = Path(__file__).parents[3]
ARTIFACT_DIR = PROJECT_DIR / ".scratch/ds4-agent-1p1d-profile/artifacts/ticket-02"


def _pinned_inputs() -> tuple[dict, list[dict]]:
    plan = json.loads((ARTIFACT_DIR / "workload_plan.json").read_text())
    turns = pq.read_table(ARTIFACT_DIR / "rendered_turns.parquet").to_pylist()
    return plan, turns


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
        if item["reasoning_mode"] == "no_think"
        and item["selection_quantile"] == 0.0
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

    scheduled = Counter()
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
    with pytest.raises(ValueError, match="divisible"):
        build_prefill_points(
            plan,
            turns,
            block_size=16,
            token_budget=4096,
            homogeneous_prefix_tokens=4097,
            seed=20260715,
        )


def test_point_id_covers_planned_chunks_and_planner_algorithm() -> None:
    point = _build_pinned_points()[0]
    changed_chunk = copy.deepcopy(point.canonical_payload)
    changed_chunk["planned_chunks"][0]["scheduled_tokens_by_request"][0] = (
        "r0",
        changed_chunk["planned_chunks"][0]["scheduled_tokens_by_request"][0][1]
        - 1,
    )
    changed_planner = copy.deepcopy(point.canonical_payload)
    changed_planner["planner_digest"] = "0" * 64

    assert make_point_id(changed_chunk) != point.point_id
    assert make_point_id(changed_planner) != point.point_id
