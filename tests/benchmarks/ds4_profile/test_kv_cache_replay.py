# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

assert os.environ["PYTHONHASHSEED"] == "0"
assert os.environ["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] == "0"

pytestmark = pytest.mark.skip_global_cleanup


def _replay_turn(index: int, tokens: list[int], trajectory_id: str = "task:no_think"):
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
    first = kv_cache_replay.make_request(_replay_turn(0, [1, 1, 2, 2]), 2, "first")
    second = kv_cache_replay.make_request(_replay_turn(1, [1, 1, 9, 9]), 2, "second")

    assert calls == [kv_cache_replay.sha256]
    assert first.block_hashes[0] == second.block_hashes[0]


def test_native_events_preserve_byte_sha_hashes() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import make_manager, make_request
    from vllm.distributed.kv_events import BlockStored

    manager = make_manager(capacity_blocks=2, block_size=2, max_model_len=32)
    request = make_request(_replay_turn(0, [1, 1, 2, 2]), 2, "request")
    hit, hit_tokens, _ = manager.get_computed_blocks(request)
    assert (
        manager.allocate_slots(request, request.num_tokens, hit_tokens, hit) is not None
    )

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
    from benchmarks.ds4_profile.kv_cache_replay import (
        make_manager,
        make_request,
        observe_block_pool,
    )
    from vllm.distributed.kv_events import BlockRemoved

    manager = make_manager(capacity_blocks=3, block_size=2, max_model_len=32)
    first = make_request(_replay_turn(0, [1, 1, 2, 2, 3]), 2, "first")
    hit, hit_tokens, _ = manager.get_computed_blocks(first)
    assert manager.allocate_slots(first, first.num_tokens, hit_tokens, hit) is not None
    manager.free(first)
    manager.take_events()

    second = make_request(_replay_turn(1, [1, 1, 9, 9, 8]), 2, "second")
    hit, hit_tokens, _ = manager.get_computed_blocks(second)
    with observe_block_pool(manager) as calls:
        assert (
            manager.allocate_slots(
                second, second.num_tokens - hit_tokens, hit_tokens, hit
            )
            is not None
        )
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
    assert removed
    free = next(call for call in calls if call.operation == "free")
    assert free.block_ids == tuple(reversed(second_ids))


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
        row["miss_class"] for row in result.event_rows if row["cache_outcome"] == "miss"
    }
    evictions = [
        row
        for row in result.event_rows
        if row["operation"] == "evict" and row["event_source"] == "native"
    ]
    assert result.status == "passed"
    assert result.eviction_count == len(evictions)
    assert result.eviction_count > 0
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


def test_replay_records_status_occupancy_and_lookup_timing() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import replay_session

    result = replay_session(
        run_id="fixture-run",
        turns=[_replay_turn(0, [1, 1, 2, 2, 3])],
        capacity_blocks=3,
        block_size=2,
        max_model_len=32,
    )

    lookup_rows = [row for row in result.event_rows if row["operation"] == "lookup"]
    allocation = next(
        row
        for row in result.event_rows
        if row["event_source"] == "observer" and row["operation"] == "allocate"
    )
    free = next(row for row in result.event_rows if row["operation"] == "free")

    assert all(row["status"] == "passed" for row in result.event_rows)
    assert (
        sum(row["duration_ns"] for row in lookup_rows)
        == result.turn_rows[0]["lookup_time_ns"]
    )
    assert allocation["active_blocks_after"] > allocation["active_blocks_before"]
    assert free["active_blocks_after"] == 0
    assert free["cached_resident_blocks_after"] > 0
    assert (
        result.turn_rows[0]["cached_resident_blocks_after_free"]
        == free["cached_resident_blocks_after"]
    )


def test_replay_marks_non_atomic_admission_failure_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.ds4_profile import kv_cache_replay

    original_allocate_slots = kv_cache_replay.KVCacheManager.allocate_slots

    def allocate_then_fail(manager, *args, **kwargs):
        assert original_allocate_slots(manager, *args, **kwargs) is not None
        return None

    monkeypatch.setattr(
        kv_cache_replay.KVCacheManager,
        "allocate_slots",
        allocate_then_fail,
    )
    result = kv_cache_replay.replay_session(
        run_id="fixture-run",
        turns=[_replay_turn(0, [1, 1, 2, 2])],
        capacity_blocks=2,
        block_size=2,
        max_model_len=32,
    )

    assert result.status == "invalid"
    assert "non-atomic admission failure" in result.error
    assert all(row["status"] == "invalid" for row in result.event_rows)
    assert any(
        row["event_source"] == "observer" and row["operation"] == "allocate"
        for row in result.event_rows
    )
    assert any(
        row["event_source"] == "native" and row["operation"] == "store"
        for row in result.event_rows
    )
    assert result.turn_rows[0]["status"] == "invalid"


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
        if row["turn_index"] == 1 and row["cache_outcome"] == "manager_forced_recompute"
    ]
    assert len(forced) == 1
    assert forced[0]["block_position"] == 1
    assert forced[0]["miss_class"] is None
    assert result.turn_rows[1]["manager_forced_recompute_blocks"] == 1
    assert result.turn_rows[1]["capacity_miss_blocks"] == 0


def test_replay_counts_duplicate_hashes_as_distinct_resident_blocks() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import replay_session

    result = replay_session(
        run_id="fixture-run",
        turns=[
            _replay_turn(0, [1, 1, 2, 2]),
            _replay_turn(1, [1, 1, 2, 2]),
        ],
        capacity_blocks=3,
        block_size=2,
        max_model_len=32,
    )

    assert result.status == "passed"
    assert result.turn_rows[1]["manager_forced_recompute_blocks"] == 1
    assert result.turn_rows[1]["cached_resident_blocks_after_free"] == 3


def test_replay_pairs_duplicate_hash_evictions_by_occurrence() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import replay_session

    result = replay_session(
        run_id="fixture-run",
        turns=[
            _replay_turn(0, [1, 1, 2, 2]),
            _replay_turn(1, [1, 1, 2, 2]),
            _replay_turn(2, [9, 9, 8, 8, 7, 7]),
        ],
        capacity_blocks=3,
        block_size=2,
        max_model_len=32,
    )

    evictions = [
        row
        for row in result.event_rows
        if row["turn_index"] == 2
        and row["operation"] == "evict"
        and row["event_source"] == "native"
    ]
    assert result.status == "passed"
    assert len(evictions) == 3
    assert len({row["observer_call_index"] for row in evictions}) == 3


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


def test_selected_turn_manifest_is_ordered_and_content_addressed() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import _selected_turn_manifest

    turns = [
        _replay_turn(1, [1, 1, 2, 2, 3]),
        _replay_turn(0, [1, 1, 9, 9]),
    ]

    manifest = _selected_turn_manifest(turns, block_size=2)

    expected_token_hash = hashlib.sha256(
        json.dumps([1, 1, 9, 9], separators=(",", ":")).encode()
    ).hexdigest()
    assert [row["turn_index"] for row in manifest] == [0, 1]
    assert set(manifest[0]) == {
        "trajectory_id",
        "turn_index",
        "prompt_tokens",
        "prompt_token_ids_sha256",
        "block_hashes_sha256",
    }
    assert manifest[0]["prompt_token_ids_sha256"] == expected_token_hash
    assert len(manifest[0]["block_hashes_sha256"]) == 64


def test_selection_plan_uses_real_replay_and_is_deterministic(tmp_path: Path) -> None:
    from benchmarks.ds4_profile.kv_cache_replay import build_selection_plan

    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("tokenizer")
    artifacts = {}
    for name in (
        "manifest",
        "normalized_turns",
        "normalized_provenance",
        "rendered_turns",
        "workload_provenance",
    ):
        path = tmp_path / name
        path.write_text(name)
        artifacts[name] = str(path)
    config = {
        "artifacts": artifacts,
        "tokenizer": {"path": str(tokenizer)},
        "replay": {"block_size": 2, "max_model_len": 32},
    }
    turns = [
        _replay_turn(0, [1, 1, 2, 2, 3]),
        _replay_turn(1, [1, 1, 9, 9, 8, 8]),
        _replay_turn(2, [1, 1, 2, 2, 7]),
    ]

    first = build_selection_plan(config, turns)
    second = build_selection_plan(config, turns)

    assert first == second
    assert first["selected"]["trajectory_id"] == "task:no_think"
    assert first["selected"]["capacity_blocks"] == 3
    assert first["selected"]["eviction_count"] > 0
    assert (
        first["selected"]["input_set_sha256"]
        == hashlib.sha256(
            json.dumps(first["inputs"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert (
        first["sha256"]
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in first.items() if key != "sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )


def test_verify_pinned_selection_rejects_drift() -> None:
    from benchmarks.ds4_profile.kv_cache_replay import (
        _canonical_json_sha256,
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


def test_verify_pinned_selection_rejects_config_input_path_drift(
    tmp_path: Path,
) -> None:
    from benchmarks.ds4_profile.kv_cache_replay import (
        build_selection_plan,
        verify_pinned_selection,
    )

    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("tokenizer")
    artifacts = {}
    for name in (
        "manifest",
        "normalized_turns",
        "normalized_provenance",
        "rendered_turns",
        "workload_provenance",
    ):
        path = tmp_path / name
        path.write_text(name)
        artifacts[name] = str(path)
    config = {
        "artifacts": artifacts,
        "tokenizer": {"path": str(tokenizer)},
        "replay": {"block_size": 2, "max_model_len": 32},
    }
    turns = [
        _replay_turn(0, [1, 1, 2, 2, 3]),
        _replay_turn(1, [1, 1, 9, 9, 8, 8]),
        _replay_turn(2, [1, 1, 2, 2, 7]),
    ]
    plan = build_selection_plan(config, turns)
    selected = plan["selected"]
    config["selection"] = {
        "status": "pinned",
        "trajectory_id": selected["trajectory_id"],
        "reasoning_mode": selected["reasoning_mode"],
        "capacity_blocks": selected["capacity_blocks"],
        "input_set_sha256": selected["input_set_sha256"],
        "planning_sha256": plan["sha256"],
        "turn_manifest_sha256": selected["turn_manifest_sha256"],
    }
    replacement = tmp_path / "replacement-manifest"
    replacement.write_text("manifest")
    config["artifacts"]["manifest"] = str(replacement)

    with pytest.raises(ValueError, match="planning input paths do not match"):
        verify_pinned_selection(config, plan)


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


def test_load_full_turns_reads_only_ticket_02_scalar_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.ds4_profile import kv_cache_replay, workloads

    row = _turn_row("task:no_think", 0, [1, 2, 3, 4])
    monkeypatch.setattr(workloads, "render_turns", lambda **_: [row])
    ticket_02 = dict(row)
    ticket_02.pop("_prompt_token_ids")
    ticket_02["execution_prompt_token_ids"] = [1, 2, 3, 4]
    ticket_02["execution_completion_token_ids"] = [5, 6]
    rendered_path = tmp_path / "rendered_turns.parquet"
    pq.write_table(pa.Table.from_pylist([ticket_02]), rendered_path)
    selected_columns: list[list[str]] = []
    read_table = kv_cache_replay.pq.read_table

    def record_selected_columns(path: str, *, columns: list[str]) -> pa.Table:
        selected_columns.append(columns)
        return read_table(path, columns=columns)

    monkeypatch.setattr(kv_cache_replay.pq, "read_table", record_selected_columns)
    config = {
        "artifacts": {
            "manifest": str(tmp_path / "manifest.json"),
            "normalized_turns": str(tmp_path / "turns.parquet"),
            "rendered_turns": str(rendered_path),
        },
        "tokenizer": {"path": str(tmp_path / "tokenizer")},
        "replay": {"block_size": 4},
    }

    kv_cache_replay.load_full_turns(config)

    assert selected_columns == [list(kv_cache_replay.SCALAR_TURN_FIELDS)]
    assert "execution_prompt_token_ids" not in selected_columns[0]
    assert "execution_completion_token_ids" not in selected_columns[0]


def test_load_full_turns_rejects_duplicate_ticket_02_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.ds4_profile import workloads
    from benchmarks.ds4_profile.kv_cache_replay import load_full_turns

    row = _turn_row("task:no_think", 0, [1, 2, 3, 4])
    monkeypatch.setattr(workloads, "render_turns", lambda **_: [row])
    ticket_02 = dict(row)
    ticket_02.pop("_prompt_token_ids")
    rendered_path = tmp_path / "rendered_turns.parquet"
    pq.write_table(pa.Table.from_pylist([ticket_02, ticket_02]), rendered_path)
    config = {
        "artifacts": {
            "manifest": str(tmp_path / "manifest.json"),
            "normalized_turns": str(tmp_path / "turns.parquet"),
            "rendered_turns": str(rendered_path),
        },
        "tokenizer": {"path": str(tmp_path / "tokenizer")},
        "replay": {"block_size": 4},
    }

    with pytest.raises(ValueError, match="Ticket 02 duplicate key"):
        load_full_turns(config)


def test_load_full_turns_rejects_extra_ticket_02_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.ds4_profile import workloads
    from benchmarks.ds4_profile.kv_cache_replay import load_full_turns

    row = _turn_row("task:no_think", 0, [1, 2, 3, 4])
    monkeypatch.setattr(workloads, "render_turns", lambda **_: [row])
    ticket_02_rows = []
    for turn_index in (0, 1):
        ticket_02 = _turn_row("task:no_think", turn_index, [1, 2, 3, 4])
        ticket_02.pop("_prompt_token_ids")
        ticket_02_rows.append(ticket_02)
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

    with pytest.raises(ValueError, match="Ticket 02 key set mismatch"):
        load_full_turns(config)
