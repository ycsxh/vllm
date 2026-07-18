# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

assert os.environ["PYTHONHASHSEED"] == "0"
assert os.environ["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] == "0"


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
    first_ids = manager.get_block_ids("first")[0]
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
    free = next(call for call in calls if call.operation == "free")
    assert free.block_ids == tuple(reversed(second_ids))
    assert first_ids[-1] not in second_ids or len(set(first_ids)) < len(first_ids)


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
