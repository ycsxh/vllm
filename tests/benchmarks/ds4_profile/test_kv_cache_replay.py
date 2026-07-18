# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
