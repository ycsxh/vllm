# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

import benchmarks.ds4_profile.workloads as workloads
from benchmarks.ds4_profile.workloads import (
    build_artifacts,
    build_workload_plan,
    map_execution_tokens,
    render_turns,
)

PROJECT_DIR = Path(__file__).parents[3]
ARTIFACT_DIR = PROJECT_DIR / ".scratch/ds4-agent-1p1d-profile"
SNAPSHOT_DIR = ARTIFACT_DIR / "snapshot" / "4da61f3d06b48b6817a62b99e9c47035c8e59787"
TOKENIZER_DIR = (
    ARTIFACT_DIR
    / "tokenizers"
    / "deepseek-v4-flash"
    / "60d8d70770c6776ff598c94bb586a859a38244f1"
)
TURNS_PATH = ARTIFACT_DIR / "artifacts/ticket-01/turns.parquet"
QWEN_TOKENIZER_DIR = (
    ARTIFACT_DIR
    / "tokenizers"
    / "qwen2.5-coder-7b-instruct"
    / "c03e6d358207e414f1eca0bb1891e29f1db0e242"
)


def test_deepseek_encoding_falls_back_to_installed_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = PROJECT_DIR / "vllm/tokenizers/deepseek_v4_encoding.py"

    class Distribution:
        def locate_file(self, path: str) -> Path:
            assert path == "vllm/tokenizers/deepseek_v4_encoding.py"
            return source_path

    monkeypatch.setattr(workloads, "__file__", str(tmp_path / "workloads.py"))
    monkeypatch.setattr(
        workloads.importlib.metadata,
        "distribution",
        lambda _: Distribution(),
    )

    encoding = workloads._load_deepseek_v4_encoding()

    assert encoding.dsml_token == "｜DSML｜"


def test_pinned_real_turn_prompts_match_source_token_usage() -> None:
    turns = render_turns(
        manifest_path=SNAPSHOT_DIR / "manifest.json",
        normalized_turns_path=TURNS_PATH,
        tokenizer_path=TOKENIZER_DIR,
    )

    assert len(turns) == 667
    assert {turn["reasoning_mode"] for turn in turns} == {
        "no_think",
        "think_high",
    }
    assert all(turn["prompt_tokens"] == turn["source_prompt_tokens"] for turn in turns)


def test_rendered_turn_segments_have_exact_tool_ready_boundaries() -> None:
    turns = render_turns(
        manifest_path=SNAPSHOT_DIR / "manifest.json",
        normalized_turns_path=TURNS_PATH,
        tokenizer_path=TOKENIZER_DIR,
    )

    for turn in turns:
        assert turn["completion_tokens"] == sum(
            turn[column]
            for column in (
                "reasoning_tokens",
                "assistant_content_tokens",
                "tool_call_tokens",
                "completion_framing_tokens",
            )
        )
        assert turn["cumulative_context_tokens"] == (
            turn["prompt_tokens"]
            + turn["completion_tokens"]
            + turn["tool_result_tokens"]
        )
        assert turn["tokens_until_tool_ready"] == (
            turn["completion_tokens"] - turn["completion_framing_tokens"]
        )

    first_no_think = turns[0]
    first_think_high = next(
        turn for turn in turns if turn["reasoning_mode"] == "think_high"
    )
    assert {
        key: first_no_think[key]
        for key in (
            "prompt_tokens",
            "reasoning_tokens",
            "assistant_content_tokens",
            "tool_call_tokens",
            "completion_framing_tokens",
            "completion_tokens",
            "tool_result_tokens",
            "cumulative_context_tokens",
            "tokens_until_tool_ready",
        )
    } == {
        "prompt_tokens": 1690,
        "reasoning_tokens": 0,
        "assistant_content_tokens": 14,
        "tool_call_tokens": 114,
        "completion_framing_tokens": 1,
        "completion_tokens": 129,
        "tool_result_tokens": 563,
        "cumulative_context_tokens": 2382,
        "tokens_until_tool_ready": 128,
    }
    assert {
        key: first_think_high[key]
        for key in (
            "prompt_tokens",
            "reasoning_tokens",
            "assistant_content_tokens",
            "tool_call_tokens",
            "completion_framing_tokens",
            "completion_tokens",
            "tool_result_tokens",
            "cumulative_context_tokens",
            "tokens_until_tool_ready",
        )
    } == {
        "prompt_tokens": 1690,
        "reasoning_tokens": 101,
        "assistant_content_tokens": 0,
        "tool_call_tokens": 49,
        "completion_framing_tokens": 1,
        "completion_tokens": 151,
        "tool_result_tokens": 835,
        "cumulative_context_tokens": 2676,
        "tokens_until_tool_ready": 150,
    }


def test_adjacent_turn_prefixes_are_block_aligned_and_uniquely_attributed() -> None:
    turns = render_turns(
        manifest_path=SNAPSHOT_DIR / "manifest.json",
        normalized_turns_path=TURNS_PATH,
        tokenizer_path=TOKENIZER_DIR,
        block_size=16,
    )

    for turn in turns:
        assert turn["reusable_prefix_tokens"] % 16 == 0
        assert turn["new_prefill_tokens"] == (
            turn["prompt_tokens"] - turn["reusable_prefix_tokens"]
        )
        assert turn["prefix_reuse_ratio"] == (
            turn["reusable_prefix_tokens"] / turn["prompt_tokens"]
        )
        assert turn["ideal_prefix_hit"] == (turn["reusable_prefix_tokens"] > 0)
        assert turn["reusable_prefix_tokens"] == sum(
            turn[column]
            for column in (
                "global_reusable_tokens",
                "task_reusable_tokens",
                "session_reusable_tokens",
            )
        )

    second_no_think = turns[1]
    assert {
        key: second_no_think[key]
        for key in (
            "exact_lcp_tokens",
            "reusable_prefix_tokens",
            "new_prefill_tokens",
            "prefix_reuse_ratio",
            "ideal_prefix_hit",
            "reuse_distance_turns",
            "global_prefix_tokens",
            "task_prefix_tokens",
            "global_reusable_tokens",
            "task_reusable_tokens",
            "session_reusable_tokens",
        )
    } == {
        "exact_lcp_tokens": 1690,
        "reusable_prefix_tokens": 1680,
        "new_prefill_tokens": 702,
        "prefix_reuse_ratio": 1680 / 2382,
        "ideal_prefix_hit": True,
        "reuse_distance_turns": 1,
        "global_prefix_tokens": 305,
        "task_prefix_tokens": 1689,
        "global_reusable_tokens": 320,
        "task_reusable_tokens": 1360,
        "session_reusable_tokens": 0,
    }
    assert second_no_think["prefix_source_spans"] == [
        {
            "source": "global",
            "token_start": 0,
            "token_end": 305,
            "attributed_block_start": 0,
            "attributed_block_end": 320,
        },
        {
            "source": "task",
            "token_start": 305,
            "token_end": 1680,
            "attributed_block_start": 320,
            "attributed_block_end": 1680,
        },
    ]


def test_qwen_execution_mapping_is_legal_deterministic_and_prefix_exact() -> None:
    logical_sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 8],
        [9, 10],
    ]

    mapped = map_execution_tokens(
        logical_sequences,
        source_vocab_size=16,
        tokenizer_path=QWEN_TOKENIZER_DIR,
        seed=20260715,
    )
    repeated = map_execution_tokens(
        list(reversed(logical_sequences)),
        source_vocab_size=16,
        tokenizer_path=QWEN_TOKENIZER_DIR,
        seed=20260715,
    )

    assert list(reversed(repeated)) == mapped
    assert [len(sequence) for sequence in mapped] == [5, 4, 2]
    assert mapped[0][:3] == mapped[1][:3]
    assert mapped[0][3] != mapped[1][3]

    tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER_DIR, local_files_only=True)
    mapped_ids = {token for sequence in mapped for token in sequence}
    assert mapped_ids.isdisjoint(tokenizer.all_special_ids)
    assert all(0 <= token < len(tokenizer) for token in mapped_ids)
    assert all(
        tokenizer.convert_ids_to_tokens(token) is not None for token in mapped_ids
    )
    assert all(
        tokenizer.decode([token], skip_special_tokens=False) for token in mapped_ids
    )


def test_workload_plan_is_complete_deterministic_and_budget_bounded() -> None:
    turns = render_turns(
        manifest_path=SNAPSHOT_DIR / "manifest.json",
        normalized_turns_path=TURNS_PATH,
        tokenizer_path=TOKENIZER_DIR,
        block_size=16,
    )

    plan = build_workload_plan(turns, seed=20260715)
    repeated = build_workload_plan(turns, seed=20260715)
    different_seed = build_workload_plan(turns, seed=20260716)

    assert plan == repeated
    assert plan["mixed_batches"] != different_seed["mixed_batches"]
    assert [
        (point["batch_size"], point["per_request_scheduled_tokens"])
        for point in plan["p_homogeneous"]
    ] == [
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
    ]
    assert all(
        point["total_scheduled_tokens"] <= 4096 for point in plan["p_homogeneous"]
    )

    assert len(plan["trace_quantile_buckets"]) == 10
    assert {bucket["reasoning_mode"] for bucket in plan["trace_quantile_buckets"]} == {
        "no_think",
        "think_high",
    }
    assert len(plan["exact_replays"]) == 10
    assert (
        len(
            {
                (replay["trajectory_id"], replay["turn_index"])
                for replay in plan["exact_replays"]
            }
        )
        == 10
    )
    assert len(plan["cache_interleavings"]) == 40
    assert {interleaving["mode"] for interleaving in plan["cache_interleavings"]} == {
        "serial",
        "round_robin",
        "seeded_random",
    }
    assert {
        interleaving["session_concurrency"]
        for interleaving in plan["cache_interleavings"]
    } == {1, 2, 4, 8}
    assert {
        interleaving["reasoning_mode"] for interleaving in plan["cache_interleavings"]
    } == {"no_think", "think_high"}
    assert {
        interleaving["seed"]
        for interleaving in plan["cache_interleavings"]
        if interleaving["mode"] == "seeded_random"
    } == {20260715, 20260716, 20260717}
    assert all(
        len(interleaving["trajectory_ids"]) == interleaving["session_concurrency"]
        for interleaving in plan["cache_interleavings"]
    )

    assert {
        (batch["composition"], batch["batch_size"]) for batch in plan["mixed_batches"]
    } == {
        (composition, batch_size)
        for composition in ("similar", "random", "high_skew")
        for batch_size in (2, 4, 8)
    }
    assert all(
        batch["total_scheduled_tokens"]
        == sum(turn["new_prefill_tokens"] for turn in batch["turns"])
        <= 4096
        for batch in plan["mixed_batches"]
    )
    for batch_size in (2, 4, 8):
        similar = next(
            batch
            for batch in plan["mixed_batches"]
            if batch["composition"] == "similar" and batch["batch_size"] == batch_size
        )
        high_skew = next(
            batch
            for batch in plan["mixed_batches"]
            if batch["composition"] == "high_skew" and batch["batch_size"] == batch_size
        )
        similar_lengths = [turn["new_prefill_tokens"] for turn in similar["turns"]]
        high_skew_lengths = [turn["new_prefill_tokens"] for turn in high_skew["turns"]]
        assert max(high_skew_lengths) - min(high_skew_lengths) >= (
            max(similar_lengths) - min(similar_lengths)
        )


def test_artifact_build_is_compact_versioned_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    kwargs = {
        "manifest_path": SNAPSHOT_DIR / "manifest.json",
        "normalized_turns_path": TURNS_PATH,
        "ds4_tokenizer_path": TOKENIZER_DIR,
        "qwen_tokenizer_path": QWEN_TOKENIZER_DIR,
        "block_size": 16,
        "seed": 20260715,
    }

    build_artifacts(output_dir=first_output, **kwargs)
    build_artifacts(output_dir=second_output, **kwargs)

    for name in ("rendered_turns.parquet", "workload_plan.json", "provenance.json"):
        assert (first_output / name).read_bytes() == (second_output / name).read_bytes()

    table = pq.read_table(first_output / "rendered_turns.parquet")
    plan = json.loads((first_output / "workload_plan.json").read_text())
    provenance = json.loads((first_output / "provenance.json").read_text())
    assert table.num_rows == 667
    assert table.schema.metadata == {b"schema_version": b"1.0.0"}
    assert {
        "prompt_tokens",
        "completion_tokens",
        "exact_lcp_tokens",
        "reusable_prefix_tokens",
        "prefix_source_spans",
        "execution_prompt_token_ids",
        "execution_completion_token_ids",
    }.issubset(table.column_names)

    selected_keys = {
        (replay["trajectory_id"], replay["turn_index"])
        for replay in plan["exact_replays"]
    }
    selected_keys.update(
        (turn["trajectory_id"], turn["turn_index"])
        for batch in plan["mixed_batches"]
        for turn in batch["turns"]
    )
    rows = table.to_pylist()
    rows_with_execution_tokens = [
        row for row in rows if row["execution_prompt_token_ids"] is not None
    ]
    assert {
        (row["trajectory_id"], row["turn_index"]) for row in rows_with_execution_tokens
    } == selected_keys
    assert all(
        len(row["execution_prompt_token_ids"]) == row["prompt_tokens"]
        and len(row["execution_completion_token_ids"]) == row["completion_tokens"]
        for row in rows_with_execution_tokens
    )
    assert all(
        row["execution_prompt_token_ids"] is None
        and row["execution_completion_token_ids"] is None
        for row in rows
        if (row["trajectory_id"], row["turn_index"]) not in selected_keys
    )
    assert provenance == {
        "artifact_schema_version": "1.0.0",
        "block_size": 16,
        "dataset_revision": "4da61f3d06b48b6817a62b99e9c47035c8e59787",
        "ds4_tokenizer": {
            "repo_id": "deepseek-ai/DeepSeek-V4-Flash",
            "revision": "60d8d70770c6776ff598c94bb586a859a38244f1",
        },
        "execution_mapping_seed": 20260715,
        "manifest_sha256": hashlib.sha256(
            (SNAPSHOT_DIR / "manifest.json").read_bytes()
        ).hexdigest(),
        "normalized_turns_sha256": hashlib.sha256(TURNS_PATH.read_bytes()).hexdigest(),
        "qwen_tokenizer": {
            "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        },
        "rendered_turn_count": 667,
        "selected_execution_turn_count": len(selected_keys),
        "workload_seed": 20260715,
    }


def test_workload_cli_emits_public_artifact_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.workloads",
            "--manifest",
            str(SNAPSHOT_DIR / "manifest.json"),
            "--normalized-turns",
            str(TURNS_PATH),
            "--ds4-tokenizer",
            str(TOKENIZER_DIR),
            "--qwen-tokenizer",
            str(QWEN_TOKENIZER_DIR),
            "--output-dir",
            str(output_dir),
            "--block-size",
            "16",
            "--seed",
            "20260715",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert {path.name for path in output_dir.iterdir()} == {
        "provenance.json",
        "rendered_turns.parquet",
        "workload_plan.json",
    }
