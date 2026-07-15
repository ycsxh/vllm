# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pinned"


def test_pinned_fixture_emits_normalized_turn_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "normalized"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(FIXTURE_DIR / "manifest.json"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    table = pq.read_table(output_dir / "turns.parquet")
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert table.num_rows == 4
    assert set(table.column("reasoning_mode").to_pylist()) == {
        "no_think",
        "think_high",
    }
    assert provenance == {
        "dataset_repo_id": "Yi30/deepseek-v4-swebench-trajectories",
        "dataset_repo_type": "model",
        "dataset_revision": "4da61f3d06b48b6817a62b99e9c47035c8e59787",
        "file_count": 2,
        "parser_version": "1.0.0",
        "pilot_coverage": {
            "domains": ["astropy"],
            "reasoning_modes": ["no_think", "think_high"],
            "trajectory_count": 20,
            "unique_task_count": 10,
        },
        "schema_version": "1.0.0",
        "source_files": [
            {
                "path": "data/no_think/astropy__astropy-12907.traj.json",
                "sha256": (
                    "f75519fb12540da413a8b844bb6a975fa9635b12c82a743eb5995d7c4d145432"
                ),
            },
            {
                "path": "data/think_high/astropy__astropy-12907.traj.json",
                "sha256": (
                    "808fdd40dd5327611fbc86d74c94606387c3d6ed3cd1979d469e4b1ec61e9630"
                ),
            },
        ],
        "trajectory_count": 2,
        "turn_count": 4,
    }


def test_normalized_turn_preserves_tool_structure_and_source_metrics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "normalized"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(FIXTURE_DIR / "manifest.json"),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    table = pq.read_table(output_dir / "turns.parquet")
    assert table.schema.metadata == {
        b"dataset_revision": b"4da61f3d06b48b6817a62b99e9c47035c8e59787",
        b"parser_version": b"1.0.0",
        b"pilot_coverage": (
            b'{"domains":["astropy"],"reasoning_modes":["no_think",'
            b'"think_high"],"trajectory_count":20,"unique_task_count":10}'
        ),
        b"schema_version": b"1.0.0",
    }
    first_turn = table.to_pylist()[0]
    assert first_turn == {
        "assistant_content": "",
        "assistant_message_index": 2,
        "completion_tokens": 12,
        "dataset_revision": "4da61f3d06b48b6817a62b99e9c47035c8e59787",
        "message_count": 3,
        "parallel_tool_call_count": 2,
        "parser_version": "1.0.0",
        "prompt_tokens": 40,
        "reasoning_content": None,
        "reasoning_mode": "no_think",
        "reasoning_tokens": None,
        "schema_version": "1.0.0",
        "source_agent_type": "minisweagent.agents.default.DefaultAgent",
        "source_model": "hosted_vllm/deepseek-ai/DeepSeek-V4-Flash",
        "source_path": "data/no_think/astropy__astropy-12907.traj.json",
        "source_sha256": (
            "f75519fb12540da413a8b844bb6a975fa9635b12c82a743eb5995d7c4d145432"
        ),
        "source_trajectory_format": "mini-swe-agent-1.1",
        "task_id": "astropy__astropy-12907",
        "tool_call_count": 2,
        "tool_calls": [
            {
                "arguments": (
                    '{"command":"sed -n \'1,80p\' astropy/modeling/separable.py"}'
                ),
                "id": "call_read",
                "name": "bash",
                "type": "function",
            },
            {
                "arguments": (
                    '{"command":"rg \'separability_matrix\' astropy/modeling"}'
                ),
                "id": "call_search",
                "name": "bash",
                "type": "function",
            },
        ],
        "tool_duration_ms": 30.0,
        "tool_ready": True,
        "tool_results": [
            {
                "content": "def separability_matrix(...): ...",
                "duration_ms": 18.5,
                "tool_call_id": "call_read",
            },
            {
                "content": ("astropy/modeling/separable.py:42:def separability_matrix"),
                "duration_ms": 11.5,
                "tool_call_id": "call_search",
            },
        ],
        "total_tokens": 52,
        "trajectory_id": "astropy__astropy-12907:no_think",
        "turn_count": 2,
        "turn_index": 0,
    }


@pytest.mark.parametrize(
    ("config_key", "value", "error"),
    [
        (
            "model",
            {"model_name": "hosted_vllm/Qwen/Qwen3-32B"},
            "not a DS4 trajectory",
        ),
        (
            "agent_type",
            "sweagent.agent.agents.Agent",
            "not a mini-swe-agent trajectory",
        ),
    ],
)
def test_non_ds4_or_non_mini_swe_agent_source_is_rejected(
    tmp_path: Path,
    config_key: str,
    value: object,
    error: str,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = snapshot_dir / manifest["files"][0]["path"]
    trajectory = json.loads(source_path.read_text())
    trajectory["info"]["config"][config_key] = value
    source_path.write_text(json.dumps(trajectory))
    manifest["files"][0]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "normalized"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert error in result.stderr


def test_manifest_requires_immutable_revision_and_never_mutates_raw_source(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = snapshot_dir / manifest["files"][0]["path"]
    original_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest["dataset"]["revision"] = "main"
    manifest_path.write_text(json.dumps(manifest))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "normalized"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "dataset revision must be a full immutable commit" in result.stderr
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_sha256


def test_parallel_tool_call_is_not_ready_when_any_arguments_are_incomplete(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = snapshot_dir / manifest["files"][0]["path"]
    trajectory = json.loads(source_path.read_text())
    trajectory["messages"][2]["tool_calls"][1]["function"].pop("arguments")
    source_path.write_text(json.dumps(trajectory))
    manifest["files"][0]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    output_dir = tmp_path / "normalized"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    first_turn = pq.read_table(output_dir / "turns.parquet").to_pylist()[0]
    assert first_turn["tool_call_count"] == 2
    assert first_turn["parallel_tool_call_count"] == 2
    assert first_turn["tool_calls"][1]["arguments"] is None
    assert first_turn["tool_ready"] is False


def test_non_mini_swe_agent_trajectory_format_is_rejected(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = snapshot_dir / manifest["files"][0]["path"]
    trajectory = json.loads(source_path.read_text())
    trajectory["trajectory_format"] = "swe-agent-1.1"
    source_path.write_text(json.dumps(trajectory))
    manifest["files"][0]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "normalized"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "unsupported trajectory format" in result.stderr


def test_unknown_reasoning_mode_is_rejected(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original_path = snapshot_dir / manifest["files"][0]["path"]
    baseline_path = snapshot_dir / "data/baseline" / original_path.name
    baseline_path.parent.mkdir(parents=True)
    original_path.replace(baseline_path)
    manifest["files"][0]["path"] = str(baseline_path.relative_to(snapshot_dir))
    manifest_path.write_text(json.dumps(manifest))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "normalized"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "unsupported reasoning mode: baseline" in result.stderr


def test_outputs_are_deterministic_and_source_hashes_are_enforced(
    tmp_path: Path,
) -> None:
    output_dirs = [tmp_path / "first", tmp_path / "second"]
    source_paths = [
        FIXTURE_DIR / entry["path"]
        for entry in json.loads((FIXTURE_DIR / "manifest.json").read_text())["files"]
    ]
    source_hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths
    ]

    for output_dir in output_dirs:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.ds4_profile.normalize",
                "--manifest",
                str(FIXTURE_DIR / "manifest.json"),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    assert (output_dirs[0] / "turns.parquet").read_bytes() == (
        output_dirs[1] / "turns.parquet"
    ).read_bytes()
    assert (output_dirs[0] / "provenance.json").read_bytes() == (
        output_dirs[1] / "provenance.json"
    ).read_bytes()
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths] == (
        source_hashes
    )

    snapshot_dir = tmp_path / "tampered"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    source_path = snapshot_dir / "data/no_think/astropy__astropy-12907.traj.json"
    source_path.write_text(source_path.read_text() + "\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(snapshot_dir / "manifest.json"),
            "--output-dir",
            str(tmp_path / "tampered-output"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "SHA-256 mismatch" in result.stderr


def test_any_non_ds4_assistant_response_is_rejected(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = snapshot_dir / manifest["files"][0]["path"]
    trajectory = json.loads(source_path.read_text())
    trajectory["messages"][2]["extra"]["response"]["model"] = "Qwen/Qwen3-32B"
    source_path.write_text(json.dumps(trajectory))
    manifest["files"][0]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "normalized"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "assistant response is not from DS4" in result.stderr


def test_complete_pilot_requires_all_20_paired_trajectories(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(FIXTURE_DIR / "manifest.json"),
            "--output-dir",
            str(tmp_path / "normalized"),
            "--require-complete-pilot",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "complete pilot requires 20 trajectories, found 2" in result.stderr


def test_null_completion_token_details_are_preserved_as_missing(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_DIR, snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_path = snapshot_dir / manifest["files"][0]["path"]
    trajectory = json.loads(source_path.read_text())
    usage = trajectory["messages"][2]["extra"]["response"]["usage"]
    usage["completion_tokens_details"] = None
    source_path.write_text(json.dumps(trajectory))
    manifest["files"][0]["sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    output_dir = tmp_path / "normalized"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.normalize",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    first_turn = pq.read_table(output_dir / "turns.parquet").to_pylist()[0]
    assert first_turn["reasoning_tokens"] is None
