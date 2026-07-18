# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_fixture_inputs(tmp_path: Path) -> Path:
    rendered_turns_path = tmp_path / "rendered_turns.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "trajectory_id": "task:no_think",
                    "turn_index": 3,
                    "execution_prompt_token_ids": list(range(256)),
                    "execution_completion_token_ids": list(range(301, 321)),
                }
            ]
        ),
        rendered_turns_path,
    )
    workload_plan_path = tmp_path / "workload_plan.json"
    workload_plan_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "exact_replays": [
                    {
                        "trajectory_id": "task:no_think",
                        "turn_index": 3,
                        "prompt_tokens": 256,
                    }
                ],
            }
        )
    )
    config_path = tmp_path / "profile-spine.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "fixture-run",
                "artifacts": {
                    "rendered_turns": str(rendered_turns_path),
                    "workload_plan": str(workload_plan_path),
                },
                "model": {
                    "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "revision": "model-revision",
                    "tokenizer": "/models/qwen-tokenizer",
                },
                "profile": {
                    "block_size": 16,
                    "max_num_batched_tokens": 4096,
                    "measured_repetitions": 10,
                    "noisy_cv_threshold": 0.05,
                    "prefill_chunk_tokens": 128,
                    "warmup_repetitions": 3,
                },
                "roles": {"decode": {"gpu": 1}, "prefill": {"gpu": 0}},
                "source": {"commit": "abc123", "dirty": False},
            }
        )
    )
    return config_path


def test_fixture_cli_emits_the_complete_versioned_artifact_contract(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "result"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "fixture",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert {path.name for path in output_dir.iterdir()} == {
        "aggregates.parquet",
        "provenance.json",
        "raw_samples.parquet",
        "result.md",
        "run-config.json",
    }

    raw = pq.read_table(output_dir / "raw_samples.parquet")
    aggregates = pq.read_table(output_dir / "aggregates.parquet")
    assert raw.schema.metadata == {b"schema_version": b"1.0.0"}
    assert aggregates.schema.metadata == {b"schema_version": b"1.0.0"}
    assert raw.num_rows == 30
    assert aggregates.num_rows == 2

    raw_rows = raw.to_pylist()
    aggregate_rows = aggregates.to_pylist()
    assert {row["run_id"] for row in raw_rows + aggregate_rows} == {"fixture-run"}
    assert {row["point_id"] for row in raw_rows} == {
        "decode-b1-t1",
        "prefill-b1-t128",
    }
    assert {row["point_id"] for row in aggregate_rows} == {
        "decode-b1-t1",
        "prefill-b1-t128",
    }
    assert len({row["sample_id"] for row in raw_rows}) == raw.num_rows
    assert all(row["sample_count"] == 10 for row in aggregate_rows)

    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["artifact_schema_version"] == "1.0.0"
    assert provenance["hardware_validated"] is False
    assert provenance["run_id"] == "fixture-run"
    assert provenance["status"] == "fixture-only"
    assert provenance["source"] == {"commit": "abc123", "dirty": False}

    frozen_config = json.loads((output_dir / "run-config.json").read_text())
    assert frozen_config["run_id"] == "fixture-run"
    assert frozen_config["profile"]["measured_repetitions"] == 10
    report = (output_dir / "result.md").read_text()
    assert "Fixture only" in report
    assert "Hardware validated: no" in report
    assert "raw_samples.parquet" in report
    assert "aggregates.parquet" in report


def test_fixture_records_sampled_token_discard_and_teacher_forced_injection(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "result"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "fixture",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rows = pq.read_table(output_dir / "raw_samples.parquet").to_pylist()
    decode_steps = [
        row
        for row in rows
        if row["role"] == "decode" and row["phase"] in {"warmup", "steady"}
    ]
    assert [row["injected_token_id"] for row in decode_steps] == list(range(302, 315))
    assert all(row["sampled_token_id"] is not None for row in decode_steps)
    assert all(
        row["sampled_token_id"] != row["injected_token_id"] for row in decode_steps
    )
    assert all(row["injected_token_id"] is None for row in rows[:2])


def test_validate_cli_rejects_cross_artifact_identifier_mismatch(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "result"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "fixture",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    valid = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "validate",
            "--result-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr

    raw_path = output_dir / "raw_samples.parquet"
    raw = pq.read_table(raw_path)
    run_ids = pa.array(["wrong-run"] * raw.num_rows, type=pa.string())
    run_id_index = raw.schema.get_field_index("run_id")
    tampered = raw.set_column(run_id_index, raw.schema.field(run_id_index), run_ids)
    pq.write_table(tampered, raw_path)

    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "validate",
            "--result-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert invalid.returncode == 2
    assert "run_id" in invalid.stderr


def test_gpu_worker_plan_targets_the_low_level_compiled_teacher_forced_path(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    plan_path = tmp_path / "decode-plan.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "gpu-worker",
            "--config",
            str(config_path),
            "--role",
            "decode",
            "--output",
            str(plan_path),
            "--inspect-plan",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(plan_path.read_text())
    assert plan["runner_boundary"] == ("vllm.v1.worker.gpu_worker.Worker.execute_model")
    assert plan["engine"]["enforce_eager"] is False
    assert plan["engine"]["compilation_mode"] == "VLLM_COMPILE"
    assert plan["engine"]["cudagraph_enabled"] is True
    assert plan["setup_prefill_chunks"] == [256]
    assert plan["scheduled_tokens_per_step"] == 1
    assert plan["initial_teacher_forced_token_id"] == 301
    assert plan["teacher_forced_token_ids"] == list(range(302, 315))
    assert plan["sample_phases"] == {"steady": 10, "warmup": 3}


def test_gpu_worker_failure_is_recorded_without_claiming_hardware_validation(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    worker_path = tmp_path / "worker.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "gpu-worker",
            "--config",
            str(config_path),
            "--role",
            "decode",
            "--output",
            str(worker_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    worker = json.loads(worker_path.read_text())
    assert worker["hardware_validated"] is False
    assert worker["role"] == "decode"
    assert worker["runner_boundary"] == (
        "vllm.v1.worker.gpu_worker.Worker.execute_model"
    )
    assert worker["status"] == "failed"
    assert worker["samples"] == []
    assert "error" in worker


def _write_passed_worker_results(
    tmp_path: Path, config_path: Path
) -> tuple[list[Path], Path]:
    fixture_dir = tmp_path / "fixture"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "fixture",
            "--config",
            str(config_path),
            "--output-dir",
            str(fixture_dir),
        ],
        check=True,
    )
    rows = pq.read_table(fixture_dir / "raw_samples.parquet").to_pylist()
    worker_paths = []
    for role in ("prefill", "decode"):
        role_rows = [row for row in rows if row["role"] == role]
        for row in role_rows:
            row["runner_boundary"] = "vllm.v1.worker.gpu_worker.Worker.execute_model"
        worker_path = tmp_path / f"{role}.json"
        worker_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "run_id": "fixture-run",
                    "hardware_validated": True,
                    "role": role,
                    "runner_boundary": (
                        "vllm.v1.worker.gpu_worker.Worker.execute_model"
                    ),
                    "model_runner_implementation": "GPUModelRunner",
                    "compile_enabled": True,
                    "cudagraph_enabled": True,
                    "cudagraph_observations": ["FULL"],
                    "samples": role_rows,
                    "status": "passed",
                }
            )
        )
        worker_paths.append(worker_path)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "hardware": {"driver": "test-driver"},
                "checks": {"nvidia_smi": {"status": "passed"}},
            }
        )
    )
    return worker_paths, preflight_path


def test_assemble_cli_merges_both_gpu_roles_into_the_public_artifact_contract(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    worker_paths, preflight_path = _write_passed_worker_results(tmp_path, config_path)
    output_dir = tmp_path / "gpu-result"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "assemble",
            "--config",
            str(config_path),
            "--preflight",
            str(preflight_path),
            "--worker-result",
            str(worker_paths[0]),
            "--worker-result",
            str(worker_paths[1]),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    raw = pq.read_table(output_dir / "raw_samples.parquet")
    aggregates = pq.read_table(output_dir / "aggregates.parquet")
    assert raw.num_rows == 30
    assert aggregates.num_rows == 2
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["hardware_validated"] is True
    assert provenance["status"] == "passed"
    assert provenance["preflight"]["hardware"]["driver"] == "test-driver"
    assert {worker["role"] for worker in provenance["workers"]} == {
        "decode",
        "prefill",
    }
    report = (output_dir / "result.md").read_text()
    assert "Hardware validated: yes" in report


def test_assemble_cli_rejects_a_worker_with_incomplete_steady_samples(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    worker_paths, preflight_path = _write_passed_worker_results(tmp_path, config_path)
    decode = json.loads(worker_paths[1].read_text())
    decode["samples"] = decode["samples"][:-1]
    worker_paths[1].write_text(json.dumps(decode))
    output_dir = tmp_path / "invalid-gpu-result"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "assemble",
            "--config",
            str(config_path),
            "--preflight",
            str(preflight_path),
            "--worker-result",
            str(worker_paths[0]),
            "--worker-result",
            str(worker_paths[1]),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["hardware_validated"] is False
    assert provenance["status"] == "invalid"


def test_container_profile_spine_plan_binds_both_workers_and_assembles_results(
    tmp_path: Path,
) -> None:
    profile_config = _write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "ticket-04-result"
    container_config = (
        Path(__file__).parents[3]
        / "benchmarks/ds4_profile/config/container-contract.json"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "profile-spine",
            "--config",
            str(container_config),
            "--profile-config",
            str(profile_config),
            "--output-dir",
            str(output_dir),
            "--print-plan",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("benchmarks.ds4_profile.profile_spine gpu-worker") == 2
    assert "--role prefill" in result.stdout
    assert "--role decode" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=1" in result.stdout
    assert "--membind=0" in result.stdout
    assert "--membind=1" in result.stdout
    assert "benchmarks.ds4_profile.profile_spine assemble" in result.stdout
    assert str(output_dir) in result.stdout
    assert not output_dir.exists()


def test_server_profile_config_pins_inputs_and_production_measurement_rules() -> None:
    config_path = (
        Path(__file__).parents[3] / "benchmarks/ds4_profile/config/profile-spine.json"
    )

    config = json.loads(config_path.read_text())

    assert config["schema_version"] == "1.0.0"
    assert config["run_id"] is None
    assert config["artifacts"] == {
        "rendered_turns": "/mnt/ds4/ticket-02/rendered_turns.parquet",
        "workload_plan": "/mnt/ds4/ticket-02/workload_plan.json",
    }
    assert config["model"] == {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "tokenizer": (
            "/mnt/ds4/tokenizers/qwen2.5-coder-7b-instruct/"
            "c03e6d358207e414f1eca0bb1891e29f1db0e242"
        ),
    }
    assert config["profile"] == {
        "block_size": 16,
        "max_num_batched_tokens": 4096,
        "measured_repetitions": 10,
        "noisy_cv_threshold": 0.05,
        "prefill_chunk_tokens": 128,
        "warmup_repetitions": 3,
    }


@pytest.mark.skipif(
    os.environ.get("DS4_PROFILE_SPINE_GPU_SMOKE") != "1",
    reason="requires the documented dual-RTX-3090 container runtime",
)
def test_profile_spine_executes_both_gpu_roles_without_latency_thresholds(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "gpu-profile-spine"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "profile-spine",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    raw_rows = pq.read_table(output_dir / "raw_samples.parquet").to_pylist()
    assert {row["role"] for row in raw_rows} == {"decode", "prefill"}
    assert sum(row["phase"] == "steady" for row in raw_rows) == 20
    decode_rows = [
        row
        for row in raw_rows
        if row["role"] == "decode" and row["phase"] in {"warmup", "steady"}
    ]
    assert len(decode_rows) == 13
    assert all(row["sampled_token_id"] is not None for row in decode_rows)
    assert all(
        row["sampled_token_id"] != row["injected_token_id"] for row in decode_rows
    )
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["hardware_validated"] is True
    assert provenance["status"] == "passed"
    assert all(worker["compile_enabled"] for worker in provenance["workers"])
    assert all(worker["cudagraph_enabled"] for worker in provenance["workers"])
    assert all(
        worker["runner_boundary"] == "vllm.v1.worker.gpu_worker.Worker.execute_model"
        for worker in provenance["workers"]
    )
