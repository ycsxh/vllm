# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


def test_v2_point_id_covers_every_workload_dimension() -> None:
    from benchmarks.ds4_profile import profile_spine

    payload = {
        "workload_family": "homogeneous",
        "selector": "b2-t512",
        "requests": [{
            "request_key": "r0",
            "trajectory_id": None,
            "turn_index": None,
            "reasoning_mode": None,
            "context_tokens": 4608,
            "cached_tokens": 4096,
            "new_tokens": 512,
            "token_digest": "a" * 64,
        }],
        "composition": "none",
        "seed": 20260715,
        "batch_size": 1,
        "chunk_budget": 4096,
        "cache_condition": "prefix_hit",
        "block_size": 16,
        "homogeneous_prefix_tokens": 4096,
        "capacity_target": "native",
        "planner_digest": "b" * 64,
        "planned_chunks": [{
            "chunk_index": 0,
            "scheduled_tokens_by_request": [["r0", 512]],
        }],
    }
    original = profile_spine.make_point_id(payload)
    changed = copy.deepcopy(payload)
    changed["requests"][0]["cached_tokens"] = 4080
    assert original.startswith("p2-")
    assert profile_spine.make_point_id(changed) != original
    assert profile_spine.make_comparison_id(
        payload
    ) == profile_spine.make_comparison_id(
        {
            **payload,
            "cache_condition": "full_recompute",
            "planned_chunks": [
                {"chunk_index": 0, "scheduled_tokens_by_request": [["r0", 4096]]},
                {"chunk_index": 1, "scheduled_tokens_by_request": [["r0", 512]]},
            ],
        }
    )
    changed_chunk = copy.deepcopy(payload)
    changed_chunk["planned_chunks"][0]["scheduled_tokens_by_request"][0][1] = 511
    assert profile_spine.make_point_id(changed_chunk) != original
    changed_planner = {**payload, "planner_digest": "c" * 64}
    assert profile_spine.make_point_id(changed_planner) != original


def _v2_row(schema: pa.Schema, **values: object) -> dict[str, object]:
    row: dict[str, object] = {}
    for field in schema:
        if pa.types.is_string(field.type):
            row[field.name] = ""
        elif pa.types.is_boolean(field.type):
            row[field.name] = False
        elif pa.types.is_integer(field.type):
            row[field.name] = 0
        elif pa.types.is_floating(field.type):
            row[field.name] = 0.0
        elif pa.types.is_list(field.type):
            row[field.name] = []
        else:
            raise AssertionError(f"fixture default missing for {field}")
    row.update(values)
    return row


def _write_v2_result(tmp_path: Path) -> Path:
    from benchmarks.ds4_profile import profile_spine

    output_dir = tmp_path / "v2-result"
    output_dir.mkdir()
    points = []
    for index in range(34):
        for cache_condition in ("prefix_hit", "full_recompute"):
            payload = {
                "workload_family": "homogeneous",
                "selector": f"fixture-{index}",
                "requests": [{
                    "request_key": "r0",
                    "trajectory_id": None,
                    "turn_index": None,
                    "reasoning_mode": None,
                    "context_tokens": 512,
                    "cached_tokens": 0,
                    "new_tokens": 512,
                    "token_digest": f"{index:064x}",
                }],
                "composition": "none",
                "seed": 1,
                "batch_size": 1,
                "chunk_budget": 4096,
                "cache_condition": cache_condition,
                "block_size": 16,
                "homogeneous_prefix_tokens": 4096,
                "capacity_target": "native",
                "planner_digest": "b" * 64,
                "planned_chunks": [{
                    "chunk_index": 0,
                    "scheduled_tokens_by_request": [["r0", 512]],
                }],
            }
            points.append(
                {
                    "point_id": profile_spine.make_point_id(payload),
                    "comparison_id": profile_spine.make_comparison_id(payload),
                    "canonical_payload": payload,
                }
            )
    point_ids = [point["point_id"] for point in points]
    config = {
        "schema_version": "2.0.0",
        "run_id": "v2-fixture",
        "run_kind": "full",
        "points": points,
        "canonical_full_manifest": point_ids,
        "expected_manifest": point_ids,
        "profile": {"noisy_cv_threshold": 0.05},
    }
    raw_rows = []
    turn_rows = []
    evidence_rows = []
    aggregate_rows = []
    for point in points:
        payload = point["canonical_payload"]
        for phase, count in (("warmup", 3), ("steady", 10)):
            for ordinal in range(count):
                elapsed = 1.0 if phase == "warmup" else 10.0 + ordinal
                raw_rows.append(
                    _v2_row(
                        profile_spine.V2_RAW_SAMPLE_SCHEMA,
                        schema_version="2.0.0",
                        run_id="v2-fixture",
                        point_id=point["point_id"],
                        comparison_id=point["comparison_id"],
                        sample_id=(
                            f"v2-fixture:{point['point_id']}:{phase}:{ordinal}:0"
                        ),
                        role="prefill",
                        workload_family="homogeneous",
                        selector=payload["selector"],
                        composition="none",
                        cache_condition=payload["cache_condition"],
                        planner_digest=payload["planner_digest"],
                        phase=phase,
                        ordinal=ordinal,
                        chunk_count=1,
                        row_kind="chunk",
                        status="passed",
                        allocation_state="allocated",
                        planned_scheduled_tokens_by_request=[
                            {"request_key": "r0", "scheduled_tokens": 512}
                        ],
                        actual_scheduled_tokens_by_request=[
                            {"request_key": "r0", "scheduled_tokens": 512}
                        ],
                        cache_reset_completed=True,
                        cache_reset_empty=True,
                        requested_kv_blocks=1,
                        allocatable_kv_blocks=1,
                        allocated_kv_blocks=1,
                        kv_block_bytes=1,
                        requested_kv_bytes=1,
                        allocated_kv_bytes=1,
                        scheduled_tokens=512,
                        context_tokens=512,
                        new_tokens=512,
                        runner_wall_time_ms=elapsed,
                        cuda_model_time_ms=elapsed,
                        runtime_mode="FULL",
                    )
                )
                turn_rows.append(
                    _v2_row(
                        profile_spine.V2_TURN_SAMPLE_SCHEMA,
                        schema_version="2.0.0",
                        run_id="v2-fixture",
                        point_id=point["point_id"],
                        comparison_id=point["comparison_id"],
                        sample_id=f"v2-fixture:{point['point_id']}:{phase}:{ordinal}",
                        role="prefill",
                        workload_family="homogeneous",
                        selector=payload["selector"],
                        composition="none",
                        cache_condition=payload["cache_condition"],
                        planner_digest=payload["planner_digest"],
                        phase=phase,
                        ordinal=ordinal,
                        status="passed",
                        allocation_state="allocated",
                        chunk_count=1,
                        scheduled_tokens=512,
                        context_tokens=512,
                        new_tokens=512,
                        requested_kv_blocks=1,
                        allocated_kv_blocks=1,
                        requested_kv_bytes=1,
                        allocated_kv_bytes=1,
                        runner_wall_time_ms=elapsed,
                        cuda_model_time_ms=elapsed,
                        throughput_tokens_per_s=100.0 + ordinal,
                        runtime_mode="FULL",
                    )
                )
                if payload["cache_condition"] == "prefix_hit":
                    evidence_rows.append(
                        _v2_row(
                            profile_spine.V2_PREFIX_EVIDENCE_SCHEMA,
                            schema_version="2.0.0",
                            run_id="v2-fixture",
                            point_id=point["point_id"],
                            phase=phase,
                            ordinal=ordinal,
                            request_key="r0",
                            kv_cache_group="0",
                            prime_completed=True,
                            prime_synchronized=True,
                        )
                    )
        values = list(range(10, 20))
        throughput = list(range(100, 110))
        mean = sum(values) / len(values)
        throughput_mean = sum(throughput) / len(throughput)
        aggregate_rows.append(
            _v2_row(
                profile_spine.V2_AGGREGATE_SCHEMA,
                schema_version="2.0.0",
                run_id="v2-fixture",
                point_id=point["point_id"],
                comparison_id=point["comparison_id"],
                role="prefill",
                sample_count=10,
                runner_wall_time_median_ms=14.5,
                runner_wall_time_p90_ms=18.1,
                runner_wall_time_mean_ms=mean,
                runner_wall_time_cv=statistics.pstdev(values) / mean,
                throughput_median_tokens_per_s=104.5,
                throughput_p90_tokens_per_s=108.1,
                throughput_mean_tokens_per_s=throughput_mean,
                throughput_cv=statistics.pstdev(throughput) / throughput_mean,
            )
        )
    comparison_rows = []
    for index in range(34):
        hit = points[2 * index]
        miss = points[2 * index + 1]
        comparison_rows.append(
            _v2_row(
                profile_spine.V2_COMPARISON_SCHEMA,
                schema_version="2.0.0",
                run_id="v2-fixture",
                comparison_id=hit["comparison_id"],
                prefix_hit_point_id=hit["point_id"],
                full_recompute_point_id=miss["point_id"],
                prefix_hit_median_ms=14.5,
                full_recompute_median_ms=14.5,
                recompute_penalty_ms=0.0,
                recompute_penalty_ratio=1.0,
            )
        )
    for name, rows, schema in (
        ("raw_samples.parquet", raw_rows, profile_spine.V2_RAW_SAMPLE_SCHEMA),
        ("turn_samples.parquet", turn_rows, profile_spine.V2_TURN_SAMPLE_SCHEMA),
        ("aggregates.parquet", aggregate_rows, profile_spine.V2_AGGREGATE_SCHEMA),
        ("comparisons.parquet", comparison_rows, profile_spine.V2_COMPARISON_SCHEMA),
        (
            "prefix_evidence.parquet",
            evidence_rows,
            profile_spine.V2_PREFIX_EVIDENCE_SCHEMA,
        ),
    ):
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), output_dir / name)
    (output_dir / "run-config.json").write_text(json.dumps(config))
    (output_dir / "provenance.json").write_text(json.dumps({"run_id": "v2-fixture"}))
    (output_dir / "result.md").write_text("fixture\n")
    return output_dir


def test_v1_result_still_validates(tmp_path: Path) -> None:
    from benchmarks.ds4_profile import profile_spine

    config_path = _write_fixture_inputs(tmp_path)
    output_dir = tmp_path / "v1-result"
    profile_spine._write_fixture_result(config_path, output_dir)
    profile_spine._validate_result_dir(output_dir)


def test_v2_validator_rejects_unknown_schema_version(tmp_path: Path) -> None:
    from benchmarks.ds4_profile import profile_spine

    output_dir = _write_v2_result(tmp_path)
    config = json.loads((output_dir / "run-config.json").read_text())
    config["schema_version"] = "3.0.0"
    (output_dir / "run-config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unsupported schema_version"):
        profile_spine._validate_result_dir(output_dir)


def test_v2_validator_rejects_each_unknown_enum(tmp_path: Path) -> None:
    from benchmarks.ds4_profile import profile_spine

    output_dir = _write_v2_result(tmp_path)
    raw = pq.read_table(output_dir / "raw_samples.parquet").to_pylist()
    raw[0]["role"] = "decode"
    pq.write_table(
        pa.Table.from_pylist(raw, schema=profile_spine.V2_RAW_SAMPLE_SCHEMA),
        output_dir / "raw_samples.parquet",
    )
    with pytest.raises(ValueError, match="unknown role"):
        profile_spine._validate_result_dir(output_dir)


def test_v2_validator_recomputes_aggregate_statistics(tmp_path: Path) -> None:
    from benchmarks.ds4_profile import profile_spine

    output_dir = _write_v2_result(tmp_path)
    aggregates = pq.read_table(output_dir / "aggregates.parquet").to_pylist()
    aggregates[0]["runner_wall_time_median_ms"] = 1.0
    pq.write_table(
        pa.Table.from_pylist(aggregates, schema=profile_spine.V2_AGGREGATE_SCHEMA),
        output_dir / "aggregates.parquet",
    )
    with pytest.raises(
        ValueError, match="aggregate statistics do not match turn samples"
    ):
        profile_spine._validate_result_dir(output_dir)


def test_v2_validator_requires_exact_manifest_and_coordinates(tmp_path: Path) -> None:
    from benchmarks.ds4_profile import profile_spine

    output_dir = _write_v2_result(tmp_path)
    raw = pq.read_table(output_dir / "raw_samples.parquet").to_pylist()
    raw.pop(0)
    pq.write_table(
        pa.Table.from_pylist(raw, schema=profile_spine.V2_RAW_SAMPLE_SCHEMA),
        output_dir / "raw_samples.parquet",
    )
    with pytest.raises(ValueError, match="exact planned chunk coordinates"):
        profile_spine._validate_result_dir(output_dir)


def _fake_gpu_worker(
    *, state_token_id: int, cached_token_id: int | None
) -> SimpleNamespace:
    prev_sampled_token_ids = (
        None if cached_token_id is None else torch.tensor([[cached_token_id]])
    )
    input_batch = SimpleNamespace(
        num_prompt_tokens=[2],
        prev_sampled_token_ids=prev_sampled_token_ids,
        req_id_to_index={"request": 0},
        token_ids_cpu=torch.zeros((1, 4), dtype=torch.int64),
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        input_ids=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int64)),
        requests={"request": SimpleNamespace(output_token_ids=[state_token_id])},
    )
    return SimpleNamespace(model_runner=runner)


def test_prefill_sample_detection_checks_inner_request_tokens() -> None:
    from benchmarks.ds4_profile.gpu_profile import _has_sampled_tokens

    assert not _has_sampled_tokens(SimpleNamespace(sampled_token_ids=[[]]))
    assert _has_sampled_tokens(SimpleNamespace(sampled_token_ids=[[17]]))


def test_config_requires_cuda_graph_capture_for_both_profile_points() -> None:
    from benchmarks.ds4_profile.profile_spine import _resolve_config

    config = {
        "profile": {"prefill_chunk_tokens": 128},
        "runtime": {
            "compilation": {
                "capture_sizes": [1],
                "compile_sizes": [1, 128],
            }
        },
    }
    replay = {
        "execution_completion_token_ids": [2],
        "execution_prompt_token_ids": [1],
    }

    with pytest.raises(ValueError, match="capture_sizes"):
        _resolve_config(config, replay)


@pytest.mark.parametrize(
    ("state_token_id", "cached_token_id"),
    [(17, None), (-1, 17)],
)
def test_teacher_forcing_replaces_sync_and_async_token_state(
    state_token_id: int, cached_token_id: int | None
) -> None:
    from benchmarks.ds4_profile.gpu_profile import _inject_teacher_forced_token

    worker = _fake_gpu_worker(
        state_token_id=state_token_id,
        cached_token_id=cached_token_id,
    )
    _inject_teacher_forced_token(worker, "request", 17, 23)

    runner = worker.model_runner
    assert runner.requests["request"].output_token_ids == [23]
    assert runner.input_batch.token_ids_cpu[0, 2].item() == 23
    if cached_token_id is not None:
        assert runner.input_batch.prev_sampled_token_ids[0, 0].item() == 23


def test_next_gpu_input_must_consume_the_injected_token() -> None:
    from benchmarks.ds4_profile.gpu_profile import (
        _assert_teacher_forced_input,
        _inject_teacher_forced_token,
    )

    worker = _fake_gpu_worker(state_token_id=-1, cached_token_id=17)
    _inject_teacher_forced_token(worker, "request", 17, 23)
    runner = worker.model_runner

    runner.input_ids.gpu.copy_(runner.input_batch.prev_sampled_token_ids[:, 0])
    _assert_teacher_forced_input(worker, "request", 23)

    runner.input_ids.gpu[0] = 17
    with pytest.raises(RuntimeError, match="did not consume"):
        _assert_teacher_forced_input(worker, "request", 23)


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
                "runtime": {
                    "compilation": {
                        "capture_sizes": [1, 128],
                        "compile_sizes": [1, 128],
                        "cudagraph_mode": "FULL_AND_PIECEWISE",
                        "mode": "VLLM_COMPILE",
                    },
                    "dtype": "half",
                    "enable_chunked_prefill": True,
                    "enable_prefix_caching": True,
                    "enforce_eager": False,
                    "gpu_memory_utilization": 0.9,
                    "kv_cache_dtype": "auto",
                    "max_num_seqs": 1,
                    "sampling": {"max_tokens": 1, "temperature": 0.0},
                    "seed": 0,
                    "skip_tokenizer_init": True,
                    "tensor_parallel_size": 1,
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
    assert all(row["sampled_token_discarded"] for row in decode_steps)
    assert all(row["new_tokens"] == 1 for row in decode_steps)
    assert all(row["cached_tokens"] > 0 for row in decode_steps)
    assert all(row["context_tokens"] > row["prompt_tokens"] for row in decode_steps)
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
    assert plan["engine"]["block_size"] == 16
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
    assert worker["samples"][0]["status"] == "failed"
    assert worker["samples"][0]["phase"] == "startup"
    assert worker["failure"]["point_id"] == "decode-b1-t1"
    assert "error" in worker["failure"]


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
            if row["phase"] in {"warmup", "steady"}:
                row["cudagraph_runtime_mode"] = (
                    "FULL" if role == "decode" else "PIECEWISE"
                )
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
                    "cudagraph_observations": [
                        {
                            "num_padded_tokens": 1 if role == "decode" else 128,
                            "num_paddings": 0,
                            "num_unpadded_tokens": 1 if role == "decode" else 128,
                            "runtime_mode": (
                                "FULL" if role == "decode" else "PIECEWISE"
                            ),
                        }
                    ],
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
    assert provenance["run_parameters"]["runtime"]["dtype"] == "half"
    assert provenance["run_parameters"]["runtime"]["effective_max_model_len"] == 8192
    assert {worker["role"] for worker in provenance["workers"]} == {
        "decode",
        "prefill",
    }
    report = (output_dir / "result.md").read_text()
    assert "Hardware validated: yes" in report
    frozen_config = json.loads((output_dir / "run-config.json").read_text())
    assert frozen_config["runtime"]["effective_max_model_len"] == 8192


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


def test_assemble_cli_rejects_missing_cudagraph_runtime_evidence(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    worker_paths, preflight_path = _write_passed_worker_results(tmp_path, config_path)
    decode = json.loads(worker_paths[1].read_text())
    decode["cudagraph_observations"] = []
    for row in decode["samples"]:
        row["cudagraph_runtime_mode"] = None
    worker_paths[1].write_text(json.dumps(decode))
    output_dir = tmp_path / "invalid-cudagraph-result"

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


def test_assemble_accepts_a_discarded_sample_that_matches_the_injected_token(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    worker_paths, preflight_path = _write_passed_worker_results(tmp_path, config_path)
    decode = json.loads(worker_paths[1].read_text())
    decode_step = next(
        row for row in decode["samples"] if row["phase"] in {"warmup", "steady"}
    )
    decode_step["sampled_token_id"] = decode_step["injected_token_id"]
    worker_paths[1].write_text(json.dumps(decode))
    output_dir = tmp_path / "equal-token-result"

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
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["hardware_validated"] is True


def test_assemble_cli_preserves_partial_rows_and_failed_point(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture_inputs(tmp_path)
    worker_paths, preflight_path = _write_passed_worker_results(tmp_path, config_path)
    decode = json.loads(worker_paths[1].read_text())
    decode["status"] = "failed"
    decode["hardware_validated"] = False
    decode["samples"] = decode["samples"][:5]
    decode["samples"][-1]["status"] = "failed"
    decode["samples"][-1]["phase"] = "setup"
    decode["samples"][-1]["ordinal"] = 0
    decode["samples"][-1]["sample_id"] = "fixture-run:decode-b1-t1:setup:0"
    decode["samples"][-1]["error"] = "RuntimeError: injected failure"
    decode["failure"] = {
        "error": "RuntimeError: injected failure",
        "ordinal": 0,
        "phase": "setup",
        "point_id": "decode-b1-t1",
    }
    worker_paths[1].write_text(json.dumps(decode))
    output_dir = tmp_path / "partial-result"

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

    assert result.returncode == 2, result.stderr
    rows = pq.read_table(output_dir / "raw_samples.parquet").to_pylist()
    failed = [row for row in rows if row["status"] == "failed"]
    assert len(rows) == 20
    assert failed[0]["point_id"] == "decode-b1-t1"
    assert failed[0]["phase"] == "setup"
    assert failed[0]["error"] == "RuntimeError: injected failure"


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
    assert config["runtime"] == {
        "compilation": {
            "capture_sizes": [1, 128],
            "compile_sizes": [1, 128],
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "mode": "VLLM_COMPILE",
        },
        "dtype": "half",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "gpu_memory_utilization": 0.9,
        "kv_cache_dtype": "auto",
        "max_num_seqs": 1,
        "sampling": {"max_tokens": 1, "temperature": 0.0},
        "seed": 0,
        "skip_tokenizer_init": True,
        "tensor_parallel_size": 1,
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
    assert all(row["sampled_token_discarded"] for row in decode_rows)
    assert all(
        row["cudagraph_runtime_mode"] in {"FULL", "PIECEWISE"} for row in decode_rows
    )
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["hardware_validated"] is True
    assert provenance["status"] == "passed"
    assert all(worker["compile_enabled"] for worker in provenance["workers"])
    assert all(worker["cudagraph_enabled"] for worker in provenance["workers"])
    assert all(worker["cudagraph_observations"] for worker in provenance["workers"])
    assert all(
        worker["runner_boundary"] == "vllm.v1.worker.gpu_worker.Worker.execute_model"
        for worker in provenance["workers"]
    )
