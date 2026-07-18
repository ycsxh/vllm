# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "1.0.0"

RAW_SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("point_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("phase", pa.string(), nullable=False),
        pa.field("ordinal", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("runner_boundary", pa.string(), nullable=False),
        pa.field("batch_size", pa.int32(), nullable=False),
        pa.field("num_scheduled_tokens", pa.int32(), nullable=False),
        pa.field("prompt_tokens", pa.int32(), nullable=False),
        pa.field("context_tokens", pa.int32(), nullable=False),
        pa.field("cached_tokens", pa.int32(), nullable=False),
        pa.field("new_tokens", pa.int32(), nullable=False),
        pa.field("sampled_token_id", pa.int64()),
        pa.field("injected_token_id", pa.int64()),
        pa.field("sampled_token_discarded", pa.bool_(), nullable=False),
        pa.field("runner_wall_time_ms", pa.float64()),
        pa.field("cuda_model_time_ms", pa.float64()),
        pa.field("derived_time_ms", pa.float64()),
        pa.field("compile_enabled", pa.bool_(), nullable=False),
        pa.field("cudagraph_enabled", pa.bool_(), nullable=False),
        pa.field("cudagraph_runtime_mode", pa.string()),
        pa.field("error", pa.string()),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode()},
)

AGGREGATE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("point_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("sample_count", pa.int32(), nullable=False),
        pa.field("runner_wall_time_median_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_p90_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_mean_ms", pa.float64(), nullable=False),
        pa.field("runner_wall_time_cv", pa.float64(), nullable=False),
        pa.field("noisy", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": SCHEMA_VERSION.encode()},
)


def _load_replay(config: dict[str, Any]) -> dict[str, Any]:
    plan = json.loads(Path(config["artifacts"]["workload_plan"]).read_text())
    replay_ref = min(
        plan["exact_replays"], key=lambda item: item.get("prompt_tokens", 0)
    )
    rows = pq.read_table(
        config["artifacts"]["rendered_turns"],
        columns=[
            "trajectory_id",
            "turn_index",
            "execution_prompt_token_ids",
            "execution_completion_token_ids",
        ],
    ).to_pylist()
    replay = next(
        (
            row
            for row in rows
            if row["trajectory_id"] == replay_ref["trajectory_id"]
            and row["turn_index"] == replay_ref["turn_index"]
        ),
        None,
    )
    if replay is None:
        raise ValueError("the selected exact replay is absent from rendered turns")
    if not replay["execution_prompt_token_ids"]:
        raise ValueError("the selected exact replay has no execution prompt tokens")
    if not replay["execution_completion_token_ids"]:
        raise ValueError("the selected exact replay has no execution completion tokens")
    return replay


def _resolve_config(config: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    runtime = resolved["runtime"]
    runtime["effective_max_model_len"] = max(
        8192,
        len(replay["execution_prompt_token_ids"])
        + len(replay["execution_completion_token_ids"])
        + 1,
    )
    compilation = runtime["compilation"]
    required_compile_sizes = {
        1,
        resolved["profile"]["prefill_chunk_tokens"],
    }
    if not required_compile_sizes.issubset(compilation["compile_sizes"]):
        raise ValueError(
            "compile_sizes must cover decode and the configured prefill chunk"
        )
    return resolved


def make_sample_row(
    *,
    run_id: str,
    point_id: str,
    role: str,
    phase: str,
    ordinal: int,
    scheduled_tokens: int,
    prompt_tokens: int,
    elapsed_ms: float | None,
    sampled_token_id: int | None = None,
    injected_token_id: int | None = None,
    runner_boundary: str = "deterministic-fixture",
    cuda_model_time_ms: float | None = None,
    cached_tokens: int = 0,
    context_tokens: int = 0,
    new_tokens: int | None = None,
    sampled_token_discarded: bool = False,
    cudagraph_runtime_mode: str | None = None,
    compile_enabled: bool = True,
    cudagraph_enabled: bool = True,
    status: str = "passed",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "point_id": point_id,
        "sample_id": f"{run_id}:{point_id}:{phase}:{ordinal}",
        "role": role,
        "phase": phase,
        "ordinal": ordinal,
        "status": status,
        "runner_boundary": runner_boundary,
        "batch_size": 1,
        "num_scheduled_tokens": scheduled_tokens,
        "prompt_tokens": prompt_tokens,
        "context_tokens": context_tokens,
        "cached_tokens": cached_tokens,
        "new_tokens": scheduled_tokens if new_tokens is None else new_tokens,
        "sampled_token_id": sampled_token_id,
        "injected_token_id": injected_token_id,
        "sampled_token_discarded": sampled_token_discarded,
        "runner_wall_time_ms": elapsed_ms,
        "cuda_model_time_ms": cuda_model_time_ms,
        "derived_time_ms": None,
        "compile_enabled": compile_enabled,
        "cudagraph_enabled": cudagraph_enabled,
        "cudagraph_runtime_mode": cudagraph_runtime_mode,
        "error": error,
    }


def _fixture_samples(config: dict[str, Any], replay: dict[str, Any]) -> list[dict]:
    profile = config["profile"]
    run_id = config["run_id"]
    points = (
        (
            "prefill",
            f"prefill-b1-t{profile['prefill_chunk_tokens']}",
            profile["prefill_chunk_tokens"],
        ),
        ("decode", "decode-b1-t1", 1),
    )
    rows = []
    prompt_tokens = len(replay["execution_prompt_token_ids"])
    completion_token_ids = replay["execution_completion_token_ids"]
    for point_offset, (role, point_id, scheduled_tokens) in enumerate(points):
        decode_step_index = 0
        for phase, count, base_ms in (
            ("startup", 1, 100.0),
            ("capture", 1, 50.0),
            ("warmup", profile["warmup_repetitions"], 5.0),
            ("steady", profile["measured_repetitions"], 10.0),
        ):
            for ordinal in range(count):
                is_decode_step = role == "decode" and phase in {"warmup", "steady"}
                sampled_token_id = (
                    100_000 + decode_step_index if is_decode_step else None
                )
                injected_token_id = (
                    completion_token_ids[decode_step_index + 1]
                    if is_decode_step
                    else None
                )
                rows.append(
                    make_sample_row(
                        run_id=run_id,
                        point_id=point_id,
                        role=role,
                        phase=phase,
                        ordinal=ordinal,
                        scheduled_tokens=scheduled_tokens,
                        prompt_tokens=prompt_tokens,
                        elapsed_ms=base_ms + point_offset + ordinal / 10,
                        sampled_token_id=sampled_token_id,
                        injected_token_id=injected_token_id,
                        sampled_token_discarded=is_decode_step,
                        context_tokens=(
                            prompt_tokens + decode_step_index + 1
                            if is_decode_step
                            else scheduled_tokens
                        ),
                        cached_tokens=(
                            prompt_tokens + decode_step_index if is_decode_step else 0
                        ),
                        cudagraph_runtime_mode=("FULL" if is_decode_step else None),
                    )
                )
                if is_decode_step:
                    decode_step_index += 1
    return rows


def _gpu_worker_plan(
    config: dict[str, Any], replay: dict[str, Any], role: str
) -> dict[str, Any]:
    profile = config["profile"]
    runtime = config["runtime"]
    prompt_token_count = len(replay["execution_prompt_token_ids"])
    max_batched_tokens = profile["max_num_batched_tokens"]
    setup_prefill_chunks = []
    remaining = prompt_token_count
    while remaining:
        chunk_tokens = min(remaining, max_batched_tokens)
        setup_prefill_chunks.append(chunk_tokens)
        remaining -= chunk_tokens
    step_count = profile["warmup_repetitions"] + profile["measured_repetitions"]
    completion_token_ids = replay["execution_completion_token_ids"]
    required_decode_tokens = step_count + 1
    if role == "decode" and len(completion_token_ids) < required_decode_tokens:
        raise ValueError(
            "the selected exact replay does not contain enough decode tokens "
            f"for setup plus {step_count} profile steps"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": config["run_id"],
        "role": role,
        "runner_boundary": "vllm.v1.worker.gpu_worker.Worker.execute_model",
        "engine": {
            "block_size": profile["block_size"],
            "compilation_mode": runtime["compilation"]["mode"],
            "cudagraph_enabled": (runtime["compilation"]["cudagraph_mode"] != "NONE"),
            "enforce_eager": runtime["enforce_eager"],
            "max_num_batched_tokens": max_batched_tokens,
            "max_num_seqs": runtime["max_num_seqs"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
            "dtype": runtime["dtype"],
            "kv_cache_dtype": runtime["kv_cache_dtype"],
            "effective_max_model_len": runtime["effective_max_model_len"],
        },
        "sample_phases": {
            "steady": profile["measured_repetitions"],
            "warmup": profile["warmup_repetitions"],
        },
        "scheduled_tokens_per_step": (
            1 if role == "decode" else profile["prefill_chunk_tokens"]
        ),
        "setup_prefill_chunks": setup_prefill_chunks,
        "initial_teacher_forced_token_id": (
            completion_token_ids[0] if role == "decode" else None
        ),
        "teacher_forced_token_ids": (
            completion_token_ids[1:required_decode_tokens] if role == "decode" else []
        ),
    }


def _aggregate_rows(
    samples: list[dict[str, Any]], noisy_cv_threshold: float
) -> list[dict[str, Any]]:
    point_ids = sorted({row["point_id"] for row in samples})
    aggregates = []
    for point_id in point_ids:
        point_samples = [
            row
            for row in samples
            if row["point_id"] == point_id
            and row["phase"] == "steady"
            and row["status"] == "passed"
        ]
        values = [row["runner_wall_time_ms"] for row in point_samples]
        if not values or any(value is None for value in values):
            continue
        mean = statistics.fmean(values)
        cv = statistics.pstdev(values) / mean if mean else 0.0
        p90 = statistics.quantiles(values, n=10, method="inclusive")[8]
        aggregates.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": point_samples[0]["run_id"],
                "point_id": point_id,
                "role": point_samples[0]["role"],
                "sample_count": len(values),
                "runner_wall_time_median_ms": statistics.median(values),
                "runner_wall_time_p90_ms": p90,
                "runner_wall_time_mean_ms": mean,
                "runner_wall_time_cv": cv,
                "noisy": cv > noisy_cv_threshold,
            }
        )
    return aggregates


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _result_markdown(
    *, status_label: str, hardware_validated: bool, aggregates: list[dict[str, Any]]
) -> str:
    lines = [
        "# DS4 profile spine result",
        "",
        f"Status: {status_label}",
        "",
        f"Hardware validated: {'yes' if hardware_validated else 'no'}",
        "",
        "| Point | Samples | Median runner wall time (ms) | Noisy |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in aggregates:
        lines.append(
            f"| `{row['point_id']}` | {row['sample_count']} | "
            f"{row['runner_wall_time_median_ms']:.3f} | "
            f"{'yes' if row['noisy'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Artifacts: `raw_samples.parquet`, `aggregates.parquet`, "
            "`provenance.json`, and `run-config.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_result_artifacts(
    *,
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    provenance: dict[str, Any],
    status_label: str,
    output_dir: Path,
) -> None:
    aggregates = _aggregate_rows(samples, config["profile"]["noisy_cv_threshold"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging-"
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        pq.write_table(
            pa.Table.from_pylist(samples, schema=RAW_SAMPLE_SCHEMA),
            staging_dir / "raw_samples.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(aggregates, schema=AGGREGATE_SCHEMA),
            staging_dir / "aggregates.parquet",
        )
        _write_json(staging_dir / "run-config.json", config)
        _write_json(staging_dir / "provenance.json", provenance)
        (staging_dir / "result.md").write_text(
            _result_markdown(
                status_label=status_label,
                hardware_validated=provenance["hardware_validated"],
                aggregates=aggregates,
            )
        )
        _validate_result_dir(staging_dir)
        output_dir.mkdir()
        for path in staging_dir.iterdir():
            path.replace(output_dir / path.name)


def _validate_result_dir(result_dir: Path) -> None:
    required_files = {
        "aggregates.parquet",
        "provenance.json",
        "raw_samples.parquet",
        "result.md",
        "run-config.json",
    }
    missing = sorted(
        name for name in required_files if not (result_dir / name).is_file()
    )
    if missing:
        raise ValueError(f"missing result artifacts: {', '.join(missing)}")

    raw = pq.read_table(result_dir / "raw_samples.parquet")
    aggregates = pq.read_table(result_dir / "aggregates.parquet")
    if raw.schema != RAW_SAMPLE_SCHEMA:
        raise ValueError("raw_samples.parquet does not match the versioned schema")
    if aggregates.schema != AGGREGATE_SCHEMA:
        raise ValueError("aggregates.parquet does not match the versioned schema")

    raw_rows = raw.to_pylist()
    aggregate_rows = aggregates.to_pylist()
    config = json.loads((result_dir / "run-config.json").read_text())
    provenance = json.loads((result_dir / "provenance.json").read_text())
    run_id = config["run_id"]
    observed_run_ids = {row["run_id"] for row in raw_rows + aggregate_rows} | {
        provenance.get("run_id")
    }
    if observed_run_ids != {run_id}:
        raise ValueError(
            f"run_id mismatch: expected {run_id}, found {sorted(observed_run_ids)}"
        )

    sample_ids = [row["sample_id"] for row in raw_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    if any(
        row["sample_id"]
        != f"{row['run_id']}:{row['point_id']}:{row['phase']}:{row['ordinal']}"
        for row in raw_rows
    ):
        raise ValueError("sample_id values must match their sample coordinates")
    phases = {row["phase"] for row in raw_rows}
    if not phases.issubset({"startup", "capture", "setup", "warmup", "steady"}):
        raise ValueError(f"unknown sample phase: {sorted(phases)}")

    aggregate_point_ids = {row["point_id"] for row in aggregate_rows}
    passed_steady_point_ids = {
        row["point_id"]
        for row in raw_rows
        if row["phase"] == "steady" and row["status"] == "passed"
    }
    if aggregate_point_ids != passed_steady_point_ids:
        raise ValueError("aggregate point_id values do not match passed steady samples")
    for aggregate in aggregate_rows:
        steady_count = sum(
            row["point_id"] == aggregate["point_id"]
            and row["phase"] == "steady"
            and row["status"] == "passed"
            for row in raw_rows
        )
        if aggregate["sample_count"] != steady_count:
            raise ValueError(
                f"sample_count mismatch for point_id {aggregate['point_id']}"
            )


def _write_fixture_result(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version {SCHEMA_VERSION}")
    replay = _load_replay(config)
    config = _resolve_config(config, replay)
    samples = _fixture_samples(config, replay)
    _write_result_artifacts(
        config=config,
        samples=samples,
        provenance={
            "artifact_schema_version": SCHEMA_VERSION,
            "hardware_validated": False,
            "run_id": config["run_id"],
            "source": config["source"],
            "status": "fixture-only",
        },
        status_label="Fixture only",
        output_dir=output_dir,
    )
    print(output_dir)


def _worker_samples_valid(
    worker: dict[str, Any], config: dict[str, Any], replay: dict[str, Any]
) -> bool:
    role = worker.get("role")
    if role not in {"prefill", "decode"}:
        return False
    plan = _gpu_worker_plan(config, replay, role)
    samples = worker.get("samples", [])
    expected_point_id = (
        "decode-b1-t1"
        if role == "decode"
        else f"prefill-b1-t{config['profile']['prefill_chunk_tokens']}"
    )
    expected_counts = {
        "startup": 1,
        "capture": 1,
        "warmup": config["profile"]["warmup_repetitions"],
        "steady": config["profile"]["measured_repetitions"],
    }
    if any(
        sum(sample.get("phase") == phase for sample in samples) != expected_count
        for phase, expected_count in expected_counts.items()
    ):
        return False
    observations = worker.get("cudagraph_observations", [])
    if not observations or any(
        not isinstance(observation, dict)
        or observation.get("runtime_mode") not in {"FULL", "PIECEWISE"}
        for observation in observations
    ):
        return False
    measured_samples = [
        sample for sample in samples if sample["phase"] in {"warmup", "steady"}
    ]
    if any(
        sample.get("cudagraph_runtime_mode") not in {"FULL", "PIECEWISE"}
        for sample in measured_samples
    ):
        return False
    if any(
        sample.get("schema_version") != SCHEMA_VERSION
        or sample.get("run_id") != config["run_id"]
        or sample.get("role") != role
        or sample.get("point_id") != expected_point_id
        or sample.get("runner_boundary") != plan["runner_boundary"]
        or sample.get("status") != "passed"
        or sample.get("compile_enabled") is not True
        or sample.get("cudagraph_enabled") is not True
        for sample in samples
    ):
        return False
    if role == "decode":
        decode_samples = [
            sample for sample in samples if sample["phase"] in {"warmup", "steady"}
        ]
        if [sample["injected_token_id"] for sample in decode_samples] != plan[
            "teacher_forced_token_ids"
        ]:
            return False
        if any(
            sample["sampled_token_id"] is None
            or sample.get("sampled_token_discarded") is not True
            for sample in decode_samples
        ):
            return False
    return True


def _assemble_gpu_result(
    *,
    config_path: Path,
    preflight_path: Path,
    worker_paths: list[Path],
    output_dir: Path,
) -> bool:
    config = json.loads(config_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    workers = [json.loads(path.read_text()) for path in worker_paths]
    replay = _load_replay(config)
    config = _resolve_config(config, replay)
    roles = {worker.get("role") for worker in workers}
    workers_valid = (
        roles == {"prefill", "decode"}
        and all(worker.get("schema_version") == SCHEMA_VERSION for worker in workers)
        and all(worker.get("run_id") == config["run_id"] for worker in workers)
        and all(worker.get("status") == "passed" for worker in workers)
        and all(worker.get("hardware_validated") is True for worker in workers)
        and all(worker.get("compile_enabled") is True for worker in workers)
        and all(worker.get("cudagraph_enabled") is True for worker in workers)
        and all(_worker_samples_valid(worker, config, replay) for worker in workers)
    )
    hardware_validated = preflight.get("status") == "ready" and workers_valid
    samples = [sample for worker in workers for sample in worker.get("samples", [])]
    worker_summaries = [
        {key: value for key, value in worker.items() if key != "samples"}
        for worker in workers
    ]
    provenance = {
        "artifact_schema_version": SCHEMA_VERSION,
        "hardware_validated": hardware_validated,
        "invocation": [sys.executable, *sys.argv],
        "model": config["model"],
        "preflight": preflight,
        "roles": config["roles"],
        "run_id": config["run_id"],
        "run_parameters": {
            "profile": config["profile"],
            "runtime": config["runtime"],
        },
        "source": config["source"],
        "status": "passed" if hardware_validated else "invalid",
        "workers": worker_summaries,
    }
    _write_result_artifacts(
        config=config,
        samples=samples,
        provenance=provenance,
        status_label="Passed" if hardware_validated else "Invalid",
        output_dir=output_dir,
    )
    print(output_dir)
    return hardware_validated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DS4 profile spine.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--config", type=Path, required=True)
    fixture.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--result-dir", type=Path, required=True)
    gpu_worker = subparsers.add_parser("gpu-worker")
    gpu_worker.add_argument("--config", type=Path, required=True)
    gpu_worker.add_argument("--role", choices=("prefill", "decode"), required=True)
    gpu_worker.add_argument("--output", type=Path, required=True)
    gpu_worker.add_argument("--inspect-plan", action="store_true")
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--config", type=Path, required=True)
    assemble.add_argument("--preflight", type=Path, required=True)
    assemble.add_argument("--worker-result", type=Path, action="append", required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fixture":
            _write_fixture_result(args.config, args.output_dir)
        elif args.command == "validate":
            _validate_result_dir(args.result_dir)
            print(args.result_dir)
        elif args.command == "gpu-worker":
            config = json.loads(args.config.read_text())
            replay = _load_replay(config)
            config = _resolve_config(config, replay)
            plan = _gpu_worker_plan(config, replay, args.role)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.inspect_plan:
                _write_json(args.output, plan)
                print(args.output)
                return
            try:
                from benchmarks.ds4_profile.gpu_profile import run_gpu_worker

                worker_result = run_gpu_worker(config, replay, plan)
                returncode = 0 if worker_result["status"] == "passed" else 2
            except Exception as error:
                worker_result = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": config["run_id"],
                    "hardware_validated": False,
                    "role": args.role,
                    "runner_boundary": plan["runner_boundary"],
                    "samples": [],
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                returncode = 2
            _write_json(args.output, worker_result)
            print(args.output)
            if returncode:
                raise SystemExit(returncode)
        elif args.command == "assemble":
            passed = _assemble_gpu_result(
                config_path=args.config,
                preflight_path=args.preflight,
                worker_paths=args.worker_result,
                output_dir=args.output_dir,
            )
            if not passed:
                raise SystemExit(2)
    except (KeyError, OSError, ValueError) as error:
        print(f"validation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
