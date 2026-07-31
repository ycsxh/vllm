# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-audit raw P-only chunked-prefill artifacts and build their report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.ds4_profile.chunked_prefill import (
    MODEL,
    UNSUPPORTED_ERROR_MARKERS,
    ChunkedPrefillExecution,
    ChunkedPrefillPoint,
    build_engine_config,
    derive_chunked_prefill_sample,
    group_points_by_token_budget,
    load_chunked_prefill_plan,
    prepare_requests,
    summarize_point_runs,
    summarize_run_samples,
)
from benchmarks.ds4_profile.fixed_batch import (
    BatchObservation,
    batch_observation_from_dict,
)

CSV_FIELDS = (
    "point_id",
    "status",
    "status_reason",
    "metric_label",
    "batch_size",
    "requested_hit_ratio",
    "planned_cached_tokens",
    "aligned_hit_ratio",
    "actual_cached_tokens",
    "max_num_batched_tokens",
    "execution_mode",
    "global_sample_count",
    "noisy",
    "prefill_completion_p50_mean_ms",
    "prefill_completion_p50_cv",
    "prefill_completion_global_p50_ms",
    "prefill_completion_global_p90_ms",
    "context_iterations_p50_mean",
    "context_iterations_p50_cv",
    "context_iterations_global_p50",
    "context_iterations_global_p90",
    "computed_token_throughput_p50_mean_per_s",
    "computed_token_throughput_p50_cv",
    "computed_token_throughput_global_p50_per_s",
    "computed_token_throughput_global_p90_per_s",
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Files produced by one deterministic report build."""

    summary_csv: Path
    report_md: Path
    plot_paths: tuple[Path, ...]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load JSON from {path}") from error


def _load_json_object(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _audit_cache_warm_observation(
    point: ChunkedPrefillPoint,
    observation: BatchObservation,
) -> None:
    if len(observation.requests) != 1:
        raise ValueError(f"{point.id} cache warm returned the wrong request count")
    request = observation.requests[0]
    if request.prompt_tokens != point.cached_tokens + 1:
        raise ValueError(f"{point.id} cache warm prompt length differs")
    if request.cached_tokens != 0:
        raise ValueError(f"{point.id} cache warm unexpectedly reused cached tokens")
    if len(request.output_token_ids) != 1:
        raise ValueError(f"{point.id} cache warm output length differs")
    if not observation.iterations:
        raise ValueError(f"{point.id} cache warm lacks iteration evidence")
    for iteration in observation.iterations:
        if not math.isfinite(iteration.elapsed_ms) or iteration.elapsed_ms <= 0:
            raise ValueError(f"{point.id} cache warm latency is not positive")
        if (
            iteration.context_requests != 1
            or not 0 < iteration.context_tokens <= point.max_num_batched_tokens
        ):
            raise ValueError(f"{point.id} cache warm context work exceeds its budget")
        if iteration.generation_requests or iteration.generation_tokens:
            raise ValueError(f"{point.id} cache warm must be context-only")
    if sum(item.context_tokens for item in observation.iterations) != (
        point.cached_tokens + 1
    ):
        raise ValueError(f"{point.id} cache warm context-token total differs")


def _audit_cache_warm_observations(
    point: ChunkedPrefillPoint,
    raw_observations: object,
    *,
    expected_count: int | None = None,
) -> None:
    if not isinstance(raw_observations, list):
        raise ValueError(f"{point.id} cache-warm observations must be a list")
    if expected_count is None:
        expected_count = point.batch_size if point.cached_tokens else 0
    if len(raw_observations) != expected_count:
        raise ValueError(f"{point.id} cache-warm observation count differs")
    for value in raw_observations:
        _audit_cache_warm_observation(
            point,
            batch_observation_from_dict(value),
        )


def _audit_sample_artifact(
    point: ChunkedPrefillPoint,
    path: Path,
) -> tuple[dict[str, float], int]:
    artifact = _load_json_object(path)
    if set(artifact) != {
        "cache_warm_observations",
        "derived_sample",
        "observation",
    }:
        raise ValueError(f"{point.id} sample artifact fields differ")
    _audit_cache_warm_observations(
        point,
        artifact["cache_warm_observations"],
    )
    observation = batch_observation_from_dict(artifact["observation"])
    sample = derive_chunked_prefill_sample(point, observation)
    if artifact["derived_sample"] != sample:
        raise ValueError(f"{point.id} stored derived sample differs from raw data")
    return sample, observation.requests[0].cached_tokens


def _expected_point_summary(
    point: ChunkedPrefillPoint,
    run_summaries: list[dict[str, dict[str, float]]],
    measured_samples: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "point_id": point.id,
        "metric_label": "P-side prefill-completion latency",
        "batch_size": point.batch_size,
        "requested_hit_ratio": point.hit_ratio,
        "planned_cached_tokens": point.cached_tokens,
        "aligned_hit_ratio": point.aligned_hit_ratio,
        "max_num_batched_tokens": point.max_num_batched_tokens,
        **summarize_point_runs(run_summaries, measured_samples),
    }


def _audit_valid_point(
    point: ChunkedPrefillPoint,
    point_dir: Path,
    expected_engine_dir: Path,
) -> tuple[dict[str, Any], int]:
    if _load_json_object(point_dir / "point.json") != asdict(point):
        raise ValueError(f"{point.id} artifact differs from the explicit plan")
    status = _load_json_object(point_dir / "status.json")
    if status.get("status") != "valid":
        raise ValueError(f"{point.id} status changed during report audit")
    reference = _load_json_object(point_dir / "engine-reference.json")
    if reference.get(
        "max_num_batched_tokens"
    ) != point.max_num_batched_tokens or reference.get("engine_dir") != str(
        Path("engines") / expected_engine_dir.name
    ):
        raise ValueError(f"{point.id} engine reference differs from its group")

    run_summaries = []
    measured_samples = []
    actual_cached_tokens = set()
    for run_index in range(1, point.repetitions + 1):
        run_dir = point_dir / f"run-{run_index:02d}"
        for batch_index in range(1, point.warmup_batches + 1):
            _audit_sample_artifact(
                point,
                run_dir / f"warmup-{batch_index:02d}.json",
            )
        run_samples = []
        for batch_index in range(1, point.measured_batches + 1):
            sample, actual_cached = _audit_sample_artifact(
                point,
                run_dir / f"measured-{batch_index:02d}.json",
            )
            run_samples.append(sample)
            measured_samples.append(sample)
            actual_cached_tokens.add(actual_cached)
        run_summary = summarize_run_samples(run_samples)
        if _load_json_object(run_dir / "run-summary.json") != run_summary:
            raise ValueError(f"{point.id} stored run summary differs from raw data")
        run_summaries.append(run_summary)
    expected_summary = _expected_point_summary(
        point,
        run_summaries,
        measured_samples,
    )
    if _load_json_object(point_dir / "point-summary.json") != expected_summary:
        raise ValueError(f"{point.id} stored point summary differs from raw data")
    if actual_cached_tokens != {point.cached_tokens}:
        raise ValueError(f"{point.id} actual cached-token counts disagree")
    return expected_summary, actual_cached_tokens.pop()


def _audit_expected_execution(
    execution: ChunkedPrefillExecution,
    *,
    expected_model_revision: str,
    expected_tokenizer_revision: str,
    expected_vllm_commit: str,
) -> None:
    expected = {
        "model_revision": expected_model_revision,
        "tokenizer_revision": expected_tokenizer_revision,
        "vllm_commit": expected_vllm_commit,
    }
    for name, value in expected.items():
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"expected {name} must be a full revision")
        if getattr(execution, name) != value:
            label = "vLLM commit" if name == "vllm_commit" else name.replace("_", " ")
            raise ValueError(f"resolved execution differs from expected {label}")


def _audit_runtime_invocation(
    engine_dir: Path,
    engine_config: dict[str, Any],
    execution: ChunkedPrefillExecution,
) -> None:
    invocation = _load_json_object(engine_dir / "runtime-invocation.json")
    expected_command = [
        "numactl",
        f"--physcpubind={execution.p_cpu_affinity}",
        f"--membind={execution.p_numa_node}",
        sys.executable,
        "-m",
        "benchmarks.ds4_profile.fixed_batch_runtime",
        "--serve",
        "--engine-config",
        str(engine_dir / "engine-config.json"),
        "--point-dir",
        str(engine_dir),
    ]
    if invocation.get("command") != expected_command:
        raise ValueError("runtime invocation command differs from frozen placement")
    environment = invocation.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("runtime invocation environment is missing")
    expected_environment = {
        **engine_config["runtime_environment"],
        "CUDA_VISIBLE_DEVICES": execution.p_gpu,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if any(
        environment.get(name) != value for name, value in expected_environment.items()
    ):
        raise ValueError("runtime invocation environment differs from provenance")
    required_paths = (
        "CUDA_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "LD_LIBRARY_PATH",
        "PATH",
    )
    if any(
        not isinstance(environment.get(name), str) or not environment[name]
        for name in required_paths
    ):
        raise ValueError("runtime invocation lacks a required environment path")


def _expected_failure_status(error: object) -> str:
    error_text = str(error).lower()
    if any(marker in error_text for marker in UNSUPPORTED_ERROR_MARKERS):
        return "unsupported"
    return "failed"


def _audit_partial_sample(
    point: ChunkedPrefillPoint,
    path: Path,
) -> str:
    partial = _load_json_object(path)
    if set(partial) != {
        "failure_stage",
        "cache_warm_index",
        "cache_warm_observations",
        "observation",
    }:
        raise ValueError(f"{point.id} partial-sample fields differ")
    stage = partial["failure_stage"]
    valid_stages = {
        "wait_idle_before_reset",
        "cache_reset_runtime",
        "cache_reset_rejected",
        "cache_warm_runtime",
        "cache_warm_validation",
        "wait_idle_after_cache_warm",
        "target_runtime",
        "target_validation",
    }
    if stage not in valid_stages:
        raise ValueError(f"{point.id} partial-sample stage is invalid")
    warm_index = partial["cache_warm_index"]
    if stage in {"cache_warm_runtime", "cache_warm_validation"}:
        if (
            isinstance(warm_index, bool)
            or not isinstance(warm_index, int)
            or not 1 <= warm_index <= point.batch_size
            or not point.cached_tokens
        ):
            raise ValueError(f"{point.id} partial cache-warm index is invalid")
        expected_warm_count = warm_index - 1
    else:
        if warm_index is not None:
            raise ValueError(f"{point.id} partial cache-warm index differs")
        expected_warm_count = (
            point.batch_size
            if point.cached_tokens
            and stage
            in {
                "wait_idle_after_cache_warm",
                "target_runtime",
                "target_validation",
            }
            else 0
        )
    _audit_cache_warm_observations(
        point,
        partial["cache_warm_observations"],
        expected_count=expected_warm_count,
    )
    raw_observation = partial["observation"]
    if stage not in {"cache_warm_validation", "target_validation"}:
        if raw_observation is not None:
            raise ValueError(f"{point.id} runtime failure has an observation")
        return stage
    observation = batch_observation_from_dict(raw_observation)
    try:
        if stage == "cache_warm_validation":
            _audit_cache_warm_observation(point, observation)
        else:
            derive_chunked_prefill_sample(point, observation)
    except ValueError:
        return stage
    raise ValueError(f"{point.id} retained invalid observation is valid")


def _expected_sample_names(prefix: str, count: int) -> set[str]:
    return {f"{prefix}-{index:02d}.json" for index in range(1, count + 1)}


def _audit_complete_run(
    point: ChunkedPrefillPoint,
    run_dir: Path,
) -> None:
    warm_paths = sorted(run_dir.glob("warmup-*.json"))
    measured_paths = sorted(run_dir.glob("measured-*.json"))
    if {path.name for path in warm_paths} != _expected_sample_names(
        "warmup", point.warmup_batches
    ):
        raise ValueError(f"{point.id} completed run warmup topology differs")
    if {path.name for path in measured_paths} != _expected_sample_names(
        "measured", point.measured_batches
    ):
        raise ValueError(f"{point.id} completed run measured topology differs")
    for path in warm_paths:
        _audit_sample_artifact(point, path)
    measured_samples = [
        _audit_sample_artifact(point, path)[0] for path in measured_paths
    ]
    summary_path = run_dir / "run-summary.json"
    if not summary_path.is_file() or _load_json_object(
        summary_path
    ) != summarize_run_samples(measured_samples):
        raise ValueError(f"{point.id} completed run summary differs from raw data")
    if any(
        (run_dir / filename).exists()
        for filename in ("failure.json", "partial-sample.json")
    ):
        raise ValueError(f"{point.id} completed run has failure evidence")


def _audit_failure_record(
    point: ChunkedPrefillPoint,
    point_dir: Path,
) -> None:
    status = _load_json_object(point_dir / "status.json")
    failure_path = point_dir / "point-failure.json"
    if not failure_path.is_file():
        raise ValueError(f"{point.id} lacks point-failure evidence")
    failure = _load_json_object(failure_path)
    for name in ("status", "phase", "error_type", "error"):
        if not isinstance(failure.get(name), str) or not failure[name]:
            raise ValueError(f"{point.id} point-failure field {name} is invalid")
    if failure["status"] not in {"failed", "unsupported"}:
        raise ValueError(f"{point.id} point-failure status is invalid")
    if failure["status"] != _expected_failure_status(failure["error"]):
        raise ValueError(f"{point.id} point-failure classification differs")
    if any(status.get(name) != value for name, value in failure.items()):
        raise ValueError(f"{point.id} status differs from point-failure evidence")

    engine_dir = (
        point_dir.parents[1] / "engines" / f"budget-{point.max_num_batched_tokens:04d}"
    )
    engine_failure_path = engine_dir / "engine-failure.json"
    if engine_failure_path.is_file():
        if _load_json_object(engine_failure_path) != failure:
            raise ValueError(f"{point.id} engine and point failures differ")
        if status != failure or failure["phase"] != "runtime_initialization":
            raise ValueError(f"{point.id} initialization failure evidence differs")
        if any(point_dir.glob("run-*")):
            raise ValueError(f"{point.id} initialization failure has run artifacts")
        if any(
            (point_dir / filename).exists()
            for filename in (
                "point-summary.json",
                "partial-sample.json",
                "invalid-observation.json",
            )
        ):
            raise ValueError(
                f"{point.id} initialization failure has sample or summary evidence"
            )
        return

    if failure["phase"] not in {
        "warmup",
        "measured",
        "run_summary",
        "point_summary",
    }:
        raise ValueError(f"{point.id} point-failure phase is invalid")
    completed = status.get("completed_repetitions")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= point.repetitions
        or status.get("required_repetitions") != point.repetitions
    ):
        raise ValueError(f"{point.id} failure repetition counts differ")
    if failure["phase"] == "point_summary":
        if completed != point.repetitions or failure.get("batch") != 0:
            raise ValueError(f"{point.id} point-summary failure position differs")
    elif completed >= point.repetitions:
        raise ValueError(f"{point.id} failure exceeds the repetition plan")
    run_dirs = sorted(point_dir.glob("run-*"))
    expected_run_names = {
        f"run-{index:02d}"
        for index in range(
            1,
            completed + (0 if failure["phase"] == "point_summary" else 1) + 1,
        )
    }
    if {path.name for path in run_dirs} != expected_run_names:
        raise ValueError(f"{point.id} retained failure run topology differs")
    for run_dir in run_dirs[:completed]:
        _audit_complete_run(point, run_dir)

    if failure["phase"] == "point_summary":
        summary_path = point_dir / "point-summary.json"
        if summary_path.exists():
            run_summaries = [
                _load_json_object(run_dir / "run-summary.json") for run_dir in run_dirs
            ]
            measured_samples = [
                _audit_sample_artifact(point, path)[0]
                for run_dir in run_dirs
                for path in sorted(run_dir.glob("measured-*.json"))
            ]
            if _load_json_object(summary_path) != _expected_point_summary(
                point,
                run_summaries,
                measured_samples,
            ):
                raise ValueError(f"{point.id} failed point summary differs")
        return

    active_run = run_dirs[-1]
    active_failure = active_run / "failure.json"
    if not active_failure.is_file() or _load_json_object(active_failure) != failure:
        raise ValueError(f"{point.id} active run failure evidence differs")
    batch = failure.get("batch")
    if failure["phase"] == "run_summary":
        if batch != 0:
            raise ValueError(f"{point.id} run-summary failure batch differs")
        warm_count = point.warmup_batches
        measured_count = point.measured_batches
        if (active_run / "partial-sample.json").exists():
            raise ValueError(f"{point.id} run-summary failure has partial evidence")
    else:
        limit = (
            point.warmup_batches
            if failure["phase"] == "warmup"
            else point.measured_batches
        )
        if (
            isinstance(batch, bool)
            or not isinstance(batch, int)
            or not 1 <= batch <= limit
        ):
            raise ValueError(f"{point.id} failure batch differs")
        warm_count = batch - 1 if failure["phase"] == "warmup" else point.warmup_batches
        measured_count = batch - 1 if failure["phase"] == "measured" else 0
        partial_path = active_run / "partial-sample.json"
        if not partial_path.is_file():
            raise ValueError(f"{point.id} lacks partial-sample evidence")
        failure_stage = _audit_partial_sample(point, partial_path)
        invalid_path = active_run / "invalid-observation.json"
        if failure_stage in {"cache_warm_validation", "target_validation"}:
            if not invalid_path.is_file():
                raise ValueError(f"{point.id} lacks invalid-observation evidence")
            invalid = _load_json_object(invalid_path)
            if invalid.pop("validation_stage", None) != failure_stage:
                raise ValueError(f"{point.id} invalid-observation stage differs")
            if invalid != _load_json_object(partial_path)["observation"]:
                raise ValueError(f"{point.id} invalid observations differ")
        elif invalid_path.exists():
            raise ValueError(f"{point.id} runtime failure has invalid-observation")
    warm_paths = sorted(active_run.glob("warmup-*.json"))
    measured_paths = sorted(active_run.glob("measured-*.json"))
    if {path.name for path in warm_paths} != _expected_sample_names(
        "warmup", warm_count
    ) or {path.name for path in measured_paths} != _expected_sample_names(
        "measured", measured_count
    ):
        raise ValueError(f"{point.id} active run sample topology differs")
    for path in (*warm_paths, *measured_paths):
        _audit_sample_artifact(point, path)
    if (active_run / "run-summary.json").exists():
        raise ValueError(f"{point.id} active failed run has a summary")
    if (point_dir / "point-summary.json").exists():
        raise ValueError(f"{point.id} failed point unexpectedly has a summary")


def _audit_frozen_context(
    points: tuple[ChunkedPrefillPoint, ...],
    results_dir: Path,
    hardware: dict[str, Any],
    *,
    expected_model_revision: str,
    expected_tokenizer_revision: str,
    expected_vllm_commit: str,
) -> dict[str, Any]:
    groups = group_points_by_token_budget(points)
    resolved = _load_json_object(results_dir / "resolved-plan.json")
    expected_groups = [
        {
            "max_num_batched_tokens": budget,
            "point_ids": [point.id for point in group],
        }
        for budget, group in groups
    ]
    if (
        resolved.get("schema_version") != 1
        or resolved.get("points") != [asdict(point) for point in points]
        or resolved.get("engine_groups") != expected_groups
    ):
        raise ValueError("resolved plan differs from the report plan")
    execution_value = resolved.get("execution")
    if not isinstance(execution_value, dict):
        raise ValueError("resolved execution provenance is missing")
    try:
        execution = ChunkedPrefillExecution(**execution_value)
    except (TypeError, ValueError) as error:
        raise ValueError("resolved execution provenance is invalid") from error
    if execution.vllm_dirty:
        raise ValueError("hardware report requires a clean vLLM checkout")
    _audit_expected_execution(
        execution,
        expected_model_revision=expected_model_revision,
        expected_tokenizer_revision=expected_tokenizer_revision,
        expected_vllm_commit=expected_vllm_commit,
    )
    order = _load_json_object(results_dir / "execution-order.json")
    if order.get("points") != [
        {
            "index": index,
            "point_id": point.id,
            "max_num_batched_tokens": point.max_num_batched_tokens,
        }
        for index, point in enumerate(points, start=1)
    ]:
        raise ValueError("point execution order differs from the explicit plan")

    expected_engine_names = {f"budget-{budget:04d}" for budget, _group in groups}
    actual_engine_names = {
        path.name for path in (results_dir / "engines").iterdir() if path.is_dir()
    }
    if actual_engine_names != expected_engine_names:
        raise ValueError("retained engine groups differ from the explicit plan")
    actual_point_names = {
        path.name for path in (results_dir / "points").iterdir() if path.is_dir()
    }
    if actual_point_names != {point.id for point in points}:
        raise ValueError("retained point directories differ from the explicit plan")

    point_statuses = {}
    for point in points:
        point_dir = results_dir / "points" / point.id
        if _load_json_object(point_dir / "point.json") != asdict(point):
            raise ValueError(f"{point.id} artifact differs from the explicit plan")
        expected_requests = json.loads(
            json.dumps(
                [
                    asdict(request)
                    for request in prepare_requests(
                        point,
                        seed=execution.seed,
                    )
                ]
            )
        )
        if _load_json(point_dir / "requests.json") != expected_requests:
            raise ValueError(f"{point.id} deterministic requests differ")
        reference = _load_json_object(point_dir / "engine-reference.json")
        expected_engine_name = f"budget-{point.max_num_batched_tokens:04d}"
        if reference != {
            "max_num_batched_tokens": point.max_num_batched_tokens,
            "engine_dir": str(Path("engines") / expected_engine_name),
        }:
            raise ValueError(f"{point.id} engine reference differs from its group")
        status = _load_json_object(point_dir / "status.json").get("status")
        if status not in {"valid", "failed", "unsupported"}:
            raise ValueError(f"{point.id} has an invalid retained status")
        point_statuses[point.id] = status
    if "failed" in point_statuses.values():
        expected_status = "failed"
    elif "unsupported" in point_statuses.values():
        expected_status = "valid_with_unsupported"
    else:
        expected_status = "valid"
    if _load_json_object(results_dir / "status.json") != {
        "status": expected_status,
        "point_statuses": point_statuses,
    }:
        raise ValueError("top-level status differs from retained point statuses")

    runtime_provenance = []
    gpu_models = set()
    for budget, group in groups:
        engine_dir = results_dir / "engines" / f"budget-{budget:04d}"
        engine_config = _load_json_object(engine_dir / "engine-config.json")
        if engine_config != build_engine_config(budget, execution):
            raise ValueError(f"budget {budget} engine config differs from provenance")
        _audit_runtime_invocation(engine_dir, engine_config, execution)
        provenance_path = engine_dir / "provenance.json"
        if not provenance_path.is_file():
            failure = _load_json_object(engine_dir / "engine-failure.json")
            if failure.get("status") not in {"failed", "unsupported"} or any(
                point_statuses[point.id] == "valid" for point in group
            ):
                raise ValueError(f"budget {budget} lacks valid runtime provenance")
            continue
        provenance = _load_json_object(provenance_path)
        if (
            provenance.get("model") != MODEL
            or provenance.get("execution") != execution_value
        ):
            raise ValueError(f"budget {budget} runtime provenance is inconsistent")
        runtime = provenance.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError(f"budget {budget} runtime provenance is invalid")
        if runtime.get("runner_boundary") != "vllm.LLM.generate":
            raise ValueError("runtime runner boundary is not public LLM.generate")
        if runtime.get("cache_reset_boundary") != "vllm.LLM.reset_prefix_cache":
            raise ValueError("runtime cache-reset boundary is not public")
        if runtime.get("iteration_source") != "enable_logging_iteration_details":
            raise ValueError("runtime iteration-detail provenance is invalid")
        if runtime.get("visible_gpu_count") != 1:
            raise ValueError("P engine did not isolate one visible GPU")
        if runtime.get("visible_gpu_model") != hardware["gpu_model"]:
            raise ValueError("runtime GPU model differs from frozen hardware")
        if (
            not isinstance(runtime.get("nvidia_driver"), str)
            or not runtime["nvidia_driver"]
            or not isinstance(runtime.get("gpu_uuid"), str)
            or not runtime["gpu_uuid"].startswith("GPU-")
        ):
            raise ValueError("runtime physical-GPU provenance is incomplete")
        versions = runtime.get("runtime_versions")
        if not isinstance(versions, dict) or any(
            not isinstance(versions.get(name), str) or not versions[name]
            for name in ("python", "torch", "vllm", "cuda")
        ):
            raise ValueError("runtime version provenance is incomplete")
        runtime_provenance.append(runtime)
        gpu_models.add(runtime["visible_gpu_model"])
    if gpu_models and gpu_models != {hardware["gpu_model"]}:
        raise ValueError("hardware GPU model differs from runtime provenance")
    runtime_versions = {
        json.dumps(runtime.get("runtime_versions"), sort_keys=True)
        for runtime in runtime_provenance
    }
    if len(runtime_versions) > 1:
        raise ValueError("runtime versions differ across engine groups")
    return {
        "execution": execution_value,
        "runtime_provenance": runtime_provenance,
    }


def _row_for_point(
    point: ChunkedPrefillPoint,
    results_dir: Path,
) -> dict[str, Any]:
    point_dir = results_dir / "points" / point.id
    status_value = _load_json_object(point_dir / "status.json")
    status = status_value.get("status")
    if status not in {"valid", "failed", "unsupported"}:
        raise ValueError(f"{point.id} has an invalid retained status")
    row = dict.fromkeys(CSV_FIELDS, "")
    row.update(
        {
            "point_id": point.id,
            "status": status,
            "status_reason": status_value.get(
                "error",
                status_value.get("reason", ""),
            ),
            "metric_label": "P-side prefill-completion latency",
            "batch_size": point.batch_size,
            "requested_hit_ratio": point.hit_ratio,
            "planned_cached_tokens": point.cached_tokens,
            "aligned_hit_ratio": point.aligned_hit_ratio,
            "max_num_batched_tokens": point.max_num_batched_tokens,
            "execution_mode": point.execution_mode,
        }
    )
    if status != "valid":
        _audit_failure_record(point, point_dir)
        return row
    engine_dir = results_dir / "engines" / f"budget-{point.max_num_batched_tokens:04d}"
    summary, actual_cached = _audit_valid_point(
        point,
        point_dir,
        engine_dir,
    )
    latency = summary["metrics"]["prefill_completion_latency_ms"]
    iterations = summary["metrics"]["context_iterations"]
    throughput = summary["metrics"]["computed_token_throughput_per_s"]
    row.update(
        {
            "actual_cached_tokens": actual_cached,
            "global_sample_count": summary["global_sample_count"],
            "noisy": summary["noisy"],
            "prefill_completion_p50_mean_ms": latency["mean"],
            "prefill_completion_p50_cv": latency["cv"],
            "prefill_completion_global_p50_ms": latency["global_p50"],
            "prefill_completion_global_p90_ms": latency["global_p90"],
            "context_iterations_p50_mean": iterations["mean"],
            "context_iterations_p50_cv": iterations["cv"],
            "context_iterations_global_p50": iterations["global_p50"],
            "context_iterations_global_p90": iterations["global_p90"],
            "computed_token_throughput_p50_mean_per_s": throughput["mean"],
            "computed_token_throughput_p50_cv": throughput["cv"],
            "computed_token_throughput_global_p50_per_s": throughput["global_p50"],
            "computed_token_throughput_global_p90_per_s": throughput["global_p90"],
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _series(
    rows: list[dict[str, Any]],
    metric: str,
) -> list[tuple[str, list[tuple[float, float, bool]]]]:
    valid = [row for row in rows if row["status"] == "valid" and row[metric] != ""]
    keys = sorted(
        {(int(row["batch_size"]), float(row["requested_hit_ratio"])) for row in valid}
    )
    return [
        (
            f"B={batch_size}, hit={hit_ratio:.0%}",
            sorted(
                (
                    float(row["max_num_batched_tokens"]),
                    float(row[metric]),
                    bool(row["noisy"]),
                )
                for row in valid
                if int(row["batch_size"]) == batch_size
                and float(row["requested_hit_ratio"]) == hit_ratio
            ),
        )
        for batch_size, hit_ratio in keys
    ]


def _status_annotations(rows: list[dict[str, Any]]) -> list[str]:
    return [
        (
            f"{row['status']}: budget={row['max_num_batched_tokens']}, "
            f"B={row['batch_size']}, "
            f"hit={float(row['requested_hit_ratio']):.0%} "
            f"({row['point_id']})"
        )
        for row in rows
        if row["status"] != "valid"
    ]


def _write_svg(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float, bool]]]],
    annotations: list[str],
) -> None:
    width = 900
    plot_height = 500
    legend_rows = max(1, (len(series) + 2) // 3)
    height = plot_height + 22 * (legend_rows + len(annotations))
    left, right, top, bottom = 90, 35, 55, 75
    points = [point for _, values in series for point in values]
    x_values = [point[0] for point in points] or [512.0, 4_096.0]
    y_values = [point[1] for point in points] or [0.0, 1.0]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        y_min, y_max = 0.0, max(1.0, y_max * 1.2)

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (plot_height - top - bottom)

    colors = (
        "#2563eb",
        "#dc2626",
        "#059669",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
        "#15803d",
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{plot_height - bottom}" '
        f'x2="{width - right}" y2="{plot_height - bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{plot_height - bottom}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{plot_height - 24}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">'
        "Batch-wide max_num_batched_tokens</text>",
        f'<text x="20" y="{plot_height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 20 {plot_height / 2})" '
        f'font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>',
    ]
    for index, (label, values) in enumerate(series):
        color = colors[index % len(colors)]
        coordinates = " ".join(
            f"{x_position(x):.2f},{y_position(y):.2f}" for x, y, _ in values
        )
        if coordinates:
            elements.append(
                f'<polyline points="{coordinates}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        for x, y, noisy in values:
            style = (
                f'fill="white" stroke="{color}" stroke-width="3"'
                if noisy
                else f'fill="{color}"'
            )
            note = " (noisy)" if noisy else ""
            elements.append(
                f'<circle cx="{x_position(x):.2f}" '
                f'cy="{y_position(y):.2f}" r="4" {style}>'
                f"<title>budget={x:g}, {y:g}{note}</title></circle>"
            )
        legend_x = left + (index % 3) * 260
        legend_y = plot_height + 18 + (index // 3) * 22
        elements.append(
            f'<text x="{legend_x}" y="{legend_y}" '
            f'font-family="sans-serif" font-size="12" fill="{color}">'
            f"{html.escape(label)}</text>"
        )
    annotation_start = plot_height + 18 + legend_rows * 22
    for index, annotation in enumerate(annotations):
        elements.append(
            f'<text x="{left}" y="{annotation_start + index * 18}" '
            'font-family="sans-serif" font-size="12" fill="#b91c1c">'
            f"{html.escape(annotation)}</text>"
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    hardware: dict[str, Any],
    frozen_context: dict[str, Any],
) -> None:
    execution = frozen_context["execution"]
    lines = [
        "# DS4 P-only chunked-prefill profile",
        "",
        "The primary metric is P-engine-local prefill-completion latency: the "
        "sum of all validated context-iteration elapsed times for one exact "
        "target batch. It is a P-side TTFT proxy, not client-observed 1P1D TTFT.",
        "",
        "It excludes cache preparation, KV transfer, the D worker, proxy/network "
        "latency, queueing outside the offline engine, client timing, and the "
        "one-token completion phase.",
        "",
        "## Frozen scope",
        "",
        f"- Hardware: {hardware['gpu_count']} x {hardware['gpu_model']} "
        f"({hardware['topology']})",
        "- Model: Qwen/Qwen3.5-4B BF16, TP=1",
        f"- Model revision: `{execution['model_revision']}`",
        f"- Tokenizer revision: `{execution['tokenizer_revision']}`",
        f"- vLLM commit: `{execution['vllm_commit']}` "
        f"(dirty={execution['vllm_dirty']})",
        f"- Attention backend: `{execution['attention_backend']}`",
        f"- P placement: GPU `{execution['p_gpu']}`, CPUs "
        f"`{execution['p_cpu_affinity']}`, NUMA `{execution['p_numa_node']}`",
        "- Input: fixed 13,723-token prompts; output: one ignored-EOS token",
        "- Requested hits: 0%, 75%, and 90%; aligned cached tokens: 0, "
        "10,240, and 12,160",
        "- Chunked prefill on; prefix caching on; max_num_seqs=4; token budget "
        "is batch-wide, not per request",
        "- Runtime provenance: "
        f"`{json.dumps(frozen_context['runtime_provenance'], sort_keys=True)}`",
        "",
        "## Audited points",
        "",
        "| Point | Budget | B | Requested hit | Aligned hit | Actual cached | "
        "Status | P-prefill p50 mean ms | Iterations p50 mean | Noisy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        latency = (
            "-"
            if row["prefill_completion_p50_mean_ms"] == ""
            else f"{float(row['prefill_completion_p50_mean_ms']):.4f}"
        )
        iterations = (
            "-"
            if row["context_iterations_p50_mean"] == ""
            else f"{float(row['context_iterations_p50_mean']):.2f}"
        )
        actual = (
            "-"
            if row["actual_cached_tokens"] == ""
            else str(row["actual_cached_tokens"])
        )
        lines.append(
            f"| {row['point_id']} | {row['max_num_batched_tokens']} | "
            f"{row['batch_size']} | "
            f"{float(row['requested_hit_ratio']):.0%} | "
            f"{float(row['aligned_hit_ratio']):.4%} | {actual} | "
            f"{row['status']} | {latency} | {iterations} | "
            f"{row['noisy'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![P prefill completion latency](p-prefill-completion-latency.svg)",
            "",
            "![P context iterations](p-context-iterations.svg)",
            "",
            "![P computed-token throughput](p-computed-token-throughput.svg)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_chunked_prefill_report(
    plan_path: Path,
    results_dir: Path,
    report_dir: Path,
    *,
    hardware: dict[str, Any],
    expected_model_revision: str,
    expected_tokenizer_revision: str,
    expected_vllm_commit: str,
) -> ReportArtifacts:
    """Recompute every result from raw observations and build the report."""
    if set(hardware) != {"gpu_count", "gpu_model", "topology"}:
        raise ValueError("hardware must contain gpu_count, gpu_model, and topology")
    if (
        hardware["gpu_count"] != 2
        or hardware["gpu_model"] != "NVIDIA GeForce RTX 3090"
        or hardware["topology"] != "P=GPU0/NUMA0 on dual RTX 3090"
    ):
        raise ValueError("hardware must be the frozen dual RTX 3090 target")
    if report_dir.exists():
        raise FileExistsError(f"report directory already exists: {report_dir}")
    points = load_chunked_prefill_plan(plan_path)
    frozen_context = _audit_frozen_context(
        points,
        results_dir,
        hardware,
        expected_model_revision=expected_model_revision,
        expected_tokenizer_revision=expected_tokenizer_revision,
        expected_vllm_commit=expected_vllm_commit,
    )
    rows = [_row_for_point(point, results_dir) for point in points]
    report_dir.mkdir(parents=True)
    summary_csv = report_dir / "summary.csv"
    report_md = report_dir / "report.md"
    _write_csv(summary_csv, rows)
    plot_specs = (
        (
            "p-prefill-completion-latency.svg",
            "P-side prefill-completion latency",
            "Batch context latency sum (ms)",
            "prefill_completion_p50_mean_ms",
        ),
        (
            "p-context-iterations.svg",
            "P context-iteration count",
            "Context iterations",
            "context_iterations_p50_mean",
        ),
        (
            "p-computed-token-throughput.svg",
            "P computed-token throughput",
            "Computed tokens/s",
            "computed_token_throughput_p50_mean_per_s",
        ),
    )
    plot_paths = []
    annotations = _status_annotations(rows)
    for filename, title, y_label, metric in plot_specs:
        plot_path = report_dir / filename
        _write_svg(
            plot_path,
            title=title,
            y_label=y_label,
            series=_series(rows, metric),
            annotations=annotations,
        )
        plot_paths.append(plot_path)
    _write_report(
        report_md,
        rows,
        hardware,
        frozen_context,
    )
    return ReportArtifacts(
        summary_csv=summary_csv,
        report_md=report_md,
        plot_paths=tuple(plot_paths),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 3090")
    parser.add_argument(
        "--topology",
        default="P=GPU0/NUMA0 on dual RTX 3090",
    )
    parser.add_argument("--expected-model-revision", required=True)
    parser.add_argument("--expected-tokenizer-revision", required=True)
    parser.add_argument("--expected-vllm-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_chunked_prefill_report(
        args.plan,
        args.results_dir,
        args.report_dir,
        hardware={
            "gpu_count": args.gpu_count,
            "gpu_model": args.gpu_model,
            "topology": args.topology,
        },
        expected_model_revision=args.expected_model_revision,
        expected_tokenizer_revision=args.expected_tokenizer_revision,
        expected_vllm_commit=args.expected_vllm_commit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
