# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run explicit fixed-batch P and D node profiles for the DS4 experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import regex as re

MODEL = "Qwen/Qwen3.5-4B"
INPUT_TOKENS = 12_800
BLOCK_TOKENS = 128
CACHE_PAGE_TOKENS = 640
P_HIT_TOKENS = {0.0: 0, 0.75: 9_600, 0.9: 11_520}
P_BATCH_SIZES = {1, 2, 4, 8, 16}
D_BATCH_SIZES = {1, 2, 4, 8, 16}
PRIMARY_REPETITIONS = 3
PRIMARY_WARMUP_BATCHES = 5
PRIMARY_MEASURED_BATCHES = 10
NOISY_CV = 0.05
FIXED_BATCH_RUNTIME_ENVIRONMENT = {
    "FLASHINFER_JIT_VERBOSE": "0",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_KV_CACHE_LAYOUT": "HND",
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "0",
    "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
}
CAPACITY_ERROR_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "insufficient kv cache",
    "no available memory for the cache blocks",
    "larger than the available kv cache memory",
)
CPU_LIST = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")
GPU_INDEX = re.compile(r"\d+")


@dataclass(frozen=True)
class FixedBatchExecution:
    """Frozen runtime provenance shared by every point in one invocation."""

    model_revision: str
    tokenizer_revision: str
    vllm_commit: str
    attention_backend: str
    vllm_dirty: bool = False
    gpu_memory_utilization: float = 0.9
    seed: int = 17
    role_gpus: tuple[tuple[str, str], ...] = (
        ("P", "0"),
        ("D", "1"),
    )
    role_cpu_affinity: tuple[tuple[str, str], ...] | None = None
    role_numa_nodes: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self) -> None:
        full_revision = re.compile(r"[0-9a-f]{40}")
        for name, value in (
            ("model_revision", self.model_revision),
            ("tokenizer_revision", self.tokenizer_revision),
            ("vllm_commit", self.vllm_commit),
        ):
            if not full_revision.fullmatch(value):
                raise ValueError(f"{name} must be a full 40-character revision")
        if not self.attention_backend:
            raise ValueError("attention_backend must be non-empty")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if set(dict(self.role_gpus)) != {"P", "D"}:
            raise ValueError("role_gpus must define exactly P and D")
        if any(not GPU_INDEX.fullmatch(gpu) for _, gpu in self.role_gpus):
            raise ValueError("each role GPU must be one physical GPU index")
        if len(set(dict(self.role_gpus).values())) != 2:
            raise ValueError("P and D must use distinct physical GPUs")
        if (self.role_cpu_affinity is None) != (self.role_numa_nodes is None):
            raise ValueError("CPU affinity and NUMA nodes must be provided together")
        if self.role_cpu_affinity is not None and (
            set(dict(self.role_cpu_affinity)) != {"P", "D"}
            or set(dict(self.role_numa_nodes or ())) != {"P", "D"}
        ):
            raise ValueError("CPU affinity and NUMA nodes must define P and D")
        if self.role_cpu_affinity is not None and any(
            not CPU_LIST.fullmatch(cpu_list) for _, cpu_list in self.role_cpu_affinity
        ):
            raise ValueError("CPU affinity must use Linux CPU-list syntax")
        if self.role_numa_nodes is not None and any(
            isinstance(node, bool) or not isinstance(node, int) or node < 0
            for _, node in self.role_numa_nodes
        ):
            raise ValueError("NUMA nodes must be nonnegative integers")


@dataclass(frozen=True)
class FixedBatchPoint:
    """One explicit fixed-batch experiment point."""

    id: str
    role: Literal["P", "D"]
    batch_size: int
    input_tokens: int
    hit_ratio: float | None
    output_tokens: int
    repetitions: int
    warmup_batches: int
    measured_batches: int
    execution_mode: Literal["primary", "feasibility_smoke", "diagnostic"]
    requires_supported_point: str | None = None

    @property
    def cached_tokens(self) -> int:
        return 0 if self.hit_ratio is None else P_HIT_TOKENS[self.hit_ratio]

    @property
    def computed_tokens(self) -> int:
        return self.input_tokens - self.cached_tokens


@dataclass(frozen=True)
class PreparedRequest:
    """A deterministic target prompt and its optional cache-warm prefix."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    planned_cached_tokens: int
    warm_prompt_token_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class IterationRecord:
    """One public iteration-detail log record."""

    elapsed_ms: float
    context_requests: int
    context_tokens: int
    generation_requests: int
    generation_tokens: int


@dataclass(frozen=True)
class RequestObservation:
    """Observable result for one generated request."""

    request_id: str
    prompt_tokens: int
    cached_tokens: int
    output_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class BatchObservation:
    """Returned outputs and iteration details for one exact batch."""

    wall_time_ms: float
    requests: tuple[RequestObservation, ...]
    iterations: tuple[IterationRecord, ...]


class ObservationValidationError(ValueError):
    """A validation failure that retains the offending public observation."""

    def __init__(
        self,
        message: str,
        observation: BatchObservation,
        validation_stage: str,
    ) -> None:
        super().__init__(message)
        self.observation = observation
        self.validation_stage = validation_stage


class FixedBatchRuntime(Protocol):
    """Public offline-runtime boundary used by the fixed-batch runner."""

    def provenance(self) -> dict[str, Any]: ...

    def wait_idle(self) -> None: ...

    def reset_prefix_cache(self) -> bool: ...

    def start_profile(self, profile_prefix: str) -> None: ...

    def stop_profile(self) -> None: ...

    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[
    [FixedBatchPoint, dict[str, Any], Path],
    FixedBatchRuntime,
]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load JSON object from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_point(value: object) -> FixedBatchPoint:
    if not isinstance(value, dict):
        raise ValueError("each point must be a JSON object")
    common_fields = {
        "id",
        "role",
        "batch_size",
        "input_tokens",
        "output_tokens",
        "repetitions",
        "warmup_batches",
        "measured_batches",
        "execution_mode",
    }
    optional_fields = {"hit_ratio", "requires_supported_point"}
    unexpected = sorted(value.keys() - common_fields - optional_fields)
    missing = sorted(common_fields - value.keys())
    if unexpected:
        raise ValueError(f"point contains unexpected fields: {', '.join(unexpected)}")
    if missing:
        raise ValueError(f"point is missing fields: {', '.join(missing)}")

    point_id = value["id"]
    if not isinstance(point_id, str) or not point_id:
        raise ValueError("point id must be a non-empty string")
    role = value["role"]
    if role not in {"P", "D"}:
        raise ValueError("role must be P or D")
    input_tokens = _positive_int(value["input_tokens"], "input_tokens")
    if input_tokens != INPUT_TOKENS:
        raise ValueError(f"input_tokens must be exactly {INPUT_TOKENS}")
    batch_size = _positive_int(value["batch_size"], "batch_size")
    allowed_batches = P_BATCH_SIZES if role == "P" else D_BATCH_SIZES
    if batch_size not in allowed_batches:
        raise ValueError(f"unsupported {role} batch_size: {batch_size}")
    output_tokens = _positive_int(value["output_tokens"], "output_tokens")
    expected_output = 1 if role == "P" else 128
    if output_tokens != expected_output:
        raise ValueError(f"{role} output_tokens must be exactly {expected_output}")

    hit_ratio_value = value.get("hit_ratio")
    if role == "P":
        if (
            isinstance(hit_ratio_value, bool)
            or not isinstance(hit_ratio_value, (int, float))
            or float(hit_ratio_value) not in P_HIT_TOKENS
        ):
            raise ValueError("P hit_ratio must be 0, 0.75, or 0.9")
        hit_ratio = float(hit_ratio_value)
    else:
        if "hit_ratio" in value:
            raise ValueError("D points must not define hit_ratio")
        hit_ratio = None

    execution_mode = value["execution_mode"]
    if execution_mode not in {"primary", "feasibility_smoke", "diagnostic"}:
        raise ValueError(
            "execution_mode must be primary, feasibility_smoke, or diagnostic"
        )
    repetitions = _positive_int(value["repetitions"], "repetitions")
    warmup_batches = _positive_int(value["warmup_batches"], "warmup_batches")
    measured_batches = _positive_int(value["measured_batches"], "measured_batches")
    if execution_mode == "primary" and (
        repetitions != PRIMARY_REPETITIONS
        or warmup_batches != PRIMARY_WARMUP_BATCHES
        or measured_batches != PRIMARY_MEASURED_BATCHES
    ):
        raise ValueError("primary points require 3 runs, 5 warmups, and 10 batches")
    if execution_mode != "primary" and (
        repetitions != 1 or warmup_batches != 1 or measured_batches != 1
    ):
        raise ValueError("smoke and diagnostic points require one 1/1/1 run")
    requires_supported_point = value.get("requires_supported_point")
    if requires_supported_point is not None and (
        not isinstance(requires_supported_point, str) or not requires_supported_point
    ):
        raise ValueError("requires_supported_point must be a non-empty string")
    return FixedBatchPoint(
        id=point_id,
        role=role,
        batch_size=batch_size,
        input_tokens=input_tokens,
        hit_ratio=hit_ratio,
        output_tokens=output_tokens,
        repetitions=repetitions,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
        execution_mode=execution_mode,
        requires_supported_point=requires_supported_point,
    )


def load_fixed_batch_plan(path: Path) -> tuple[FixedBatchPoint, ...]:
    """Load an explicit point list without generating parameter combinations."""
    value = _load_json_object(path)
    if set(value) != {"schema_version", "points"}:
        raise ValueError("plan must contain only schema_version and points")
    if value["schema_version"] != 1:
        raise ValueError("unsupported plan schema_version")
    raw_points = value["points"]
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("points must be a non-empty list")
    points = tuple(_parse_point(point) for point in raw_points)
    ids = [point.id for point in points]
    if len(ids) != len(set(ids)):
        raise ValueError("point ids must be unique")
    prior_points: dict[str, FixedBatchPoint] = {}
    for point in points:
        requires_smoke = (
            point.role == "P"
            and point.batch_size == 16
            and point.hit_ratio in {0.0, 0.75}
            and point.execution_mode == "primary"
        )
        if requires_smoke and point.requires_supported_point is None:
            raise ValueError(
                "P B=16 at 0% or 75% requires a matching feasibility smoke"
            )
        if point.requires_supported_point is not None:
            smoke = prior_points.get(point.requires_supported_point)
            if (
                smoke is None
                or smoke.execution_mode != "feasibility_smoke"
                or smoke.role != point.role
                or smoke.batch_size != point.batch_size
                or smoke.input_tokens != point.input_tokens
                or smoke.hit_ratio != point.hit_ratio
                or smoke.output_tokens != point.output_tokens
            ):
                raise ValueError(
                    "requires_supported_point must name an earlier matching "
                    "feasibility smoke"
                )
        prior_points[point.id] = point
    return points


def _prompt_tokens(seed: int, request_index: int) -> tuple[int, ...]:
    state = (seed + request_index * 0x9E3779B9) & 0xFFFFFFFF
    tokens = []
    for _ in range(INPUT_TOKENS):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        tokens.append(1_024 + state % 100_000)
    return tuple(tokens)


def prepare_requests(
    point: FixedBatchPoint,
    *,
    seed: int,
) -> tuple[PreparedRequest, ...]:
    """Construct exact, stable prompts with request-isolated first blocks."""
    requests = []
    for index in range(point.batch_size):
        prompt = _prompt_tokens(seed, index)
        cached_tokens = point.cached_tokens
        warm_prompt = prompt[: cached_tokens + 1] if cached_tokens else None
        requests.append(
            PreparedRequest(
                request_id=f"{point.id}-request-{index:02d}",
                prompt_token_ids=prompt,
                planned_cached_tokens=cached_tokens,
                warm_prompt_token_ids=warm_prompt,
            )
        )
    first_blocks = {request.prompt_token_ids[:BLOCK_TOKENS] for request in requests}
    if len(first_blocks) != len(requests):
        raise RuntimeError("deterministic request isolation blocks collided")
    return tuple(requests)


def _engine_config(
    point: FixedBatchPoint,
    execution: FixedBatchExecution,
    point_dir: Path,
) -> dict[str, Any]:
    warm_tokens = point.cached_tokens + 1 if point.cached_tokens else 0
    max_model_len = point.input_tokens + point.output_tokens
    cpu_affinity = (
        None
        if execution.role_cpu_affinity is None
        else dict(execution.role_cpu_affinity)[point.role]
    )
    numa_node = (
        None
        if execution.role_numa_nodes is None
        else dict(execution.role_numa_nodes)[point.role]
    )
    config = {
        "model": MODEL,
        "model_revision": execution.model_revision,
        "tokenizer_revision": execution.tokenizer_revision,
        "vllm_commit": execution.vllm_commit,
        "vllm_dirty": execution.vllm_dirty,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "device": "cuda",
        "cuda_visible_devices": dict(execution.role_gpus)[point.role],
        "cpu_affinity": cpu_affinity,
        "numa_node": numa_node,
        "attention_backend": execution.attention_backend,
        "block_size": BLOCK_TOKENS,
        "cache_alignment_tokens": CACHE_PAGE_TOKENS,
        "kv_cache_dtype": "bfloat16",
        "language_model_only": True,
        "mamba_cache_mode": "align",
        "enable_prefix_caching": True,
        "enable_chunked_prefill": False,
        "max_model_len": max_model_len,
        "max_num_seqs": point.batch_size,
        "max_num_batched_tokens": max(
            max_model_len,
            point.batch_size * point.computed_tokens,
            warm_tokens,
        ),
        "gpu_memory_utilization": execution.gpu_memory_utilization,
        "seed": execution.seed,
        "async_scheduling": False,
        "enable_logging_iteration_details": True,
        "runtime_environment": dict(FIXED_BATCH_RUNTIME_ENVIRONMENT),
    }
    if point.execution_mode == "diagnostic":
        config["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str((point_dir / "traces").resolve()),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "ignore_frontend": True,
            "delay_iterations": 0 if point.role == "P" else 1,
            "max_iterations": 1 if point.role == "P" else 127,
        }
    return config


def _observation_dict(observation: BatchObservation) -> dict[str, Any]:
    return {
        "wall_time_ms": observation.wall_time_ms,
        "requests": [asdict(request) for request in observation.requests],
        "iterations": [asdict(iteration) for iteration in observation.iterations],
    }


def batch_observation_from_dict(value: object) -> BatchObservation:
    """Parse a retained runtime observation for independent artifact audit."""
    if not isinstance(value, dict):
        raise ValueError("batch observation must be an object")
    try:
        wall_time_ms = float(value["wall_time_ms"])
        raw_requests = value["requests"]
        raw_iterations = value["iterations"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("batch observation is missing required fields") from error
    if (
        wall_time_ms < 0
        or not isinstance(raw_requests, list)
        or not isinstance(raw_iterations, list)
    ):
        raise ValueError("batch observation fields are invalid")
    try:
        requests = tuple(
            RequestObservation(
                request_id=request["request_id"],
                prompt_tokens=request["prompt_tokens"],
                cached_tokens=request["cached_tokens"],
                output_token_ids=tuple(request["output_token_ids"]),
            )
            for request in raw_requests
        )
        iterations = tuple(
            IterationRecord(
                elapsed_ms=iteration["elapsed_ms"],
                context_requests=iteration["context_requests"],
                context_tokens=iteration["context_tokens"],
                generation_requests=iteration["generation_requests"],
                generation_tokens=iteration["generation_tokens"],
            )
            for iteration in raw_iterations
        )
    except (KeyError, TypeError) as error:
        raise ValueError("batch observation records are invalid") from error
    return BatchObservation(
        wall_time_ms=wall_time_ms,
        requests=requests,
        iterations=iterations,
    )


def _validate_outputs(
    observation: BatchObservation,
    *,
    request_count: int,
    prompt_tokens: int,
    output_tokens: int,
) -> None:
    if len(observation.requests) != request_count:
        raise ValueError("runtime returned the wrong request count")
    for request in observation.requests:
        if request.prompt_tokens != prompt_tokens:
            raise ValueError("runtime returned an unexpected prompt length")
        if len(request.output_token_ids) != output_tokens:
            raise ValueError("runtime returned an unexpected output length")


def _run_p_batch(
    point: FixedBatchPoint,
    requests: tuple[PreparedRequest, ...],
    runtime: FixedBatchRuntime,
    *,
    profile_prefix: str | None = None,
) -> tuple[BatchObservation, tuple[dict[str, float], ...]]:
    runtime.wait_idle()
    if not runtime.reset_prefix_cache():
        raise RuntimeError("prefix-cache reset failed")
    for request in requests:
        if request.warm_prompt_token_ids is None:
            continue
        warm = runtime.generate(
            (request.warm_prompt_token_ids,),
            max_tokens=1,
            ignore_eos=True,
        )
        try:
            _validate_outputs(
                warm,
                request_count=1,
                prompt_tokens=point.cached_tokens + 1,
                output_tokens=1,
            )
        except ValueError as error:
            raise ObservationValidationError(
                str(error),
                warm,
                "cache_warm",
            ) from error
    runtime.wait_idle()
    if profile_prefix is not None:
        runtime.start_profile(profile_prefix)
    try:
        target = runtime.generate(
            tuple(request.prompt_token_ids for request in requests),
            max_tokens=1,
            ignore_eos=True,
        )
    finally:
        if profile_prefix is not None:
            runtime.stop_profile()
    try:
        samples = derive_batch_metric_samples(point, target)
    except ValueError as error:
        raise ObservationValidationError(
            str(error),
            target,
            "P_target",
        ) from error
    return target, samples


def _derive_p_metric_samples(
    point: FixedBatchPoint,
    observation: BatchObservation,
) -> tuple[dict[str, float], ...]:
    _validate_outputs(
        observation,
        request_count=point.batch_size,
        prompt_tokens=point.input_tokens,
        output_tokens=1,
    )
    if any(
        request.cached_tokens != point.cached_tokens for request in observation.requests
    ):
        raise ValueError("target cached-token count differs from the point plan")
    if len(observation.iterations) != 1:
        raise ValueError("P target must execute in exactly one iteration")
    iteration = observation.iterations[0]
    if (
        iteration.context_requests != point.batch_size
        or iteration.context_tokens != point.batch_size * point.computed_tokens
        or iteration.generation_requests != 0
        or iteration.generation_tokens != 0
    ):
        raise ValueError("P target iteration composition differs from the point plan")
    seconds = iteration.elapsed_ms / 1_000
    if seconds <= 0:
        raise ValueError("iteration latency must be positive")
    return (
        {
            "latency_ms": iteration.elapsed_ms,
            "request_throughput_per_s": point.batch_size / seconds,
            "computed_token_throughput_per_s": (
                point.batch_size * point.computed_tokens / seconds
            ),
        },
    )


def _run_d_batch(
    point: FixedBatchPoint,
    requests: tuple[PreparedRequest, ...],
    runtime: FixedBatchRuntime,
    *,
    profile_prefix: str | None = None,
) -> tuple[BatchObservation, tuple[dict[str, float], ...]]:
    runtime.wait_idle()
    if not runtime.reset_prefix_cache():
        raise RuntimeError("prefix-cache reset failed")
    if profile_prefix is not None:
        runtime.start_profile(profile_prefix)
    try:
        target = runtime.generate(
            tuple(request.prompt_token_ids for request in requests),
            max_tokens=point.output_tokens,
            ignore_eos=True,
        )
    finally:
        if profile_prefix is not None:
            runtime.stop_profile()
    try:
        samples = derive_batch_metric_samples(point, target)
    except ValueError as error:
        raise ObservationValidationError(
            str(error),
            target,
            "D_target",
        ) from error
    return target, samples


def _derive_d_metric_samples(
    point: FixedBatchPoint,
    observation: BatchObservation,
) -> tuple[dict[str, float], ...]:
    _validate_outputs(
        observation,
        request_count=point.batch_size,
        prompt_tokens=point.input_tokens,
        output_tokens=point.output_tokens,
    )
    if len(observation.iterations) != point.output_tokens:
        raise ValueError("D target must contain one setup and 127 decode iterations")
    setup = observation.iterations[0]
    if (
        setup.context_requests != point.batch_size
        or setup.context_tokens != point.batch_size * point.input_tokens
        or setup.generation_requests != 0
        or setup.generation_tokens != 0
    ):
        raise ValueError("D setup iteration composition differs from the point plan")
    decode_iterations = observation.iterations[1:]
    if any(
        iteration.context_requests != 0
        or iteration.context_tokens != 0
        or iteration.generation_requests != point.batch_size
        or iteration.generation_tokens != point.batch_size
        for iteration in decode_iterations
    ):
        raise ValueError("D decode iteration composition differs from the point plan")
    if any(iteration.elapsed_ms <= 0 for iteration in decode_iterations):
        raise ValueError("decode iteration latency must be positive")
    steady_iterations = decode_iterations[1:]
    if len(steady_iterations) != 126:
        raise ValueError("D target must retain 126 steady decode iterations")
    latencies = [iteration.elapsed_ms for iteration in steady_iterations]
    return tuple(
        {
            "latency_ms": latency,
            "output_token_throughput_per_s": (point.batch_size / (latency / 1_000)),
        }
        for latency in latencies
    )


def derive_batch_metric_samples(
    point: FixedBatchPoint,
    observation: BatchObservation,
) -> tuple[dict[str, float], ...]:
    """Validate one target and preserve every primary metric sample."""
    if point.role == "P":
        return _derive_p_metric_samples(point, observation)
    return _derive_d_metric_samples(point, observation)


def derive_first_decode_metric_sample(
    point: FixedBatchPoint,
    observation: BatchObservation,
) -> dict[str, float]:
    """Return the separately reported first pure-decode step."""
    if point.role != "D":
        raise ValueError("first-decode metrics apply only to D points")
    _derive_d_metric_samples(point, observation)
    iteration = observation.iterations[1]
    return {
        "first_decode_latency_ms": iteration.elapsed_ms,
        "first_decode_output_token_throughput_per_s": (
            point.batch_size / (iteration.elapsed_ms / 1_000)
        ),
    }


def _sample_artifact(
    point: FixedBatchPoint,
    observation: BatchObservation,
    metric_samples: tuple[dict[str, float], ...],
) -> dict[str, Any]:
    artifact = {
        "observation": _observation_dict(observation),
        "derived_samples": metric_samples,
        "derived_summary": summarize_run_samples(metric_samples),
    }
    if point.role == "D":
        artifact.update(
            {
                "setup_iteration": asdict(observation.iterations[0]),
                "first_decode_iteration": asdict(observation.iterations[1]),
                "first_decode_sample": derive_first_decode_metric_sample(
                    point,
                    observation,
                ),
                "steady_decode_iterations": [
                    asdict(iteration) for iteration in observation.iterations[2:]
                ],
            }
        )
    return artifact


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_run_samples(
    samples: Sequence[dict[str, float]],
) -> dict[str, Any]:
    """Summarize the measured samples from one repetition."""
    if not samples:
        raise ValueError("a run requires measured samples")
    summary = {}
    for metric in samples[0]:
        values = [sample[metric] for sample in samples]
        summary[metric] = {
            "p50": _percentile(values, 50),
            "p90": _percentile(values, 90),
            "p95": _percentile(values, 95),
        }
    return summary


def summarize_point_runs(
    run_summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize every run percentile and label cross-run variation."""
    if not run_summaries:
        raise ValueError("a point requires run summaries")
    metrics: dict[str, Any] = {}
    noisy_metrics = []
    for metric in run_summaries[0]:
        percentile_summaries = {}
        for percentile in ("p50", "p90", "p95"):
            values = [float(run[metric][percentile]) for run in run_summaries]
            mean = statistics.fmean(values)
            stddev = statistics.stdev(values) if len(values) > 1 else 0.0
            cv = stddev / mean if mean else 0.0
            noisy = cv > NOISY_CV
            percentile_summaries[percentile] = {
                "values": values,
                "mean": mean,
                "cv": cv,
                "noisy": noisy,
            }
            if noisy:
                noisy_metrics.append(f"{metric}.{percentile}")
        p50 = percentile_summaries["p50"]
        metrics[metric] = {
            "run_p50_values": p50["values"],
            "mean": p50["mean"],
            "cv": p50["cv"],
            "noisy": p50["noisy"],
            "percentiles": percentile_summaries,
        }
    return {
        "status": "valid",
        "metrics": metrics,
        "noisy_metrics": noisy_metrics,
    }


def _run_point(
    point: FixedBatchPoint,
    point_dir: Path,
    execution: FixedBatchExecution,
    runtime_factory: RuntimeFactory,
) -> dict[str, Any]:
    requests = prepare_requests(point, seed=execution.seed)
    engine_config = _engine_config(point, execution, point_dir)
    point_dir.mkdir(parents=True, exist_ok=False)
    _write_json(point_dir / "point.json", asdict(point))
    _write_json(
        point_dir / "requests.json",
        [asdict(request) for request in requests],
    )
    _write_json(point_dir / "engine-config.json", engine_config)
    runtime: FixedBatchRuntime | None = None
    run_summaries = []
    active_run_dir: Path | None = None
    active_phase = "runtime_initialization"
    active_batch = 0
    try:
        runtime = runtime_factory(point, engine_config, point_dir)
        _write_json(
            point_dir / "provenance.json",
            {
                "model": MODEL,
                "execution": asdict(execution),
                "runtime": runtime.provenance(),
            },
        )
        run_batch = _run_p_batch if point.role == "P" else _run_d_batch
        for run_index in range(1, point.repetitions + 1):
            run_dir = point_dir / f"run-{run_index:02d}"
            run_dir.mkdir()
            active_run_dir = run_dir
            for batch_index in range(1, point.warmup_batches + 1):
                active_phase = "warmup"
                active_batch = batch_index
                observation, metric_samples = run_batch(
                    point,
                    requests,
                    runtime,
                )
                _write_json(
                    run_dir / f"warmup-{batch_index:02d}.json",
                    _sample_artifact(point, observation, metric_samples),
                )
            measured = []
            first_decode_measured = []
            for batch_index in range(1, point.measured_batches + 1):
                active_phase = "measured"
                active_batch = batch_index
                observation, metric_samples = run_batch(
                    point,
                    requests,
                    runtime,
                    profile_prefix=(
                        point.id if point.execution_mode == "diagnostic" else None
                    ),
                )
                measured.extend(metric_samples)
                if point.role == "D":
                    first_decode_measured.append(
                        derive_first_decode_metric_sample(point, observation)
                    )
                _write_json(
                    run_dir / f"measured-{batch_index:02d}.json",
                    _sample_artifact(point, observation, metric_samples),
                )
            run_summary = summarize_run_samples(measured)
            if first_decode_measured:
                run_summary.update(summarize_run_samples(first_decode_measured))
            run_summaries.append(run_summary)
            _write_json(run_dir / "run-summary.json", run_summary)
        summary = {
            "point_id": point.id,
            "role": point.role,
            "batch_size": point.batch_size,
            "hit_ratio": point.hit_ratio,
            "metric_label": (
                "P-side TTFT proxy" if point.role == "P" else "D-side TPOT proxy"
            ),
            **summarize_point_runs(run_summaries),
        }
        _write_json(point_dir / "point-summary.json", summary)
        _write_json(
            point_dir / "status.json",
            {
                "status": "valid",
                "completed_repetitions": len(run_summaries),
                "required_repetitions": point.repetitions,
            },
        )
        return summary
    except Exception as error:
        error_text = str(error)
        lowered = error_text.lower()
        status = (
            "unsupported"
            if any(marker in lowered for marker in CAPACITY_ERROR_MARKERS)
            else "failed"
        )
        retained_error = (
            error.__cause__ if isinstance(error, ObservationValidationError) else error
        )
        failure = {
            "status": status,
            "phase": active_phase,
            "batch": active_batch,
            "error_type": type(retained_error).__name__,
            "error": error_text,
        }
        if active_run_dir is not None:
            if isinstance(error, ObservationValidationError):
                _write_json(
                    active_run_dir / "invalid-observation.json",
                    {
                        "validation_stage": error.validation_stage,
                        **_observation_dict(error.observation),
                    },
                )
            _write_json(active_run_dir / "failure.json", failure)
        _write_json(point_dir / "point-failure.json", failure)
        _write_json(
            point_dir / "status.json",
            {
                **failure,
                "completed_repetitions": len(run_summaries),
                "required_repetitions": point.repetitions,
            },
        )
        return failure
    finally:
        if runtime is not None:
            runtime.close()


def run_fixed_batch_profile(
    plan_path: Path,
    results_dir: Path,
    *,
    execution: FixedBatchExecution,
    runtime_factory: RuntimeFactory | None = None,
) -> dict[str, Any]:
    """Run one explicit fixed-batch plan and retain its validated artifacts."""
    if runtime_factory is None:
        if execution.vllm_dirty:
            raise ValueError("production execution requires a clean vLLM checkout")
        if execution.role_cpu_affinity is None or execution.role_numa_nodes is None:
            raise ValueError("production execution requires CPU and NUMA placement")
        from benchmarks.ds4_profile.fixed_batch_runtime import (
            SubprocessRuntimeFactory,
        )

        runtime_factory = SubprocessRuntimeFactory()
    points = load_fixed_batch_plan(plan_path)
    if results_dir.exists():
        raise FileExistsError(f"results directory already exists: {results_dir}")
    results_dir.mkdir(parents=True)
    _write_json(
        results_dir / "resolved-plan.json",
        {
            "schema_version": 1,
            "execution": asdict(execution),
            "points": [asdict(point) for point in points],
        },
    )
    point_statuses = {}
    for point in points:
        if (
            point.requires_supported_point is not None
            and point_statuses[point.requires_supported_point] != "valid"
        ):
            point_dir = results_dir / "points" / point.id
            point_dir.mkdir(parents=True)
            _write_json(point_dir / "point.json", asdict(point))
            _write_json(
                point_dir / "status.json",
                {
                    "status": "unsupported",
                    "reason": (
                        "required feasibility smoke did not complete successfully"
                    ),
                    "requires_supported_point": point.requires_supported_point,
                },
            )
            point_statuses[point.id] = "unsupported"
            continue
        point_result = _run_point(
            point,
            results_dir / "points" / point.id,
            execution,
            runtime_factory,
        )
        point_statuses[point.id] = point_result["status"]
    if "failed" in point_statuses.values():
        status = "failed"
    elif "unsupported" in point_statuses.values():
        status = "valid_with_unsupported"
    else:
        status = "valid"
    result = {
        "status": status,
        "point_statuses": point_statuses,
    }
    _write_json(results_dir / "status.json", result)
    return result


def _git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return commit, dirty


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--p-gpu", default="0")
    parser.add_argument("--d-gpu", default="1")
    parser.add_argument("--prefill-cpus", required=True)
    parser.add_argument("--prefill-numa-node", required=True, type=int)
    parser.add_argument("--decode-cpus", required=True)
    parser.add_argument("--decode-numa-node", required=True, type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    vllm_commit, vllm_dirty = _git_state(repo_root)
    execution = FixedBatchExecution(
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        vllm_commit=vllm_commit,
        vllm_dirty=vllm_dirty,
        attention_backend=args.attention_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        role_gpus=(("P", args.p_gpu), ("D", args.d_gpu)),
        role_cpu_affinity=(
            ("P", args.prefill_cpus),
            ("D", args.decode_cpus),
        ),
        role_numa_nodes=(
            ("P", args.prefill_numa_node),
            ("D", args.decode_numa_node),
        ),
    )
    if args.dry_run:
        points = load_fixed_batch_plan(args.plan)
        resolved = {
            "schema_version": 1,
            "execution": asdict(execution),
            "points": [asdict(point) for point in points],
            "engine_configs": [
                _engine_config(
                    point,
                    execution,
                    args.results_dir / "points" / point.id,
                )
                for point in points
            ],
        }
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0
    result = run_fixed_batch_profile(
        args.plan,
        args.results_dir,
        execution=execution,
    )
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
