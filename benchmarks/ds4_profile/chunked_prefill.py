# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the DS4 P-only chunked-prefill profile."""

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

from benchmarks.ds4_profile.fixed_batch import (
    CAPACITY_ERROR_MARKERS,
    FIXED_BATCH_RUNTIME_ENVIRONMENT,
    BatchObservation,
)

MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_REVISION = MODEL_REVISION
INPUT_TOKENS = 13_723
BLOCK_TOKENS = 128
CACHE_PAGE_TOKENS = 640
HIT_RATIOS = {0.0, 0.75, 0.9}
BATCH_SIZES = {1, 2, 4}
TOKEN_BUDGETS = {512, 1_024, 2_048, 4_096}
PRIMARY_REPETITIONS = 3
PRIMARY_WARMUP_BATCHES = 1
PRIMARY_MEASURED_BATCHES = 3
NOISY_CV = 0.05
ATTENTION_BACKEND = "FLASH_ATTN"
P_GPU = "0"
P_CPU_AFFINITY = "0,2,4,6,8,10"
P_NUMA_NODE = 0
CPU_LIST = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")
GPU_INDEX = re.compile(r"\d+")


@dataclass(frozen=True)
class ChunkedPrefillExecution:
    """Frozen P-side runtime provenance shared by one invocation."""

    model_revision: str
    tokenizer_revision: str
    vllm_commit: str
    attention_backend: str
    p_gpu: str
    p_cpu_affinity: str
    p_numa_node: int
    vllm_dirty: bool = False
    gpu_memory_utilization: float = 0.9
    seed: int = 17

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
        if not GPU_INDEX.fullmatch(self.p_gpu):
            raise ValueError("p_gpu must be one physical GPU index")
        if not CPU_LIST.fullmatch(self.p_cpu_affinity):
            raise ValueError("p_cpu_affinity must use Linux CPU-list syntax")
        if (
            isinstance(self.p_numa_node, bool)
            or not isinstance(self.p_numa_node, int)
            or self.p_numa_node < 0
        ):
            raise ValueError("p_numa_node must be a nonnegative integer")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.attention_backend != ATTENTION_BACKEND:
            raise ValueError(f"attention_backend must be {ATTENTION_BACKEND}")
        if self.model_revision != MODEL_REVISION:
            raise ValueError(f"model_revision must be {MODEL_REVISION}")
        if self.tokenizer_revision != TOKENIZER_REVISION:
            raise ValueError(f"tokenizer_revision must be {TOKENIZER_REVISION}")
        if self.p_gpu != P_GPU:
            raise ValueError("the accepted P placement requires physical GPU 0")
        if self.p_cpu_affinity != P_CPU_AFFINITY:
            raise ValueError(f"the accepted P CPU affinity must be {P_CPU_AFFINITY}")
        if self.p_numa_node != P_NUMA_NODE:
            raise ValueError("the accepted P placement requires NUMA node 0")


@dataclass(frozen=True)
class ChunkedPrefillPoint:
    """One explicit P-only chunked-prefill experiment point."""

    id: str
    batch_size: int
    input_tokens: int
    hit_ratio: float
    output_tokens: int
    max_num_batched_tokens: int
    repetitions: int
    warmup_batches: int
    measured_batches: int
    execution_mode: Literal["primary", "feasibility_smoke"]

    @property
    def cached_tokens(self) -> int:
        """Return the requested hit floored to complete HMA cache pages."""
        return (
            int(self.input_tokens * self.hit_ratio) // CACHE_PAGE_TOKENS
        ) * CACHE_PAGE_TOKENS

    @property
    def computed_tokens(self) -> int:
        """Return the context tokens computed for each request."""
        return self.input_tokens - self.cached_tokens

    @property
    def aligned_hit_ratio(self) -> float:
        """Return the physical cache-hit ratio implied by complete pages."""
        return self.cached_tokens / self.input_tokens


@dataclass(frozen=True)
class PreparedRequest:
    """A deterministic target prompt and its cache-preparation prefix."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    planned_cached_tokens: int
    warm_prompt_token_ids: tuple[int, ...] | None


class PartialSampleError(RuntimeError):
    """Carry raw setup or invalid-target evidence out of a failed sample."""

    def __init__(
        self,
        error: Exception,
        *,
        failure_stage: str,
        cache_warm_observations: tuple[BatchObservation, ...],
        cache_warm_index: int | None = None,
        observation: BatchObservation | None = None,
    ) -> None:
        super().__init__(str(error))
        self.original_error = error
        self.failure_stage = failure_stage
        self.cache_warm_observations = cache_warm_observations
        self.cache_warm_index = cache_warm_index
        self.observation = observation


class ChunkedPrefillRuntime(Protocol):
    """Public offline runtime boundary for one persistent budget group."""

    def provenance(self) -> dict[str, Any]: ...

    def wait_idle(self) -> None: ...

    def reset_prefix_cache(self) -> bool: ...

    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[
    [int, dict[str, Any], Path],
    ChunkedPrefillRuntime,
]


def _load_json_object(path: Path) -> dict[str, object]:
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


def _parse_point(value: object) -> ChunkedPrefillPoint:
    if not isinstance(value, dict):
        raise ValueError("each point must be a JSON object")
    fields = {
        "id",
        "batch_size",
        "input_tokens",
        "hit_ratio",
        "output_tokens",
        "max_num_batched_tokens",
        "repetitions",
        "warmup_batches",
        "measured_batches",
        "execution_mode",
    }
    unexpected = sorted(value.keys() - fields)
    missing = sorted(fields - value.keys())
    if unexpected:
        raise ValueError(f"point contains unexpected fields: {', '.join(unexpected)}")
    if missing:
        raise ValueError(f"point is missing fields: {', '.join(missing)}")
    try:
        point_id = value["id"]
        input_tokens = _positive_int(value["input_tokens"], "input_tokens")
        batch_size = _positive_int(value["batch_size"], "batch_size")
        output_tokens = _positive_int(value["output_tokens"], "output_tokens")
        token_budget = _positive_int(
            value["max_num_batched_tokens"],
            "max_num_batched_tokens",
        )
        repetitions = _positive_int(value["repetitions"], "repetitions")
        warmup_batches = _positive_int(value["warmup_batches"], "warmup_batches")
        measured_batches = _positive_int(
            value["measured_batches"],
            "measured_batches",
        )
        hit_ratio_value = value["hit_ratio"]
        execution_mode = value["execution_mode"]
    except KeyError as error:
        raise ValueError(f"point is missing field: {error.args[0]}") from error
    if not isinstance(point_id, str) or not point_id:
        raise ValueError("point id must be a non-empty string")
    if input_tokens != INPUT_TOKENS:
        raise ValueError(f"input_tokens must be exactly {INPUT_TOKENS}")
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"unsupported batch_size: {batch_size}")
    if output_tokens != 1:
        raise ValueError("output_tokens must be exactly 1")
    if token_budget not in TOKEN_BUDGETS:
        raise ValueError(f"unsupported max_num_batched_tokens: {token_budget}")
    if (
        isinstance(hit_ratio_value, bool)
        or not isinstance(hit_ratio_value, (int, float))
        or float(hit_ratio_value) not in HIT_RATIOS
    ):
        raise ValueError("hit_ratio must be 0, 0.75, or 0.9")
    if execution_mode not in {"primary", "feasibility_smoke"}:
        raise ValueError("execution_mode must be primary or feasibility_smoke")
    sample_shape = (repetitions, warmup_batches, measured_batches)
    if execution_mode == "primary" and sample_shape != (
        PRIMARY_REPETITIONS,
        PRIMARY_WARMUP_BATCHES,
        PRIMARY_MEASURED_BATCHES,
    ):
        raise ValueError("primary points require 3 runs, 1 warmup, and 3 batches")
    if execution_mode == "feasibility_smoke" and sample_shape != (1, 1, 1):
        raise ValueError("smoke points require one run, warmup, and measured batch")
    return ChunkedPrefillPoint(
        id=point_id,
        batch_size=batch_size,
        input_tokens=input_tokens,
        hit_ratio=float(hit_ratio_value),
        output_tokens=output_tokens,
        max_num_batched_tokens=token_budget,
        repetitions=repetitions,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
        execution_mode=execution_mode,
    )


def load_chunked_prefill_plan(path: Path) -> tuple[ChunkedPrefillPoint, ...]:
    """Load an explicit point list without expanding parameter combinations."""
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
    return points


def group_points_by_token_budget(
    points: tuple[ChunkedPrefillPoint, ...],
) -> tuple[tuple[int, tuple[ChunkedPrefillPoint, ...]], ...]:
    """Group explicit points for persistent engines without changing order."""
    grouped: dict[int, list[ChunkedPrefillPoint]] = {}
    for point in points:
        grouped.setdefault(point.max_num_batched_tokens, []).append(point)
    return tuple((budget, tuple(group)) for budget, group in grouped.items())


def build_engine_config(
    token_budget: int,
    execution: ChunkedPrefillExecution,
) -> dict[str, Any]:
    """Build the frozen public-LLM configuration for one budget group."""
    return {
        "model": MODEL,
        "model_revision": execution.model_revision,
        "tokenizer_revision": execution.tokenizer_revision,
        "vllm_commit": execution.vllm_commit,
        "vllm_dirty": execution.vllm_dirty,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "device": "cuda",
        "cuda_visible_devices": execution.p_gpu,
        "cpu_affinity": execution.p_cpu_affinity,
        "numa_node": execution.p_numa_node,
        "attention_backend": execution.attention_backend,
        "block_size": BLOCK_TOKENS,
        "cache_alignment_tokens": CACHE_PAGE_TOKENS,
        "kv_cache_dtype": "bfloat16",
        "language_model_only": True,
        "mamba_cache_mode": "align",
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_model_len": INPUT_TOKENS + 1,
        "max_num_seqs": max(BATCH_SIZES),
        "max_num_batched_tokens": token_budget,
        "gpu_memory_utilization": execution.gpu_memory_utilization,
        "seed": execution.seed,
        "async_scheduling": False,
        "enable_logging_iteration_details": True,
        "runtime_environment": dict(FIXED_BATCH_RUNTIME_ENVIRONMENT),
    }


def _prompt_tokens(seed: int, request_index: int) -> tuple[int, ...]:
    state = (seed + request_index * 0x9E3779B9) & 0xFFFFFFFF
    tokens = []
    for _ in range(INPUT_TOKENS):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        tokens.append(1_024 + state % 100_000)
    return tuple(tokens)


def prepare_requests(
    point: ChunkedPrefillPoint,
    *,
    seed: int,
) -> tuple[PreparedRequest, ...]:
    """Construct exact stable prompts with request-isolated first blocks."""
    requests = []
    for index in range(point.batch_size):
        prompt = _prompt_tokens(seed, index)
        warm_prompt = prompt[: point.cached_tokens + 1] if point.cached_tokens else None
        requests.append(
            PreparedRequest(
                request_id=f"{point.id}-request-{index:02d}",
                prompt_token_ids=prompt,
                planned_cached_tokens=point.cached_tokens,
                warm_prompt_token_ids=warm_prompt,
            )
        )
    first_blocks = {request.prompt_token_ids[:BLOCK_TOKENS] for request in requests}
    if len(first_blocks) != len(requests):
        raise RuntimeError("deterministic request isolation blocks collided")
    return tuple(requests)


def derive_chunked_prefill_sample(
    point: ChunkedPrefillPoint,
    observation: BatchObservation,
) -> dict[str, float]:
    """Validate one target batch and derive its P-engine-local metrics."""
    if len(observation.requests) != point.batch_size:
        raise ValueError("runtime returned the wrong request count")
    for request in observation.requests:
        if request.prompt_tokens != point.input_tokens:
            raise ValueError("runtime returned an unexpected prompt length")
        if request.cached_tokens != point.cached_tokens:
            raise ValueError("target cached-token count differs from the point plan")
        if len(request.output_token_ids) != point.output_tokens:
            raise ValueError("runtime returned an unexpected output length")
    if not observation.iterations:
        raise ValueError("target emitted no context iterations")
    for iteration in observation.iterations:
        if not math.isfinite(iteration.elapsed_ms) or iteration.elapsed_ms <= 0:
            raise ValueError("context iteration latency must be positive and finite")
        if not 0 < iteration.context_requests <= point.batch_size:
            raise ValueError("context request count differs from the point plan")
        if not 0 < iteration.context_tokens <= point.max_num_batched_tokens:
            raise ValueError("context iteration exceeds the batch-wide token budget")
        if iteration.generation_requests or iteration.generation_tokens:
            raise ValueError("target iterations must contain only context work")
    expected_context_tokens = point.batch_size * point.computed_tokens
    context_tokens = sum(
        iteration.context_tokens for iteration in observation.iterations
    )
    if context_tokens != expected_context_tokens:
        raise ValueError("total context tokens differ from the point plan")
    latency_ms = sum(iteration.elapsed_ms for iteration in observation.iterations)
    return {
        "prefill_completion_latency_ms": latency_ms,
        "context_iterations": float(len(observation.iterations)),
        "computed_token_throughput_per_s": context_tokens / (latency_ms / 1_000),
    }


def _observation_dict(observation: BatchObservation) -> dict[str, Any]:
    return {
        "wall_time_ms": observation.wall_time_ms,
        "requests": [asdict(request) for request in observation.requests],
        "iterations": [asdict(iteration) for iteration in observation.iterations],
    }


def _validate_warm_observation(
    observation: BatchObservation,
    *,
    prompt_tokens: int,
    token_budget: int,
) -> None:
    if len(observation.requests) != 1:
        raise ValueError("cache warm returned the wrong request count")
    request = observation.requests[0]
    if request.prompt_tokens != prompt_tokens:
        raise ValueError("cache warm returned an unexpected prompt length")
    if request.cached_tokens != 0:
        raise ValueError("cache warm unexpectedly reused cached tokens")
    if len(request.output_token_ids) != 1:
        raise ValueError("cache warm returned an unexpected output length")
    if not observation.iterations:
        raise ValueError("cache warm emitted no iteration-detail records")
    for iteration in observation.iterations:
        if not math.isfinite(iteration.elapsed_ms) or iteration.elapsed_ms <= 0:
            raise ValueError("cache warm iteration latency must be positive and finite")
        if (
            iteration.context_requests != 1
            or not 0 < iteration.context_tokens <= token_budget
        ):
            raise ValueError("cache warm context work differs from its budget")
        if iteration.generation_requests or iteration.generation_tokens:
            raise ValueError("cache warm iterations must contain only context work")
    if sum(item.context_tokens for item in observation.iterations) != prompt_tokens:
        raise ValueError("cache warm context tokens differ from its prompt")


def _run_sample(
    point: ChunkedPrefillPoint,
    requests: tuple[PreparedRequest, ...],
    runtime: ChunkedPrefillRuntime,
) -> tuple[tuple[BatchObservation, ...], BatchObservation, dict[str, float]]:
    cache_warm_observations: list[BatchObservation] = []
    try:
        runtime.wait_idle()
    except Exception as error:
        raise PartialSampleError(
            error,
            failure_stage="wait_idle_before_reset",
            cache_warm_observations=(),
        ) from error
    try:
        reset = runtime.reset_prefix_cache()
    except Exception as error:
        raise PartialSampleError(
            error,
            failure_stage="cache_reset_runtime",
            cache_warm_observations=(),
        ) from error
    if not reset:
        error = RuntimeError("prefix-cache reset failed")
        raise PartialSampleError(
            error,
            failure_stage="cache_reset_rejected",
            cache_warm_observations=(),
        ) from error
    for warm_index, request in enumerate(requests, start=1):
        if request.warm_prompt_token_ids is None:
            continue
        try:
            warm = runtime.generate(
                (request.warm_prompt_token_ids,),
                max_tokens=1,
                ignore_eos=True,
            )
        except Exception as error:
            raise PartialSampleError(
                error,
                failure_stage="cache_warm_runtime",
                cache_warm_observations=tuple(cache_warm_observations),
                cache_warm_index=warm_index,
            ) from error
        try:
            _validate_warm_observation(
                warm,
                prompt_tokens=point.cached_tokens + 1,
                token_budget=point.max_num_batched_tokens,
            )
        except ValueError as error:
            raise PartialSampleError(
                error,
                failure_stage="cache_warm_validation",
                cache_warm_observations=tuple(cache_warm_observations),
                cache_warm_index=warm_index,
                observation=warm,
            ) from error
        cache_warm_observations.append(warm)
    try:
        runtime.wait_idle()
    except Exception as error:
        raise PartialSampleError(
            error,
            failure_stage="wait_idle_after_cache_warm",
            cache_warm_observations=tuple(cache_warm_observations),
        ) from error
    try:
        target = runtime.generate(
            tuple(request.prompt_token_ids for request in requests),
            max_tokens=1,
            ignore_eos=True,
        )
    except Exception as error:
        raise PartialSampleError(
            error,
            failure_stage="target_runtime",
            cache_warm_observations=tuple(cache_warm_observations),
        ) from error
    try:
        sample = derive_chunked_prefill_sample(point, target)
    except ValueError as error:
        raise PartialSampleError(
            error,
            failure_stage="target_validation",
            cache_warm_observations=tuple(cache_warm_observations),
            observation=target,
        ) from error
    return tuple(cache_warm_observations), target, sample


def _sample_artifact(
    cache_warm_observations: tuple[BatchObservation, ...],
    target: BatchObservation,
    sample: dict[str, float],
) -> dict[str, Any]:
    return {
        "cache_warm_observations": [
            _observation_dict(observation) for observation in cache_warm_observations
        ],
        "observation": _observation_dict(target),
        "derived_sample": sample,
    }


def _partial_sample_artifact(error: PartialSampleError) -> dict[str, Any]:
    return {
        "failure_stage": error.failure_stage,
        "cache_warm_index": error.cache_warm_index,
        "cache_warm_observations": [
            _observation_dict(observation)
            for observation in error.cache_warm_observations
        ],
        "observation": (
            None if error.observation is None else _observation_dict(error.observation)
        ),
    }


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
) -> dict[str, dict[str, float]]:
    """Summarize the three measured target batches from one repetition."""
    if not samples:
        raise ValueError("a run requires measured samples")
    return {
        metric: {
            "p50": _percentile([sample[metric] for sample in samples], 50),
            "p90": _percentile([sample[metric] for sample in samples], 90),
        }
        for metric in samples[0]
    }


def summarize_point_runs(
    run_summaries: Sequence[dict[str, dict[str, float]]],
    samples: Sequence[dict[str, float]],
) -> dict[str, Any]:
    """Summarize run p50 stability and the global nine-sample distribution."""
    if not run_summaries or not samples:
        raise ValueError("a point requires run summaries and measured samples")
    metrics = {}
    for metric in run_summaries[0]:
        run_p50_values = [summary[metric]["p50"] for summary in run_summaries]
        mean = statistics.fmean(run_p50_values)
        stddev = statistics.stdev(run_p50_values) if len(run_p50_values) > 1 else 0.0
        values = [sample[metric] for sample in samples]
        metrics[metric] = {
            "run_p50_values": run_p50_values,
            "mean": mean,
            "cv": stddev / mean if mean else 0.0,
            "global_p50": _percentile(values, 50),
            "global_p90": _percentile(values, 90),
        }
    latency = metrics["prefill_completion_latency_ms"]
    return {
        "status": "valid",
        "global_sample_count": len(samples),
        "metrics": metrics,
        "noisy": latency["cv"] > NOISY_CV,
    }


def _classify_failure(error: BaseException) -> Literal["failed", "unsupported"]:
    error_text = str(error).lower()
    if any(marker in error_text for marker in CAPACITY_ERROR_MARKERS):
        return "unsupported"
    return "failed"


def _run_point(
    point: ChunkedPrefillPoint,
    point_dir: Path,
    execution: ChunkedPrefillExecution,
    runtime: ChunkedPrefillRuntime,
    engine_dir: Path,
) -> dict[str, Any]:
    requests = prepare_requests(point, seed=execution.seed)
    point_dir.mkdir(parents=True, exist_ok=False)
    _write_json(point_dir / "point.json", asdict(point))
    _write_json(
        point_dir / "requests.json",
        [asdict(request) for request in requests],
    )
    _write_json(
        point_dir / "engine-reference.json",
        {
            "max_num_batched_tokens": point.max_num_batched_tokens,
            "engine_dir": str(Path("engines") / engine_dir.name),
        },
    )
    run_summaries = []
    measured_samples = []
    active_run_dir: Path | None = None
    active_phase = "point_initialization"
    active_batch = 0
    try:
        for run_index in range(1, point.repetitions + 1):
            run_dir = point_dir / f"run-{run_index:02d}"
            run_dir.mkdir()
            active_run_dir = run_dir
            for batch_index in range(1, point.warmup_batches + 1):
                active_phase = "warmup"
                active_batch = batch_index
                cache_warm, target, sample = _run_sample(
                    point,
                    requests,
                    runtime,
                )
                _write_json(
                    run_dir / f"warmup-{batch_index:02d}.json",
                    _sample_artifact(cache_warm, target, sample),
                )
            run_samples = []
            for batch_index in range(1, point.measured_batches + 1):
                active_phase = "measured"
                active_batch = batch_index
                cache_warm, target, sample = _run_sample(
                    point,
                    requests,
                    runtime,
                )
                run_samples.append(sample)
                measured_samples.append(sample)
                _write_json(
                    run_dir / f"measured-{batch_index:02d}.json",
                    _sample_artifact(cache_warm, target, sample),
                )
            active_phase = "run_summary"
            active_batch = 0
            run_summary = summarize_run_samples(run_samples)
            _write_json(run_dir / "run-summary.json", run_summary)
            run_summaries.append(run_summary)
        active_run_dir = None
        active_phase = "point_summary"
        active_batch = 0
        summary = {
            "point_id": point.id,
            "metric_label": "P-side prefill-completion latency",
            "batch_size": point.batch_size,
            "requested_hit_ratio": point.hit_ratio,
            "planned_cached_tokens": point.cached_tokens,
            "aligned_hit_ratio": point.aligned_hit_ratio,
            "max_num_batched_tokens": point.max_num_batched_tokens,
            **summarize_point_runs(run_summaries, measured_samples),
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
        status = _classify_failure(error)
        retained_error = (
            error.original_error if isinstance(error, PartialSampleError) else error
        )
        failure = {
            "status": status,
            "phase": active_phase,
            "batch": active_batch,
            "error_type": type(retained_error).__name__,
            "error": error_text,
        }
        if active_run_dir is not None:
            if isinstance(error, PartialSampleError):
                partial = _partial_sample_artifact(error)
                _write_json(
                    active_run_dir / "partial-sample.json",
                    partial,
                )
            if isinstance(error, PartialSampleError) and error.observation is not None:
                _write_json(
                    active_run_dir / "invalid-observation.json",
                    {
                        "validation_stage": error.failure_stage,
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


def run_chunked_prefill_profile(
    plan_path: Path,
    results_dir: Path,
    *,
    execution: ChunkedPrefillExecution,
    runtime_factory: RuntimeFactory | None = None,
) -> dict[str, Any]:
    """Run an explicit plan with one persistent P engine per token budget."""
    if runtime_factory is None:
        if execution.vllm_dirty:
            raise ValueError("production execution requires a clean vLLM checkout")
        from benchmarks.ds4_profile.chunked_prefill_runtime import (
            GroupedSubprocessRuntimeFactory,
        )

        runtime_factory = GroupedSubprocessRuntimeFactory()
    points = load_chunked_prefill_plan(plan_path)
    groups = group_points_by_token_budget(points)
    if results_dir.exists():
        raise FileExistsError(f"results directory already exists: {results_dir}")
    results_dir.mkdir(parents=True)
    engine_groups = [
        {
            "max_num_batched_tokens": budget,
            "point_ids": [point.id for point in group],
        }
        for budget, group in groups
    ]
    _write_json(
        results_dir / "resolved-plan.json",
        {
            "schema_version": 1,
            "execution": asdict(execution),
            "points": [asdict(point) for point in points],
            "engine_groups": engine_groups,
        },
    )
    _write_json(
        results_dir / "execution-order.json",
        {
            "points": [
                {
                    "index": index,
                    "point_id": point.id,
                    "max_num_batched_tokens": point.max_num_batched_tokens,
                }
                for index, point in enumerate(points, start=1)
            ]
        },
    )
    point_statuses = {}
    for budget, group in groups:
        engine_dir = results_dir / "engines" / f"budget-{budget:04d}"
        engine_dir.mkdir(parents=True)
        engine_config = build_engine_config(budget, execution)
        _write_json(engine_dir / "engine-config.json", engine_config)
        runtime: ChunkedPrefillRuntime | None = None
        try:
            runtime = runtime_factory(budget, engine_config, engine_dir)
            _write_json(
                engine_dir / "provenance.json",
                {
                    "model": MODEL,
                    "execution": asdict(execution),
                    "runtime": runtime.provenance(),
                },
            )
            for point in group:
                result = _run_point(
                    point,
                    results_dir / "points" / point.id,
                    execution,
                    runtime,
                    engine_dir,
                )
                point_statuses[point.id] = result["status"]
        except Exception as error:
            error_text = str(error)
            status = _classify_failure(error)
            failure = {
                "status": status,
                "phase": "runtime_initialization",
                "error_type": type(error).__name__,
                "error": error_text,
            }
            _write_json(engine_dir / "engine-failure.json", failure)
            for point in group:
                point_dir = results_dir / "points" / point.id
                point_dir.mkdir(parents=True, exist_ok=False)
                _write_json(point_dir / "point.json", asdict(point))
                _write_json(
                    point_dir / "requests.json",
                    [
                        asdict(request)
                        for request in prepare_requests(
                            point,
                            seed=execution.seed,
                        )
                    ],
                )
                _write_json(
                    point_dir / "engine-reference.json",
                    {
                        "max_num_batched_tokens": budget,
                        "engine_dir": str(Path("engines") / engine_dir.name),
                    },
                )
                _write_json(point_dir / "point-failure.json", failure)
                _write_json(point_dir / "status.json", failure)
                point_statuses[point.id] = status
        finally:
            if runtime is not None:
                runtime.close()
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
    parser.add_argument("--prefill-cpus", required=True)
    parser.add_argument("--prefill-numa-node", required=True, type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    vllm_commit, vllm_dirty = _git_state(repo_root)
    execution = ChunkedPrefillExecution(
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        vllm_commit=vllm_commit,
        vllm_dirty=vllm_dirty,
        attention_backend=args.attention_backend,
        p_gpu=args.p_gpu,
        p_cpu_affinity=args.prefill_cpus,
        p_numa_node=args.prefill_numa_node,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    if args.dry_run:
        points = load_chunked_prefill_plan(args.plan)
        groups = group_points_by_token_budget(points)
        resolved = {
            "schema_version": 1,
            "execution": asdict(execution),
            "points": [asdict(point) for point in points],
            "engine_groups": [
                {
                    "max_num_batched_tokens": budget,
                    "point_ids": [point.id for point in group],
                }
                for budget, group in groups
            ],
            "engine_configs": [
                build_engine_config(budget, execution) for budget, _group in groups
            ],
        }
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0
    result = run_chunked_prefill_profile(
        args.plan,
        args.results_dir,
        execution=execution,
    )
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
