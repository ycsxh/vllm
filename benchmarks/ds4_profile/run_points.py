# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run explicit controlled-cache points against the fixed DS4 1P1D runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import regex as re

from benchmarks.ds4_profile.run_pd import (
    EFFECTIVE_HMA_PAGE_TOKENS,
    PORTS,
    RUNTIME_ENVIRONMENT_NAMES,
    LaunchConfig,
    SubprocessRuntime,
    build_plan,
    running_plan,
)

MODEL = "Qwen/Qwen3.5-4B"
POINT_FIELDS = {
    "hit_ratio",
    "id",
    "max_concurrency",
    "max_num_batched_tokens",
    "num_prompts",
    "output_tokens",
    "repetitions",
    "source_request_id",
}


class Tokenizer(Protocol):
    """Tokenizer operations required to construct controlled prompts."""

    name_or_path: str
    init_kwargs: dict[str, Any]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


TokenizerLoader = Callable[..., Tokenizer]


class PointRuntime(Protocol):
    """External HTTP and benchmark-command boundary for a measured point."""

    def get_text(self, url: str, timeout: float) -> str: ...

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> Any: ...

    def run(
        self,
        command: tuple[str, ...],
        *,
        environment: dict[str, str],
        cwd: Path,
    ) -> None: ...

    def sleep(self, seconds: float) -> None: ...


class SubprocessPointRuntime(SubprocessRuntime):
    """Production point runtime using the exact child environment."""

    def run(
        self,
        command: tuple[str, ...],
        *,
        environment: dict[str, str],
        cwd: Path,
    ) -> None:
        subprocess.run(command, check=True, cwd=cwd, env=environment)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class ExperimentPoint:
    """One explicitly selected experiment point."""

    id: str
    source_request_id: str
    hit_ratio: float
    max_num_batched_tokens: int
    max_concurrency: int
    output_tokens: int
    num_prompts: int
    repetitions: int


@dataclass(frozen=True)
class PreparedRequest:
    """One measured request and its exact planned warm prefix."""

    request_id: str
    prompt: str
    prompt_ids: list[int]
    input_tokens: int
    isolation_ids: list[int]
    planned_cached_tokens: int
    warm_prefix: str | None


@dataclass(frozen=True)
class PreparedPoint:
    """A point expanded only into its explicit repeated request set."""

    point: ExperimentPoint
    requests: tuple[PreparedRequest, ...]
    isolation_block_tokens: int
    cache_alignment_tokens: int


PROMETHEUS_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\s+\d+)?$"
)
PROMETHEUS_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')
PROMPT_TOKENS = "vllm:prompt_tokens_by_source_total"
NIXL_FAILURE_COUNTERS = (
    "vllm:nixl_num_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total",
    "vllm:nixl_num_kv_expired_reqs_total",
)
POINT_SUMMARY_METRICS = (
    "actual_p_hit_ratio",
    "output_throughput",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load JSON object from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_point(value: object) -> ExperimentPoint:
    if not isinstance(value, dict):
        raise ValueError("each point must be a JSON object")
    unexpected = sorted(value.keys() - POINT_FIELDS)
    missing = sorted(POINT_FIELDS - value.keys())
    if unexpected:
        raise ValueError(f"point contains unexpected fields: {', '.join(unexpected)}")
    if missing:
        raise ValueError(f"point is missing fields: {', '.join(missing)}")

    point_id = value["id"]
    source_request_id = value["source_request_id"]
    if not isinstance(point_id, str) or not point_id:
        raise ValueError("point id must be a non-empty string")
    if not isinstance(source_request_id, str) or not source_request_id:
        raise ValueError("source_request_id must be a non-empty string")
    hit_ratio = value["hit_ratio"]
    if (
        isinstance(hit_ratio, bool)
        or not isinstance(hit_ratio, (int, float))
        or not 0 <= hit_ratio < 1
    ):
        raise ValueError("hit_ratio must be a number in [0, 1)")

    num_prompts = _positive_int(value["num_prompts"], "num_prompts")
    if not 20 <= num_prompts <= 50:
        raise ValueError("num_prompts must be between 20 and 50")
    repetitions = _positive_int(value["repetitions"], "repetitions")
    if repetitions != 3:
        raise ValueError("repetitions must be exactly 3 for Ticket 3")
    return ExperimentPoint(
        id=point_id,
        source_request_id=source_request_id,
        hit_ratio=float(hit_ratio),
        max_num_batched_tokens=_positive_int(
            value["max_num_batched_tokens"], "max_num_batched_tokens"
        ),
        max_concurrency=_positive_int(value["max_concurrency"], "max_concurrency"),
        output_tokens=_positive_int(value["output_tokens"], "output_tokens"),
        num_prompts=num_prompts,
        repetitions=repetitions,
    )


def load_experiment_plan(path: Path) -> tuple[ExperimentPoint, ...]:
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
    point_ids = [point.id for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("point ids must be unique")
    return points


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"failed to read {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} must not be empty")
    return rows


def _decode_exact(tokenizer: Tokenizer, token_ids: list[int], purpose: str) -> str:
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if (
        not isinstance(text, str)
        or tokenizer.encode(text, add_special_tokens=False) != token_ids
    ):
        raise ValueError(f"{purpose} does not round trip through the tokenizer")
    return text


def _isolation_block(
    tokenizer: Tokenizer,
    source_request_id: str,
    sample_index: int,
    source_prompt: str,
    block_size: int,
) -> tuple[str, list[int]]:
    identity = f"{source_request_id}@sample-{sample_index:03d}"
    seed = hashlib.sha256(identity.encode()).hexdigest() + " ds4 isolation "
    seed_ids = tokenizer.encode(seed * (block_size + 1), add_special_tokens=False)
    if len(seed_ids) < block_size:
        raise ValueError("tokenizer cannot construct a full isolation block")
    isolation_ids = seed_ids[:block_size]
    isolation_text = _decode_exact(tokenizer, isolation_ids, "isolation block")
    source_ids = tokenizer.encode(source_prompt, add_special_tokens=False)
    combined_ids = tokenizer.encode(
        isolation_text + source_prompt, add_special_tokens=False
    )
    if combined_ids != isolation_ids + source_ids:
        raise ValueError("isolation block changes at the prompt boundary")
    return isolation_text, isolation_ids


def prepare_point(
    point: ExperimentPoint,
    prepared_dir: Path,
    *,
    block_size: int,
    cache_alignment_tokens: int | None = None,
    tokenizer: Tokenizer,
) -> PreparedPoint:
    """Construct deterministic isolated requests for one explicit point."""
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    if cache_alignment_tokens is None:
        cache_alignment_tokens = block_size
    if (
        isinstance(cache_alignment_tokens, bool)
        or not isinstance(cache_alignment_tokens, int)
        or cache_alignment_tokens <= 0
        or cache_alignment_tokens % block_size
    ):
        raise ValueError(
            "cache_alignment_tokens must be a positive multiple of block_size"
        )
    provenance = _load_json_object(prepared_dir / "provenance.json")
    tokenizer_provenance = provenance.get("tokenizer")
    resolved_tokenizer_revision = tokenizer.init_kwargs.get(
        "_ds4_resolved_revision", tokenizer.init_kwargs.get("_commit_hash")
    )
    if tokenizer.name_or_path != MODEL or tokenizer_provenance != {
        "model": MODEL,
        "revision": resolved_tokenizer_revision,
    }:
        raise ValueError("prepared data and loaded tokenizer provenance differ")

    dataset_rows = _load_json_lines(prepared_dir / "dataset.jsonl")
    sidecar_rows = _load_json_lines(prepared_dir / "rows.jsonl")
    if len(dataset_rows) != len(sidecar_rows) or provenance.get("row_count") != len(
        dataset_rows
    ):
        raise ValueError("prepared dataset artifacts have inconsistent row counts")

    source_matches = [
        (dataset, sidecar)
        for dataset, sidecar in zip(dataset_rows, sidecar_rows)
        if sidecar.get("request_id") == point.source_request_id
    ]
    if len(source_matches) != 1:
        raise ValueError("source_request_id must identify exactly one prepared row")
    dataset, sidecar = source_matches[0]
    source_prompt = dataset.get("prompt")
    prompt_ids = sidecar.get("prompt_ids")
    if (
        not isinstance(source_prompt, str)
        or not source_prompt
        or not isinstance(prompt_ids, list)
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in prompt_ids
        )
        or sidecar.get("input_tokens") != len(prompt_ids)
        or tokenizer.encode(source_prompt, add_special_tokens=False) != prompt_ids
    ):
        raise ValueError("prepared source prompt and token IDs are inconsistent")

    requests = []
    isolation_blocks: set[tuple[int, ...]] = set()
    for sample_index in range(point.num_prompts):
        isolation_text, isolation_ids = _isolation_block(
            tokenizer,
            point.source_request_id,
            sample_index,
            source_prompt,
            block_size,
        )
        isolation_key = tuple(isolation_ids)
        if isolation_key in isolation_blocks:
            raise ValueError("isolation blocks must be request-unique")
        isolation_blocks.add(isolation_key)
        prompt = isolation_text + source_prompt
        combined_ids = tokenizer.encode(prompt, add_special_tokens=False)
        planned_cached_tokens = (
            int(len(combined_ids) * point.hit_ratio // cache_alignment_tokens)
            * cache_alignment_tokens
        )
        warm_prefix = None
        if planned_cached_tokens:
            # Disaggregated P computes prompt_len - 1 tokens. Include the next
            # measured token so the planned prefix fills its final cache page.
            warm_prefix = _decode_exact(
                tokenizer,
                combined_ids[: planned_cached_tokens + 1],
                "warm prefix",
            )
        requests.append(
            PreparedRequest(
                request_id=(f"{point.source_request_id}@sample-{sample_index:03d}"),
                prompt=prompt,
                prompt_ids=combined_ids,
                input_tokens=len(combined_ids),
                isolation_ids=isolation_ids,
                planned_cached_tokens=planned_cached_tokens,
                warm_prefix=warm_prefix,
            )
        )
    if point.hit_ratio > 0 and not any(
        request.planned_cached_tokens for request in requests
    ):
        raise ValueError("nonzero hit point does not span one effective cache page")
    return PreparedPoint(
        point,
        tuple(requests),
        isolation_block_tokens=block_size,
        cache_alignment_tokens=cache_alignment_tokens,
    )


def prepare_experiment(
    points: tuple[ExperimentPoint, ...],
    prepared_dir: Path,
    *,
    block_size: int,
    cache_alignment_tokens: int | None = None,
    tokenizer: Tokenizer,
) -> tuple[PreparedPoint, ...]:
    """Prepare every explicit point and reject aligned duplicate conditions."""
    prepared_points = tuple(
        prepare_point(
            point,
            prepared_dir,
            block_size=block_size,
            cache_alignment_tokens=cache_alignment_tokens,
            tokenizer=tokenizer,
        )
        for point in points
    )
    aligned_conditions: dict[tuple[Any, ...], str] = {}
    for prepared in prepared_points:
        point = prepared.point
        condition = (
            point.source_request_id,
            point.max_num_batched_tokens,
            point.max_concurrency,
            point.output_tokens,
            point.num_prompts,
            tuple(request.planned_cached_tokens for request in prepared.requests),
        )
        previous_id = aligned_conditions.get(condition)
        if previous_id is not None:
            raise ValueError(
                f"points {previous_id} and {point.id} have the same aligned prefix"
            )
        aligned_conditions[condition] = point.id
    return prepared_points


def _parse_metrics(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE.fullmatch(line.strip())
        if match is None:
            continue
        labels = tuple(sorted(PROMETHEUS_LABEL.findall(match.group("labels") or "")))
        key = (match.group("name"), labels)
        samples[key] = samples.get(key, 0.0) + float(match.group("value"))
    return samples


def _metric_total(
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
    *,
    source: str | None = None,
) -> float:
    values = []
    for (sample_name, labels), value in samples.items():
        if sample_name != name:
            continue
        label_map = dict(labels)
        if source is None or label_map.get("source") == source:
            values.append(value)
    if not values:
        suffix = f' with source="{source}"' if source is not None else ""
        raise ValueError(f"required metric {name}{suffix} is absent")
    return sum(values)


def _metric_source_deltas(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
) -> dict[str, float]:
    sources = set()
    for sample_name, labels in set(before) | set(after):
        if sample_name != name:
            continue
        source = dict(labels).get("source")
        if source is not None:
            sources.add(source)
    if not sources:
        raise ValueError(f"required labeled metric {name} is absent")
    return {
        source: _delta(before, after, name, source=source) for source in sorted(sources)
    }


def _delta(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
    *,
    source: str | None = None,
) -> float:
    value = _metric_total(after, name, source=source) - _metric_total(
        before, name, source=source
    )
    if value < 0:
        raise ValueError(f"metric {name} decreased during the measured run")
    return value


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _detailed_arrays(
    point: PreparedPoint, official_result: dict[str, Any]
) -> tuple[list[float], list[list[float]], list[int]]:
    required_arrays = (
        "errors",
        "generated_texts",
        "input_lens",
        "itls",
        "output_lens",
        "start_times",
        "ttfts",
    )
    expected_count = len(point.requests)
    for name in required_arrays:
        value = official_result.get(name)
        if not isinstance(value, list) or len(value) != expected_count:
            raise ValueError(f"official detailed result has invalid {name}")
    if official_result.get("completed") != expected_count:
        raise ValueError("official result did not complete every request")
    if official_result.get("failed") != 0:
        raise ValueError("official result reports failed requests")
    if any(official_result["errors"]):
        raise ValueError("official detailed result contains request errors")

    expected_input_lens = [request.input_tokens for request in point.requests]
    if official_result["input_lens"] != expected_input_lens:
        raise ValueError("official input lengths differ from the frozen prompts")
    output_lens = official_result["output_lens"]
    if any(
        isinstance(length, bool)
        or not isinstance(length, int)
        or length != point.point.output_tokens
        for length in output_lens
    ):
        raise ValueError("official output lengths differ from the explicit point")
    ttfts = official_result["ttfts"]
    itls = official_result["itls"]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        for value in ttfts
    ) or any(
        not isinstance(samples, list)
        or len(samples) != output_length - 1
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            for value in samples
        )
        for samples, output_length in zip(itls, output_lens)
    ):
        raise ValueError("official latency samples are missing or invalid")
    return [float(value) for value in ttfts], itls, output_lens


def derive_run_result(
    point: PreparedPoint,
    official_result: dict[str, Any],
    *,
    p_metrics_before: str,
    p_metrics_after: str,
    d_metrics_before: str,
    d_metrics_after: str,
    block_size: int,
) -> dict[str, Any]:
    """Validate one repetition and derive auditable request-level summaries."""
    p_before = _parse_metrics(p_metrics_before)
    p_after = _parse_metrics(p_metrics_after)
    d_before = _parse_metrics(d_metrics_before)
    d_after = _parse_metrics(d_metrics_after)
    p_source_deltas = _metric_source_deltas(p_before, p_after, PROMPT_TOKENS)
    try:
        p_local_compute = p_source_deltas["local_compute"]
        p_local_hit = p_source_deltas["local_cache_hit"]
    except KeyError as error:
        raise ValueError(
            "P prompt token metrics lack required cache sources"
        ) from error
    d_external = _delta(d_before, d_after, PROMPT_TOKENS, source="external_kv_transfer")
    expected_cached = sum(request.planned_cached_tokens for request in point.requests)
    if expected_cached == 0:
        if p_local_hit != 0:
            raise ValueError("0% point observed an unintended local cache hit")
    elif p_local_hit == 0:
        raise ValueError("nonzero planned P cache hit was not observed")
    elif abs(p_local_hit - expected_cached) > block_size * len(point.requests):
        raise ValueError(
            "actual P cache hit differs by more than one block per request"
        )
    if d_external <= 0:
        raise ValueError("D external KV transfer did not increase")

    failure_deltas = {}
    for name in NIXL_FAILURE_COUNTERS:
        value = _delta(p_before, p_after, name) + _delta(d_before, d_after, name)
        failure_deltas[name] = value
        if value:
            raise ValueError(f"failed or expired NIXL counter increased: {name}")

    ttfts, itls, output_lens = _detailed_arrays(point, official_result)
    p_total = sum(p_source_deltas.values())
    if p_total <= 0:
        raise ValueError("P prompt token metrics did not increase")
    total_input_tokens = sum(request.input_tokens for request in point.requests)
    output_throughput = official_result.get("output_throughput")
    if (
        isinstance(output_throughput, bool)
        or not isinstance(output_throughput, (int, float))
        or output_throughput < 0
    ):
        raise ValueError("official output throughput is missing or invalid")
    derived: dict[str, Any] = {
        "status": "valid",
        "point_id": point.point.id,
        "completed": len(point.requests),
        "failed": 0,
        "requested_hit_ratio": point.point.hit_ratio,
        "aligned_planned_hit_ratio": expected_cached / total_input_tokens,
        "planned_cached_tokens": expected_cached,
        "p_local_compute_tokens": p_local_compute,
        "p_local_cache_hit_tokens": p_local_hit,
        "p_prompt_tokens_by_source": p_source_deltas,
        "actual_p_hit_ratio": p_local_hit / p_total,
        "d_external_kv_transfer_tokens": d_external,
        "nixl_failure_deltas": failure_deltas,
        "output_throughput": float(output_throughput),
    }
    for percentile in (50, 90, 95):
        derived[f"p{percentile}_ttft_ms"] = _percentile(ttfts, percentile) * 1000
    if point.point.output_tokens > 1:
        tpots = [
            sum(float(value) for value in samples) / (output_length - 1)
            for samples, output_length in zip(itls, output_lens)
        ]
        for percentile in (50, 90, 95):
            derived[f"p{percentile}_tpot_ms"] = _percentile(tpots, percentile) * 1000
    return derived


def summarize_point_runs(
    point_id: str, runs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Calculate point-level means and CVs from measured run summaries."""
    if not runs:
        raise ValueError("point summary requires measured runs")
    metrics = {}
    noisy_metrics = []
    for name in POINT_SUMMARY_METRICS:
        values = [run.get(name) for run in runs]
        present = [value is not None for value in values]
        if not any(present):
            continue
        if not all(present) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            for value in values
        ):
            raise ValueError(f"run summaries have inconsistent metric {name}")
        numeric_values = [float(value) for value in values]
        mean = statistics.fmean(numeric_values)
        standard_deviation = (
            statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0
        )
        cv = standard_deviation / mean if mean else 0.0
        noisy = cv > 0.05
        metrics[name] = {
            "values": numeric_values,
            "mean": mean,
            "cv": cv,
            "noisy": noisy,
        }
        if noisy:
            noisy_metrics.append(name)
    return {
        "status": "valid",
        "point_id": point_id,
        "repetitions": len(runs),
        "metrics": metrics,
        "noisy_metrics": noisy_metrics,
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def write_point_inputs(point: PreparedPoint, point_dir: Path) -> Path:
    """Freeze one point and its exact measured CustomDataset."""
    point_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = point_dir / "dataset.jsonl"
    dataset_content = "".join(
        json.dumps({"prompt": request.prompt}, separators=(",", ":"), sort_keys=True)
        + "\n"
        for request in point.requests
    )
    dataset_path.write_text(dataset_content, encoding="utf-8")
    _write_json(
        point_dir / "point.json",
        {
            "schema_version": 1,
            "isolation_block_tokens": point.isolation_block_tokens,
            "cache_alignment_tokens": point.cache_alignment_tokens,
            "point": {
                "id": point.point.id,
                "source_request_id": point.point.source_request_id,
                "hit_ratio": point.point.hit_ratio,
                "max_num_batched_tokens": point.point.max_num_batched_tokens,
                "max_concurrency": point.point.max_concurrency,
                "output_tokens": point.point.output_tokens,
                "num_prompts": point.point.num_prompts,
                "repetitions": point.point.repetitions,
            },
            "requests": [
                {
                    "request_id": request.request_id,
                    "input_tokens": request.input_tokens,
                    "prompt_ids": request.prompt_ids,
                    "isolation_ids": request.isolation_ids,
                    "planned_cached_tokens": request.planned_cached_tokens,
                }
                for request in point.requests
            ],
            "dataset_sha256": hashlib.sha256(dataset_content.encode()).hexdigest(),
        },
    )
    return dataset_path


def build_benchmark_command(
    point: PreparedPoint,
    *,
    dataset_path: Path,
    run_dir: Path,
    repo_root: Path,
    tokenizer_path: Path,
) -> tuple[str, ...]:
    """Build the one supported official ``vllm bench serve`` invocation."""
    return (
        str(repo_root / ".venv/bin/python"),
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{PORTS['proxy']}",
        "--endpoint",
        "/v1/completions",
        "--model",
        MODEL,
        "--tokenizer",
        str(tokenizer_path),
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(dataset_path),
        "--num-prompts",
        str(point.point.num_prompts),
        "--max-concurrency",
        str(point.point.max_concurrency),
        "--request-rate",
        "inf",
        "--ready-check-timeout-sec",
        "0",
        "--num-warmups",
        "0",
        "--skip-chat-template",
        "--disable-shuffle",
        "--no-oversample",
        "--custom-output-len",
        str(point.point.output_tokens),
        "--ignore-eos",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(run_dir),
        "--result-filename",
        "bench-result.json",
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "50,90,95,99",
        "--disable-tqdm",
    )


def _wait_idle(runtime: PointRuntime, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        idle = True
        for role in ("prefill", "decode"):
            metrics = _parse_metrics(
                runtime.get_text(
                    f"http://127.0.0.1:{PORTS[role]}/metrics",
                    min(5.0, timeout),
                )
            )
            running = _metric_total(metrics, "vllm:num_requests_running")
            waiting = _metric_total(metrics, "vllm:num_requests_waiting")
            idle = idle and running == 0 and waiting == 0
        if idle:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for P and D to become idle")
        runtime.sleep(min(0.25, timeout))


def _reset_cache(
    runtime: PointRuntime,
    role: str,
    *,
    timeout: float,
    attempts: int = 3,
) -> None:
    url = f"http://127.0.0.1:{PORTS[role]}/reset_prefix_cache"
    for attempt in range(attempts):
        response = runtime.post_json(url, {}, timeout)
        if isinstance(response, dict) and response.get("success") is True:
            return
        if attempt + 1 < attempts:
            runtime.sleep(0.25)
    raise RuntimeError(f"{role} prefix-cache reset failed after {attempts} attempts")


def run_repetition(
    point: PreparedPoint,
    run_dir: Path,
    *,
    dataset_path: Path,
    repo_root: Path,
    tokenizer_path: Path,
    runtime_environment: dict[str, str],
    runtime: PointRuntime | None = None,
    block_size: int,
    request_timeout: float,
) -> dict[str, Any]:
    """Run one controlled repetition and retain any artifacts on failure."""
    runtime = runtime or SubprocessPointRuntime()
    run_dir.mkdir(parents=True, exist_ok=False)
    command = build_benchmark_command(
        point,
        dataset_path=dataset_path,
        run_dir=run_dir,
        repo_root=repo_root,
        tokenizer_path=tokenizer_path,
    )
    _write_json(run_dir / "invocation.json", {"command": list(command)})
    try:
        _wait_idle(runtime, request_timeout)
        _reset_cache(runtime, "prefill", timeout=request_timeout)
        _reset_cache(runtime, "decode", timeout=request_timeout)
        request_url = f"http://127.0.0.1:{PORTS['proxy']}/v1/completions"
        for request in point.requests:
            if request.warm_prefix is None:
                continue
            runtime.post_json(
                request_url,
                {
                    "model": MODEL,
                    "prompt": request.warm_prefix,
                    "max_tokens": 1,
                    "temperature": 0,
                    "seed": 0,
                    "ignore_eos": True,
                    "stream": False,
                },
                request_timeout,
            )
        _wait_idle(runtime, request_timeout)
        _reset_cache(runtime, "decode", timeout=request_timeout)

        metrics_before = {}
        for short_role, role in (("p", "prefill"), ("d", "decode")):
            metrics_before[short_role] = runtime.get_text(
                f"http://127.0.0.1:{PORTS[role]}/metrics", request_timeout
            )
            (run_dir / f"{short_role}-metrics-before.txt").write_text(
                metrics_before[short_role], encoding="utf-8"
            )
        runtime.run(
            command,
            environment=runtime_environment,
            cwd=repo_root,
        )
        official_result = _load_json_object(run_dir / "bench-result.json")
        metrics_after = {}
        for short_role, role in (("p", "prefill"), ("d", "decode")):
            metrics_after[short_role] = runtime.get_text(
                f"http://127.0.0.1:{PORTS[role]}/metrics", request_timeout
            )
            (run_dir / f"{short_role}-metrics-after.txt").write_text(
                metrics_after[short_role], encoding="utf-8"
            )
        derived = derive_run_result(
            point,
            official_result,
            p_metrics_before=metrics_before["p"],
            p_metrics_after=metrics_after["p"],
            d_metrics_before=metrics_before["d"],
            d_metrics_after=metrics_after["d"],
            block_size=block_size,
        )
        _write_json(run_dir / "derived.json", derived)
        return derived
    except BaseException as error:
        _write_json(
            run_dir / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def execute_point(
    point: PreparedPoint,
    base_config: LaunchConfig,
    results_dir: Path,
    *,
    tokenizer_path: Path,
    runtime: Any | None = None,
    block_size: int | None = None,
    cache_alignment_tokens: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Own one point-specific deployment and its three measured repetitions."""
    runtime = runtime or SubprocessPointRuntime()
    if block_size is None:
        block_size = point.isolation_block_tokens
    if cache_alignment_tokens is None:
        cache_alignment_tokens = point.cache_alignment_tokens
    if (
        block_size != point.isolation_block_tokens
        or cache_alignment_tokens != point.cache_alignment_tokens
    ):
        raise ValueError("execution cache geometry differs from prepared point")
    point_dir = results_dir / "points" / point.point.id
    dataset_path = write_point_inputs(point, point_dir)
    config = replace(
        base_config,
        run_dir=point_dir,
        max_num_batched_tokens=point.point.max_num_batched_tokens,
        expected_remote_tokens=None,
    )
    plan = build_plan(config)
    _write_json(point_dir / "server-plan.json", plan.as_dict())
    _write_json(
        point_dir / "provenance.json",
        {
            "model": MODEL,
            "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "tokenizer_path": str(tokenizer_path),
            "vllm_commit": config.vllm_commit,
            "vllm_dirty": config.vllm_dirty,
            "topology": plan.as_dict()["topology"],
            "compatibility": plan.as_dict()["compatibility"],
        },
    )
    try:
        with running_plan(plan, runtime):
            canary_responses = []
            for request in point.requests[: point.point.max_concurrency]:
                canary_responses.append(
                    runtime.post_json(
                        f"http://127.0.0.1:{PORTS['proxy']}/v1/completions",
                        {
                            "model": MODEL,
                            "prompt": request.prompt,
                            "max_tokens": point.point.output_tokens,
                            "temperature": 0,
                            "seed": 0,
                            "ignore_eos": True,
                            "stream": False,
                        },
                        config.request_timeout,
                    )
                )
            _wait_idle(runtime, config.request_timeout)
            _reset_cache(runtime, "prefill", timeout=config.request_timeout)
            _reset_cache(runtime, "decode", timeout=config.request_timeout)
            _write_json(
                point_dir / "engine-warmup.json",
                {
                    "request_count": len(canary_responses),
                    "responses": canary_responses,
                    "caches_reset": True,
                },
            )

            results = []
            for repetition in range(1, point.point.repetitions + 1):
                results.append(
                    run_repetition(
                        point,
                        point_dir / f"run-{repetition:02d}",
                        dataset_path=dataset_path,
                        repo_root=config.repo_root,
                        tokenizer_path=tokenizer_path,
                        runtime_environment=config.runtime_environment,
                        runtime=runtime,
                        block_size=cache_alignment_tokens,
                        request_timeout=config.request_timeout,
                    )
                )
            summary = summarize_point_runs(point.point.id, results)
            _write_json(point_dir / "point-summary.json", summary)
            _write_json(
                point_dir / "status.json",
                {
                    "status": "valid",
                    "completed_repetitions": len(results),
                    "required_repetitions": point.point.repetitions,
                    "noisy_metrics": summary["noisy_metrics"],
                },
            )
            return tuple(results)
    except BaseException as error:
        _write_json(
            point_dir / "point-failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _load_tokenizer(model: str, *, revision: str) -> Tokenizer:
    from transformers import utils

    from benchmarks.ds4_profile.prepare_dataset import (
        _load_tokenizer as load_dataset_tokenizer,
    )
    from benchmarks.ds4_profile.prepare_dataset import _validate_tokenizer

    tokenizer = load_dataset_tokenizer(model, revision=revision)
    _validate_tokenizer(tokenizer, model, revision)
    tokenizer_config = utils.cached_file(
        model,
        "tokenizer_config.json",
        revision=revision,
        local_files_only=True,
    )
    if not isinstance(tokenizer_config, str):
        raise ValueError("pinned tokenizer config is absent from the local cache")
    tokenizer.init_kwargs["_ds4_snapshot_path"] = str(
        Path(tokenizer_config).absolute().parent
    )
    return tokenizer


def _tokenizer_snapshot_path(tokenizer: Tokenizer, revision: str) -> Path:
    value = tokenizer.init_kwargs.get("_ds4_snapshot_path")
    if not isinstance(value, str):
        raise ValueError("loaded tokenizer lacks its immutable snapshot path")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(
            "tokenizer snapshot path must be an existing absolute directory"
        )
    resolved = path.resolve()
    if resolved.name != revision:
        raise ValueError("tokenizer snapshot path does not match the pinned revision")
    return resolved


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
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--prefill-cpus", required=True)
    parser.add_argument("--prefill-numa-node", required=True, type=int)
    parser.add_argument("--decode-cpus", required=True)
    parser.add_argument("--decode-numa-node", required=True, type=int)
    parser.add_argument("--readiness-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _base_launch_config(
    args: argparse.Namespace,
    repo_root: Path,
    *,
    invocation: tuple[str, ...],
) -> LaunchConfig:
    commit, dirty = _git_state(repo_root)
    return LaunchConfig(
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        attention_backend=args.attention_backend,
        prefill_cpus=args.prefill_cpus,
        prefill_numa_node=args.prefill_numa_node,
        decode_cpus=args.decode_cpus,
        decode_numa_node=args.decode_numa_node,
        run_dir=args.results_dir.resolve(),
        repo_root=repo_root,
        vllm_commit=commit,
        vllm_dirty=dirty,
        runtime_environment={
            name: os.environ[name]
            for name in RUNTIME_ENVIRONMENT_NAMES
            if name in os.environ
        },
        expected_remote_tokens=None,
        readiness_timeout=args.readiness_timeout,
        request_timeout=args.request_timeout,
        shutdown_timeout=args.shutdown_timeout,
        launcher_invocation=invocation,
    )


def _dry_run_summary(
    points: tuple[PreparedPoint, ...],
    base_config: LaunchConfig,
    results_dir: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    point_summaries = []
    for point in points:
        config = replace(
            base_config,
            run_dir=results_dir / "points" / point.point.id,
            max_num_batched_tokens=point.point.max_num_batched_tokens,
            expected_remote_tokens=None,
        )
        point_summaries.append(
            {
                "id": point.point.id,
                "request_count": len(point.requests),
                "planned_cached_tokens": sum(
                    request.planned_cached_tokens for request in point.requests
                ),
                "server_plan": build_plan(config).as_dict(),
                "benchmark_command": list(
                    build_benchmark_command(
                        point,
                        dataset_path=(
                            results_dir / "points" / point.point.id / "dataset.jsonl"
                        ),
                        run_dir=results_dir / "points" / point.point.id / "run-01",
                        repo_root=base_config.repo_root,
                        tokenizer_path=tokenizer_path,
                    )
                ),
            }
        )
    return {
        "schema_version": 1,
        "tokenizer_path": str(tokenizer_path),
        "points": point_summaries,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    tokenizer_loader: TokenizerLoader = _load_tokenizer,
) -> int:
    """Run the explicit Ticket 3 plan and return its process exit status."""
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    cli_args = tuple(sys.argv[1:] if argv is None else argv)
    invocation = (
        str(repo_root / ".venv/bin/python"),
        "-m",
        __package__ + ".run_points",
        *cli_args,
    )
    tokenizer = tokenizer_loader(MODEL, revision=args.tokenizer_revision)
    points = prepare_experiment(
        load_experiment_plan(args.plan),
        args.prepared_dir,
        block_size=128,
        cache_alignment_tokens=EFFECTIVE_HMA_PAGE_TOKENS,
        tokenizer=tokenizer,
    )
    tokenizer_path = _tokenizer_snapshot_path(tokenizer, args.tokenizer_revision)
    base_config = _base_launch_config(args, repo_root, invocation=invocation)
    results_dir = args.results_dir.resolve()
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_summary(
                    points,
                    base_config,
                    results_dir,
                    tokenizer_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if base_config.vllm_dirty:
        raise ValueError("execution requires a clean vLLM working tree")
    if results_dir.exists():
        raise ValueError("results directory must not already exist")

    results_dir.mkdir(parents=True)
    shutil.copytree(args.prepared_dir, results_dir / "prepared")
    _write_json(results_dir / "plan.json", _load_json_object(args.plan))
    manifest = {
        "schema_version": 1,
        "status": "running",
        "vllm_commit": base_config.vllm_commit,
        "vllm_dirty": base_config.vllm_dirty,
        "model_revision": base_config.model_revision,
        "tokenizer_revision": base_config.tokenizer_revision,
        "tokenizer_path": str(tokenizer_path),
        "invocation": list(invocation),
        "points": [],
    }
    _write_json(results_dir / "run-manifest.json", manifest)
    failed = False
    for point_index, point in enumerate(points):
        try:
            results = execute_point(
                point,
                base_config,
                results_dir,
                tokenizer_path=tokenizer_path,
            )
            manifest["points"].append(
                {
                    "id": point.point.id,
                    "status": "valid",
                    "completed_repetitions": len(results),
                }
            )
        except KeyboardInterrupt as error:
            manifest["points"].append(
                {
                    "id": point.point.id,
                    "status": "interrupted",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            manifest["status"] = "interrupted"
            _write_json(results_dir / "run-manifest.json", manifest)
            return 130
        except Exception as error:
            failed = True
            manifest["points"].append(
                {
                    "id": point.point.id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if point_index == 0:
                break
    manifest["status"] = "failed" if failed else "valid"
    _write_json(results_dir / "run-manifest.json", manifest)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
