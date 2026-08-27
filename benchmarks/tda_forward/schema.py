# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Frozen configuration and run-matrix contracts for Issue #18."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config") / "issue18.json"
RunMode = Literal["oracle", "disabled", "enabled", "healthy", "stalled"]
RUN_SOURCE_ARTIFACTS = (
    "workload.json",
    "manifest.json",
    "logs/prefill.log",
    "logs/decode.log",
    "logs/proxy.log",
    "raw/responses.jsonl",
    "raw/cpu.jsonl",
    "raw/measurement-window.json",
    "raw/process-exits.json",
    "raw/proxy-state.json",
    "raw/prefill-metrics-before.txt",
    "raw/prefill-metrics-after.txt",
    "raw/decode-metrics-before.txt",
    "raw/decode-metrics-after.txt",
    "raw/prefill-instrumentation.json",
    "raw/decode-instrumentation.json",
)


@dataclass(frozen=True)
class WorkloadConfig:
    session_count: int
    turns_per_session: int
    warmup_turns: int
    max_tokens: int
    request_timeout_seconds: float
    run_timeout_seconds: float
    shared_prefix_words: int
    pressure_words: int
    g_seconds: float


@dataclass(frozen=True)
class PublisherConfig:
    healthy_hwm: int
    healthy_queue_size: int
    stalled_hwm: int
    stalled_queue_size: int
    stalled_consumer_delay_seconds: float


@dataclass(frozen=True)
class ThresholdConfig:
    minimum_throughput_ratio: float
    maximum_latency_ratio: float
    maximum_stalled_enqueue_p99_ms: float


@dataclass(frozen=True)
class AcceptanceConfig:
    raw: dict[str, Any]
    seeds: tuple[int, ...]
    workload: WorkloadConfig
    publisher: PublisherConfig
    thresholds: ThresholdConfig

    @property
    def deployment(self) -> dict[str, Any]:
        return self.raw["deployment"]


@dataclass(frozen=True)
class RunSpec:
    seed: int
    mode: RunMode
    model: str
    block_size: int
    max_model_len: int
    max_num_batched_tokens: int
    gpu_memory_utilization: float
    session_count: int
    turns_per_session: int
    max_tokens: int
    events_enabled: bool
    consumer_delay_seconds: float
    publisher_hwm: int
    publisher_queue_size: int
    conclusion_scope: str

    @property
    def run_id(self) -> str:
        return f"{self.mode}-seed-{self.seed}"


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AcceptanceConfig:
    """Load and validate the committed acceptance definition."""
    raw = _object(json.loads(path.read_text(encoding="utf-8")), "config")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported acceptance config schema")
    seeds = raw.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(set(seeds)) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("at least three distinct integer seeds are required")
    deployment = _object(raw.get("deployment"), "deployment")
    workload_value = _object(raw.get("workload"), "workload")
    publisher_value = _object(raw.get("publisher"), "publisher")
    threshold_value = _object(raw.get("thresholds"), "thresholds")
    config = AcceptanceConfig(
        raw=raw,
        seeds=tuple(seeds),
        workload=WorkloadConfig(**workload_value),
        publisher=PublisherConfig(**publisher_value),
        thresholds=ThresholdConfig(**threshold_value),
    )
    positive = (
        deployment["block_size"],
        deployment["max_model_len"],
        deployment["max_num_batched_tokens"],
        config.workload.session_count,
        config.workload.turns_per_session,
        config.workload.max_tokens,
    )
    if any(isinstance(value, bool) or value <= 0 for value in positive):
        raise ValueError("deployment and workload sizes must be positive")
    if not 0 < config.thresholds.minimum_throughput_ratio <= 1:
        raise ValueError("minimum throughput ratio must be in (0, 1]")
    if config.thresholds.maximum_latency_ratio < 1:
        raise ValueError("maximum latency ratio must be at least 1")
    if config.thresholds.maximum_stalled_enqueue_p99_ms <= 0:
        raise ValueError("stalled enqueue threshold must be positive")
    return config


def _run(config: AcceptanceConfig, seed: int, mode: RunMode) -> RunSpec:
    deployment = config.deployment
    stalled = mode == "stalled"
    enabled = mode not in {"oracle", "disabled"}
    publisher = config.publisher
    return RunSpec(
        seed=seed,
        mode=mode,
        model=config.raw["model"],
        block_size=deployment["block_size"],
        max_model_len=deployment["max_model_len"],
        max_num_batched_tokens=deployment["max_num_batched_tokens"],
        gpu_memory_utilization=deployment["gpu_memory_utilization"],
        session_count=config.workload.session_count,
        turns_per_session=config.workload.turns_per_session,
        max_tokens=config.workload.max_tokens,
        events_enabled=enabled,
        consumer_delay_seconds=(
            publisher.stalled_consumer_delay_seconds if stalled else 0.0
        ),
        publisher_hwm=(publisher.stalled_hwm if stalled else publisher.healthy_hwm),
        publisher_queue_size=(
            publisher.stalled_queue_size if stalled else publisher.healthy_queue_size
        ),
        conclusion_scope=(
            "liveness_only"
            if stalled
            else ("correctness_oracle" if mode == "oracle" else "controlled")
        ),
    )


def comparison_pair(config: AcceptanceConfig, seed: int) -> tuple[RunSpec, RunSpec]:
    """Return the matched disabled/enabled pair for one fixed seed."""
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not in the fixed seed set")
    return _run(config, seed, "disabled"), _run(config, seed, "enabled")


def build_run_matrix(config: AcceptanceConfig) -> tuple[RunSpec, ...]:
    """Build controlled repetitions plus healthy and saturation drives."""
    runs = [_run(config, seed, "oracle") for seed in config.seeds]
    runs.extend(run for seed in config.seeds for run in comparison_pair(config, seed))
    runs.extend(_run(config, seed, "healthy") for seed in config.seeds)
    runs.append(_run(config, config.seeds[0], "stalled"))
    return tuple(runs)


def comparison_fingerprint(run: RunSpec) -> str:
    """Hash every field that must match across overhead controls."""
    value = asdict(run)
    for name in (
        "mode",
        "events_enabled",
        "consumer_delay_seconds",
        "publisher_hwm",
        "publisher_queue_size",
        "conclusion_scope",
    ):
        value.pop(name)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
