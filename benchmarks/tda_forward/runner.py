# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single entry point for Issue #18 target-server acceptance evidence."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import regex as re

from benchmarks.tda_forward.metrics import distribution
from benchmarks.tda_forward.report import build_report
from benchmarks.tda_forward.schema import (
    DEFAULT_CONFIG_PATH,
    RUN_SOURCE_ARTIFACTS,
    AcceptanceConfig,
    RunSpec,
    build_run_matrix,
    comparison_fingerprint,
    load_config,
)
from benchmarks.tda_forward.workload import (
    Session,
    Turn,
    TurnResult,
    build_workload,
    run_sessions,
)

_SHA = re.compile(r"[0-9a-f]{40}")
_REQUIRED_ENV = (
    "TDA_MODEL_REVISION",
    "TDA_TOKENIZER_REVISION",
    "TDA_ATTENTION_BACKEND",
    "TDA_PREFILL_CPUS",
    "TDA_PREFILL_NUMA_NODE",
    "TDA_DECODE_CPUS",
    "TDA_DECODE_NUMA_NODE",
)
_RUNTIME_ENV = (
    "CUDA_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "NO_PROXY",
    "PATH",
    "TRANSFORMERS_OFFLINE",
    "VLLM_SSM_CONV_STATE_LAYOUT",
    "no_proxy",
)


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    command: tuple[str, ...]
    environment: dict[str, str]
    log_path: Path

    def public(self) -> dict[str, Any]:
        allow = {
            "CUDA_VISIBLE_DEVICES",
            "PYTHONPATH",
            "TDA_ACCEPTANCE_STATS_PATH",
            "TDA_ACCEPTANCE_PUBLISHER_DRAIN_DELAY_SECONDS",
            "TDA_ACCEPTANCE_WINDOW_PATH",
            "UCX_NET_DEVICES",
            "VLLM_KV_CACHE_LAYOUT",
            "VLLM_NIXL_SIDE_CHANNEL_HOST",
            "VLLM_NIXL_SIDE_CHANNEL_PORT",
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
            *_RUNTIME_ENV,
        }
        return {
            "name": self.name,
            "command": list(self.command),
            "environment": {
                key: value
                for key, value in sorted(self.environment.items())
                if key in allow
            },
            "log_path": str(self.log_path),
        }


@dataclass
class Child:
    spec: ProcessSpec
    process: subprocess.Popen[bytes]
    output: Any


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_hashes(run_dir: Path) -> dict[str, str]:
    hashes = {}
    for relative in RUN_SOURCE_ARTIFACTS:
        path = run_dir / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def _wait_ready(children: list[Child], name: str, url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for child in children:
            status = child.process.poll()
            if status is not None:
                raise RuntimeError(
                    f"{child.spec.name} exited before {name} readiness: {status}"
                )
        try:
            _get_text(url, min(2.0, max(0.1, deadline - time.monotonic())))
            return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {name}: {url}")


def _start(spec: ProcessSpec) -> Child:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    output = spec.log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            spec.command,
            env=spec.environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        output.close()
        raise
    return Child(spec, process, output)


def _stop(children: list[Child], timeout: float = 30.0) -> dict[str, int]:
    before = {
        child.spec.name: child.process.returncode
        for child in children
        if child.process.poll() is not None
    }
    for child in reversed(children):
        if child.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    for child in reversed(children):
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.process.wait(timeout=max(0.0, deadline - time.monotonic()))
    for child in reversed(children):
        if child.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.process.pid, signal.SIGKILL)
            child.process.wait(timeout=5)
        child.output.close()
    return before


def _server_command(
    repo_root: Path,
    config: AcceptanceConfig,
    run: RunSpec,
    *,
    role: str,
    port: int,
) -> tuple[str, ...]:
    deployment = config.deployment
    transfer = json.dumps(
        {
            "kv_connector": "NixlConnector",
            "kv_load_failure_policy": "fail",
            "kv_role": role,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    command = (
        str(repo_root / ".venv/bin/python"),
        "-m",
        "benchmarks.tda_forward.instrumented_server",
        "serve",
        run.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--revision",
        os.environ["TDA_MODEL_REVISION"],
        "--tokenizer-revision",
        os.environ["TDA_TOKENIZER_REVISION"],
        "--dtype",
        deployment["dtype"],
        "--kv-cache-dtype",
        deployment["kv_cache_dtype"],
        "--tensor-parallel-size",
        str(deployment["tensor_parallel_size"]),
        "--language-model-only",
        "--attention-backend",
        os.environ["TDA_ATTENTION_BACKEND"],
        "--block-size",
        str(run.block_size),
        "--max-model-len",
        str(run.max_model_len),
        "--max-num-batched-tokens",
        str(run.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(run.gpu_memory_utilization),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--mamba-cache-mode",
        "align",
        "--kv-transfer-config",
        transfer,
    )
    if role == "kv_consumer" and run.events_enabled:
        ports = config.raw["ports"]
        events = json.dumps(
            {
                "enable_kv_cache_events": True,
                "publisher": "zmq",
                "endpoint": f"tcp://*:{ports['decode_events']}",
                "hwm": run.publisher_hwm,
                "max_queue_size": run.publisher_queue_size,
                "topic": "tda-acceptance",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        command += ("--kv-events-config", events)
    return command


def _process_specs(
    repo_root: Path,
    config: AcceptanceConfig,
    run: RunSpec,
    run_dir: Path,
) -> tuple[ProcessSpec, ...]:
    ports = config.raw["ports"]
    base_environment = dict(os.environ)
    base_environment.update(
        {
            "UCX_NET_DEVICES": "all",
            "VLLM_KV_CACHE_LAYOUT": "HND",
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "0",
        }
    )
    processes = []
    for name, gpu, role, port, cpus, numa, side_port in (
        (
            "prefill",
            "0",
            "kv_producer",
            ports["prefill"],
            os.environ["TDA_PREFILL_CPUS"],
            os.environ["TDA_PREFILL_NUMA_NODE"],
            ports["prefill_side_channel"],
        ),
        (
            "decode",
            "1",
            "kv_consumer",
            ports["decode"],
            os.environ["TDA_DECODE_CPUS"],
            os.environ["TDA_DECODE_NUMA_NODE"],
            ports["decode_side_channel"],
        ),
    ):
        environment = {
            **base_environment,
            "CUDA_VISIBLE_DEVICES": gpu,
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_port),
            "TDA_ACCEPTANCE_STATS_PATH": str(
                run_dir / "raw" / f"{name}-instrumentation.json"
            ),
            "TDA_ACCEPTANCE_WINDOW_PATH": str(
                run_dir / "raw" / f"{name}-measurement-window"
            ),
            "TDA_ACCEPTANCE_PUBLISHER_DRAIN_DELAY_SECONDS": (
                str(run.consumer_delay_seconds)
                if name == "decode" and run.mode == "stalled"
                else "0"
            ),
        }
        command = (
            "numactl",
            f"--physcpubind={cpus}",
            f"--membind={numa}",
            *_server_command(repo_root, config, run, role=role, port=port),
        )
        processes.append(
            ProcessSpec(name, command, environment, run_dir / "logs" / f"{name}.log")
        )

    proxy = (
        str(repo_root / ".venv/bin/python"),
        "-m",
        "tda_forward.proxy",
        "--native",
        "--model",
        run.model,
        "--tokenizer-revision",
        os.environ["TDA_TOKENIZER_REVISION"],
        "--g",
        str(config.workload.g_seconds),
        "--block-size",
        str(run.block_size),
        "--prefill-url",
        f"http://127.0.0.1:{ports['prefill']}",
        "--decode-url",
        f"http://127.0.0.1:{ports['decode']}",
        "--port",
        str(ports["proxy"]),
    )
    if run.events_enabled:
        proxy += (
            "--event-endpoint",
            f"tcp://127.0.0.1:{ports['decode_events']}",
            "--event-topic",
            "tda-acceptance",
        )
        if run.events_enabled:
            proxy += (
                "--event-subscriber-hwm",
                str(config.publisher.healthy_hwm),
            )
    else:
        proxy += ("--disable-events",)
    proxy_environment = {
        **base_environment,
        "PYTHONPATH": os.pathsep.join(
            (
                str(repo_root / "examples/disaggregated/tda_forward/src"),
                str(repo_root),
            )
        ),
    }
    processes.append(
        ProcessSpec(
            "proxy",
            proxy,
            proxy_environment,
            run_dir / "logs" / "proxy.log",
        )
    )
    return tuple(processes)


def _git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return commit, dirty


def _git_remotes(repo_root: Path) -> dict[str, str]:
    remotes = {}
    for name in ("origin", "upstream"):
        result = subprocess.run(
            ("git", "remote", "get-url", name),
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            remotes[name] = result.stdout.strip()
    return remotes


def _target_hardware() -> list[dict[str, str]]:
    result = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi query failed: {result.stderr.strip()}")
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"malformed nvidia-smi row: {line!r}")
        gpus.append(dict(zip(("index", "name", "driver_version"), fields)))
    if len(gpus) != 2 or any("RTX 3090" not in gpu["name"] for gpu in gpus):
        raise RuntimeError("Issue #18 requires exactly two NVIDIA RTX 3090 GPUs")
    return gpus


def _capture_command(command: tuple[str, ...]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": list(command), "error": repr(error)}


def _environment_evidence(
    repo_root: Path,
    commit: str,
    remotes: dict[str, str],
    gpus: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "vllm_commit": commit,
        "git_remotes": remotes,
        "gpus": gpus,
        "platform": platform.platform(),
        "python": sys.version,
        "model_revision": os.environ["TDA_MODEL_REVISION"],
        "tokenizer_revision": os.environ["TDA_TOKENIZER_REVISION"],
        "selected_environment": {name: os.environ[name] for name in _REQUIRED_ENV},
        "runtime_environment": {
            name: os.environ[name] for name in _RUNTIME_ENV if name in os.environ
        },
        "commands": [
            _capture_command(("uname", "-a")),
            _capture_command(("nvidia-smi", "-q")),
            _capture_command(("nvidia-smi", "topo", "-m")),
            _capture_command(("nvcc", "--version")),
            _capture_command(
                (
                    str(repo_root / ".venv/bin/python"),
                    "-c",
                    (
                        "import nixl, torch; "
                        "print({'nixl': getattr(nixl, '__version__', 'unknown'), "
                        "'torch': torch.__version__, 'cuda': torch.version.cuda})"
                    ),
                )
            ),
        ],
    }


def _descendants(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text(encoding="utf-8").split()
            parents[int(fields[0])] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


def _cpu_seconds(root_pid: int) -> float | None:
    if not Path("/proc").exists():
        return None
    ticks = os.sysconf("SC_CLK_TCK")
    total = 0
    for pid in _descendants(root_pid):
        try:
            fields = (Path("/proc") / str(pid) / "stat").read_text().split()
            total += int(fields[13]) + int(fields[14])
        except (OSError, ValueError, IndexError):
            continue
    return total / ticks


async def _sample_cpu(
    children: list[Child],
    destination: Path,
    stop: asyncio.Event,
) -> None:
    with destination.open("w", encoding="utf-8") as output:
        while not stop.is_set():
            sample = {
                "timestamp": time.time(),
                "processes": {
                    child.spec.name: _cpu_seconds(child.process.pid)
                    for child in children
                },
            }
            output.write(json.dumps(sample, sort_keys=True) + "\n")
            output.flush()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=1.0)


def _choice_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return ""
    text = ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        value = choice.get("text")
        if isinstance(value, str):
            text += value
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            text += delta["content"]
    return text


async def _send_turn(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    turn: Turn,
    raw_output: Any,
    payload: dict[str, Any] | None = None,
) -> TurnResult:
    started = time.perf_counter()
    first_token_at = None
    chunks = []
    output_text = ""
    output_chunks = 0
    completion_tokens = None
    request_id = f"{turn.session_id}-{turn.scenario}-turn-{turn.turn_index}"
    try:
        async with client.stream(
            "POST",
            url,
            json=payload or turn.payload(model),
            headers={"X-Request-Id": request_id},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                chunks.append(chunk)
                usage = chunk.get("usage")
                if isinstance(usage, dict) and isinstance(
                    usage.get("completion_tokens"), int
                ):
                    completion_tokens = usage["completion_tokens"]
                text = _choice_text(chunk)
                if text:
                    output_text += text
                    output_chunks += 1
                    first_token_at = first_token_at or time.perf_counter()
        ended = time.perf_counter()
        if first_token_at is None:
            raise RuntimeError("response completed without an output token")
        ttft_ms = (first_token_at - started) * 1000
        output_tokens = completion_tokens or output_chunks
        tpot_ms = (
            (ended - first_token_at) * 1000 / (output_tokens - 1)
            if output_tokens > 1
            else None
        )
        result = TurnResult.success(
            turn,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            output_tokens=output_tokens,
            elapsed_ms=(ended - started) * 1000,
            output_sha256=hashlib.sha256(output_text.encode()).hexdigest(),
        )
        raw = {
            "request_id": request_id,
            "turn": asdict(turn),
            "result": result.as_dict(),
            "chunks": chunks,
        }
    except Exception as error:
        result = TurnResult(
            turn.session_id,
            turn.turn_index,
            turn.scenario,
            "failed",
            None,
            None,
            0,
            (time.perf_counter() - started) * 1000,
            None,
            repr(error),
        )
        raw = {
            "request_id": request_id,
            "turn": asdict(turn),
            "result": result.as_dict(),
            "chunks": chunks,
        }
    raw_output.write(json.dumps(raw, sort_keys=True) + "\n")
    raw_output.flush()
    return result


def _publisher_stats(instrumentation: dict[str, Any]) -> dict[str, Any]:
    publishers = instrumentation.get("publishers", [])
    if len(publishers) != 1 or not isinstance(publishers[0], dict):
        raise ValueError("expected exactly one Decode event publisher")
    publisher = publishers[0]
    if publisher.get("stats_error"):
        raise ValueError(f"publisher stats unavailable: {publisher['stats_error']}")
    integer_fields = (
        "enqueued_batches",
        "enqueued_events",
        "enqueued_blocks",
        "published_batches",
        "published_events",
        "published_blocks",
        "published_bytes",
        "dropped_batches",
        "dropped_events",
        "dropped_blocks",
        "queue_depth",
        "queue_high_watermark",
    )
    if any(
        isinstance(publisher.get(name), bool)
        or not isinstance(publisher.get(name), int)
        or publisher[name] < 0
        for name in integer_fields
    ):
        raise ValueError("publisher stats are incomplete or malformed")
    samples = instrumentation.get("publisher_samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("publisher queue samples are missing")
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(
            sample.get("timestamp"), (int, float)
        ):
            raise ValueError("publisher queue sample is malformed")
        sample_publishers = sample.get("publishers")
        if not isinstance(sample_publishers, list) or len(sample_publishers) != 1:
            raise ValueError("publisher queue sample has the wrong publisher count")
        sample_publisher = sample_publishers[0]
        if (
            not isinstance(sample_publisher, dict)
            or sample_publisher.get("stats_error")
            or any(
                isinstance(sample_publisher.get(name), bool)
                or not isinstance(sample_publisher.get(name), int)
                or sample_publisher[name] < 0
                for name in ("queue_depth", "queue_high_watermark")
            )
        ):
            raise ValueError("publisher queue sample stats are malformed")
    return publisher


def _require_timing_samples(
    instrumentation: dict[str, Any], name: str
) -> dict[str, Any]:
    value = instrumentation.get(name)
    if not isinstance(value, dict) or not isinstance(value.get("raw"), list):
        raise ValueError(f"{name} samples are missing")
    if not value["raw"]:
        raise ValueError(f"{name} samples are empty")
    summary = value.get("summary")
    if not isinstance(summary, dict) or summary.get("count") != len(value["raw"]):
        raise ValueError(f"{name} summary does not match raw samples")
    return value


def _require_subscriber_stats(subscriber: object) -> dict[str, Any]:
    if not isinstance(subscriber, dict):
        raise ValueError("event subscriber stats are missing")
    fields = (
        "received_batches",
        "received_events",
        "received_blocks",
        "received_bytes",
        "sequence_gaps",
    )
    if any(
        isinstance(subscriber.get(name), bool)
        or not isinstance(subscriber.get(name), int)
        or subscriber[name] < 0
        for name in fields
    ):
        raise ValueError("event subscriber stats are incomplete or malformed")
    return subscriber


def _scenario_observations(proxy_state: dict[str, Any]) -> dict[str, Any]:
    records = proxy_state.get("records", [])
    by_request = {
        record.get("request_id"): record
        for record in records
        if isinstance(record, dict)
    }
    observations = {
        "full_reuse": False,
        "partial_reuse": False,
        "zero_reuse": False,
        "evict_d": False,
        "capacity_recovery": False,
        "shared_prefix": False,
        "d_local_attempts": [],
    }
    shared_hits = set()
    capacity_seen = False
    later_after_capacity = False
    by_session: dict[str, list[dict[str, Any]]] = {}
    for record in by_request.values():
        by_session.setdefault(str(record.get("session_id")), []).append(record)
    evict_followed_by_prefill = False
    for records_for_session in by_session.values():
        ordered = sorted(
            records_for_session,
            key=lambda record: int(record.get("turn_sequence") or 0),
        )
        for previous, current in zip(ordered, ordered[1:], strict=False):
            evict_followed_by_prefill |= (
                previous.get("cache_action") == "EVICT_D"
                and current.get("execution_path") == "P_SIDE_AP"
            )
    for request_id, record in by_request.items():
        if record.get("used_d_binding"):
            observations["d_local_attempts"].append(record)
            if record.get("execution_path") != "D_LOCAL_AP":
                continue
            actual = record.get("actual_local_cached_tokens")
            prompt = record.get("prompt_tokens")
            if isinstance(actual, int) and isinstance(prompt, int) and prompt > 0:
                ratio = actual / prompt
                observations["full_reuse"] |= ratio >= 0.9
                observations["partial_reuse"] |= 0 < ratio < 0.9
            observations["zero_reuse"] |= (
                record.get("local_cache_status") == "D_MISS"
                and actual == 0
                and record.get("locally_computed_tokens") == prompt
            )
            if request_id and "shared" in str(request_id) and actual:
                shared_hits.add(record.get("session_id"))
            delay = record.get("capacity_delay_ms")
            if isinstance(delay, (int, float)) and delay > 0:
                capacity_seen = True
            elif capacity_seen:
                later_after_capacity = True
        acknowledgement = record.get("cache_action_ack", {})
        observations["evict_d"] |= (
            record.get("cache_action") == "EVICT_D"
            and isinstance(acknowledgement, dict)
            and acknowledgement.get("status") in {"EVICTED", "DEFERRED"}
        )
    observations["evict_d"] &= evict_followed_by_prefill
    observations["capacity_recovery"] = capacity_seen and later_after_capacity
    observations["shared_prefix"] = len(shared_hits) >= 2
    return observations


def _d_attempt_invalid_reasons(attempts: list[dict[str, Any]]) -> list[str]:
    reasons = []
    for record in attempts:
        request_id = record.get("request_id", "unknown")
        actual = record.get("actual_local_cached_tokens")
        prompt = record.get("prompt_tokens")
        computed = record.get("locally_computed_tokens")
        estimate = record.get("proxy_estimated_cached_tokens")
        if record.get("execution_path") != "D_LOCAL_AP":
            reasons.append(f"{request_id}: D binding did not execute on Decode")
            continue
        if not (
            isinstance(actual, int)
            and isinstance(prompt, int)
            and isinstance(computed, int)
            and isinstance(estimate, int)
        ):
            reasons.append(f"{request_id}: incomplete D-local token reconciliation")
            continue
        if actual + computed != prompt:
            reasons.append(
                f"{request_id}: cached plus computed tokens do not reconcile"
            )
        if record.get("estimate_error") != actual - estimate:
            reasons.append(f"{request_id}: signed estimate error does not reconcile")
        expected_status = "D_HIT" if actual > 0 else "D_MISS"
        if record.get("local_cache_status") != expected_status:
            reasons.append(f"{request_id}: D_HIT/D_MISS disagrees with actual reuse")
        expected_ratio = actual / prompt if prompt else None
        if record.get("hit_ratio") != expected_ratio:
            reasons.append(f"{request_id}: hit ratio does not reconcile")
    return reasons


def _run_summary(
    run: RunSpec,
    results: list[TurnResult],
    elapsed_seconds: float,
    proxy_state: dict[str, Any],
    instrumentation: dict[str, Any],
    unexpected_exits: dict[str, int],
    cpu: dict[str, Any],
) -> dict[str, Any]:
    measured = [
        result
        for result in results
        if result.turn_index > 0 and result.status == "completed"
    ]
    failed = [result for result in results if result.status != "completed"]
    if not isinstance(instrumentation.get("measurement_window_started_at"), float):
        raise ValueError("instrumentation measurement window did not start")
    scheduler_timing = _require_timing_samples(instrumentation, "scheduler_step_ms")
    publisher = _publisher_stats(instrumentation) if run.events_enabled else {}
    if run.events_enabled:
        _require_timing_samples(instrumentation, "event_construction_ms")
        _require_timing_samples(instrumentation, "event_enqueue_ms")
    event_pump = proxy_state.get("event_pump") or {}
    subscriber = event_pump.get("subscriber") or {}
    if run.events_enabled:
        subscriber = _require_subscriber_stats(subscriber)
        lag = event_pump.get("event_lag_seconds")
        if not isinstance(lag, dict) or not lag.get("raw"):
            raise ValueError("event lag samples are missing")
    mirror = proxy_state.get("mirror") or {}
    mirror_reason = mirror.get("invalid_reason")
    sequence_gaps = max(
        int(mirror.get("gap_count") or 0),
        int(subscriber.get("sequence_gaps") or 0),
    )
    overflow = int(publisher.get("dropped_batches") or 0)
    malformed = int(bool(mirror_reason and "malformed" in str(mirror_reason)))
    invalid = []
    if failed:
        invalid.append(f"{len(failed)} requests failed")
    truncated = [
        result
        for result in results
        if result.status == "completed" and result.output_tokens != run.max_tokens
    ]
    if truncated:
        invalid.append(f"{len(truncated)} requests returned an unexpected token count")
    expected_stalled_gap = (
        run.mode == "stalled"
        and sequence_gaps > 0
        and str(mirror_reason).startswith("event sequence gap:")
    )
    if mirror_reason and not expected_stalled_gap:
        invalid.append(str(mirror_reason))
    if event_pump.get("failure"):
        invalid.append(str(event_pump["failure"]))
    if unexpected_exits:
        invalid.append(f"unexpected process exits: {unexpected_exits}")
    if run.conclusion_scope == "controlled":
        if overflow:
            invalid.append("publisher overflow in controlled run")
        if sequence_gaps:
            invalid.append("sequence gap in controlled run")
    scenarios = _scenario_observations(proxy_state)
    invalid.extend(_d_attempt_invalid_reasons(scenarios["d_local_attempts"]))
    if run.mode == "healthy":
        for name in (
            "full_reuse",
            "partial_reuse",
            "zero_reuse",
            "evict_d",
            "capacity_recovery",
            "shared_prefix",
        ):
            if not scenarios[name]:
                invalid.append(f"required scenario not observed: {name}")
    ttft = [result.ttft_ms for result in measured if result.ttft_ms is not None]
    tpot = [result.tpot_ms for result in measured if result.tpot_ms is not None]
    scheduler = scheduler_timing["summary"]
    output_hashes = {
        f"{result.session_id}:{result.turn_index}": result.output_sha256
        for result in results
        if result.status == "completed" and result.output_sha256 is not None
    }
    if len(output_hashes) != sum(result.status == "completed" for result in results):
        invalid.append("one or more completed responses lack an output digest")
    serving = {
        "completed_requests": len(measured),
        "failed_requests": len(failed),
        "measurement_seconds": elapsed_seconds,
        "request_throughput": len(measured) / elapsed_seconds,
        "ttft_ms": distribution(ttft),
        "tpot_ms": distribution(tpot),
        "ttft_p95_ms": distribution(ttft)["p95"],
        "tpot_p95_ms": distribution(tpot)["p95"],
        "scheduler_step_p95_ms": scheduler.get("p95"),
    }
    return {
        "schema_version": 1,
        "run_id": run.run_id,
        "mode": run.mode,
        "seed": run.seed,
        "status": "invalid" if invalid else "valid",
        "conclusion_scope": run.conclusion_scope,
        "comparison_fingerprint": comparison_fingerprint(run),
        "invalid_reasons": sorted(set(invalid)),
        "serving": serving,
        "output_hashes": output_hashes,
        "event_plane": {
            "construction_ms": instrumentation.get("event_construction_ms"),
            "enqueue_ms": instrumentation.get("event_enqueue_ms"),
            "publisher": publisher,
            "publisher_samples": instrumentation.get("publisher_samples"),
            "proxy_event_lag_seconds": event_pump.get("event_lag_seconds"),
            "subscriber": subscriber,
            "mirror_metrics": mirror.get("metrics"),
        },
        "cpu": cpu,
        "integrity": {
            "publisher_overflow": overflow,
            "first_publisher_overflow_at": instrumentation.get(
                "first_publisher_overflow_at"
            ),
            "sequence_gaps": sequence_gaps,
            "first_sequence_gap_at": mirror.get("first_gap_at"),
            "malformed_events": malformed,
            "proxy_restarts": int("proxy" in unexpected_exits),
            "decode_restarts": int("decode" in unexpected_exits),
        },
        "scenarios": scenarios,
        "timed_out": False,
    }


async def _drive_run(
    config: AcceptanceConfig,
    run: RunSpec,
    run_dir: Path,
    children: list[Child],
) -> tuple[list[TurnResult], float, dict[str, Any]]:
    ports = config.raw["ports"]
    proxy_url = f"http://127.0.0.1:{ports['proxy']}"
    sessions = build_workload(config.workload, run.seed)
    _write_json(
        run_dir / "workload.json",
        {
            "seed": run.seed,
            "sessions": [asdict(session) for session in sessions],
        },
    )
    raw_path = run_dir / "raw" / "responses.jsonl"
    timeout = httpx.Timeout(config.workload.request_timeout_seconds)
    cpu_stop = asyncio.Event()
    cpu_task: asyncio.Task[None] | None = None
    try:
        with raw_path.open("w", encoding="utf-8") as raw_output:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if run.mode == "oracle":
                    for role in ("prefill", "decode"):
                        (run_dir / "raw" / f"{role}-measurement-window").touch()
                    cpu_task = asyncio.create_task(
                        _sample_cpu(children, run_dir / "raw" / "cpu.jsonl", cpu_stop)
                    )

                    async def drive_oracle() -> list[TurnResult]:
                        oracle_results = []
                        for session in sessions:
                            for turn in session.turns:
                                payload = turn.payload(run.model)
                                payload.pop("session_id")
                                payload.pop("t_pred")
                                oracle_results.append(
                                    await _send_turn(
                                        client,
                                        f"http://127.0.0.1:{ports['decode']}"
                                        "/v1/completions",
                                        run.model,
                                        turn,
                                        raw_output,
                                        payload,
                                    )
                                )
                        return oracle_results

                    started = time.perf_counter()
                    results = await asyncio.wait_for(
                        drive_oracle(), timeout=config.workload.run_timeout_seconds
                    )
                    elapsed = time.perf_counter() - started
                    _write_json(
                        run_dir / "raw" / "measurement-window.json",
                        {"elapsed_seconds": elapsed, "started_after_warmup": True},
                    )
                    state = _get_json(
                        f"{proxy_url}/acceptance/state",
                        config.workload.request_timeout_seconds,
                    )
                    _write_json(run_dir / "raw" / "proxy-state.json", state)
                    return results, elapsed, state
                send = lambda turn: _send_turn(
                    client,
                    f"{proxy_url}/v1/completions",
                    run.model,
                    turn,
                    raw_output,
                )
                warmup_sessions = tuple(
                    Session(
                        session.session_id,
                        session.turns[: config.workload.warmup_turns],
                    )
                    for session in sessions
                )
                measured_sessions = tuple(
                    Session(
                        session.session_id,
                        session.turns[config.workload.warmup_turns :],
                    )
                    for session in sessions
                )
                warmup = await asyncio.wait_for(
                    run_sessions(warmup_sessions, send),
                    timeout=config.workload.run_timeout_seconds,
                )
                for role in ("prefill", "decode"):
                    (run_dir / "raw" / f"{role}-measurement-window").touch()
                cpu_task = asyncio.create_task(
                    _sample_cpu(children, run_dir / "raw" / "cpu.jsonl", cpu_stop)
                )
                started = time.perf_counter()
                measured = await asyncio.wait_for(
                    run_sessions(measured_sessions, send),
                    timeout=config.workload.run_timeout_seconds,
                )
                elapsed = time.perf_counter() - started
                results = [*warmup, *measured]
                _write_json(
                    run_dir / "raw" / "measurement-window.json",
                    {"elapsed_seconds": elapsed, "started_after_warmup": True},
                )
        state = _get_json(
            f"{proxy_url}/acceptance/state",
            config.workload.request_timeout_seconds,
        )
        _write_json(run_dir / "raw" / "proxy-state.json", state)
        return results, elapsed, state
    finally:
        cpu_stop.set()
        if cpu_task is not None:
            await cpu_task


def _cpu_summary(path: Path) -> dict[str, Any]:
    samples = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(samples) < 2:
        raise ValueError("CPU evidence requires at least two measurement samples")
    elapsed = samples[-1]["timestamp"] - samples[0]["timestamp"]
    if elapsed <= 0:
        raise ValueError("CPU evidence has a nonpositive measurement interval")
    means = {}
    names = set(samples[0]["processes"]) & set(samples[-1]["processes"])
    if names != {"prefill", "decode", "proxy"}:
        raise ValueError("CPU evidence is missing a required process tree")
    for name in names:
        before = samples[0]["processes"][name]
        after = samples[-1]["processes"][name]
        if before is None or after is None:
            raise ValueError(f"CPU evidence is unavailable for {name}")
        means[name] = max(0.0, (after - before) / elapsed)
    return {
        "samples": len(samples),
        "measurement_seconds": elapsed,
        "mean_cores": means,
        "total_mean_cores": sum(means.values()),
    }


def execute_run(
    repo_root: Path,
    config: AcceptanceConfig,
    run: RunSpec,
    evidence_dir: Path,
) -> dict[str, Any]:
    """Execute one immutable target-server run and write a fail-closed summary."""
    commit, dirty = _git_state(repo_root)
    if dirty:
        raise ValueError("target-server execution requires a clean working tree")
    for name in _REQUIRED_ENV:
        if not os.environ.get(name):
            raise ValueError(f"missing required environment variable: {name}")
    for name in ("TDA_MODEL_REVISION", "TDA_TOKENIZER_REVISION"):
        if _SHA.fullmatch(os.environ[name]) is None:
            raise ValueError(f"{name} must be a full lowercase commit SHA")
    remotes = _git_remotes(repo_root)
    if "ycsxh/vllm" not in remotes.get("origin", ""):
        raise ValueError("origin must be the personal fork ycsxh/vllm")
    gpus = _target_hardware()
    current_environment = _environment_evidence(repo_root, commit, remotes, gpus)
    failed_environment_commands = [
        command
        for command in current_environment["commands"]
        if command["command"][0] == str(repo_root / ".venv/bin/python")
        and command.get("returncode") != 0
    ]
    if failed_environment_commands:
        raise RuntimeError(
            "target environment command failed: "
            f"{failed_environment_commands[0]['command']}"
        )
    environment_path = evidence_dir / "environment.json"
    frozen_config_path = evidence_dir / "config.json"
    if environment_path.exists() != frozen_config_path.exists():
        raise ValueError("experiment environment/config pair is incomplete")
    if environment_path.exists():
        existing_environment = json.loads(environment_path.read_text(encoding="utf-8"))
        for field in (
            "vllm_commit",
            "git_remotes",
            "gpus",
            "model_revision",
            "tokenizer_revision",
            "selected_environment",
            "runtime_environment",
        ):
            if existing_environment.get(field) != current_environment.get(field):
                raise ValueError(f"experiment environment changed: {field}")
    if (
        frozen_config_path.exists()
        and json.loads(frozen_config_path.read_text(encoding="utf-8")) != config.raw
    ):
        raise ValueError("experiment configuration changed")
    run_dir = evidence_dir / "runs" / run.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    specs = _process_specs(repo_root, config, run, run_dir)
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "run": asdict(run),
            "comparison_fingerprint": comparison_fingerprint(run),
            "vllm_commit": commit,
            "model_revision": os.environ["TDA_MODEL_REVISION"],
            "tokenizer_revision": os.environ["TDA_TOKENIZER_REVISION"],
            "processes": [spec.public() for spec in specs],
        },
    )
    if not environment_path.exists():
        evidence_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            environment_path,
            current_environment,
        )
        frozen_config_path.write_text(
            json.dumps(config.raw, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    children: list[Child] = []
    unexpected_exits: dict[str, int] = {}
    try:
        children.extend(_start(spec) for spec in specs[:2])
        ports = config.raw["ports"]
        for name in ("prefill", "decode"):
            _wait_ready(
                children,
                name,
                f"http://127.0.0.1:{ports[name]}/v1/models",
                900,
            )
        for role in ("prefill", "decode"):
            metrics = _get_text(
                f"http://127.0.0.1:{ports[role]}/metrics",
                config.workload.request_timeout_seconds,
            )
            (run_dir / "raw" / f"{role}-metrics-before.txt").write_text(
                metrics, encoding="utf-8"
            )
        children.append(_start(specs[2]))
        _wait_ready(
            children,
            "proxy",
            f"http://127.0.0.1:{ports['proxy']}/health",
            900,
        )
        results, elapsed, proxy_state = asyncio.run(
            _drive_run(config, run, run_dir, children)
        )
        for role in ("prefill", "decode"):
            metrics = _get_text(
                f"http://127.0.0.1:{ports[role]}/metrics",
                config.workload.request_timeout_seconds,
            )
            (run_dir / "raw" / f"{role}-metrics-after.txt").write_text(
                metrics, encoding="utf-8"
            )
    except BaseException as error:
        shutdown_exception: BaseException | None = None
        try:
            unexpected_exits = _stop(children)
        except BaseException as stop_error:
            shutdown_exception = stop_error
        _write_json(
            run_dir / "invalid-run.json",
            {
                "status": "invalid",
                "reason": repr(error),
                "shutdown_error": (
                    repr(shutdown_exception) if shutdown_exception is not None else None
                ),
                "run_id": run.run_id,
                "timestamp": time.time(),
            },
        )
        if shutdown_exception is not None:
            raise error from shutdown_exception
        raise

    try:
        unexpected_exits = _stop(children)
        _write_json(run_dir / "raw" / "process-exits.json", unexpected_exits)
        instrumentation_path = run_dir / "raw" / "decode-instrumentation.json"
        if not instrumentation_path.exists():
            raise RuntimeError("Decode instrumentation was not flushed during shutdown")
        instrumentation = json.loads(instrumentation_path.read_text(encoding="utf-8"))
        summary = _run_summary(
            run,
            results,
            elapsed,
            proxy_state,
            instrumentation,
            unexpected_exits,
            _cpu_summary(run_dir / "raw" / "cpu.jsonl"),
        )
        summary["source_sha256"] = _source_hashes(run_dir)
        _write_json(run_dir / "run-summary.json", summary)
        return summary
    except BaseException as error:
        _write_json(
            run_dir / "invalid-run.json",
            {
                "status": "invalid",
                "reason": repr(error),
                "run_id": run.run_id,
                "timestamp": time.time(),
            },
        )
        raise


def _find_run(config: AcceptanceConfig, run_id: str) -> RunSpec:
    matches = [run for run in build_run_matrix(config) if run.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"unknown run id: {run_id}")
    return matches[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run = subparsers.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--evidence-dir", required=True, type=Path)
    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--evidence-dir", required=True, type=Path)
    report = subparsers.add_parser("report")
    report.add_argument("--evidence-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(DEFAULT_CONFIG_PATH)
    repo_root = Path(__file__).resolve().parents[2]
    if args.command == "plan":
        value = {
            "config": config.raw,
            "runs": [asdict(run) for run in build_run_matrix(config)],
            "required_environment": list(_REQUIRED_ENV),
        }
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        execute_run(
            repo_root,
            config,
            _find_run(config, args.run_id),
            args.evidence_dir,
        )
        return 0
    if args.command == "run-all":
        for run in build_run_matrix(config):
            execute_run(repo_root, config, run, args.evidence_dir)
        build_report(args.evidence_dir, DEFAULT_CONFIG_PATH)
        return 0
    if args.command == "report":
        build_report(args.evidence_dir, DEFAULT_CONFIG_PATH)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
