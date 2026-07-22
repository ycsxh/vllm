# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed Qwen3.5-4B 1P1D NIXL feasibility launcher."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

MODEL = "Qwen/Qwen3.5-4B"
PORTS = {
    "prefill": 8100,
    "decode": 8200,
    "proxy": 8000,
    "prefill_side_channel": 5600,
    "decode_side_channel": 5601,
}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
CPU_LIST = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")


@dataclass(frozen=True)
class LaunchConfig:
    """Inputs that must be frozen before constructing a launch plan."""

    model_revision: str
    tokenizer_revision: str
    attention_backend: str
    prefill_cpus: str
    prefill_numa_node: int
    decode_cpus: str
    decode_numa_node: int
    run_dir: Path
    repo_root: Path
    vllm_commit: str
    vllm_dirty: bool
    readiness_timeout: float = 900.0
    request_timeout: float = 300.0
    shutdown_timeout: float = 30.0
    launcher_invocation: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessSpec:
    """One fixed child process in the 1P1D deployment."""

    name: str
    command: tuple[str, ...]
    environment: dict[str, str]
    log_path: Path

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable process description."""
        return {
            "name": self.name,
            "command": list(self.command),
            "environment": dict(sorted(self.environment.items())),
            "log_path": str(self.log_path),
        }


@dataclass(frozen=True)
class LaunchPlan:
    """Deterministic launch and smoke-test contract."""

    config: LaunchConfig
    processes: tuple[ProcessSpec, ...]
    smoke_request: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return the complete public plan without executing it."""
        return {
            "schema_version": 1,
            "model": MODEL,
            "model_revision": self.config.model_revision,
            "tokenizer_revision": self.config.tokenizer_revision,
            "vllm_commit": self.config.vllm_commit,
            "vllm_dirty": self.config.vllm_dirty,
            "ports": PORTS,
            "topology": {
                "prefill": {
                    "gpu": 0,
                    "cpus": self.config.prefill_cpus,
                    "numa_node": self.config.prefill_numa_node,
                    "tensor_parallel_size": 1,
                },
                "decode": {
                    "gpu": 1,
                    "cpus": self.config.decode_cpus,
                    "numa_node": self.config.decode_numa_node,
                    "tensor_parallel_size": 1,
                },
            },
            "compatibility": {
                "dtype": "bfloat16",
                "weight_quantization": None,
                "kv_cache_dtype": "bfloat16",
                "kv_cache_layout": "HND",
                "attention_backend": self.config.attention_backend,
                "block_size": 128,
                "mamba_cache_mode": "align",
                "max_num_batched_tokens": 4096,
                "prefix_caching": True,
                "chunked_prefill": True,
                "nixl_load_failure_policy": "fail",
                "speculative_decoding": False,
            },
            "timeouts_seconds": {
                "readiness": self.config.readiness_timeout,
                "request": self.config.request_timeout,
                "shutdown": self.config.shutdown_timeout,
            },
            "launcher_invocation": list(self.config.launcher_invocation),
            "smoke_request": self.smoke_request,
            "processes": [process.as_dict() for process in self.processes],
        }


class Runtime(Protocol):
    """Public process and HTTP adapter used by ``execute_plan``."""

    def start(self, process: ProcessSpec) -> Any:
        """Start one process and return its opaque handle."""
        ...

    def wait_ready(
        self, name: str, url: str, timeout: float, handles: list[Any]
    ) -> None:
        """Wait for one endpoint or raise on timeout/child exit."""
        ...

    def get_text(self, url: str, timeout: float) -> str:
        """Fetch a text endpoint."""
        ...

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        """Post JSON and return the decoded response."""
        ...

    def stop(self, handles: list[Any], timeout: float) -> None:
        """Stop every started child within the timeout."""
        ...


@dataclass
class _Child:
    process: subprocess.Popen[bytes]
    log_file: Any
    name: str


class SubprocessRuntime:
    """Production runtime with isolated child process groups."""

    def start(self, process: ProcessSpec) -> _Child:
        """Start one child in a new process session with a dedicated log."""
        process.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = process.log_path.open("ab", buffering=0)
        environment = os.environ.copy()
        environment.update(process.environment)
        try:
            child = subprocess.Popen(
                process.command,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            log_file.close()
            raise
        return _Child(child, log_file, process.name)

    def wait_ready(
        self, name: str, url: str, timeout: float, handles: list[_Child]
    ) -> None:
        """Poll an HTTP endpoint while failing on an exited child."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for handle in handles:
                status = handle.process.poll()
                if status is not None:
                    raise RuntimeError(
                        f"{handle.name} exited before {name} readiness: {status}"
                    )
            try:
                self.get_text(url, min(2.0, max(0.1, deadline - time.monotonic())))
                return
            except (OSError, urllib.error.URLError):
                time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for {name}: {url}")

    def get_text(self, url: str, timeout: float) -> str:
        """Fetch UTF-8 text using the standard library."""
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        """Post JSON using the standard library."""
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def stop(self, handles: list[_Child], timeout: float) -> None:
        """Terminate process groups, then kill groups that miss the deadline."""
        started_at = time.monotonic()
        graceful_deadline = started_at + timeout / 2
        deadline = started_at + timeout
        for handle in reversed(handles):
            if handle.process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(handle.process.pid, signal.SIGTERM)
        for handle in reversed(handles):
            remaining = max(0.0, graceful_deadline - time.monotonic())
            with contextlib.suppress(subprocess.TimeoutExpired):
                handle.process.wait(timeout=remaining)
        for handle in reversed(handles):
            if handle.process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(handle.process.pid, signal.SIGKILL)
        survivors = []
        for handle in reversed(handles):
            if handle.process.poll() is None:
                try:
                    handle.process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    survivors.append(handle.name)
            handle.log_file.close()
        if survivors:
            names = ", ".join(survivors)
            raise TimeoutError(f"process groups survived cleanup timeout: {names}")


def _server_command(config: LaunchConfig, port: int, role: str) -> tuple[str, ...]:
    transfer = json.dumps(
        {
            "kv_connector": "NixlConnector",
            "kv_load_failure_policy": "fail",
            "kv_role": role,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    python = str(config.repo_root / ".venv/bin/python")
    return (
        python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--tokenizer",
        MODEL,
        "--revision",
        config.model_revision,
        "--tokenizer-revision",
        config.tokenizer_revision,
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--language-model-only",
        "--attention-backend",
        config.attention_backend,
        "--block-size",
        "128",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        "4096",
        "--port",
        str(port),
        "--kv-transfer-config",
        transfer,
    )


def build_plan(config: LaunchConfig) -> LaunchPlan:
    """Build the one supported 1P1D plan and reject unfrozen inputs."""
    for name, value in (
        ("model revision", config.model_revision),
        ("tokenizer revision", config.tokenizer_revision),
        ("vLLM commit", config.vllm_commit),
    ):
        if FULL_REVISION.fullmatch(value) is None:
            raise ValueError(f"{name} must be a full 40-character lowercase SHA")
    if min(config.prefill_numa_node, config.decode_numa_node) < 0:
        raise ValueError("NUMA nodes must be non-negative")
    if config.prefill_numa_node == config.decode_numa_node:
        raise ValueError("prefill and decode NUMA nodes must be explicit and distinct")
    for role, cpus in (
        ("prefill", config.prefill_cpus),
        ("decode", config.decode_cpus),
    ):
        if CPU_LIST.fullmatch(cpus) is None:
            raise ValueError(f"{role} CPUs must use numactl CPU-list syntax")
    if (
        min(
            config.readiness_timeout,
            config.request_timeout,
            config.shutdown_timeout,
        )
        <= 0
    ):
        raise ValueError("all timeouts must be positive")

    server_dir = config.run_dir / "server"
    common_environment = {
        "UCX_NET_DEVICES": "all",
        "VLLM_KV_CACHE_LAYOUT": "HND",
        "VLLM_SERVER_DEV_MODE": "1",
    }
    processes = []
    for name, gpu, cpus, node, port, side_port, role in (
        (
            "prefill",
            0,
            config.prefill_cpus,
            config.prefill_numa_node,
            PORTS["prefill"],
            PORTS["prefill_side_channel"],
            "kv_producer",
        ),
        (
            "decode",
            1,
            config.decode_cpus,
            config.decode_numa_node,
            PORTS["decode"],
            PORTS["decode_side_channel"],
            "kv_consumer",
        ),
    ):
        environment = {
            **common_environment,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_port),
        }
        command = (
            "numactl",
            f"--physcpubind={cpus}",
            f"--membind={node}",
            *_server_command(config, port, role),
        )
        processes.append(
            ProcessSpec(name, command, environment, server_dir / f"{name[0]}.log")
        )

    proxy_script = (
        config.repo_root
        / "examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py"
    )
    proxy_command = (
        str(config.repo_root / ".venv/bin/python"),
        str(proxy_script),
        "--model",
        MODEL,
        "--prefill",
        f"127.0.0.1:{PORTS['prefill']}",
        "--decode",
        f"127.0.0.1:{PORTS['decode']}",
        "--port",
        str(PORTS["proxy"]),
    )
    processes.append(ProcessSpec("proxy", proxy_command, {}, server_dir / "proxy.log"))
    prompt = "Explain deterministic cache transfer in one sentence. " * 32
    smoke_request = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 16,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
    }
    return LaunchPlan(config, tuple(processes), smoke_request)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _generated_text(response: Any) -> str:
    try:
        return response["choices"][0]["text"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("smoke response lacks choices[0].text") from error


def _existing_launcher_artifacts(run_dir: Path) -> list[Path]:
    artifacts = [
        run_dir / name
        for name in (
            "launch-plan.json",
            "provenance.json",
            "cold-response.json",
            "repeated-response.json",
            "smoke-result.json",
        )
        if (run_dir / name).exists()
    ]
    server_dir = run_dir / "server"
    for pattern in ("p.log", "d.log", "proxy.log", "*-metrics-*.txt"):
        artifacts.extend(server_dir.glob(pattern))
    return sorted(set(artifacts))


def execute_plan(plan: LaunchPlan, runtime: Runtime | None = None) -> dict[str, Any]:
    """Execute the fixed smoke and always clean up every started process."""
    if plan.config.vllm_dirty:
        raise ValueError("execution requires a clean vLLM working tree")
    runtime = runtime or SubprocessRuntime()
    run_dir = plan.config.run_dir
    existing_artifacts = _existing_launcher_artifacts(run_dir)
    if existing_artifacts:
        paths = ", ".join(str(path.relative_to(run_dir)) for path in existing_artifacts)
        raise ValueError(f"run directory already contains launcher artifacts: {paths}")
    server_dir = run_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "launch-plan.json", plan.as_dict())
    _write_json(
        run_dir / "provenance.json",
        {
            "model": MODEL,
            "model_revision": plan.config.model_revision,
            "tokenizer_revision": plan.config.tokenizer_revision,
            "vllm_commit": plan.config.vllm_commit,
            "vllm_dirty": plan.config.vllm_dirty,
            "launcher_invocation": list(plan.config.launcher_invocation),
            "topology": plan.as_dict()["topology"],
            "compatibility": plan.as_dict()["compatibility"],
            "runtime_evidence_files": {
                "cuda_topology": "server/gpus.txt and server/topology.txt",
                "nixl_and_cuda_logs": "server/p.log and server/d.log",
                "nixl_metrics": "server/{p,d}-metrics-{before,cold,repeated}.txt",
            },
        },
    )

    handles: list[Any] = []
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal_handler_installed = False

    def interrupt_for_cleanup(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    try:
        signal.signal(signal.SIGTERM, interrupt_for_cleanup)
        signal_handler_installed = True
    except ValueError:
        pass
    try:
        for process in plan.processes[:2]:
            handles.append(runtime.start(process))
        for name, port in (
            ("prefill", PORTS["prefill"]),
            ("decode", PORTS["decode"]),
        ):
            runtime.wait_ready(
                name,
                f"http://127.0.0.1:{port}/v1/models",
                plan.config.readiness_timeout,
                handles,
            )

        handles.append(runtime.start(plan.processes[2]))
        runtime.wait_ready(
            "proxy",
            f"http://127.0.0.1:{PORTS['proxy']}/status",
            plan.config.readiness_timeout,
            handles,
        )
        metric_urls = {
            "p": f"http://127.0.0.1:{PORTS['prefill']}/metrics",
            "d": f"http://127.0.0.1:{PORTS['decode']}/metrics",
        }
        metrics: dict[str, dict[str, str]] = {"p": {}, "d": {}}
        for role, url in metric_urls.items():
            metrics[role]["before"] = runtime.get_text(url, plan.config.request_timeout)
            _write_text(
                server_dir / f"{role}-metrics-before.txt",
                metrics[role]["before"],
            )

        request_url = f"http://127.0.0.1:{PORTS['proxy']}/v1/completions"
        cold = runtime.post_json(
            request_url, plan.smoke_request, plan.config.request_timeout
        )
        _write_json(run_dir / "cold-response.json", cold)
        for role, url in metric_urls.items():
            metrics[role]["cold"] = runtime.get_text(url, plan.config.request_timeout)
            _write_text(server_dir / f"{role}-metrics-cold.txt", metrics[role]["cold"])
        repeated = runtime.post_json(
            request_url, plan.smoke_request, plan.config.request_timeout
        )
        _write_json(run_dir / "repeated-response.json", repeated)
        for role, url in metric_urls.items():
            metrics[role]["repeated"] = runtime.get_text(
                url, plan.config.request_timeout
            )
            _write_text(
                server_dir / f"{role}-metrics-repeated.txt",
                metrics[role]["repeated"],
            )

        cold_text = _generated_text(cold)
        repeated_text = _generated_text(repeated)
        if cold_text != repeated_text:
            raise RuntimeError("cold and repeated greedy outputs differ")
        result = {
            "status": "smoke_requests_completed",
            "outputs_identical": True,
            "cold_response": cold,
            "repeated_response": repeated,
            "gate_a": "pending_metric_review",
        }
        _write_json(run_dir / "smoke-result.json", result)
        return result
    finally:
        active_error = sys.exception()
        try:
            try:
                runtime.stop(handles, plan.config.shutdown_timeout)
            except BaseException as cleanup_error:
                if active_error is None:
                    raise
                active_error.add_note(f"cleanup also failed: {cleanup_error}")
        finally:
            if signal_handler_installed:
                signal.signal(signal.SIGTERM, previous_sigterm)


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
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--prefill-cpus", required=True)
    parser.add_argument("--prefill-numa-node", required=True, type=int)
    parser.add_argument("--decode-cpus", required=True)
    parser.add_argument("--decode-numa-node", required=True, type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--readiness-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its process exit status."""
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    commit, dirty = _git_state(repo_root)
    invocation = (str(repo_root / ".venv/bin/python"), "-m", __package__ + ".run_pd")
    if argv is None:
        invocation += tuple(sys.argv[1:])
    else:
        invocation += tuple(argv)
    config = LaunchConfig(
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        attention_backend=args.attention_backend,
        prefill_cpus=args.prefill_cpus,
        prefill_numa_node=args.prefill_numa_node,
        decode_cpus=args.decode_cpus,
        decode_numa_node=args.decode_numa_node,
        run_dir=args.run_dir.resolve(),
        repo_root=repo_root,
        vllm_commit=commit,
        vllm_dirty=dirty,
        readiness_timeout=args.readiness_timeout,
        request_timeout=args.request_timeout,
        shutdown_timeout=args.shutdown_timeout,
        launcher_invocation=invocation,
    )
    plan = build_plan(config)
    if args.dry_run:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return 0
    try:
        execute_plan(plan)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
