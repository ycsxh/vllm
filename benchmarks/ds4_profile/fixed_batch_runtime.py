# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public offline-vLLM adapter for the DS4 fixed-batch runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import regex as re

from benchmarks.ds4_profile.fixed_batch import (
    BatchObservation,
    FixedBatchPoint,
    IterationRecord,
    RequestObservation,
)

ITERATION_PATTERN = re.compile(
    r"Iteration\(\d+\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>\d+(?:\.\d+)?) ms"
)


def parse_iteration_records(text: str) -> tuple[IterationRecord, ...]:
    """Parse the built-in EngineCore iteration-detail log format."""
    return tuple(
        IterationRecord(
            elapsed_ms=float(match["elapsed_ms"]),
            context_requests=int(match["context_requests"]),
            context_tokens=int(match["context_tokens"]),
            generation_requests=int(match["generation_requests"]),
            generation_tokens=int(match["generation_tokens"]),
        )
        for match in ITERATION_PATTERN.finditer(text)
    )


def _write_logging_config(point_dir: Path) -> Path:
    log_path = point_dir / "iteration-details.log"
    config_path = point_dir / "logging-config.json"
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(log_path),
                "formatter": "plain",
                "level": "INFO",
            }
        },
        "loggers": {
            "vllm": {
                "handlers": ["file"],
                "level": "INFO",
                "propagate": False,
            }
        },
    }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path


def _query_nvidia_gpu(physical_index: str) -> dict[str, Any]:
    result = subprocess.run(
        (
            "nvidia-smi",
            f"--id={physical_index}",
            "--query-gpu=name,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    if len(fields) != 3 or not all(fields):
        raise RuntimeError("nvidia-smi returned incomplete GPU provenance")
    return {
        "visible_gpu_count": 1,
        "visible_gpu_model": fields[0],
        "nvidia_driver": fields[1],
        "gpu_uuid": fields[2],
    }


class OfflineLLMRuntime:
    """Translate public ``LLM`` operations into auditable batch observations."""

    def __init__(
        self,
        engine_config: dict[str, Any],
        point_dir: Path,
        *,
        llm: Any | None = None,
        tokens_prompt_factory: Callable[[list[int]], Any] | None = None,
        sampling_params_factory: Callable[..., Any] | None = None,
        runtime_versions: dict[str, str] | None = None,
    ) -> None:
        self._engine_config = engine_config
        self._point_dir = point_dir
        self._iteration_log = point_dir / "iteration-details.log"
        if llm is None:
            logging_config = _write_logging_config(point_dir)
            os.environ["VLLM_LOGGING_CONFIG_PATH"] = str(logging_config)
            self._gpu_provenance = _query_nvidia_gpu(
                engine_config["cuda_visible_devices"]
            )
            import torch

            from vllm import LLM, SamplingParams, TokensPrompt, __version__

            profiler_config = engine_config.get("profiler_config")
            self._llm = LLM(
                model=engine_config["model"],
                revision=engine_config["model_revision"],
                tokenizer=engine_config["model"],
                tokenizer_revision=engine_config["tokenizer_revision"],
                dtype=engine_config["dtype"],
                kv_cache_dtype=engine_config["kv_cache_dtype"],
                tensor_parallel_size=engine_config["tensor_parallel_size"],
                language_model_only=engine_config["language_model_only"],
                attention_config={"backend": engine_config["attention_backend"]},
                block_size=engine_config["block_size"],
                enable_prefix_caching=engine_config["enable_prefix_caching"],
                enable_chunked_prefill=engine_config["enable_chunked_prefill"],
                max_model_len=engine_config["max_model_len"],
                max_num_seqs=engine_config["max_num_seqs"],
                max_num_batched_tokens=engine_config["max_num_batched_tokens"],
                gpu_memory_utilization=engine_config["gpu_memory_utilization"],
                seed=engine_config["seed"],
                mamba_cache_mode=engine_config["mamba_cache_mode"],
                enable_logging_iteration_details=True,
                profiler_config=profiler_config,
            )
            self._tokens_prompt_factory = lambda token_ids: TokensPrompt(
                prompt_token_ids=token_ids
            )
            self._sampling_params_factory = SamplingParams
            self._runtime_versions = {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "vllm": __version__,
                "cuda": str(torch.version.cuda),
            }
        else:
            if tokens_prompt_factory is None or sampling_params_factory is None:
                raise ValueError("injected LLM requires prompt and sampling factories")
            self._llm = llm
            self._tokens_prompt_factory = tokens_prompt_factory
            self._sampling_params_factory = sampling_params_factory
            self._runtime_versions = runtime_versions or {}
            self._gpu_provenance = {}
            self._iteration_log.touch()

    def provenance(self) -> dict[str, Any]:
        """Return runtime versions and the public API boundary."""
        provenance = {
            "runner_boundary": "vllm.LLM.generate",
            "cache_reset_boundary": "vllm.LLM.reset_prefix_cache",
            "iteration_source": "enable_logging_iteration_details",
            "runtime_versions": self._runtime_versions,
        }
        provenance.update(self._gpu_provenance)
        return provenance

    def wait_idle(self) -> None:
        """Offline ``generate`` is synchronous, so returning means idle."""

    def reset_prefix_cache(self) -> bool:
        """Reset the public offline prefix cache."""
        return bool(self._llm.reset_prefix_cache())

    def start_profile(self, profile_prefix: str) -> None:
        """Start an explicitly configured diagnostic trace."""
        self._llm.start_profile(profile_prefix)

    def stop_profile(self) -> None:
        """Stop and flush an explicitly configured diagnostic trace."""
        self._llm.stop_profile()

    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation:
        """Generate one exact batch and attach its newly logged iterations."""
        log_offset = self._iteration_log.stat().st_size
        prompts = [
            self._tokens_prompt_factory(list(token_ids))
            for token_ids in prompt_token_ids
        ]
        sampling_params = self._sampling_params_factory(
            max_tokens=max_tokens,
            temperature=0.0,
            seed=0,
            ignore_eos=ignore_eos,
        )
        started = time.perf_counter()
        outputs = self._llm.generate(
            prompts,
            sampling_params,
            use_tqdm=False,
        )
        wall_time_ms = (time.perf_counter() - started) * 1_000
        with self._iteration_log.open(encoding="utf-8", errors="replace") as file:
            file.seek(log_offset)
            iteration_text = file.read()
        iterations = parse_iteration_records(iteration_text)
        if not iterations:
            raise RuntimeError("vLLM emitted no iteration-detail records")
        if len(outputs) != len(prompt_token_ids):
            raise RuntimeError("vLLM returned the wrong number of requests")
        requests = []
        for output, input_ids in zip(outputs, prompt_token_ids):
            if output.num_cached_tokens is None:
                raise RuntimeError("vLLM did not return num_cached_tokens")
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM returned an unexpected completion count")
            requests.append(
                RequestObservation(
                    request_id=output.request_id,
                    prompt_tokens=len(input_ids),
                    cached_tokens=output.num_cached_tokens,
                    output_token_ids=tuple(output.outputs[0].token_ids),
                )
            )
        return BatchObservation(
            wall_time_ms=wall_time_ms,
            requests=tuple(requests),
            iterations=iterations,
        )

    def close(self) -> None:
        """The worker process owns engine teardown on process exit."""


class SubprocessOfflineRuntime:
    """Persistent JSON-lines client for one isolated offline engine process."""

    def __init__(
        self,
        point: FixedBatchPoint,
        engine_config: dict[str, Any],
        point_dir: Path,
    ) -> None:
        self._point = point
        self._point_dir = point_dir
        engine_config_path = point_dir / "engine-config.json"
        cpu_affinity = engine_config["cpu_affinity"]
        numa_node = engine_config["numa_node"]
        if not isinstance(cpu_affinity, str) or not cpu_affinity:
            raise ValueError("production runtime requires CPU affinity")
        if isinstance(numa_node, bool) or not isinstance(numa_node, int):
            raise ValueError("production runtime requires a NUMA node")
        stderr_path = point_dir / "runtime-stderr.log"
        stdout_noise_path = point_dir / "runtime-stdout.log"
        self._stderr = stderr_path.open("w", encoding="utf-8")
        self._stdout_noise = stdout_noise_path.open("w", encoding="utf-8")
        command = (
            "numactl",
            f"--physcpubind={cpu_affinity}",
            f"--membind={numa_node}",
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.fixed_batch_runtime",
            "--serve",
            "--engine-config",
            str(engine_config_path),
            "--point-dir",
            str(point_dir),
        )
        environment = {
            **os.environ,
            **engine_config["runtime_environment"],
            "CUDA_VISIBLE_DEVICES": engine_config["cuda_visible_devices"],
        }
        (point_dir / "runtime-invocation.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "environment": {
                        name: environment.get(name)
                        for name in (
                            "CUDA_HOME",
                            "CUDA_VISIBLE_DEVICES",
                            "FLASHINFER_JIT_VERBOSE",
                            "HF_HOME",
                            "HF_HUB_CACHE",
                            "HF_HUB_OFFLINE",
                            "LD_LIBRARY_PATH",
                            "PATH",
                            "TRANSFORMERS_OFFLINE",
                            "VLLM_ENABLE_V1_MULTIPROCESSING",
                            "VLLM_KV_CACHE_LAYOUT",
                            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
                            "VLLM_SSM_CONV_STATE_LAYOUT",
                        )
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        try:
            ready = self._read_response()
        except Exception:
            self._terminate()
            raise
        if ready.get("event") != "ready":
            self.close()
            raise RuntimeError("offline runtime did not report readiness")
        self._provenance = ready["provenance"]

    def _failure_tail(self) -> str:
        self._stderr.flush()
        try:
            lines = (self._point_dir / "runtime-stderr.log").read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return ""
        return "\n".join(lines.splitlines()[-20:])

    def _read_response(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        while True:
            line = self._process.stdout.readline()
            if not line:
                returncode = self._process.poll()
                detail = self._failure_tail()
                raise RuntimeError(
                    f"offline runtime exited without a response: {returncode}\n{detail}"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                self._stdout_noise.write(line)
                self._stdout_noise.flush()
                continue
            if not isinstance(value, dict):
                self._stdout_noise.write(line)
                self._stdout_noise.flush()
                continue
            if value.get("ok") is False:
                raise RuntimeError(
                    f"{value.get('error_type', 'RuntimeError')}: "
                    f"{value.get('error', '')}"
                )
            return value

    def _request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError("offline runtime exited before request")
        assert self._process.stdin is not None
        self._process.stdin.write(
            json.dumps({"operation": operation, **payload}) + "\n"
        )
        self._process.stdin.flush()
        return self._read_response()

    def provenance(self) -> dict[str, Any]:
        return self._provenance

    def wait_idle(self) -> None:
        self._request("wait_idle")

    def reset_prefix_cache(self) -> bool:
        return bool(self._request("reset_prefix_cache")["success"])

    def start_profile(self, profile_prefix: str) -> None:
        self._request("start_profile", profile_prefix=profile_prefix)

    def stop_profile(self) -> None:
        self._request("stop_profile")

    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation:
        response = self._request(
            "generate",
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
        value = response["observation"]
        return BatchObservation(
            wall_time_ms=value["wall_time_ms"],
            requests=tuple(
                RequestObservation(
                    request_id=request["request_id"],
                    prompt_tokens=request["prompt_tokens"],
                    cached_tokens=request["cached_tokens"],
                    output_token_ids=tuple(request["output_token_ids"]),
                )
                for request in value["requests"]
            ),
            iterations=tuple(
                IterationRecord(**iteration) for iteration in value["iterations"]
            ),
        )

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._request("close")
            except (BrokenPipeError, RuntimeError):
                self._terminate()
            else:
                try:
                    self._process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._terminate()
        self._stderr.close()
        self._stdout_noise.close()

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._stderr.close()
        self._stdout_noise.close()


class SubprocessRuntimeFactory:
    """Create one isolated engine process for each explicit point."""

    def __call__(
        self,
        point: FixedBatchPoint,
        engine_config: dict[str, Any],
        point_dir: Path,
    ) -> SubprocessOfflineRuntime:
        return SubprocessOfflineRuntime(point, engine_config, point_dir)


def _load_engine_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("engine config must be a JSON object")
    return value


def _response(value: dict[str, Any]) -> None:
    print(json.dumps(value), flush=True)


def _serve(engine_config_path: Path, point_dir: Path) -> int:
    runtime = OfflineLLMRuntime(
        _load_engine_config(engine_config_path),
        point_dir,
    )
    _response(
        {
            "ok": True,
            "event": "ready",
            "provenance": runtime.provenance(),
        }
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["operation"]
            if operation == "wait_idle":
                runtime.wait_idle()
                result: dict[str, Any] = {}
            elif operation == "reset_prefix_cache":
                result = {"success": runtime.reset_prefix_cache()}
            elif operation == "start_profile":
                runtime.start_profile(request["profile_prefix"])
                result = {}
            elif operation == "stop_profile":
                runtime.stop_profile()
                result = {}
            elif operation == "generate":
                observation = runtime.generate(
                    tuple(tuple(prompt) for prompt in request["prompt_token_ids"]),
                    max_tokens=request["max_tokens"],
                    ignore_eos=request["ignore_eos"],
                )
                result = {
                    "observation": {
                        "wall_time_ms": observation.wall_time_ms,
                        "requests": [asdict(item) for item in observation.requests],
                        "iterations": [asdict(item) for item in observation.iterations],
                    }
                }
            elif operation == "close":
                runtime.close()
                _response({"ok": True})
                return 0
            else:
                raise ValueError(f"unsupported operation: {operation}")
            _response({"ok": True, **result})
        except Exception as error:
            _response(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--engine-config", type=Path)
    parser.add_argument("--point-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.serve or args.engine_config is None or args.point_dir is None:
        raise ValueError("--serve, --engine-config, and --point-dir are required")
    return _serve(args.engine_config, args.point_dir)


if __name__ == "__main__":
    raise SystemExit(main())
