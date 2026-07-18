# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import regex as re


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, check=False, text=True)


def _check_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    files = []
    for artifact in config.get("artifacts", []):
        path = Path(artifact["path"])
        actual_hash = _sha256(path) if path.is_file() else None
        files.append(
            {
                "actual_sha256": actual_hash,
                "expected_sha256": artifact["sha256"],
                "name": artifact["name"],
                "path": str(path),
                "status": ("passed" if actual_hash == artifact["sha256"] else "failed"),
            }
        )
    status = "passed" if all(item["status"] == "passed" for item in files) else "failed"
    return {"files": files, "status": status}


def _check_executable(name: str) -> dict[str, str]:
    executable = shutil.which(name)
    if executable is None:
        return {
            "detail": f"{name} is not installed or not on PATH",
            "status": "failed",
        }
    return {"detail": executable, "status": "passed"}


def _check_results_dir(path_value: str | None) -> dict[str, str]:
    if path_value is None:
        return {"detail": "results_dir is not configured", "status": "failed"}
    path = Path(path_value)
    if not path.is_dir():
        return {
            "detail": f"results directory does not exist: {path}",
            "status": "failed",
        }
    try:
        with tempfile.NamedTemporaryFile(dir=path):
            pass
    except OSError as error:
        return {"detail": str(error), "status": "failed"}
    return {"detail": str(path), "status": "passed"}


def _check_nixl(expected_version: str | None) -> dict[str, Any]:
    try:
        version = importlib.metadata.version("nixl")
        modules = ["nixl", "nixl._api", "nixl._bindings"]
        for module in modules:
            importlib.import_module(module)
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        return {"detail": f"{type(error).__name__}: {error}", "status": "failed"}
    if expected_version is not None and version != expected_version:
        return {
            "detail": f"expected nixl {expected_version}, found {version}",
            "status": "failed",
        }
    return {
        "detail": {"modules": modules, "version": version},
        "status": "passed",
    }


def _check_cuda(config: dict[str, Any]) -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
        torch_cuda = torch.version.cuda
        device_count = torch.accelerator.device_count()
        available = torch.accelerator.is_available()
    except (AttributeError, ImportError) as error:
        return {"detail": f"{type(error).__name__}: {error}", "status": "failed"}
    expected_count = config.get("hardware", {}).get("gpu_count")
    expected_cuda = config.get("base", {}).get("cuda")
    cuda_prefix = ".".join(expected_cuda.split(".")[:2]) if expected_cuda else None
    matches = (
        available
        and (expected_count is None or device_count == expected_count)
        and (cuda_prefix is None or str(torch_cuda).startswith(cuda_prefix))
    )
    return {
        "detail": {"device_count": device_count, "torch_cuda": torch_cuda},
        "status": "passed" if matches else "failed",
    }


def _check_numactl() -> dict[str, Any]:
    availability = _check_executable("numactl")
    if availability["status"] == "failed":
        return availability
    result = _run(["numactl", "--hardware"])
    nodes = {
        int(match.group(1)): {int(cpu) for cpu in match.group(2).split()}
        for match in re.finditer(r"^node (\d+) cpus:\s*(.*)$", result.stdout, re.M)
    }
    role_nodes = {"decode": 1, "prefill": 0}
    role_cpus = {
        "decode": {
            int(cpu)
            for cpu in os.environ.get("DS4_DECODE_CPUSET", "").split(",")
            if cpu
        },
        "prefill": {
            int(cpu)
            for cpu in os.environ.get("DS4_PREFILL_CPUSET", "").split(",")
            if cpu
        },
    }
    matches = result.returncode == 0 and all(
        role_cpus[role] and role_cpus[role].issubset(nodes.get(node, set()))
        for role, node in role_nodes.items()
    )
    return {
        "detail": {"output": result.stdout, "role_nodes": role_nodes},
        "status": "passed" if matches else "failed",
    }


def _installed_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in ("nixl", "pyarrow", "torch", "transformers", "uv", "vllm"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _mounts() -> dict[str, dict[str, str | None]]:
    specifications = {
        "config": ("DS4_HOST_CONFIG_DIR", "/mnt/ds4/config", "ro"),
        "hf_cache": (
            "DS4_HOST_HF_CACHE_DIR",
            "/mnt/ds4/cache/huggingface",
            "rw",
        ),
        "raw": ("DS4_HOST_SNAPSHOT_DIR", "/mnt/ds4/raw", "ro"),
        "results": ("DS4_HOST_RESULTS_DIR", "/mnt/ds4/results", "rw"),
        "runtime_cache": (
            "DS4_HOST_RUNTIME_CACHE_DIR",
            "/mnt/ds4/cache/runtime",
            "rw",
        ),
        "ticket_01": ("DS4_HOST_TICKET_01_DIR", "/mnt/ds4/ticket-01", "ro"),
        "ticket_02": ("DS4_HOST_TICKET_02_DIR", "/mnt/ds4/ticket-02", "ro"),
        "tokenizers": (
            "DS4_HOST_TOKENIZERS_DIR",
            "/mnt/ds4/tokenizers",
            "ro",
        ),
        "uv_cache": ("DS4_HOST_UV_CACHE_DIR", "/mnt/ds4/cache/uv", "rw"),
    }
    return {
        name: {
            "container": container,
            "host": os.environ.get(environment),
            "mode": mode,
        }
        for name, (environment, container, mode) in specifications.items()
    }


def _build_inputs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "base": config.get("base"),
        "dependencies": config.get("dependencies"),
        "model": config.get("model"),
        "revisions": config.get("revisions"),
    }


def _artifact_path(config: dict[str, Any], name: str) -> Path:
    artifact = next(
        (item for item in config["artifacts"] if item["name"] == name), None
    )
    if artifact is None:
        raise ValueError(f"container contract does not define artifact {name}")
    return Path(artifact["path"])


def _cpu_commands(
    config: dict[str, Any],
    output_dir: Path,
    manifest_path: Path | None = None,
    ds4_tokenizer_path: Path | None = None,
    qwen_tokenizer_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    ticket_01_dir = output_dir / "ticket-01"
    ticket_02_dir = output_dir / "ticket-02"
    manifest_path = manifest_path or _artifact_path(config, "dataset_manifest")
    ds4_tokenizer_path = ds4_tokenizer_path or Path(config["tokenizers"]["ds4"])
    qwen_tokenizer_path = qwen_tokenizer_path or Path(config["tokenizers"]["qwen"])
    normalize_command = [
        sys.executable,
        "-m",
        "benchmarks.ds4_profile.normalize",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(ticket_01_dir),
        "--require-complete-pilot",
    ]
    workload_command = [
        sys.executable,
        "-m",
        "benchmarks.ds4_profile.workloads",
        "--manifest",
        str(manifest_path),
        "--normalized-turns",
        str(ticket_01_dir / "turns.parquet"),
        "--ds4-tokenizer",
        str(ds4_tokenizer_path),
        "--qwen-tokenizer",
        str(qwen_tokenizer_path),
        "--output-dir",
        str(ticket_02_dir),
        "--block-size",
        "16",
        "--seed",
        "20260715",
    ]
    return normalize_command, workload_command


def _generated_artifacts(output_dir: Path) -> dict[str, Path]:
    return {
        "normalized_turns": output_dir / "ticket-01/turns.parquet",
        "rendered_turns": output_dir / "ticket-02/rendered_turns.parquet",
        "ticket_02_provenance": output_dir / "ticket-02/provenance.json",
        "workload_plan": output_dir / "ticket-02/workload_plan.json",
    }


def _cpu_dry_run(
    config_path: Path,
    output_dir: Path,
    print_plan: bool,
    manifest_path: Path | None,
    ds4_tokenizer_path: Path | None,
    qwen_tokenizer_path: Path | None,
) -> int:
    config = json.loads(config_path.read_text())
    commands = _cpu_commands(
        config,
        output_dir,
        manifest_path,
        ds4_tokenizer_path,
        qwen_tokenizer_path,
    )
    if print_plan:
        for command in commands:
            print(shlex.join(command))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    command_results = []
    for command in commands:
        result = subprocess.run(command, check=False)
        command_results.append({"command": command, "returncode": result.returncode})
        if result.returncode != 0:
            break

    expected_hashes = {
        item["name"]: item["sha256"] for item in config.get("artifacts", [])
    }
    artifacts = []
    for name, path in _generated_artifacts(output_dir).items():
        actual_hash = _sha256(path) if path.is_file() else None
        artifacts.append(
            {
                "actual_sha256": actual_hash,
                "expected_sha256": expected_hashes.get(name),
                "name": name,
                "path": str(path),
                "status": (
                    "passed" if actual_hash == expected_hashes.get(name) else "failed"
                ),
            }
        )
    status = (
        "passed"
        if len(command_results) == len(commands)
        and all(result["returncode"] == 0 for result in command_results)
        and all(artifact["status"] == "passed" for artifact in artifacts)
        else "invalid"
    )
    provenance = {
        "artifacts": artifacts,
        "commands": command_results,
        "hardware_validated": False,
        "host_invocation": os.environ.get("DS4_HOST_INVOCATION"),
        "image": {"id": os.environ.get("DS4_IMAGE_ID", "unknown")},
        "installed_versions": _installed_versions(),
        "schema_version": config["schema_version"],
        "source": {
            "commit": os.environ.get("DS4_VLLM_COMMIT", "unknown"),
            "dirty": os.environ.get("DS4_VLLM_DIRTY", "unknown").lower() == "true",
        },
        "status": status,
    }
    with (output_dir / "provenance.json").open("w", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2, sort_keys=True)
        file.write("\n")
    return 0 if status == "passed" else 2


def _cache_model(config_path: Path, output_path: Path, print_plan: bool) -> int:
    config = json.loads(config_path.read_text())
    cache_dir = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            os.environ.get("HF_HOME", "/mnt/ds4/cache/huggingface"),
        )
    )
    command = [
        "hf",
        "download",
        config["model"]["repo_id"],
        "--revision",
        config["model"]["revision"],
        "--cache-dir",
        str(cache_dir),
    ]
    if print_plan:
        print(shlex.join(command))
        return 0

    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    try:
        snapshot_path = snapshot_download(
            repo_id=config["model"]["repo_id"],
            revision=config["model"]["revision"],
            cache_dir=cache_dir,
        )
        record = {
            "cache_dir": str(cache_dir),
            "elapsed_seconds": time.perf_counter() - started,
            "model": config["model"],
            "snapshot_path": snapshot_path,
            "status": "passed",
        }
        returncode = 0
    except Exception as error:
        record = {
            "cache_dir": str(cache_dir),
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "model": config["model"],
            "status": "failed",
        }
        returncode = 1
    _write_json(output_path, record)
    return returncode


def _exec_profile(output_path: Path, command: list[str]) -> int:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("exec requires a command after --")
    started = time.perf_counter()
    result = subprocess.run(command, check=False)
    record = {
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "hardware_validated": False,
        "host_invocation": os.environ.get("DS4_HOST_INVOCATION"),
        "image": {"id": os.environ.get("DS4_IMAGE_ID", "unknown")},
        "installed_versions": _installed_versions(),
        "returncode": result.returncode,
        "source": {
            "commit": os.environ.get("DS4_VLLM_COMMIT", "unknown"),
            "dirty": os.environ.get("DS4_VLLM_DIRTY", "unknown").lower() == "true",
        },
        "status": "passed" if result.returncode == 0 else "failed",
    }
    _write_json(output_path, record)
    return result.returncode


def _nvidia_checks(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    availability = _check_executable("nvidia-smi")
    if availability["status"] == "failed":
        return availability, {
            "devices": [],
            "driver": None,
            "p2p_read": None,
            "topology": None,
        }

    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader",
        ]
    )
    devices = []
    if query.returncode == 0:
        for row in query.stdout.splitlines():
            fields = [field.strip() for field in row.split(",", 3)]
            if len(fields) == 4:
                devices.append(
                    {
                        "driver": fields[3],
                        "index": int(fields[0]),
                        "name": fields[1],
                        "uuid": fields[2],
                    }
                )
    topology = _run(["nvidia-smi", "topo", "-m"])
    p2p = _run(["nvidia-smi", "topo", "-p2p", "r"])
    hardware = config.get("hardware", {})
    matches = (
        query.returncode == 0
        and len(devices) == hardware.get("gpu_count")
        and all(device["name"] == hardware.get("gpu_name") for device in devices)
        and topology.returncode == 0
        and hardware.get("topology") in topology.stdout
        and p2p.returncode == 0
        and hardware.get("p2p_read") in p2p.stdout
    )
    return {
        "detail": "two expected GPUs and topology detected"
        if matches
        else query.stderr,
        "status": "passed" if matches else "failed",
    }, {
        "devices": devices,
        "driver": devices[0]["driver"] if devices else None,
        "p2p_read": p2p.stdout,
        "topology": topology.stdout,
    }


def _preflight(config_path: Path, output_path: Path) -> int:
    config = json.loads(config_path.read_text())
    artifact_check = _check_artifacts(config)
    nvidia_check, hardware = _nvidia_checks(config)
    installed_versions = _installed_versions()
    checks = {
        "artifacts": artifact_check,
        "cuda": _check_cuda(config),
        "nixl": _check_nixl(config.get("dependencies", {}).get("nixl")),
        "numactl": _check_numactl(),
        "nvidia_smi": nvidia_check,
        "results": _check_results_dir(config.get("results_dir")),
        "uv": _check_executable("uv"),
    }
    status = (
        "ready"
        if all(check["status"] == "passed" for check in checks.values())
        else "invalid"
    )
    provenance = {
        "build_inputs": _build_inputs(config),
        "checks": checks,
        "hardware": hardware,
        "hardware_validated": False,
        "host_invocation": os.environ.get("DS4_HOST_INVOCATION"),
        "image": {"id": os.environ.get("DS4_IMAGE_ID", "unknown")},
        "installed_versions": installed_versions,
        "invocation": [sys.executable, *sys.argv],
        "mounts": _mounts(),
        "roles": {
            "decode": {
                "cpuset": os.environ.get("DS4_DECODE_CPUSET"),
                "gpu": os.environ.get("DS4_DECODE_GPU"),
            },
            "prefill": {
                "cpuset": os.environ.get("DS4_PREFILL_CPUSET"),
                "gpu": os.environ.get("DS4_PREFILL_GPU"),
            },
        },
        "schema_version": config["schema_version"],
        "source": {
            "commit": os.environ.get("DS4_VLLM_COMMIT", "unknown"),
            "dirty": os.environ.get("DS4_VLLM_DIRTY", "unknown").lower() == "true",
        },
        "status": status,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2, sort_keys=True)
        file.write("\n")
    print(output_path)
    return 0 if status == "ready" else 2


def _gpu_commands(config_path: Path, output_path: Path) -> list[list[str]]:
    role_settings = (
        (
            "prefill",
            os.environ.get("DS4_PREFILL_GPU", "0"),
            os.environ.get("DS4_PREFILL_CPUSET", "0,2,4,6,8,10"),
            "0",
        ),
        (
            "decode",
            os.environ.get("DS4_DECODE_GPU", "1"),
            os.environ.get("DS4_DECODE_CPUSET", "1,3,5,7,9,11"),
            "1",
        ),
    )
    return [
        [
            "numactl",
            f"--physcpubind={cpuset}",
            f"--membind={node}",
            "env",
            f"CUDA_VISIBLE_DEVICES={gpu}",
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "gpu-worker",
            "--config",
            str(config_path),
            "--role",
            role,
            "--output",
            str(output_path.with_name(f"gpu-smoke-{role}.json")),
        ]
        for role, gpu, cpuset, node in role_settings
    ]


def _load_exact_replay(config: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    plan = json.loads(_artifact_path(config, "workload_plan").read_text())
    replay = min(plan["exact_replays"], key=lambda item: item.get("prompt_tokens", 0))
    rendered_turns = pq.read_table(
        _artifact_path(config, "rendered_turns"),
        columns=["execution_prompt_token_ids", "trajectory_id", "turn_index"],
    ).to_pylist()
    row = next(
        (
            item
            for item in rendered_turns
            if item["trajectory_id"] == replay["trajectory_id"]
            and item["turn_index"] == replay["turn_index"]
        ),
        None,
    )
    if row is None or not row["execution_prompt_token_ids"]:
        raise ValueError("selected exact replay has no execution prompt token IDs")
    token_ids = row["execution_prompt_token_ids"]
    token_bytes = json.dumps(token_ids, separators=(",", ":")).encode()
    return {
        "prompt_token_count": len(token_ids),
        "prompt_token_ids": token_ids,
        "prompt_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "trajectory_id": row["trajectory_id"],
        "turn_index": row["turn_index"],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def _gpu_worker(
    config_path: Path, role: str, output_path: Path, inspect_input: bool
) -> int:
    config = json.loads(config_path.read_text())
    replay = _load_exact_replay(config)
    input_record = {
        key: value for key, value in replay.items() if key != "prompt_token_ids"
    }
    if inspect_input:
        _write_json(
            output_path,
            {
                "hardware_validated": False,
                "input": input_record,
                "role": role,
                "status": "input-ready",
            },
        )
        return 0

    started = time.perf_counter()
    try:
        from vllm import LLM, SamplingParams, TokensPrompt

        llm = LLM(
            model=config["model"]["repo_id"],
            revision=config["model"]["revision"],
            tokenizer=config["tokenizers"]["qwen"],
            dtype="half",
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            enforce_eager=True,
            gpu_memory_utilization=0.90,
            max_model_len=max(8192, replay["prompt_token_count"] + 1),
            max_num_batched_tokens=4096,
            max_num_seqs=1,
            tensor_parallel_size=1,
        )
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=replay["prompt_token_ids"])],
            SamplingParams(max_tokens=1, temperature=0.0),
            use_tqdm=False,
        )
        generated_token_count = len(outputs[0].outputs[0].token_ids)
        if generated_token_count != 1:
            raise RuntimeError(
                f"expected one generated token, found {generated_token_count}"
            )
        record = {
            "elapsed_seconds": time.perf_counter() - started,
            "generated_token_count": generated_token_count,
            "hardware_validated": False,
            "input": input_record,
            "model": config["model"],
            "role": role,
            "runner_boundary": "vllm.LLM.generate",
            "status": "passed",
        }
        returncode = 0
    except Exception as error:
        record = {
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "hardware_validated": False,
            "input": input_record,
            "model": config["model"],
            "role": role,
            "runner_boundary": "vllm.LLM.generate",
            "status": "failed",
        }
        returncode = 1
    _write_json(output_path, record)
    return returncode


def _gpu_smoke(config_path: Path, output_path: Path, print_plan: bool) -> int:
    commands = _gpu_commands(config_path, output_path)
    if print_plan:
        for command in commands:
            print(shlex.join(command))
        return 0
    preflight_path = output_path.with_name("gpu-smoke-preflight.json")
    preflight_returncode = _preflight(config_path, preflight_path)
    preflight = json.loads(preflight_path.read_text())
    if preflight_returncode != 0:
        provenance = {
            "hardware_validated": False,
            "preflight": preflight,
            "reason": "hardware preflight failed",
            "status": "skipped",
            "workers": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(provenance, file, indent=2, sort_keys=True)
            file.write("\n")
        return 2

    processes = [subprocess.Popen(command) for command in commands]
    returncodes = [process.wait() for process in processes]
    workers = []
    for command, returncode in zip(commands, returncodes):
        worker_path = Path(command[-1])
        worker = (
            json.loads(worker_path.read_text())
            if worker_path.is_file()
            else {"status": "failed", "error": "worker record is missing"}
        )
        worker["command"] = command
        worker["returncode"] = returncode
        workers.append(worker)
    passed = all(
        worker["returncode"] == 0 and worker["status"] == "passed" for worker in workers
    )
    provenance = {
        "hardware_validated": passed,
        "host_invocation": os.environ.get("DS4_HOST_INVOCATION"),
        "image": {"id": os.environ.get("DS4_IMAGE_ID", "unknown")},
        "installed_versions": _installed_versions(),
        "invocation": [sys.executable, *sys.argv],
        "preflight": preflight,
        "source": {
            "commit": os.environ.get("DS4_VLLM_COMMIT", "unknown"),
            "dirty": os.environ.get("DS4_VLLM_DIRTY", "unknown").lower() == "true",
        },
        "status": "passed" if passed else "invalid",
        "workers": workers,
    }
    _write_json(output_path, provenance)
    return 0 if passed else 2


def _profile_role_settings() -> dict[str, dict[str, str | int]]:
    return {
        "prefill": {
            "gpu": os.environ.get("DS4_PREFILL_GPU", "0"),
            "cpuset": os.environ.get("DS4_PREFILL_CPUSET", "0,2,4,6,8,10"),
            "numa_node": 0,
        },
        "decode": {
            "gpu": os.environ.get("DS4_DECODE_GPU", "1"),
            "cpuset": os.environ.get("DS4_DECODE_CPUSET", "1,3,5,7,9,11"),
            "numa_node": 1,
        },
    }


def _profile_spine_worker_commands(
    profile_config_path: Path, work_dir: Path
) -> list[list[str]]:
    return [
        [
            "numactl",
            f"--physcpubind={settings['cpuset']}",
            f"--membind={settings['numa_node']}",
            "env",
            f"CUDA_VISIBLE_DEVICES={settings['gpu']}",
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.profile_spine",
            "gpu-worker",
            "--config",
            str(profile_config_path),
            "--role",
            role,
            "--output",
            str(work_dir / f"{role}.json"),
        ]
        for role, settings in _profile_role_settings().items()
    ]


def _effective_profile_config(profile_config_path: Path) -> dict[str, Any]:
    config = json.loads(profile_config_path.read_text())
    if not config.get("run_id"):
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        config["run_id"] = f"ds4-spine-{timestamp}-{uuid.uuid4().hex[:8]}"
    dirty_value = os.environ.get(
        "DS4_VLLM_DIRTY", str(config.get("source", {}).get("dirty", True))
    ).lower()
    config["source"] = {
        "commit": os.environ.get(
            "DS4_VLLM_COMMIT", config.get("source", {}).get("commit", "unknown")
        ),
        "dirty": (
            dirty_value == "true" if dirty_value in {"true", "false"} else "unknown"
        ),
    }
    config["roles"] = _profile_role_settings()
    return config


def _profile_spine_assemble_command(
    config_path: Path,
    preflight_path: Path,
    work_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.ds4_profile.profile_spine",
        "assemble",
        "--config",
        str(config_path),
        "--preflight",
        str(preflight_path),
        "--worker-result",
        str(work_dir / "prefill.json"),
        "--worker-result",
        str(work_dir / "decode.json"),
        "--output-dir",
        str(output_dir),
    ]


def _profile_spine(
    container_config_path: Path,
    profile_config_path: Path,
    output_dir: Path | None,
    print_plan: bool,
) -> int:
    config = _effective_profile_config(profile_config_path)
    if output_dir is None:
        output_dir = Path("/mnt/ds4/results/ticket-04") / config["run_id"]
    work_dir = output_dir.parent / f".{output_dir.name}.workers"
    effective_config_path = work_dir / "run-config.json"
    preflight_path = work_dir / "preflight.json"
    commands = _profile_spine_worker_commands(effective_config_path, work_dir)
    assemble_command = _profile_spine_assemble_command(
        effective_config_path, preflight_path, work_dir, output_dir
    )
    if print_plan:
        print(
            shlex.join(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.ds4_profile.container.runtime",
                    "preflight",
                    "--config",
                    str(container_config_path),
                    "--output",
                    str(preflight_path),
                ]
            )
        )
        for command in commands:
            print(shlex.join(command))
        print(shlex.join(assemble_command))
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.workers-"
    ) as temporary_dir:
        actual_work_dir = Path(temporary_dir)
        actual_config_path = actual_work_dir / "run-config.json"
        actual_preflight_path = actual_work_dir / "preflight.json"
        _write_json(actual_config_path, config)
        preflight_returncode = _preflight(container_config_path, actual_preflight_path)
        actual_commands = _profile_spine_worker_commands(
            actual_config_path, actual_work_dir
        )
        if preflight_returncode == 0:
            processes = [subprocess.Popen(command) for command in actual_commands]
            returncodes = [process.wait() for process in processes]
        else:
            returncodes = [2, 2]
            for role in ("prefill", "decode"):
                _write_json(
                    actual_work_dir / f"{role}.json",
                    {
                        "schema_version": config["schema_version"],
                        "run_id": config["run_id"],
                        "hardware_validated": False,
                        "role": role,
                        "runner_boundary": (
                            "vllm.v1.worker.gpu_worker.Worker.execute_model"
                        ),
                        "samples": [],
                        "status": "skipped",
                        "error": "hardware preflight failed",
                    },
                )
        for command, returncode in zip(actual_commands, returncodes):
            worker_path = Path(command[-1])
            worker = json.loads(worker_path.read_text())
            worker["command"] = command
            worker["returncode"] = returncode
            _write_json(worker_path, worker)
        actual_assemble_command = _profile_spine_assemble_command(
            actual_config_path,
            actual_preflight_path,
            actual_work_dir,
            output_dir,
        )
        return subprocess.run(actual_assemble_command, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="DS4 container runtime checks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument(
        "--config", type=Path, default=Path("/mnt/ds4/config/container-contract.json")
    )
    preflight.add_argument(
        "--output", type=Path, default=Path("/mnt/ds4/results/preflight.json")
    )
    cpu_dry_run = subparsers.add_parser("cpu-dry-run")
    cpu_dry_run.add_argument(
        "--config", type=Path, default=Path("/mnt/ds4/config/container-contract.json")
    )
    cpu_dry_run.add_argument(
        "--output-dir", type=Path, default=Path("/mnt/ds4/results/cpu-dry-run")
    )
    cpu_dry_run.add_argument("--manifest", type=Path)
    cpu_dry_run.add_argument("--ds4-tokenizer", type=Path)
    cpu_dry_run.add_argument("--qwen-tokenizer", type=Path)
    cpu_dry_run.add_argument("--print-plan", action="store_true")
    gpu_smoke = subparsers.add_parser("gpu-smoke")
    gpu_smoke.add_argument(
        "--config", type=Path, default=Path("/mnt/ds4/config/container-contract.json")
    )
    gpu_smoke.add_argument(
        "--output", type=Path, default=Path("/mnt/ds4/results/gpu-smoke.json")
    )
    gpu_smoke.add_argument("--print-plan", action="store_true")
    gpu_worker = subparsers.add_parser("gpu-worker")
    gpu_worker.add_argument("--config", type=Path, required=True)
    gpu_worker.add_argument("--role", choices=("prefill", "decode"), required=True)
    gpu_worker.add_argument("--output", type=Path, required=True)
    gpu_worker.add_argument("--inspect-input", action="store_true")
    profile_spine = subparsers.add_parser("profile-spine")
    profile_spine.add_argument(
        "--config", type=Path, default=Path("/mnt/ds4/config/container-contract.json")
    )
    profile_spine.add_argument(
        "--profile-config",
        type=Path,
        default=Path("/mnt/ds4/config/profile-spine.json"),
    )
    profile_spine.add_argument("--output-dir", type=Path)
    profile_spine.add_argument("--print-plan", action="store_true")
    cache_model = subparsers.add_parser("cache-model")
    cache_model.add_argument(
        "--config", type=Path, default=Path("/mnt/ds4/config/container-contract.json")
    )
    cache_model.add_argument(
        "--output", type=Path, default=Path("/mnt/ds4/results/cache-model.json")
    )
    cache_model.add_argument("--print-plan", action="store_true")
    profile_exec = subparsers.add_parser("exec")
    profile_exec.add_argument("--output", type=Path, required=True)
    profile_exec.add_argument("profile_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "preflight":
        raise SystemExit(_preflight(args.config, args.output))
    if args.command == "cpu-dry-run":
        raise SystemExit(
            _cpu_dry_run(
                args.config,
                args.output_dir,
                args.print_plan,
                args.manifest,
                args.ds4_tokenizer,
                args.qwen_tokenizer,
            )
        )
    if args.command == "gpu-smoke":
        raise SystemExit(_gpu_smoke(args.config, args.output, args.print_plan))
    if args.command == "gpu-worker":
        raise SystemExit(
            _gpu_worker(args.config, args.role, args.output, args.inspect_input)
        )
    if args.command == "profile-spine":
        raise SystemExit(
            _profile_spine(
                args.config,
                args.profile_config,
                args.output_dir,
                args.print_plan,
            )
        )
    if args.command == "cache-model":
        raise SystemExit(_cache_model(args.config, args.output, args.print_plan))
    if args.command == "exec":
        raise SystemExit(_exec_profile(args.output, args.profile_command))


if __name__ == "__main__":
    main()
