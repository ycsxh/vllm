# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).parents[3]
CONTAINER_DIR = PROJECT_DIR / "benchmarks" / "ds4_profile" / "container"


def test_profile_image_reuses_local_vllm_image_and_installs_an_isolated_uv_env() -> (
    None
):
    dockerfile = (CONTAINER_DIR / "Dockerfile").read_text()

    assert "ARG VLLM_BASE_IMAGE" in dockerfile
    assert "FROM ${VLLM_BASE_IMAGE}" in dockerfile
    assert "COPY benchmarks/ds4_profile" in dockerfile
    assert "uv venv --system-site-packages /opt/ds4-profile" in dockerfile
    assert "uv pip install" in dockerfile
    assert "--python /opt/ds4-profile/bin/python" in dockerfile
    assert "benchmarks/ds4_profile/requirements.txt" in dockerfile
    assert "benchmarks/ds4_profile/container/requirements.txt" in dockerfile
    assert "pip install" not in dockerfile.replace("uv pip install", "")
    assert "git clone" not in dockerfile
    assert "vllm-project/vllm" not in dockerfile


def test_build_dry_run_consumes_the_checkout_and_enables_pinned_kv_dependencies(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            "bash",
            str(CONTAINER_DIR / "build.sh"),
            "--dry-run",
            "--image",
            "local/ds4-profile:test",
            "--metadata-out",
            str(tmp_path / "build.json"),
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("docker build") == 2
    assert "--target vllm-openai-base" in result.stdout
    assert "INSTALL_KV_CONNECTORS=true" in result.stdout
    assert "CUDA_VERSION=13.0.2" in result.stdout
    assert "PYTHON_VERSION=3.12" in result.stdout
    assert "UBUNTU_VERSION=22.04" in result.stdout
    assert "nvidia/cuda:13.0.2-devel-ubuntu22.04" in result.stdout
    assert "nvidia/cuda:13.0.2-base-ubuntu22.04" in result.stdout
    assert "benchmarks/ds4_profile/container/Dockerfile" in result.stdout
    assert "VLLM_BASE_IMAGE=local/ds4-profile:test-base" in result.stdout
    assert "local/ds4-profile:test" in result.stdout
    assert "git clone" not in result.stdout
    assert "github.com/vllm-project" not in result.stdout


def test_run_dry_run_exposes_both_gpu_roles_and_persistent_mounts(
    tmp_path: Path,
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "snapshot",
            "ticket-01",
            "ticket-02",
            "tokenizers",
            "config",
            "hf-cache",
            "uv-cache",
            "results",
            "runtime-cache",
        )
    }
    command = [
        "bash",
        str(CONTAINER_DIR / "run.sh"),
        "--dry-run",
        "--image",
        "local/ds4-profile:test",
    ]
    for name, path in paths.items():
        command.extend([f"--{name}-dir", str(path)])
    command.append("preflight")

    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        cwd=tmp_path,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    rendered_command = result.stdout.replace("\\", "")
    assert "docker run" in rendered_command
    assert "--gpus all" in rendered_command
    assert "--ipc=host" in rendered_command
    assert "--network=host" in rendered_command
    assert "--ulimit memlock=-1:-1" in rendered_command
    assert "--cpuset-cpus=0-11" in rendered_command
    assert "--cap-add SYS_NICE" in rendered_command
    assert "DS4_PREFILL_GPU=0" in rendered_command
    assert "DS4_DECODE_GPU=1" in rendered_command
    assert "DS4_PREFILL_CPUSET=0,2,4,6,8,10" in rendered_command
    assert "DS4_DECODE_CPUSET=1,3,5,7,9,11" in rendered_command
    assert "DS4_HOST_INVOCATION=" in rendered_command
    assert "HF_HUB_OFFLINE=1" in rendered_command
    assert ":/mnt/ds4/raw:ro" in rendered_command
    assert ":/mnt/ds4/ticket-01:ro" in rendered_command
    assert ":/mnt/ds4/ticket-02:ro" in rendered_command
    assert ":/mnt/ds4/tokenizers:ro" in rendered_command
    assert ":/mnt/ds4/config:ro" in rendered_command
    assert ":/mnt/ds4/cache/huggingface" in rendered_command
    assert ":/mnt/ds4/cache/uv" in rendered_command
    assert ":/mnt/ds4/cache/runtime" in rendered_command
    assert "VLLM_CACHE_ROOT=/mnt/ds4/cache/runtime/vllm" in rendered_command
    assert "TRITON_CACHE_DIR=/mnt/ds4/cache/runtime/triton" in rendered_command
    assert ":/mnt/ds4/results" in rendered_command
    assert "--privileged" not in rendered_command


def test_cpu_dry_run_does_not_require_an_nvidia_runtime(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(CONTAINER_DIR / "run.sh"),
            "--dry-run",
            "--image",
            "local/ds4-profile:test",
            "--snapshot-dir",
            str(tmp_path / "snapshot"),
            "--ticket-01-dir",
            str(tmp_path / "ticket-01"),
            "--ticket-02-dir",
            str(tmp_path / "ticket-02"),
            "--tokenizers-dir",
            str(tmp_path / "tokenizers"),
            "--config-dir",
            str(tmp_path / "config"),
            "--hf-cache-dir",
            str(tmp_path / "hf-cache"),
            "--uv-cache-dir",
            str(tmp_path / "uv-cache"),
            "--runtime-cache-dir",
            str(tmp_path / "runtime-cache"),
            "--results-dir",
            str(tmp_path / "results"),
            "cpu-dry-run",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--gpus" not in result.stdout
    assert "--cpuset-cpus" not in result.stdout
    assert "--cap-add" not in result.stdout
    assert "HF_HUB_OFFLINE=1" in result.stdout


def test_preflight_records_invalid_provenance_when_gpu_tools_are_absent(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "workload_plan.json"
    artifact_path.write_text('{"fixture": true}\n')
    config_path = tmp_path / "contract.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base": {
                    "cuda": "13.0.2",
                    "python": "3.12",
                    "ubuntu": "22.04",
                },
                "dependencies": {"nixl": "1.3.0", "uv": "0.11.28"},
                "artifacts": [
                    {
                        "name": "workload_plan",
                        "path": str(artifact_path),
                        "sha256": hashlib.sha256(
                            artifact_path.read_bytes()
                        ).hexdigest(),
                    }
                ],
                "hardware": {
                    "gpu_count": 2,
                    "gpu_name": "NVIDIA GeForce RTX 3090",
                    "p2p_read": "CNS",
                    "topology": "SYS",
                },
            }
        )
    )
    output_path = tmp_path / "preflight.json"
    env = os.environ.copy()
    env.update(
        {
            "DS4_IMAGE_ID": "sha256:test-image",
            "DS4_HOST_INVOCATION": "container/run.sh preflight",
            "DS4_HOST_CONFIG_DIR": "/host/config",
            "DS4_HOST_HF_CACHE_DIR": "/host/hf-cache",
            "DS4_HOST_RESULTS_DIR": "/host/results",
            "DS4_HOST_RUNTIME_CACHE_DIR": "/host/runtime-cache",
            "DS4_HOST_SNAPSHOT_DIR": "/host/snapshot",
            "DS4_HOST_TICKET_01_DIR": "/host/ticket-01",
            "DS4_HOST_TICKET_02_DIR": "/host/ticket-02",
            "DS4_HOST_TOKENIZERS_DIR": "/host/tokenizers",
            "DS4_HOST_UV_CACHE_DIR": "/host/uv-cache",
            "DS4_VLLM_COMMIT": "a" * 40,
            "DS4_VLLM_DIRTY": "true",
            "PATH": str(tmp_path / "empty-bin"),
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "preflight",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 2
    provenance = json.loads(output_path.read_text())
    assert provenance["status"] == "invalid"
    assert provenance["hardware_validated"] is False
    assert provenance["image"]["id"] == "sha256:test-image"
    assert provenance["source"] == {"commit": "a" * 40, "dirty": True}
    assert provenance["checks"]["artifacts"]["status"] == "passed"
    assert provenance["checks"]["nvidia_smi"] == {
        "detail": "nvidia-smi is not installed or not on PATH",
        "status": "failed",
    }


def test_preflight_accepts_the_pinned_dual_3090_runtime(tmp_path: Path) -> None:
    artifact_path = tmp_path / "workload_plan.json"
    artifact_path.write_text('{"fixture": true}\n')
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    config_path = tmp_path / "contract.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base": {
                    "cuda": "13.0.2",
                    "python": "3.12",
                    "ubuntu": "22.04",
                },
                "dependencies": {"nixl": "1.3.0", "uv": "0.11.28"},
                "artifacts": [
                    {
                        "name": "workload_plan",
                        "path": str(artifact_path),
                        "sha256": hashlib.sha256(
                            artifact_path.read_bytes()
                        ).hexdigest(),
                    }
                ],
                "hardware": {
                    "gpu_count": 2,
                    "gpu_name": "NVIDIA GeForce RTX 3090",
                    "p2p_read": "CNS",
                    "topology": "SYS",
                },
                "results_dir": str(results_dir),
            }
        )
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == --query-gpu=* ]]; then\n"
        "  echo '0, NVIDIA GeForce RTX 3090, GPU-prefill, 580.159.03'\n"
        "  echo '1, NVIDIA GeForce RTX 3090, GPU-decode, 580.159.03'\n"
        "elif [[ $2 == -m ]]; then\n"
        "  echo 'GPU0 X SYS 0,2,4,6,8,10 0'\n"
        "  echo 'GPU1 SYS X 1,3,5,7,9,11 1'\n"
        "else\n"
        "  echo 'GPU0 X CNS'\n"
        "  echo 'GPU1 CNS X'\n"
        "fi\n"
    )
    numactl = fake_bin / "numactl"
    numactl.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'available: 2 nodes (0-1)'\n"
        "echo 'node 0 cpus: 0 2 4 6 8 10'\n"
        "echo 'node 1 cpus: 1 3 5 7 9 11'\n"
    )
    uv = fake_bin / "uv"
    uv.write_text("#!/usr/bin/env bash\necho 'uv 0.11.28'\n")
    for executable in (nvidia_smi, numactl, uv):
        executable.chmod(0o755)
    nixl_metadata = tmp_path / "nixl-1.3.0.dist-info"
    nixl_metadata.mkdir()
    (nixl_metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: nixl\nVersion: 1.3.0\n"
    )
    nixl_package = tmp_path / "nixl"
    nixl_package.mkdir()
    for module in ("__init__", "_api", "_bindings"):
        (nixl_package / f"{module}.py").write_text("")
    torch_package = tmp_path / "torch"
    torch_package.mkdir()
    (torch_package / "__init__.py").write_text(
        "class version:\n"
        "    cuda = '13.0'\n"
        "class accelerator:\n"
        "    @staticmethod\n"
        "    def is_available(): return True\n"
        "    @staticmethod\n"
        "    def device_count(): return 2\n"
    )
    output_path = results_dir / "preflight.json"
    env = os.environ.copy()
    env.update(
        {
            "DS4_DECODE_CPUSET": "1,3,5,7,9,11",
            "DS4_DECODE_GPU": "1",
            "DS4_HOST_INVOCATION": "container/run.sh preflight",
            "DS4_HOST_CONFIG_DIR": "/host/config",
            "DS4_HOST_HF_CACHE_DIR": "/host/hf-cache",
            "DS4_HOST_RESULTS_DIR": "/host/results",
            "DS4_HOST_RUNTIME_CACHE_DIR": "/host/runtime-cache",
            "DS4_HOST_SNAPSHOT_DIR": "/host/snapshot",
            "DS4_HOST_TICKET_01_DIR": "/host/ticket-01",
            "DS4_HOST_TICKET_02_DIR": "/host/ticket-02",
            "DS4_HOST_TOKENIZERS_DIR": "/host/tokenizers",
            "DS4_HOST_UV_CACHE_DIR": "/host/uv-cache",
            "DS4_IMAGE_ID": "sha256:test-image",
            "DS4_PREFILL_CPUSET": "0,2,4,6,8,10",
            "DS4_PREFILL_GPU": "0",
            "DS4_VLLM_COMMIT": "a" * 40,
            "DS4_VLLM_DIRTY": "false",
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PYTHONPATH": str(tmp_path),
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "preflight",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    provenance = json.loads(output_path.read_text())
    assert provenance["status"] == "ready"
    assert provenance["hardware_validated"] is False
    assert provenance["checks"]["results"]["status"] == "passed"
    assert provenance["checks"]["nixl"] == {
        "detail": {
            "modules": ["nixl", "nixl._api", "nixl._bindings"],
            "version": "1.3.0",
        },
        "status": "passed",
    }
    assert provenance["checks"]["cuda"] == {
        "detail": {"device_count": 2, "torch_cuda": "13.0"},
        "status": "passed",
    }
    assert provenance["checks"]["numactl"]["status"] == "passed"
    assert provenance["checks"]["numactl"]["detail"]["role_nodes"] == {
        "decode": 1,
        "prefill": 0,
    }
    assert provenance["installed_versions"]["nixl"] == "1.3.0"
    assert provenance["host_invocation"] == "container/run.sh preflight"
    assert provenance["build_inputs"] == {
        "base": {"cuda": "13.0.2", "python": "3.12", "ubuntu": "22.04"},
        "dependencies": {"nixl": "1.3.0", "uv": "0.11.28"},
        "model": None,
        "revisions": None,
    }
    assert provenance["mounts"] == {
        "config": {
            "container": "/mnt/ds4/config",
            "host": "/host/config",
            "mode": "ro",
        },
        "hf_cache": {
            "container": "/mnt/ds4/cache/huggingface",
            "host": "/host/hf-cache",
            "mode": "rw",
        },
        "raw": {"container": "/mnt/ds4/raw", "host": "/host/snapshot", "mode": "ro"},
        "results": {
            "container": "/mnt/ds4/results",
            "host": "/host/results",
            "mode": "rw",
        },
        "runtime_cache": {
            "container": "/mnt/ds4/cache/runtime",
            "host": "/host/runtime-cache",
            "mode": "rw",
        },
        "ticket_01": {
            "container": "/mnt/ds4/ticket-01",
            "host": "/host/ticket-01",
            "mode": "ro",
        },
        "ticket_02": {
            "container": "/mnt/ds4/ticket-02",
            "host": "/host/ticket-02",
            "mode": "ro",
        },
        "tokenizers": {
            "container": "/mnt/ds4/tokenizers",
            "host": "/host/tokenizers",
            "mode": "ro",
        },
        "uv_cache": {
            "container": "/mnt/ds4/cache/uv",
            "host": "/host/uv-cache",
            "mode": "rw",
        },
    }
    assert [device["uuid"] for device in provenance["hardware"]["devices"]] == [
        "GPU-prefill",
        "GPU-decode",
    ]
    assert provenance["roles"]["prefill"] == {
        "cpuset": "0,2,4,6,8,10",
        "gpu": "0",
    }
    assert provenance["roles"]["decode"] == {
        "cpuset": "1,3,5,7,9,11",
        "gpu": "1",
    }


def test_container_contract_pins_ticket_inputs_and_execution_revisions() -> None:
    contract = json.loads(
        (
            PROJECT_DIR / "benchmarks/ds4_profile/config/container-contract.json"
        ).read_text()
    )

    assert contract["base"] == {
        "cuda": "13.0.2",
        "python": "3.12",
        "ubuntu": "22.04",
    }
    assert contract["hardware"] == {
        "gpu_count": 2,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "p2p_read": "CNS",
        "topology": "SYS",
    }
    assert contract["dependencies"] == {"nixl": "1.3.0", "uv": "0.11.28"}
    assert contract["revisions"] == {
        "dataset": "4da61f3d06b48b6817a62b99e9c47035c8e59787",
        "ds4_tokenizer": "60d8d70770c6776ff598c94bb586a859a38244f1",
        "qwen_model": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
    }
    assert {
        artifact["name"]: artifact["sha256"] for artifact in contract["artifacts"]
    } == {
        "dataset_manifest": (
            "c228d3f670b20185a3bde4be08a67b73877179795f1b899ae4f6a23d64371c6a"
        ),
        "normalized_turns": (
            "9b4d82334e9fa429c7cc38ce56eacb59e219cd7c7ccf301a1468d7ec19e2f168"
        ),
        "ticket_02_provenance": (
            "3b1eecdae4a66ae75f47a0c151f8776f3bf9a4fc927019763a041454f2df165c"
        ),
        "rendered_turns": (
            "d47933e19f63ae1df7fc3fe04e228550b41d76e5ba93c09751a307f46ba33be7"
        ),
        "workload_plan": (
            "95cec2c58ba1d005cdf8c7253c0bea305c48a8dac40e49a7afdaf74e720352d8"
        ),
    }


def test_cpu_dry_run_plan_regenerates_both_ticket_artifact_layers(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "contract.json"
    config_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": "dataset_manifest",
                        "path": "/mnt/ds4/raw/manifest.json",
                        "sha256": "unused-by-plan",
                    }
                ],
                "schema_version": 1,
                "tokenizers": {
                    "ds4": "/mnt/ds4/tokenizers/ds4-pinned",
                    "qwen": "/mnt/ds4/tokenizers/qwen-pinned",
                },
            }
        )
    )
    output_dir = tmp_path / "cpu-dry-run"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "cpu-dry-run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(tmp_path / "snapshot/manifest.json"),
            "--ds4-tokenizer",
            str(tmp_path / "tokenizers/ds4"),
            "--qwen-tokenizer",
            str(tmp_path / "tokenizers/qwen"),
            "--print-plan",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "benchmarks.ds4_profile.normalize" in result.stdout
    assert "--require-complete-pilot" in result.stdout
    assert str(output_dir / "ticket-01") in result.stdout
    assert "benchmarks.ds4_profile.workloads" in result.stdout
    assert "--normalized-turns" in result.stdout
    assert str(output_dir / "ticket-01/turns.parquet") in result.stdout
    assert str(output_dir / "ticket-02") in result.stdout
    assert str(tmp_path / "snapshot/manifest.json") in result.stdout
    assert str(tmp_path / "tokenizers/ds4") in result.stdout
    assert str(tmp_path / "tokenizers/qwen") in result.stdout
    assert not output_dir.exists()


def test_gpu_smoke_records_skipped_when_hardware_preflight_fails(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "workload_plan.json"
    artifact_path.write_text('{"fixture": true}\n')
    config_path = tmp_path / "contract.json"
    config_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": "workload_plan",
                        "path": str(artifact_path),
                        "sha256": hashlib.sha256(
                            artifact_path.read_bytes()
                        ).hexdigest(),
                    }
                ],
                "hardware": {
                    "gpu_count": 2,
                    "gpu_name": "NVIDIA GeForce RTX 3090",
                    "p2p_read": "CNS",
                    "topology": "SYS",
                },
                "results_dir": str(tmp_path),
                "schema_version": 1,
            }
        )
    )
    output_path = tmp_path / "gpu-smoke.json"
    env = os.environ.copy()
    env["PATH"] = str(tmp_path / "empty-bin")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "gpu-smoke",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 2
    provenance = json.loads(output_path.read_text())
    assert provenance["status"] == "skipped"
    assert provenance["hardware_validated"] is False
    assert provenance["reason"] == "hardware preflight failed"
    assert provenance["preflight"]["status"] == "invalid"
    assert provenance["workers"] == []


def test_gpu_smoke_plan_binds_each_role_to_its_local_gpu_and_numa_node(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "contract.json"
    config_path.write_text(json.dumps({"schema_version": 1}))
    output_path = tmp_path / "gpu-smoke.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "gpu-smoke",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--print-plan",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("gpu-worker") == 2
    assert "--physcpubind=0,2,4,6,8,10" in result.stdout
    assert "--membind=0" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert "--role prefill" in result.stdout
    assert "--physcpubind=1,3,5,7,9,11" in result.stdout
    assert "--membind=1" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=1" in result.stdout
    assert "--role decode" in result.stdout
    assert not output_path.exists()


def test_gpu_worker_loads_a_pinned_exact_replay_before_model_start(
    tmp_path: Path,
) -> None:
    workload_path = tmp_path / "workload_plan.json"
    workload_path.write_text(
        json.dumps(
            {"exact_replays": [{"trajectory_id": "task:no_think", "turn_index": 3}]}
        )
    )
    rendered_path = tmp_path / "rendered_turns.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "execution_prompt_token_ids": [11, 22, 33],
                    "trajectory_id": "task:no_think",
                    "turn_index": 3,
                }
            ]
        ),
        rendered_path,
    )
    config_path = tmp_path / "contract.json"
    config_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "name": "workload_plan",
                        "path": str(workload_path),
                        "sha256": hashlib.sha256(
                            workload_path.read_bytes()
                        ).hexdigest(),
                    },
                    {
                        "name": "rendered_turns",
                        "path": str(rendered_path),
                        "sha256": hashlib.sha256(
                            rendered_path.read_bytes()
                        ).hexdigest(),
                    },
                ],
                "model": {
                    "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                },
                "schema_version": 1,
                "tokenizers": {"qwen": "/tokenizers/qwen-pinned"},
            }
        )
    )
    output_path = tmp_path / "worker.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "gpu-worker",
            "--config",
            str(config_path),
            "--role",
            "prefill",
            "--output",
            str(output_path),
            "--inspect-input",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    result_record = json.loads(output_path.read_text())
    assert result_record["status"] == "input-ready"
    assert result_record["hardware_validated"] is False
    assert result_record["role"] == "prefill"
    assert result_record["input"] == {
        "prompt_token_count": 3,
        "prompt_token_ids_sha256": (
            "3cf46ba3daf30bf336dbbc80c5f3fd4185bf9cc3747b7ce49d58335723d3a72c"
        ),
        "trajectory_id": "task:no_think",
        "turn_index": 3,
    }


def test_model_cache_plan_pins_the_qwen_revision(tmp_path: Path) -> None:
    config_path = tmp_path / "contract.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
                },
                "schema_version": 1,
            }
        )
    )
    env = os.environ.copy()
    env["HF_HOME"] = str(tmp_path / "hf-cache")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "cache-model",
            "--config",
            str(config_path),
            "--print-plan",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "hf download Qwen/Qwen2.5-Coder-7B-Instruct" in result.stdout
    assert "--revision c03e6d358207e414f1eca0bb1891e29f1db0e242" in result.stdout
    assert f"--cache-dir {tmp_path / 'hf-cache'}" in result.stdout
    assert not (tmp_path / "hf-cache").exists()


def test_profile_exec_records_the_exact_command_and_exit_status(tmp_path: Path) -> None:
    output_path = tmp_path / "invocation.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.container.runtime",
            "exec",
            "--output",
            str(output_path),
            "--",
            sys.executable,
            "-c",
            "print('profile command ran')",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "profile command ran" in result.stdout
    provenance = json.loads(output_path.read_text())
    assert provenance["status"] == "passed"
    assert provenance["returncode"] == 0
    assert provenance["command"] == [
        sys.executable,
        "-c",
        "print('profile command ran')",
    ]
    assert provenance["hardware_validated"] is False


def test_runbook_covers_server_lifecycle_and_unvalidated_apptainer_fallback() -> None:
    runbook = (CONTAINER_DIR / "README.md").read_text()

    for required_text in (
        "Docker Engine",
        "NVIDIA Container Toolkit",
        "package manager",
        "build.sh",
        "cache-model",
        "preflight",
        "cpu-dry-run",
        "gpu-smoke",
        "offline",
        "GPU0",
        "GPU1",
        "restart",
        "CUDA",
        "NIXL",
        "UCX",
        "NUMA",
        "docker save",
        "docker-archive:",
        "not hardware-validated",
    ):
        assert required_text in runbook
    assert "curl | sh" not in runbook
    assert "git clone" not in runbook
    assert "vllm-project/vllm" not in runbook
