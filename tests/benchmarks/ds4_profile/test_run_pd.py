# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from benchmarks.ds4_profile import run_pd

REVISION = "0123456789abcdef0123456789abcdef01234567"
TOKENIZER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"


def _runtime_environment():
    return {
        "CUDA_HOME": "/opt/cuda",
        "FLASHINFER_JIT_VERBOSE": "0",
        "HF_HOME": "/models",
        "HF_HUB_CACHE": "/models/hub",
        "HF_HUB_OFFLINE": "1",
        "LD_LIBRARY_PATH": "/opt/cuda/lib",
        "LIBRARY_PATH": "/run/cuda-compat:/opt/cuda/lib",
        "NO_PROXY": "127.0.0.1,localhost",
        "PATH": "/opt/cuda/bin:/usr/bin",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
        "no_proxy": "127.0.0.1,localhost",
    }


def _config(tmp_path):
    return run_pd.LaunchConfig(
        model_revision=REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        attention_backend="FLASH_ATTN",
        prefill_cpus="0-7",
        prefill_numa_node=0,
        decode_cpus="8-15",
        decode_numa_node=1,
        run_dir=tmp_path / "run",
        repo_root=tmp_path,
        vllm_commit="a" * 40,
        vllm_dirty=False,
        runtime_environment=_runtime_environment(),
    )


def _launcher_args(tmp_path, *, dry_run=False):
    args = [
        "--model-revision",
        REVISION,
        "--tokenizer-revision",
        TOKENIZER_REVISION,
        "--attention-backend",
        "FLASH_ATTN",
        "--prefill-cpus",
        "0-7",
        "--prefill-numa-node",
        "0",
        "--decode-cpus",
        "8-15",
        "--decode-numa-node",
        "1",
        "--run-dir",
        str(tmp_path / "run"),
    ]
    if dry_run:
        args.append("--dry-run")
    return args


def _run_dry_plan(tmp_path, *, extra_environment=None):
    command = [
        sys.executable,
        "-m",
        "benchmarks.ds4_profile.run_pd",
        *_launcher_args(tmp_path, dry_run=True),
    ]
    environment = {
        **os.environ,
        **_runtime_environment(),
        **(extra_environment or {}),
    }
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(result.stdout)


def test_build_plan_freezes_fixed_topology(tmp_path):
    plan = run_pd.build_plan(_config(tmp_path)).as_dict()

    assert plan["model"] == "Qwen/Qwen3.5-4B"
    assert plan["model_revision"] == REVISION
    assert plan["tokenizer_revision"] == TOKENIZER_REVISION
    assert plan["ports"] == {
        "prefill": 8100,
        "decode": 8200,
        "proxy": 8000,
        "prefill_side_channel": 5600,
        "decode_side_channel": 5601,
    }
    assert [process["name"] for process in plan["processes"]] == [
        "prefill",
        "decode",
        "proxy",
    ]
    for process, cpus, node in (
        (plan["processes"][0], "0-7", "0"),
        (plan["processes"][1], "8-15", "1"),
    ):
        assert process["command"][:3] == [
            "numactl",
            f"--physcpubind={cpus}",
            f"--membind={node}",
        ]


def test_build_plan_freezes_fail_closed_server_configuration(tmp_path):
    prefill, decode, _ = run_pd.build_plan(_config(tmp_path)).as_dict()["processes"]

    for process, role in (
        (prefill, "kv_producer"),
        (decode, "kv_consumer"),
    ):
        command = process["command"]
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert "--language-model-only" in command
        assert "--enable-prefix-caching" in command
        assert "--enable-chunked-prefill" in command
        assert command[command.index("--mamba-cache-mode") + 1] == "align"
        transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
        assert transfer == {
            "kv_connector": "NixlConnector",
            "kv_load_failure_policy": "fail",
            "kv_role": role,
        }


def test_build_plan_freezes_proxy_routing(tmp_path):
    proxy = run_pd.build_plan(_config(tmp_path)).as_dict()["processes"][2]

    assert proxy["command"][:3] == [
        str(tmp_path / ".venv/bin/python"),
        "-m",
        "benchmarks.ds4_profile.pd_proxy",
    ]
    assert proxy["command"][proxy["command"].index("--prefill-url") + 1] == (
        "http://127.0.0.1:8100"
    )
    assert proxy["command"][proxy["command"].index("--decode-url") + 1] == (
        "http://127.0.0.1:8200"
    )
    assert proxy["command"][proxy["command"].index("--request-timeout") + 1] == (
        "300.0"
    )


def test_build_plan_freezes_smoke_contract(tmp_path):
    plan = run_pd.build_plan(_config(tmp_path)).as_dict()

    assert plan["smoke_contract"] == {
        "expected_prompt_tokens": 642,
        "effective_hma_page_tokens": 640,
        "expected_remote_tokens": 641,
    }
    assert plan["smoke_request"] == {
        "model": "Qwen/Qwen3.5-4B",
        "prompt": "Explain deterministic cache transfer in one sentence. " * 80,
        "max_tokens": 16,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
    }


def test_build_plan_freezes_complete_child_environments(tmp_path):
    plan = run_pd.build_plan(_config(tmp_path)).as_dict()
    prefill, decode, proxy = plan["processes"]
    runtime_environment = _runtime_environment()

    assert prefill["environment"] == {
        **runtime_environment,
        "CUDA_VISIBLE_DEVICES": "0",
        "UCX_NET_DEVICES": "all",
        "VLLM_KV_CACHE_LAYOUT": "HND",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "5600",
        "VLLM_SERVER_DEV_MODE": "1",
    }
    assert decode["environment"] == {
        **runtime_environment,
        "CUDA_VISIBLE_DEVICES": "1",
        "UCX_NET_DEVICES": "all",
        "VLLM_KV_CACHE_LAYOUT": "HND",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "5601",
        "VLLM_SERVER_DEV_MODE": "1",
    }
    assert proxy["environment"] == runtime_environment


def test_build_plan_rejects_unpinned_revision_before_execution(tmp_path):
    config = replace(_config(tmp_path), model_revision="main")

    try:
        run_pd.build_plan(config)
    except ValueError as error:
        assert "model revision must be a full 40-character lowercase SHA" in str(error)
    else:
        raise AssertionError("expected an unpinned revision to be rejected")


def test_build_plan_rejects_incompatible_fixed_runtime_environment(tmp_path):
    runtime_environment = _runtime_environment()
    runtime_environment["VLLM_SSM_CONV_STATE_LAYOUT"] = "SD"
    config = replace(_config(tmp_path), runtime_environment=runtime_environment)

    try:
        run_pd.build_plan(config)
    except ValueError as error:
        assert str(error) == (
            "VLLM_SSM_CONV_STATE_LAYOUT must be 'DS' for the fixed runtime, got 'SD'"
        )
    else:
        raise AssertionError("expected an incompatible runtime value to be rejected")


def test_dry_run_cli_prints_plan_without_starting_processes(tmp_path):
    plan = _run_dry_plan(tmp_path)

    assert isinstance(plan["vllm_dirty"], bool)
    assert plan["processes"][0]["name"] == "prefill"
    assert not (tmp_path / "run").exists()


def test_dry_run_cli_excludes_unrelated_ambient_environment(tmp_path):
    plan = _run_dry_plan(
        tmp_path,
        extra_environment={"AWS_SECRET_ACCESS_KEY": "must-not-be-inherited"},
    )

    for process in plan["processes"]:
        assert _runtime_environment().items() <= process["environment"].items()
        assert "AWS_SECRET_ACCESS_KEY" not in process["environment"]


def test_cli_rejects_missing_runtime_environment_before_execution(tmp_path):
    command = [
        sys.executable,
        "-m",
        "benchmarks.ds4_profile.run_pd",
        *_launcher_args(tmp_path, dry_run=True),
    ]
    environment = {**os.environ, **_runtime_environment()}
    del environment["HF_HOME"]

    result = subprocess.run(command, capture_output=True, text=True, env=environment)

    assert result.returncode != 0
    assert "ValueError: missing required runtime environment: HF_HOME" in result.stderr
    assert not (tmp_path / "run").exists()


class FakeRuntime:
    def __init__(self, fail_request=False, token_count=642):
        self.events = []
        self.fail_request = fail_request
        self.token_count = token_count

    def start(self, process):
        handle = f"handle:{process.name}"
        self.events.append(("start", process.name))
        return handle

    def wait_ready(self, name, url, timeout, handles):
        self.events.append(("ready", name, url, tuple(handles)))

    def get_text(self, url, timeout):
        self.events.append(("get", url))
        return "# deterministic fake metrics\n"

    def post_json(self, url, payload, timeout):
        if url.endswith("/tokenize"):
            self.events.append(("tokenize", url, payload))
            return {"count": self.token_count}
        self.events.append(("post", url, payload["seed"]))
        if self.fail_request:
            raise TimeoutError("injected request timeout")
        return {"choices": [{"text": "same deterministic output"}]}

    def stop(self, handles, timeout):
        self.events.append(("stop", tuple(handles), timeout))


def _start_sleeping_process(tmp_path, *, ignore_sigterm=False):
    signal_setup = (
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);" if ignore_sigterm else ""
    )
    script = (
        f"import os,signal,time;{signal_setup}"
        "print(os.getpid(), flush=True);time.sleep(60)"
    )
    log_path = tmp_path / "sleeper.log"
    runtime = run_pd.SubprocessRuntime()
    child = runtime.start(
        run_pd.ProcessSpec(
            "sleeper",
            (sys.executable, "-c", script),
            {},
            log_path,
        )
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        text = log_path.read_text().strip()
        if text:
            return runtime, child, int(text), log_path
        time.sleep(0.01)
    runtime.stop([child], timeout=1)
    raise AssertionError("sleeper did not become ready")


def _assert_process_gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise AssertionError(f"process {pid} is still running")


def _assert_log_closed(log_path):
    for fd_path in Path("/proc/self/fd").iterdir():
        try:
            if fd_path.resolve() == log_path.resolve():
                raise AssertionError(f"{log_path} is still open as {fd_path}")
        except FileNotFoundError:
            continue


def test_subprocess_runtime_uses_the_serialized_environment_only(tmp_path, monkeypatch):
    process = replace(
        run_pd.build_plan(_config(tmp_path)).processes[0],
        command=("/usr/bin/env", "-0"),
        log_path=tmp_path / "environment.log",
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-be-inherited")
    runtime = run_pd.SubprocessRuntime()
    child = runtime.start(process)
    try:
        assert child.process.wait(timeout=2) == 0
    finally:
        runtime.stop([child], timeout=1)

    entries = process.log_path.read_bytes().split(b"\0")
    child_environment = dict(entry.decode().split("=", 1) for entry in entries if entry)
    assert child_environment == process.as_dict()["environment"]
    assert "AWS_SECRET_ACCESS_KEY" not in child_environment


def test_execute_plan_runs_cold_repeat_and_cleans_up(tmp_path):
    runtime = FakeRuntime()

    result = run_pd.execute_plan(run_pd.build_plan(_config(tmp_path)), runtime)

    assert result["outputs_identical"] is True
    assert runtime.events[:6] == [
        ("start", "prefill"),
        ("start", "decode"),
        (
            "ready",
            "prefill",
            "http://127.0.0.1:8100/v1/models",
            ("handle:prefill", "handle:decode"),
        ),
        (
            "ready",
            "decode",
            "http://127.0.0.1:8200/v1/models",
            ("handle:prefill", "handle:decode"),
        ),
        (
            "tokenize",
            "http://127.0.0.1:8100/tokenize",
            {
                "model": "Qwen/Qwen3.5-4B",
                "prompt": "Explain deterministic cache transfer in one sentence. " * 80,
            },
        ),
        ("start", "proxy"),
    ]
    posts = [event for event in runtime.events if event[0] == "post"]
    assert posts == [
        ("post", "http://127.0.0.1:8000/v1/completions", 0),
        ("post", "http://127.0.0.1:8000/v1/completions", 0),
    ]
    assert runtime.events[-1] == (
        "stop",
        ("handle:prefill", "handle:decode", "handle:proxy"),
        30.0,
    )
    assert (
        json.loads((tmp_path / "run/smoke-result.json").read_text())["gate_a"]
        == "pending_metric_review"
    )


def test_execute_plan_rejects_unexpected_smoke_prompt_token_count(tmp_path):
    runtime = FakeRuntime(token_count=641)

    try:
        run_pd.execute_plan(run_pd.build_plan(_config(tmp_path)), runtime)
    except RuntimeError as error:
        assert str(error) == "smoke prompt produced 641 tokens, expected 642"
    else:
        raise AssertionError("expected the unsafe smoke prompt to be rejected")

    assert [event for event in runtime.events if event[0] == "post"] == []
    assert runtime.events[-1] == (
        "stop",
        ("handle:prefill", "handle:decode"),
        30.0,
    )


def test_execute_plan_cleans_up_all_started_processes_on_request_failure(tmp_path):
    runtime = FakeRuntime(fail_request=True)

    try:
        run_pd.execute_plan(run_pd.build_plan(_config(tmp_path)), runtime)
    except TimeoutError as error:
        assert str(error) == "injected request timeout"
    else:
        raise AssertionError("expected the injected request failure")

    assert runtime.events[-1] == (
        "stop",
        ("handle:prefill", "handle:decode", "handle:proxy"),
        30.0,
    )
    assert (tmp_path / "run/server/p-metrics-before.txt").exists()
    assert (tmp_path / "run/server/d-metrics-before.txt").exists()
    assert not (tmp_path / "run/cold-response.json").exists()


def test_execute_plan_rejects_stale_launcher_artifacts_before_start(tmp_path):
    config = _config(tmp_path)
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "cold-response.json").write_text("{}\n")
    runtime = FakeRuntime()

    try:
        run_pd.execute_plan(run_pd.build_plan(config), runtime)
    except ValueError as error:
        assert "already contains launcher artifacts" in str(error)
    else:
        raise AssertionError("expected stale launcher artifacts to be rejected")

    assert runtime.events == []


def test_subprocess_runtime_terminates_real_process_group_and_closes_log(tmp_path):
    runtime, child, pid, log_path = _start_sleeping_process(tmp_path)

    runtime.stop([child], timeout=1.0)

    _assert_process_gone(pid)
    _assert_log_closed(log_path)


def test_subprocess_runtime_kills_process_group_that_ignores_sigterm(tmp_path):
    runtime, child, pid, log_path = _start_sleeping_process(
        tmp_path, ignore_sigterm=True
    )

    runtime.stop([child], timeout=0.4)

    _assert_process_gone(pid)
    _assert_log_closed(log_path)


def test_subprocess_runtime_reports_survivor_and_closes_log(tmp_path, monkeypatch):
    runtime, child, pid, log_path = _start_sleeping_process(
        tmp_path, ignore_sigterm=True
    )
    real_killpg = os.killpg

    def suppress_sigkill(process_group, sig):
        if sig != signal.SIGKILL:
            real_killpg(process_group, sig)

    monkeypatch.setattr(os, "killpg", suppress_sigkill)
    try:
        try:
            runtime.stop([child], timeout=0.2)
        except TimeoutError as error:
            assert "sleeper" in str(error)
        else:
            raise AssertionError("expected a surviving process group to be reported")
        _assert_log_closed(log_path)
    finally:
        real_killpg(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
