# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
from dataclasses import replace

from benchmarks.ds4_profile import run_pd

REVISION = "0123456789abcdef0123456789abcdef01234567"
TOKENIZER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"


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
    )


def test_build_plan_freezes_fixed_fail_closed_topology(tmp_path):
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

    prefill, decode, proxy = plan["processes"]
    for process, gpu, cpus, node, role in (
        (prefill, "0", "0-7", "0", "kv_producer"),
        (decode, "1", "8-15", "1", "kv_consumer"),
    ):
        command = process["command"]
        assert command[:3] == [
            "numactl",
            f"--physcpubind={cpus}",
            f"--membind={node}",
        ]
        assert process["environment"]["CUDA_VISIBLE_DEVICES"] == gpu
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

    assert proxy["command"][:3] == [
        str(tmp_path / ".venv/bin/python"),
        str(
            tmp_path
            / "examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py"
        ),
        "--model",
    ]


def test_build_plan_rejects_unpinned_revision_before_execution(tmp_path):
    config = replace(_config(tmp_path), model_revision="main")

    try:
        run_pd.build_plan(config)
    except ValueError as error:
        assert "model revision must be a full 40-character lowercase SHA" in str(error)
    else:
        raise AssertionError("expected an unpinned revision to be rejected")


def test_dry_run_cli_prints_plan_without_starting_processes(tmp_path):
    command = [
        sys.executable,
        "-m",
        "benchmarks.ds4_profile.run_pd",
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
        "--dry-run",
    ]

    result = subprocess.run(command, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)

    assert isinstance(plan["vllm_dirty"], bool)
    assert plan["processes"][0]["name"] == "prefill"
    assert not (tmp_path / "run").exists()


class FakeRuntime:
    def __init__(self, fail_request=False):
        self.events = []
        self.fail_request = fail_request

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
        self.events.append(("post", url, payload["seed"]))
        if self.fail_request:
            raise TimeoutError("injected request timeout")
        return {"choices": [{"text": "same deterministic output"}]}

    def stop(self, handles, timeout):
        self.events.append(("stop", tuple(handles), timeout))


def test_execute_plan_runs_cold_repeat_and_cleans_up(tmp_path):
    runtime = FakeRuntime()

    result = run_pd.execute_plan(run_pd.build_plan(_config(tmp_path)), runtime)

    assert result["outputs_identical"] is True
    assert runtime.events[:5] == [
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
