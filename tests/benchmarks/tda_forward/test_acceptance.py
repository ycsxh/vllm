# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import importlib
import json
import sys
import types
from dataclasses import asdict, replace
from typing import Any, cast

import pytest

from benchmarks.tda_forward import metrics, report, runner, schema, workload


def test_fixed_matrix_has_three_seeds_and_matched_controls():
    config = schema.load_config(schema.DEFAULT_CONFIG_PATH)

    assert len(config.seeds) >= 3
    pairs = {(run.seed, run.mode) for run in schema.build_run_matrix(config)}
    for seed in config.seeds:
        assert (seed, "oracle") in pairs
        assert (seed, "disabled") in pairs
        assert (seed, "enabled") in pairs


def test_config_fingerprint_ignores_only_event_plane_switches():
    config = schema.load_config(schema.DEFAULT_CONFIG_PATH)
    disabled, enabled = schema.comparison_pair(config, config.seeds[0])

    assert schema.comparison_fingerprint(disabled) == schema.comparison_fingerprint(
        enabled
    )
    changed = replace(enabled, max_num_batched_tokens=2048)
    assert schema.comparison_fingerprint(disabled) != schema.comparison_fingerprint(
        changed
    )


def test_launch_plan_uses_native_nixl_and_event_controls(tmp_path, monkeypatch):
    config = schema.load_config(schema.DEFAULT_CONFIG_PATH)
    for name, value in {
        "TDA_MODEL_REVISION": "a" * 40,
        "TDA_TOKENIZER_REVISION": "b" * 40,
        "TDA_ATTENTION_BACKEND": "FLASH_ATTN",
        "TDA_PREFILL_CPUS": "0-7",
        "TDA_PREFILL_NUMA_NODE": "0",
        "TDA_DECODE_CPUS": "8-15",
        "TDA_DECODE_NUMA_NODE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    disabled, enabled = schema.comparison_pair(config, config.seeds[0])
    stalled = schema.build_run_matrix(config)[-1]

    disabled_specs = runner._process_specs(tmp_path, config, disabled, tmp_path / "d")
    enabled_specs = runner._process_specs(tmp_path, config, enabled, tmp_path / "e")
    stalled_specs = runner._process_specs(tmp_path, config, stalled, tmp_path / "s")

    for specs in (disabled_specs, enabled_specs):
        assert [spec.name for spec in specs] == ["prefill", "decode", "proxy"]
        for spec, role in zip(specs[:2], ("kv_producer", "kv_consumer"), strict=True):
            transfer = json.loads(
                spec.command[spec.command.index("--kv-transfer-config") + 1]
            )
            assert transfer["kv_connector"] == "NixlConnector"
            assert transfer["kv_role"] == role
    assert "--kv-events-config" not in disabled_specs[1].command
    assert "--disable-events" in disabled_specs[2].command
    assert "--kv-events-config" in enabled_specs[1].command
    assert "--event-endpoint" in enabled_specs[2].command
    assert (
        stalled_specs[1].environment["TDA_ACCEPTANCE_PUBLISHER_DRAIN_DELAY_SECONDS"]
        == "2.0"
    )
    assert "--event-consumer-delay" not in stalled_specs[2].command
    hwm_index = stalled_specs[2].command.index("--event-subscriber-hwm")
    assert stalled_specs[2].command[hwm_index + 1] == "100000"


def test_workload_is_deterministic_and_covers_required_scenarios():
    config = schema.load_config(schema.DEFAULT_CONFIG_PATH)

    first = workload.build_workload(config.workload, config.seeds[0])
    second = workload.build_workload(config.workload, config.seeds[0])

    assert first == second
    assert {
        "full_reuse",
        "partial_reuse",
        "zero_reuse",
        "evict_d",
        "capacity_recovery",
        "shared_prefix",
    }.issubset({turn.scenario for session in first for turn in session.turns})


def test_prometheus_parser_preserves_names_labels_and_values():
    samples = metrics.parse_prometheus(
        "# HELP vllm:num_requests_waiting waiting\n"
        'vllm:num_requests_waiting{model_name="qwen",engine="0"} 3\n'
    )

    assert (
        metrics.metric_value(
            samples,
            "vllm:num_requests_waiting",
            labels={"model_name": "qwen"},
        )
        == 3
    )
    assert metrics.distribution([1, 2, 3, 100])["p95"] == 100


@pytest.mark.asyncio
async def test_driver_serializes_each_session_and_overlaps_sessions():
    config = schema.load_config(schema.DEFAULT_CONFIG_PATH)
    sessions = workload.build_workload(config.workload, config.seeds[0])[:4]
    active: set[str] = set()
    overlap_seen = False

    async def send(turn):
        nonlocal overlap_seen
        assert turn.session_id not in active
        active.add(turn.session_id)
        overlap_seen |= len(active) > 1
        await asyncio.sleep(0.001)
        active.remove(turn.session_id)
        return workload.TurnResult.success(turn, ttft_ms=1.0, tpot_ms=2.0)

    results = await workload.run_sessions(sessions, send)

    assert overlap_seen
    assert len(results) == sum(len(session.turns) for session in sessions)


def test_report_thresholds_use_medians_and_invalidate_bad_runs(tmp_path):
    disabled = [
        _run(seed, throughput=100, ttft=100, tpot=10, scheduler=5)
        for seed in (17, 29, 43)
    ]
    enabled = [
        _run(seed, throughput=96, ttft=104, tpot=10.4, scheduler=5.2)
        for seed in (17, 29, 43)
    ]
    oracle = [
        _run(seed, throughput=1, ttft=1, tpot=1, scheduler=1) for seed in (17, 29, 43)
    ]

    verdict = report.evaluate_comparison(disabled, enabled, oracle)
    assert verdict.status == "pass"

    enabled[0]["integrity"]["sequence_gaps"] = 1
    verdict = report.evaluate_comparison(disabled, enabled, oracle)
    assert verdict.status == "invalid"
    assert "sequence gap" in " ".join(verdict.reasons)

    enabled[0]["integrity"]["sequence_gaps"] = 0
    enabled.append(enabled[0])
    assert report.evaluate_comparison(disabled, enabled, oracle).status == "invalid"

    enabled.pop()
    enabled[0]["output_hashes"] = {"request": "corrupt"}
    verdict = report.evaluate_comparison(disabled, enabled, oracle)
    assert verdict.status == "fail"
    assert "output digests" in " ".join(verdict.reasons)

    evidence = tmp_path / "run"
    evidence.mkdir()
    (evidence / "run-summary.json").write_text(json.dumps(enabled[0]), encoding="utf-8")
    loaded = report.load_run_summary(evidence)
    assert loaded["seed"] == 17


def test_report_never_promotes_missing_portable_evidence_to_pass(tmp_path):
    summary_path, report_path = report.build_report(
        tmp_path,
        schema.DEFAULT_CONFIG_PATH,
    )

    assert json.loads(summary_path.read_text())["overall"] == "incomplete"
    assert "Portable results cannot establish" in report_path.read_text()


def test_runner_does_not_accept_an_external_config_override():
    with pytest.raises(SystemExit):
        runner._parser().parse_args(["--config", "/tmp/relaxed.json", "plan"])


def test_stalled_consumer_requires_liveness_loss_visibility_and_nonblocking_enqueue():
    run: dict[str, Any] = {
        "status": "valid",
        "invalid_reasons": [],
        "conclusion_scope": "liveness_only",
        "serving": {"completed_requests": 10, "failed_requests": 0},
        "integrity": {
            "publisher_overflow": 2,
            "first_publisher_overflow_at": 1.0,
            "sequence_gaps": 1,
            "first_sequence_gap_at": 2.0,
            "malformed_events": 0,
            "proxy_restarts": 0,
            "decode_restarts": 0,
        },
        "event_plane": {"enqueue_ms": {"summary": {"p99": 1.0}}},
        "timed_out": False,
        "output_hashes": {"request": "digest"},
    }

    assert report.evaluate_stalled(run, baseline=run).status == "pass"
    run["integrity"]["first_sequence_gap_at"] = 0.5
    assert report.evaluate_stalled(run, baseline=run).status == "fail"
    run["integrity"]["first_sequence_gap_at"] = 2.0
    run["event_plane"]["enqueue_ms"]["summary"]["p99"] = 200.0
    assert report.evaluate_stalled(run).status == "fail"

    run["event_plane"]["enqueue_ms"]["summary"]["p99"] = 1.0
    run["status"] = "invalid"
    run["invalid_reasons"] = ["proxy failed"]
    assert report.evaluate_stalled(run).status == "invalid"


def test_report_binds_summary_and_manifest_to_expected_run():
    config = schema.load_config(schema.DEFAULT_CONFIG_PATH)
    expected = schema.build_run_matrix(config)[0]
    summary = {
        "run_id": expected.run_id,
        "mode": expected.mode,
        "seed": expected.seed,
        "conclusion_scope": expected.conclusion_scope,
        "comparison_fingerprint": schema.comparison_fingerprint(expected),
    }
    manifest = {
        "run": asdict(expected),
        "comparison_fingerprint": schema.comparison_fingerprint(expected),
    }

    assert not report.run_identity_reasons(summary, manifest, expected)
    summary["mode"] = "healthy"
    assert report.run_identity_reasons(summary, manifest, expected)


def test_missing_or_malformed_publisher_stats_fail_closed():
    with pytest.raises(ValueError, match="exactly one"):
        runner._publisher_stats({})
    with pytest.raises(ValueError, match="unavailable"):
        runner._publisher_stats(
            {"publishers": [{"stats_error": "boom"}], "publisher_samples": [{}]}
        )


def test_instrumentation_installs_when_spawn_imports_main_module(tmp_path, monkeypatch):
    class FakeScheduler:
        def schedule(self):
            return None

        def shutdown(self):
            return None

    class FakePublisher:
        def __init__(self):
            pass

        def publish(self, events):
            return events

        def observe_event_construction_time(self, duration):
            return duration

    scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")
    cast(Any, scheduler_module).Scheduler = FakeScheduler
    event_module = types.ModuleType("vllm.distributed.kv_events")
    cast(Any, event_module).ZmqEventPublisher = FakePublisher
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    monkeypatch.setitem(sys.modules, event_module.__name__, event_module)
    monkeypatch.setenv("TDA_ACCEPTANCE_STATS_PATH", str(tmp_path / "stats.json"))

    from benchmarks.tda_forward import instrumented_server

    importlib.reload(instrumented_server)

    assert instrumented_server._INSTALLED
    assert FakeScheduler.schedule.__name__ == "schedule"


def _run(seed, *, throughput, ttft, tpot, scheduler):
    return {
        "seed": seed,
        "comparison_fingerprint": "same",
        "status": "valid",
        "output_hashes": {f"session:{seed}": f"digest-{seed}"},
        "serving": {
            "request_throughput": throughput,
            "ttft_p95_ms": ttft,
            "tpot_p95_ms": tpot,
            "scheduler_step_p95_ms": scheduler,
        },
        "integrity": {
            "publisher_overflow": 0,
            "sequence_gaps": 0,
            "malformed_events": 0,
            "proxy_restarts": 0,
            "decode_restarts": 0,
        },
    }
