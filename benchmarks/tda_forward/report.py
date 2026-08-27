# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fail-closed Issue #18 evidence evaluation and report generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.tda_forward.metrics import distribution
from benchmarks.tda_forward.schema import (
    DEFAULT_CONFIG_PATH,
    RUN_SOURCE_ARTIFACTS,
    AcceptanceConfig,
    RunSpec,
    build_run_matrix,
    comparison_fingerprint,
    load_config,
)


@dataclass(frozen=True)
class Verdict:
    status: str
    reasons: tuple[str, ...]
    ratios: dict[str, float]


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    """Load one run only after its immutable summary exists."""
    path = run_dir / "run-summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"run summary must be an object: {path}")
    return value


def invalid_reasons(run: dict[str, Any]) -> list[str]:
    """Return controlled-run invalidators mandated by Issue #18."""
    reasons = list(run.get("invalid_reasons", []))
    if run.get("status") != "valid":
        reasons.append(f"run status is {run.get('status', 'missing')}")
    integrity = run.get("integrity", {})
    if run.get("serving", {}).get("failed_requests", 0):
        reasons.append("request failure")
    checks = (
        ("publisher_overflow", "publisher overflow"),
        ("sequence_gaps", "sequence gap"),
        ("malformed_events", "malformed event"),
        ("proxy_restarts", "Proxy restart"),
        ("decode_restarts", "Decode restart"),
    )
    for field, label in checks:
        if integrity.get(field, 0):
            reasons.append(label)
    return sorted(set(reasons))


def _median(runs: list[dict[str, Any]], name: str) -> float:
    try:
        value = statistics.median(float(run["serving"][name]) for run in runs)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing serving metric {name}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"serving metric {name} must be finite and positive")
    return value


def evaluate_comparison(
    disabled: list[dict[str, Any]],
    enabled: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    config: AcceptanceConfig | None = None,
) -> Verdict:
    """Evaluate fixed-seed medians after rejecting every invalid input run."""
    config = config or load_config()
    expected = set(config.seeds)
    if (
        len(disabled) != len(expected)
        or len(enabled) != len(expected)
        or len(oracle) != len(expected)
        or {run.get("seed") for run in disabled} != expected
        or {run.get("seed") for run in enabled} != expected
        or {run.get("seed") for run in oracle} != expected
    ):
        return Verdict("invalid", ("fixed seed set is incomplete or mismatched",), {})
    for seed in expected:
        fingerprints = {
            run.get("comparison_fingerprint")
            for run in (*oracle, *disabled, *enabled)
            if run.get("seed") == seed
        }
        if len(fingerprints) != 1 or None in fingerprints:
            return Verdict(
                "invalid",
                (f"seed {seed}: enabled/disabled configuration mismatch",),
                {},
            )
    reasons = [
        f"seed {run.get('seed')}: {reason}"
        for run in (*oracle, *disabled, *enabled)
        for reason in invalid_reasons(run)
    ]
    if reasons:
        return Verdict("invalid", tuple(sorted(set(reasons))), {})

    disabled_by_seed = {run["seed"]: run for run in disabled}
    enabled_by_seed = {run["seed"]: run for run in enabled}
    oracle_by_seed = {run["seed"]: run for run in oracle}
    output_mismatches = [
        f"seed {seed}: concurrent output digests differ from sequential oracle"
        for seed in expected
        if any(
            run.get("output_hashes") != oracle_by_seed[seed].get("output_hashes")
            for run in (disabled_by_seed[seed], enabled_by_seed[seed])
        )
    ]

    try:
        throughput_ratio = _median(enabled, "request_throughput") / _median(
            disabled, "request_throughput"
        )
        ratios = {"request_throughput": throughput_ratio}
        failed = list(output_mismatches)
        if throughput_ratio < config.thresholds.minimum_throughput_ratio:
            failed.append("enabled median throughput is below 95% of disabled")
        for name, label in (
            ("ttft_p95_ms", "p95 TTFT"),
            ("tpot_p95_ms", "p95 TPOT"),
            ("scheduler_step_p95_ms", "p95 Scheduler-step wall time"),
        ):
            baseline = _median(disabled, name)
            ratio = _median(enabled, name) / baseline
            ratios[name] = ratio
            if ratio > config.thresholds.maximum_latency_ratio:
                failed.append(f"enabled {label} increased by more than 5%")
    except ValueError as error:
        return Verdict("invalid", (str(error),), {})
    return Verdict("fail" if failed else "pass", tuple(failed), ratios)


def evaluate_stalled(
    run: dict[str, Any],
    config: AcceptanceConfig | None = None,
    baseline: dict[str, Any] | None = None,
) -> Verdict:
    """Evaluate only liveness and visible loss for the saturation drive."""
    config = config or load_config()
    invalid = []
    if run.get("status") != "valid":
        invalid.append(f"run status is {run.get('status', 'missing')}")
    invalid.extend(str(reason) for reason in run.get("invalid_reasons", []))
    integrity = run.get("integrity", {})
    serving = run.get("serving", {})
    if serving.get("failed_requests", 0):
        invalid.append("request failure during stalled-consumer drive")
    for field, label in (
        ("malformed_events", "malformed event"),
        ("proxy_restarts", "Proxy restart"),
        ("decode_restarts", "Decode restart"),
    ):
        if integrity.get(field, 0):
            invalid.append(label)
    if invalid:
        return Verdict("invalid", tuple(sorted(set(invalid))), {})

    reasons = []
    if run.get("conclusion_scope") != "liveness_only":
        reasons.append("stalled run is not marked liveness-only")
    if serving.get("completed_requests", 0) <= 0:
        reasons.append("no request completed during stalled-consumer drive")
    if integrity.get("publisher_overflow", 0) <= 0:
        reasons.append("publisher queue overflow was not observed")
    if integrity.get("sequence_gaps", 0) <= 0:
        reasons.append("consumer saturation produced no visible drop or source gap")
    overflow_at = integrity.get("first_publisher_overflow_at")
    gap_at = integrity.get("first_sequence_gap_at")
    if not (
        isinstance(overflow_at, (int, float))
        and isinstance(gap_at, (int, float))
        and overflow_at < gap_at
    ):
        reasons.append("a source gap was not observed after publisher overflow")
    if run.get("timed_out"):
        reasons.append("stalled-consumer drive exceeded its timeout")
    enqueue_p99 = (
        run.get("event_plane", {}).get("enqueue_ms", {}).get("summary", {}).get("p99")
    )
    if not isinstance(enqueue_p99, (int, float)):
        reasons.append("stalled-consumer enqueue p99 is missing")
    elif enqueue_p99 > config.thresholds.maximum_stalled_enqueue_p99_ms:
        reasons.append("Scheduler-facing publisher enqueue exceeded 50 ms p99")
    if baseline is not None and run.get("output_hashes") != baseline.get(
        "output_hashes"
    ):
        reasons.append("stalled output digests differ from disabled baseline")
    return Verdict("fail" if reasons else "pass", tuple(reasons), {})


def run_identity_reasons(
    summary: dict[str, Any], manifest: dict[str, Any], expected: RunSpec
) -> list[str]:
    """Bind a run directory and both metadata files to its committed RunSpec."""
    reasons = []
    expected_fingerprint = comparison_fingerprint(expected)
    for field, value in (
        ("run_id", expected.run_id),
        ("mode", expected.mode),
        ("seed", expected.seed),
        ("conclusion_scope", expected.conclusion_scope),
        ("comparison_fingerprint", expected_fingerprint),
    ):
        if summary.get(field) != value:
            reasons.append(f"summary {field} does not match committed run matrix")
    if manifest.get("run") != asdict(expected):
        reasons.append("manifest run does not match committed run matrix")
    if manifest.get("comparison_fingerprint") != expected_fingerprint:
        reasons.append("manifest fingerprint does not match committed run matrix")
    return reasons


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _choice_text(chunk: object) -> str:
    if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
        return ""
    text = ""
    for choice in chunk["choices"]:
        if not isinstance(choice, dict):
            continue
        if isinstance(choice.get("text"), str):
            text += choice["text"]
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            text += delta["content"]
    return text


def _audit_raw_responses(
    path: Path,
) -> tuple[
    dict[str, str],
    int,
    int,
    list[float],
    list[float],
    int,
    list[str],
]:
    hashes = {}
    measured_completed = 0
    failed = 0
    ttft = []
    tpot = []
    unexpected_tokens = 0
    reasons = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            record = json.loads(line)
            turn = record["turn"]
            result = record["result"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            reasons.append(f"responses.jsonl line {line_number} is malformed: {error}")
            continue
        if not isinstance(turn, dict) or not isinstance(result, dict):
            reasons.append(f"responses.jsonl line {line_number} is not an object")
            continue
        if result.get("status") != "completed":
            failed += 1
            continue
        key = f"{turn.get('session_id')}:{turn.get('turn_index')}"
        digest = hashlib.sha256(
            "".join(_choice_text(chunk) for chunk in record.get("chunks", [])).encode()
        ).hexdigest()
        if result.get("output_sha256") != digest:
            reasons.append(f"{key}: raw response digest does not match result")
        hashes[key] = digest
        if not turn.get("warmup"):
            measured_completed += 1
            if isinstance(result.get("ttft_ms"), (int, float)):
                ttft.append(float(result["ttft_ms"]))
            if isinstance(result.get("tpot_ms"), (int, float)):
                tpot.append(float(result["tpot_ms"]))
        if result.get("output_tokens") != turn.get("max_tokens"):
            unexpected_tokens += 1
    return (
        hashes,
        measured_completed,
        failed,
        ttft,
        tpot,
        unexpected_tokens,
        reasons,
    )


def _recompute_cpu(path: Path) -> dict[str, Any]:
    samples = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(samples) < 2:
        raise ValueError("CPU evidence requires at least two samples")
    elapsed = samples[-1]["timestamp"] - samples[0]["timestamp"]
    if elapsed <= 0:
        raise ValueError("CPU evidence has a nonpositive interval")
    names = set(samples[0]["processes"]) & set(samples[-1]["processes"])
    if names != {"prefill", "decode", "proxy"}:
        raise ValueError("CPU evidence is missing a required process tree")
    means = {}
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


def build_report(
    evidence_dir: Path,
    config_path: Path,
) -> tuple[Path, Path]:
    """Recompute acceptance from run summaries and preserve non-PASS states."""
    config = load_config(DEFAULT_CONFIG_PATH)
    supplied_config = load_config(config_path)
    summaries = []
    missing = []
    manifests: dict[str, dict[str, Any]] = {}
    audit_reasons = []
    invalid_run_reasons = []
    if supplied_config.raw != config.raw:
        audit_reasons.append("only the committed Issue #18 configuration is accepted")
    environment_path = evidence_dir / "environment.json"
    try:
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        if not isinstance(environment, dict):
            raise TypeError("environment must be an object")
    except FileNotFoundError:
        environment = {}
        missing.append("environment.json")
    except (json.JSONDecodeError, TypeError) as error:
        environment = {}
        audit_reasons.append(f"environment.json is malformed: {error}")
    frozen_config_path = evidence_dir / "config.json"
    try:
        frozen_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
        if not isinstance(frozen_config, dict):
            raise TypeError("configuration must be an object")
        if frozen_config != config.raw:
            audit_reasons.append("committed and evidence configurations differ")
    except FileNotFoundError:
        frozen_config = {}
        missing.append("config.json")
    except (json.JSONDecodeError, TypeError) as error:
        frozen_config = {}
        audit_reasons.append(f"config.json is malformed: {error}")
    gpus = environment.get("gpus", [])
    if len(gpus) != 2 or any("RTX 3090" not in gpu.get("name", "") for gpu in gpus):
        audit_reasons.append("environment is not the required dual RTX 3090 target")
    if "ycsxh/vllm" not in environment.get("git_remotes", {}).get("origin", ""):
        audit_reasons.append("environment origin is not the personal fork ycsxh/vllm")

    loaded: dict[str, dict[str, Any]] = {}

    def load_run(run_dir: Path, expected: RunSpec) -> None:
        try:
            summary = load_run_summary(run_dir)
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if not isinstance(manifest, dict):
                raise TypeError("manifest must be an object")
        except FileNotFoundError as error:
            missing.append(str(Path(error.filename).relative_to(evidence_dir)))
            invalid_path = run_dir / "invalid-run.json"
            if invalid_path.is_file():
                try:
                    invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
                    if not isinstance(invalid, dict):
                        raise TypeError("invalid-run must be an object")
                    invalid_run_reasons.append(
                        f"{expected.run_id}: {invalid.get('reason', 'runner failed')}"
                    )
                except (json.JSONDecodeError, TypeError) as invalid_error:
                    invalid_run_reasons.append(
                        f"{expected.run_id}: invalid-run.json is malformed: "
                        f"{invalid_error}"
                    )
            return
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            audit_reasons.append(f"{expected.run_id}: malformed metadata: {error}")
            return
        summaries.append(summary)
        loaded[expected.run_id] = summary
        manifests[expected.run_id] = manifest
        audit_reasons.extend(
            f"{expected.run_id}: {reason}"
            for reason in run_identity_reasons(summary, manifest, expected)
        )
        if summary.get("comparison_fingerprint") != manifest.get(
            "comparison_fingerprint"
        ):
            audit_reasons.append(
                f"{summary.get('run_id')}: summary/manifest fingerprint mismatch"
            )
        if manifest.get("vllm_commit") != environment.get("vllm_commit"):
            audit_reasons.append(
                f"{summary.get('run_id')}: vLLM commit differs from environment"
            )
        for field in ("model_revision", "tokenizer_revision"):
            if manifest.get(field) != environment.get(field):
                audit_reasons.append(
                    f"{summary.get('run_id')}: {field} differs from environment"
                )
        actual_hashes = {}
        for relative in RUN_SOURCE_ARTIFACTS:
            if not (run_dir / relative).is_file():
                audit_reasons.append(
                    f"{summary.get('run_id')}: missing artifact {relative}"
                )
            else:
                actual_hashes[relative] = _sha256(run_dir / relative)
        if summary.get("source_sha256") != actual_hashes:
            audit_reasons.append(
                f"{expected.run_id}: raw evidence digests do not match summary"
            )
        responses_path = run_dir / "raw" / "responses.jsonl"
        if responses_path.is_file():
            try:
                (
                    output_hashes,
                    completed,
                    failed,
                    ttft,
                    tpot,
                    unexpected_tokens,
                    raw_reasons,
                ) = _audit_raw_responses(responses_path)
                window = json.loads(
                    (run_dir / "raw" / "measurement-window.json").read_text()
                )
                instrumentation = json.loads(
                    (run_dir / "raw" / "decode-instrumentation.json").read_text()
                )
                proxy_state = json.loads(
                    (run_dir / "raw" / "proxy-state.json").read_text()
                )
                process_exits = json.loads(
                    (run_dir / "raw" / "process-exits.json").read_text()
                )
                elapsed = float(window["elapsed_seconds"])
                if not window.get("started_after_warmup") or elapsed <= 0:
                    raise ValueError("invalid measurement window")
                scheduler = distribution(instrumentation["scheduler_step_ms"]["raw"])
                serving = {
                    "completed_requests": completed,
                    "failed_requests": failed,
                    "measurement_seconds": elapsed,
                    "request_throughput": completed / elapsed,
                    "ttft_ms": distribution(ttft),
                    "tpot_ms": distribution(tpot),
                    "ttft_p95_ms": distribution(ttft)["p95"],
                    "tpot_p95_ms": distribution(tpot)["p95"],
                    "scheduler_step_p95_ms": scheduler["p95"],
                }
                event_pump = proxy_state.get("event_pump") or {}
                subscriber = event_pump.get("subscriber") or {}
                mirror = proxy_state.get("mirror") or {}
                publisher = (
                    instrumentation["publishers"][0] if expected.events_enabled else {}
                )
                sequence_gaps = max(
                    int(mirror.get("gap_count") or 0),
                    int(subscriber.get("sequence_gaps") or 0),
                )
                mirror_reason = mirror.get("invalid_reason")
                integrity = {
                    "publisher_overflow": int(publisher.get("dropped_batches") or 0),
                    "first_publisher_overflow_at": instrumentation.get(
                        "first_publisher_overflow_at"
                    ),
                    "sequence_gaps": sequence_gaps,
                    "first_sequence_gap_at": mirror.get("first_gap_at"),
                    "malformed_events": int(
                        bool(mirror_reason and "malformed" in str(mirror_reason))
                    ),
                    "proxy_restarts": int("proxy" in process_exits),
                    "decode_restarts": int("decode" in process_exits),
                }
                event_plane = {
                    "construction_ms": {
                        "raw": instrumentation["event_construction_ms"]["raw"],
                        "summary": distribution(
                            instrumentation["event_construction_ms"]["raw"]
                        ),
                    },
                    "enqueue_ms": {
                        "raw": instrumentation["event_enqueue_ms"]["raw"],
                        "summary": distribution(
                            instrumentation["event_enqueue_ms"]["raw"]
                        ),
                    },
                    "publisher": publisher,
                    "publisher_samples": instrumentation.get("publisher_samples"),
                    "proxy_event_lag_seconds": event_pump.get("event_lag_seconds"),
                    "subscriber": subscriber,
                    "mirror_metrics": mirror.get("metrics"),
                }
                from benchmarks.tda_forward.runner import (
                    _d_attempt_invalid_reasons,
                    _scenario_observations,
                )

                scenarios = _scenario_observations(proxy_state)
                raw_d_reasons = _d_attempt_invalid_reasons(
                    scenarios["d_local_attempts"]
                )
                if raw_d_reasons:
                    audit_reasons.extend(
                        f"{expected.run_id}: {reason}" for reason in raw_d_reasons
                    )
                if event_pump.get("failure"):
                    audit_reasons.append(f"{expected.run_id}: {event_pump['failure']}")
                if unexpected_tokens:
                    audit_reasons.append(
                        f"{expected.run_id}: {unexpected_tokens} responses have "
                        "an unexpected token count"
                    )
                recomputed = {
                    "output_hashes": output_hashes,
                    "serving": serving,
                    "event_plane": event_plane,
                    "cpu": _recompute_cpu(run_dir / "raw" / "cpu.jsonl"),
                    "integrity": integrity,
                    "scenarios": scenarios,
                }
                audit_reasons.extend(
                    f"{expected.run_id}: {name} does not match raw evidence"
                    for name, value in recomputed.items()
                    if summary.get(name) != value
                )
                audit_reasons.extend(
                    f"{expected.run_id}: {reason}" for reason in raw_reasons
                )
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                audit_reasons.append(
                    f"{expected.run_id}: raw evidence cannot be aggregated: {error}"
                )

    expected_runs = build_run_matrix(config)
    expected_ids = {run.run_id for run in expected_runs}
    runs_dir = evidence_dir / "runs"
    if runs_dir.is_dir():
        actual_ids = {path.name for path in runs_dir.iterdir() if path.is_dir()}
        for extra in sorted(actual_ids - expected_ids):
            audit_reasons.append(f"unexpected run directory: {extra}")
    for expected in expected_runs:
        load_run(runs_dir / expected.run_id, expected)
    stalled = loaded.get(f"stalled-seed-{config.seeds[0]}")

    disabled = [
        loaded[run.run_id]
        for run in expected_runs
        if run.mode == "disabled" and run.run_id in loaded
    ]
    oracle = [
        loaded[run.run_id]
        for run in expected_runs
        if run.mode == "oracle" and run.run_id in loaded
    ]
    enabled = [
        loaded[run.run_id]
        for run in expected_runs
        if run.mode == "enabled" and run.run_id in loaded
    ]
    healthy = [
        loaded[run.run_id]
        for run in expected_runs
        if run.mode == "healthy" and run.run_id in loaded
    ]
    controlled = (
        Verdict("incomplete", tuple(f"missing {path}" for path in missing), {})
        if missing
        else evaluate_comparison(disabled, enabled, oracle, config)
    )
    healthy_reasons = [
        f"healthy seed {run.get('seed')}: {reason}"
        for run in healthy
        for reason in invalid_reasons(run)
    ]
    healthy_reasons.extend(
        f"healthy seed {run.get('seed')}: required scenario not observed: {name}"
        for run in healthy
        for name in (
            "full_reuse",
            "partial_reuse",
            "zero_reuse",
            "evict_d",
            "capacity_recovery",
            "shared_prefix",
        )
        if not run.get("scenarios", {}).get(name)
    )
    oracle_by_seed = {run.get("seed"): run for run in oracle}
    healthy_reasons.extend(
        f"healthy seed {run.get('seed')}: output digests differ from sequential oracle"
        for run in healthy
        if run.get("output_hashes")
        != oracle_by_seed.get(run.get("seed"), {}).get("output_hashes")
    )
    functional = Verdict(
        "invalid" if healthy_reasons else ("incomplete" if missing else "pass"),
        tuple(healthy_reasons),
        {},
    )
    stalled_verdict = (
        Verdict("incomplete", ("stalled-consumer run is missing",), {})
        if stalled is None
        else evaluate_stalled(stalled, config, oracle_by_seed.get(config.seeds[0]))
    )
    audit = Verdict(
        (
            "invalid"
            if audit_reasons or invalid_run_reasons
            else ("incomplete" if missing else "pass")
        ),
        tuple(sorted(set((*audit_reasons, *invalid_run_reasons)))),
        {},
    )
    overall = (
        "invalid"
        if invalid_run_reasons
        else "pass"
        if not missing
        and controlled.status == "pass"
        and functional.status == "pass"
        and stalled_verdict.status == "pass"
        and audit.status == "pass"
        else (
            "incomplete"
            if missing
            else (
                "invalid"
                if "invalid"
                in {
                    controlled.status,
                    functional.status,
                    stalled_verdict.status,
                    audit.status,
                }
                else "fail"
            )
        )
    )
    result = {
        "schema_version": 1,
        "overall": overall,
        "controlled": asdict(controlled),
        "functional_stress": asdict(functional),
        "stalled_consumer": asdict(stalled_verdict),
        "evidence_audit": asdict(audit),
        "missing_runs": missing,
        "runs": summaries,
        "manifests": manifests,
        "environment": environment,
        "configuration": config.raw,
        "frozen_configuration": frozen_config,
        "notice": (
            "Only target-server evidence may produce Issue #18 PASS; portable "
            "or incomplete data is never promoted to PASS."
        ),
    }
    summary_path = evidence_dir / "acceptance-summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_path = evidence_dir / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "run_id",
                "mode",
                "seed",
                "status",
                "throughput",
                "ttft_p95_ms",
                "tpot_p95_ms",
            )
        )
        for run in summaries:
            serving = run.get("serving", {})
            writer.writerow(
                (
                    run.get("run_id"),
                    run.get("mode"),
                    run.get("seed"),
                    run.get("status"),
                    serving.get("request_throughput"),
                    serving.get("ttft_p95_ms"),
                    serving.get("tpot_p95_ms"),
                )
            )
    report_path = evidence_dir / "report.md"
    _write_markdown(report_path, result)
    return summary_path, report_path


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    controlled = result["controlled"]
    functional = result["functional_stress"]
    stalled = result["stalled_consumer"]
    audit = result["evidence_audit"]
    environment = result["environment"]
    configuration = result["configuration"]
    gpu_names = ", ".join(
        gpu.get("name", "unknown") for gpu in environment.get("gpus", [])
    )
    lines = [
        "# TDAforward Issue #18 target-server acceptance",
        "",
        (
            "> This report is recomputed from preserved target-server artifacts. "
            "Portable results cannot establish CUDA, NIXL, RTX 3090, or "
            "performance acceptance."
        ),
        "",
        "## Verdicts",
        "",
        f"- Overall Issue #18: **{result['overall'].upper()}**",
        f"- Controlled overhead: **{controlled['status'].upper()}**",
        f"- Functional stress: **{functional['status'].upper()}**",
        f"- Stalled-consumer liveness: **{stalled['status'].upper()}**",
        f"- Evidence audit: **{audit['status'].upper()}**",
        (
            "- The stalled-consumer result is excluded from routing-quality "
            "and normal-latency conclusions."
        ),
        "",
        "## Frozen target configuration",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| vLLM commit | `{environment.get('vllm_commit', 'missing')}` |",
        f"| Model | {configuration.get('model', 'missing')} |",
        f"| Model revision | `{environment.get('model_revision', 'missing')}` |",
        (
            "| Tokenizer revision | `"
            f"{environment.get('tokenizer_revision', 'missing')}` |"
        ),
        f"| GPUs | {gpu_names or 'missing'} |",
        f"| Ports | `{json.dumps(configuration.get('ports', {}), sort_keys=True)}` |",
        (
            "| Deployment | `"
            f"{json.dumps(configuration.get('deployment', {}), sort_keys=True)}` |"
        ),
        f"| Seeds | `{configuration.get('seeds', [])}` |",
        "| Launch commands | Preserved per run in `runs/*/manifest.json` |",
        "| Raw logs and measurements | Preserved per run in `runs/*/{logs,raw}` |",
        "",
        "## Controlled ratios",
        "",
        "| Metric | Enabled / disabled |",
        "| --- | ---: |",
    ]
    for name, value in controlled["ratios"].items():
        lines.append(f"| {name} | {value:.6f} |")
    lines.extend(
        (
            "",
            "## Per-run serving metrics",
            "",
            (
                "| Run | Status | req/s | TTFT p95 ms | TPOT p95 ms | "
                "Scheduler p95 ms | CPU mean cores |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for run in result["runs"]:
        serving = run.get("serving", {})
        lines.append(
            f"| {run.get('run_id')} | {run.get('status')} | "
            f"{serving.get('request_throughput')} | "
            f"{serving.get('ttft_p95_ms')} | {serving.get('tpot_p95_ms')} | "
            f"{serving.get('scheduler_step_p95_ms')} | "
            f"{run.get('cpu', {}).get('total_mean_cores')} |"
        )
    lines.extend(
        (
            "",
            "## Per-run event-plane metrics",
            "",
            (
                "| Run | construct ms p50/p95/p99 | enqueue ms p50/p95/p99 | "
                "queue HWM | batches | blocks | bytes | Proxy lag p95 s | "
                "drops | gaps |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for run in result["runs"]:
        event = run.get("event_plane", {})
        construction = (event.get("construction_ms") or {}).get("summary", {})
        enqueue = (event.get("enqueue_ms") or {}).get("summary", {})
        publisher = event.get("publisher") or {}
        lag = event.get("proxy_event_lag_seconds") or {}
        integrity = run.get("integrity", {})
        lines.append(
            f"| {run.get('run_id')} | "
            f"{construction.get('p50')}/{construction.get('p95')}/"
            f"{construction.get('p99')} | "
            f"{enqueue.get('p50')}/{enqueue.get('p95')}/{enqueue.get('p99')} | "
            f"{publisher.get('queue_high_watermark')} | "
            f"{publisher.get('published_batches')} | "
            f"{publisher.get('published_blocks')} | "
            f"{publisher.get('published_bytes')} | {lag.get('p95')} | "
            f"{integrity.get('publisher_overflow')} | "
            f"{integrity.get('sequence_gaps')} |"
        )
    lines.extend(("", "## Invalid, failed, or incomplete reasons", ""))
    reasons = (
        *controlled["reasons"],
        *functional["reasons"],
        *stalled["reasons"],
        *audit["reasons"],
    )
    lines.extend(f"- {reason}" for reason in reasons)
    if not reasons:
        lines.append("- None")
    lines.extend(
        (
            "",
            "## Provenance boundary",
            "",
            (
                "Native vLLM supplies serving, NIXL transfer, KV event vocabulary, "
                "and the bounded publisher. The pinned Dynamo-derived primitives "
                "and TDAforward-specific 1P1D mirror are documented in "
                "`examples/disaggregated/tda_forward/README.md` and "
                "`THIRD_PARTY_NOTICES.md`."
            ),
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")
