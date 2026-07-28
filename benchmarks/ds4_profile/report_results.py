# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit Ticket 4 run artifacts and generate the concise profile report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.ds4_profile.run_points import (
    MODEL,
    ExperimentPoint,
    PreparedPoint,
    PreparedRequest,
    derive_run_result,
    load_experiment_plan,
    summarize_point_runs,
)

CSV_FIELDS = (
    "point_id",
    "status",
    "status_reason",
    "source_request_id",
    "input_tokens",
    "requested_hit_ratio",
    "aligned_planned_hit_ratio_mean",
    "actual_p_hit_ratio_mean",
    "actual_p_hit_ratio_cv",
    "max_num_batched_tokens",
    "max_concurrency",
    "output_tokens",
    "execution_mode",
    "repetitions",
    "completed_requests_per_run",
    "failed_requests_per_run",
    "ttft_p50_mean_ms",
    "ttft_p50_cv",
    "ttft_p90_mean_ms",
    "ttft_p90_cv",
    "ttft_p95_mean_ms",
    "ttft_p95_cv",
    "tpot_p50_mean_ms",
    "tpot_p50_cv",
    "tpot_p90_mean_ms",
    "tpot_p90_cv",
    "tpot_p95_mean_ms",
    "tpot_p95_cv",
    "itl_p99_mean_ms",
    "itl_p99_cv",
    "output_throughput_mean",
    "output_throughput_cv",
    "noisy",
    "noisy_metrics",
    "evidence",
)
X_FIELDS = {
    "requested_hit_ratio",
    "max_num_batched_tokens",
    "max_concurrency",
    "input_tokens",
    "output_tokens",
}
PLOT_METRICS = {
    "actual_p_hit_ratio",
    "output_throughput",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
    "p99_itl_ms",
}
PLOT_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#9333ea",
    "#ea580c",
    "#0891b2",
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths written by one deterministic report build."""

    summary_csv: Path
    report_md: Path
    plot_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _AuditedPoint:
    point: ExperimentPoint
    status: str
    reason: str
    input_tokens: int | None
    runs: tuple[dict[str, Any], ...]
    summary: dict[str, Any] | None
    evidence: tuple[str, ...]

    @property
    def noisy(self) -> bool:
        return bool(self.summary and self.summary["noisy_metrics"])

    def metric_mean(self, name: str) -> float | None:
        if self.summary is None:
            return None
        metric = self.summary["metrics"].get(name)
        return None if metric is None else float(metric["mean"])


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load JSON object from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_report_metadata(path: Path) -> dict[str, Any]:
    metadata = _load_json_object(path)
    if set(metadata) != {"schema_version", "hardware", "limitations"}:
        raise ValueError(
            "report metadata must contain schema_version, hardware, and limitations"
        )
    if metadata["schema_version"] != 1:
        raise ValueError("unsupported report metadata schema_version")
    hardware = metadata["hardware"]
    if not isinstance(hardware, dict) or set(hardware) != {
        "gpu_count",
        "gpu_model",
        "interconnect",
    }:
        raise ValueError(
            "hardware metadata must contain gpu_count, gpu_model, and interconnect"
        )
    if (
        isinstance(hardware["gpu_count"], bool)
        or not isinstance(hardware["gpu_count"], int)
        or hardware["gpu_count"] <= 0
        or not isinstance(hardware["gpu_model"], str)
        or not hardware["gpu_model"]
        or not isinstance(hardware["interconnect"], str)
        or not hardware["interconnect"]
    ):
        raise ValueError("hardware metadata values are invalid")
    limitations = metadata["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(value, str) or not value for value in limitations)
    ):
        raise ValueError("report limitations must be a nonempty string list")
    return metadata


def _prepared_from_artifact(
    path: Path,
    expected_point: ExperimentPoint,
) -> PreparedPoint:
    value = _load_json_object(path)
    if value.get("schema_version") != 1:
        raise ValueError(f"{path} has an unsupported schema_version")
    point_value = value.get("point")
    if not isinstance(point_value, dict):
        raise ValueError(f"{path} lacks a point object")
    point_value = {
        **point_value,
        "execution_mode": point_value.get("execution_mode", "optimized"),
    }
    try:
        point = ExperimentPoint(**point_value)
    except TypeError as error:
        raise ValueError(f"{path} has an invalid point object") from error
    if point != expected_point:
        raise ValueError(f"{path} does not match the frozen experiment plan")
    requests_value = value.get("requests")
    if not isinstance(requests_value, list) or len(requests_value) != point.num_prompts:
        raise ValueError(f"{path} has an invalid request list")
    requests = []
    for request in requests_value:
        if not isinstance(request, dict):
            raise ValueError(f"{path} has a non-object request")
        try:
            requests.append(
                PreparedRequest(
                    request_id=request["request_id"],
                    prompt="",
                    prompt_ids=request["prompt_ids"],
                    input_tokens=request["input_tokens"],
                    isolation_ids=request["isolation_ids"],
                    planned_cached_tokens=request["planned_cached_tokens"],
                    warm_prefix=None,
                )
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path} has an invalid request") from error
    input_lengths = {request.input_tokens for request in requests}
    if len(input_lengths) != 1:
        raise ValueError(f"{path} requests must share one controlled input length")
    isolation_tokens = value.get("isolation_block_tokens")
    alignment_tokens = value.get("cache_alignment_tokens")
    if (
        isinstance(isolation_tokens, bool)
        or not isinstance(isolation_tokens, int)
        or isolation_tokens <= 0
        or isinstance(alignment_tokens, bool)
        or not isinstance(alignment_tokens, int)
        or alignment_tokens <= 0
    ):
        raise ValueError(f"{path} has invalid cache geometry")
    return PreparedPoint(
        point=point,
        requests=tuple(requests),
        isolation_block_tokens=isolation_tokens,
        cache_alignment_tokens=alignment_tokens,
    )


def _manifest_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = manifest.get("points")
    if not isinstance(values, list):
        raise ValueError("run manifest points must be a list")
    entries = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ValueError("run manifest contains an invalid point entry")
        point_id = value["id"]
        if point_id in entries:
            raise ValueError(f"run manifest repeats point {point_id}")
        entries[point_id] = value
    return entries


def _audit_valid_point(
    results_dir: Path,
    point: ExperimentPoint,
) -> _AuditedPoint:
    point_dir = results_dir / "points" / point.id
    prepared = _prepared_from_artifact(point_dir / "point.json", point)
    runs = []
    evidence = []
    for repetition in range(1, point.repetitions + 1):
        run_dir = point_dir / f"run-{repetition:02d}"
        official_path = run_dir / "bench-result.json"
        derived_path = run_dir / "derived.json"
        official = _load_json_object(official_path)
        try:
            p_before = (run_dir / "p-metrics-before.txt").read_text(encoding="utf-8")
            p_after = (run_dir / "p-metrics-after.txt").read_text(encoding="utf-8")
            d_before = (run_dir / "d-metrics-before.txt").read_text(encoding="utf-8")
            d_after = (run_dir / "d-metrics-after.txt").read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"{run_dir} lacks required metrics evidence") from error
        audited = derive_run_result(
            prepared,
            official,
            p_metrics_before=p_before,
            p_metrics_after=p_after,
            d_metrics_before=d_before,
            d_metrics_after=d_after,
            block_size=prepared.cache_alignment_tokens,
        )
        if _load_json_object(derived_path) != audited:
            raise ValueError(f"{derived_path} does not match raw evidence")
        runs.append(audited)
        evidence.extend(
            (
                str(official_path.relative_to(results_dir)),
                str(derived_path.relative_to(results_dir)),
            )
        )
    summary = summarize_point_runs(point.id, runs)
    summary_path = point_dir / "point-summary.json"
    if _load_json_object(summary_path) != summary:
        raise ValueError(f"{summary_path} does not match audited repetitions")
    evidence.append(str(summary_path.relative_to(results_dir)))
    input_tokens = prepared.requests[0].input_tokens
    return _AuditedPoint(
        point=point,
        status="valid",
        reason="",
        input_tokens=input_tokens,
        runs=tuple(runs),
        summary=summary,
        evidence=tuple(evidence),
    )


def _audit_incomplete_point(
    results_dir: Path,
    point: ExperimentPoint,
    status: str,
    reason: str,
) -> _AuditedPoint:
    point_path = results_dir / "points" / point.id / "point.json"
    input_tokens = None
    evidence = []
    if point_path.is_file():
        prepared = _prepared_from_artifact(point_path, point)
        input_tokens = prepared.requests[0].input_tokens
        evidence.append(str(point_path.relative_to(results_dir)))
    failure_path = results_dir / "points" / point.id / "point-failure.json"
    if failure_path.is_file():
        failure = _load_json_object(failure_path)
        if failure.get("status", status) != status:
            raise ValueError(f"{failure_path} status differs from the run manifest")
        if reason and failure.get("error") != reason:
            raise ValueError(f"{failure_path} reason differs from the run manifest")
        evidence.append(str(failure_path.relative_to(results_dir)))
    return _AuditedPoint(
        point=point,
        status=status,
        reason=reason,
        input_tokens=input_tokens,
        runs=(),
        summary=None,
        evidence=tuple(evidence),
    )


def _audit_results(
    results_dir: Path,
    points: tuple[ExperimentPoint, ...],
) -> tuple[tuple[_AuditedPoint, ...], dict[str, Any]]:
    manifest = _load_json_object(results_dir / "run-manifest.json")
    if manifest.get("vllm_dirty") is not False:
        raise ValueError("run manifest must record a clean vLLM checkout")
    entries = _manifest_entries(manifest)
    unknown = sorted(entries.keys() - {point.id for point in points})
    if unknown:
        raise ValueError(f"run manifest contains unknown points: {', '.join(unknown)}")
    audited = []
    for point in points:
        entry = entries.get(point.id)
        if entry is None:
            audited.append(
                _audit_incomplete_point(
                    results_dir,
                    point,
                    "unattempted",
                    "point was not attempted",
                )
            )
            continue
        status = entry.get("status")
        if status == "valid":
            audited.append(_audit_valid_point(results_dir, point))
            continue
        if status not in {"unsupported", "failed", "interrupted"}:
            raise ValueError(f"point {point.id} has unsupported status {status!r}")
        reason = entry.get("error")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"point {point.id} lacks a retained failure reason")
        audited.append(_audit_incomplete_point(results_dir, point, status, reason))
    return tuple(audited), manifest


def _metric_stat(
    point: _AuditedPoint,
    metric: str,
    statistic: str,
) -> float | None:
    if point.summary is None:
        return None
    value = point.summary["metrics"].get(metric)
    if value is None:
        return None
    return float(value[statistic])


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def _csv_row(point: _AuditedPoint) -> dict[str, str]:
    first_run = point.runs[0] if point.runs else {}
    row: dict[str, Any] = {
        "point_id": point.point.id,
        "status": point.status,
        "status_reason": point.reason,
        "source_request_id": point.point.source_request_id,
        "input_tokens": point.input_tokens,
        "requested_hit_ratio": point.point.hit_ratio,
        "aligned_planned_hit_ratio_mean": (
            None
            if not point.runs
            else sum(run["aligned_planned_hit_ratio"] for run in point.runs)
            / len(point.runs)
        ),
        "actual_p_hit_ratio_mean": _metric_stat(point, "actual_p_hit_ratio", "mean"),
        "actual_p_hit_ratio_cv": _metric_stat(point, "actual_p_hit_ratio", "cv"),
        "max_num_batched_tokens": point.point.max_num_batched_tokens,
        "max_concurrency": point.point.max_concurrency,
        "output_tokens": point.point.output_tokens,
        "execution_mode": point.point.execution_mode,
        "repetitions": len(point.runs),
        "completed_requests_per_run": first_run.get("completed"),
        "failed_requests_per_run": first_run.get("failed"),
        "ttft_p50_mean_ms": _metric_stat(point, "p50_ttft_ms", "mean"),
        "ttft_p50_cv": _metric_stat(point, "p50_ttft_ms", "cv"),
        "ttft_p90_mean_ms": _metric_stat(point, "p90_ttft_ms", "mean"),
        "ttft_p90_cv": _metric_stat(point, "p90_ttft_ms", "cv"),
        "ttft_p95_mean_ms": _metric_stat(point, "p95_ttft_ms", "mean"),
        "ttft_p95_cv": _metric_stat(point, "p95_ttft_ms", "cv"),
        "tpot_p50_mean_ms": _metric_stat(point, "p50_tpot_ms", "mean"),
        "tpot_p50_cv": _metric_stat(point, "p50_tpot_ms", "cv"),
        "tpot_p90_mean_ms": _metric_stat(point, "p90_tpot_ms", "mean"),
        "tpot_p90_cv": _metric_stat(point, "p90_tpot_ms", "cv"),
        "tpot_p95_mean_ms": _metric_stat(point, "p95_tpot_ms", "mean"),
        "tpot_p95_cv": _metric_stat(point, "p95_tpot_ms", "cv"),
        "itl_p99_mean_ms": _metric_stat(point, "p99_itl_ms", "mean"),
        "itl_p99_cv": _metric_stat(point, "p99_itl_ms", "cv"),
        "output_throughput_mean": _metric_stat(point, "output_throughput", "mean"),
        "output_throughput_cv": _metric_stat(point, "output_throughput", "cv"),
        "noisy": "true" if point.noisy else "false",
        "noisy_metrics": (
            "" if point.summary is None else ";".join(point.summary["noisy_metrics"])
        ),
        "evidence": ";".join(point.evidence),
    }
    return {name: _format_number(row[name]) for name in CSV_FIELDS}


def _write_summary(path: Path, points: tuple[_AuditedPoint, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_row(point) for point in points)


def _load_comparisons(
    plan_value: dict[str, Any],
    point_ids: set[str],
) -> tuple[dict[str, Any], ...]:
    report = plan_value.get("report")
    if not isinstance(report, dict) or set(report) != {"comparisons"}:
        raise ValueError("version 2 plan report must contain only comparisons")
    values = report["comparisons"]
    if not isinstance(values, list) or not values:
        raise ValueError("report comparisons must be a nonempty list")
    comparisons = []
    comparison_ids = set()
    for value in values:
        if not isinstance(value, dict) or set(value) != {
            "id",
            "title",
            "x",
            "metric",
            "series",
        }:
            raise ValueError("each report comparison has invalid fields")
        comparison_id = value["id"]
        if (
            not isinstance(comparison_id, str)
            or not comparison_id
            or comparison_id in comparison_ids
        ):
            raise ValueError("report comparison ids must be nonempty and unique")
        comparison_ids.add(comparison_id)
        if not isinstance(value["title"], str) or not value["title"]:
            raise ValueError(f"comparison {comparison_id} has an invalid title")
        if value["x"] not in X_FIELDS:
            raise ValueError(f"comparison {comparison_id} has an invalid x field")
        if value["metric"] not in PLOT_METRICS:
            raise ValueError(f"comparison {comparison_id} has an invalid metric")
        series_values = value["series"]
        if not isinstance(series_values, list) or not series_values:
            raise ValueError(f"comparison {comparison_id} has no series")
        for series in series_values:
            if (
                not isinstance(series, dict)
                or set(series) != {"label", "point_ids"}
                or not isinstance(series["label"], str)
                or not series["label"]
                or not isinstance(series["point_ids"], list)
                or not series["point_ids"]
                or any(
                    not isinstance(point_id, str) or point_id not in point_ids
                    for point_id in series["point_ids"]
                )
            ):
                raise ValueError(f"comparison {comparison_id} has an invalid series")
        comparisons.append(value)
    return tuple(comparisons)


def _x_value(point: _AuditedPoint, name: str) -> float | None:
    if name == "requested_hit_ratio":
        return point.point.hit_ratio
    if name == "max_num_batched_tokens":
        return float(point.point.max_num_batched_tokens)
    if name == "max_concurrency":
        return float(point.point.max_concurrency)
    if name == "input_tokens":
        return None if point.input_tokens is None else float(point.input_tokens)
    if name == "output_tokens":
        return float(point.point.output_tokens)
    raise AssertionError(f"unhandled x field {name}")


def _scaled(value: float, low: float, high: float, start: float, end: float) -> float:
    if low == high:
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def _write_plot(
    path: Path,
    comparison: dict[str, Any],
    point_map: dict[str, _AuditedPoint],
) -> None:
    selected = [
        point_map[point_id]
        for series in comparison["series"]
        for point_id in series["point_ids"]
    ]
    x_values = [
        value
        for point in selected
        if (value := _x_value(point, comparison["x"])) is not None
    ]
    y_values = [
        value
        for point in selected
        if (value := point.metric_mean(comparison["metric"])) is not None
    ]
    if not x_values:
        raise ValueError(f"comparison {comparison['id']} has no x values")
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = (min(y_values), max(y_values)) if y_values else (0.0, 1.0)
    y_padding = max((y_max - y_min) * 0.1, abs(y_max) * 0.05, 1e-9)
    y_min = max(0.0, y_min - y_padding)
    y_max += y_padding
    left, right, top, bottom = 90.0, 920.0, 75.0, 510.0

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="600" '
        'viewBox="0 0 960 600">',
        '<rect width="960" height="600" fill="white"/>',
        (
            f'<text x="480" y="34" text-anchor="middle" '
            f'font-family="sans-serif" font-size="20">'
            f"{html.escape(comparison['title'])}</text>"
        ),
        (
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" '
            'stroke="#111827"/>'
        ),
        (f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827"/>'),
        (
            f'<text x="505" y="566" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">'
            f"{html.escape(comparison['x'])}</text>"
        ),
        (
            f'<text x="24" y="292" text-anchor="middle" '
            f'transform="rotate(-90 24 292)" font-family="sans-serif" '
            f'font-size="14">{html.escape(comparison["metric"])}</text>'
        ),
        (
            f'<text x="{left}" y="{bottom + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{x_min:g}</text>'
        ),
        (
            f'<text x="{right}" y="{bottom + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{x_max:g}</text>'
        ),
        (
            f'<text x="{left - 10}" y="{bottom}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{y_min:g}</text>'
        ),
        (
            f'<text x="{left - 10}" y="{top + 4}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{y_max:g}</text>'
        ),
    ]
    legend_y = 55
    for series_index, series in enumerate(comparison["series"]):
        color = PLOT_COLORS[series_index % len(PLOT_COLORS)]
        series_points = [point_map[point_id] for point_id in series["point_ids"]]
        valid_points = [
            point
            for point in series_points
            if _x_value(point, comparison["x"]) is not None
            and point.metric_mean(comparison["metric"]) is not None
        ]
        valid_points.sort(key=lambda point: _x_value(point, comparison["x"]) or 0)
        coordinates = []
        for point in valid_points:
            x_value = _x_value(point, comparison["x"])
            y_value = point.metric_mean(comparison["metric"])
            assert x_value is not None and y_value is not None
            coordinates.append(
                (
                    point,
                    _scaled(x_value, x_min, x_max, left, right),
                    _scaled(y_value, y_min, y_max, bottom, top),
                )
            )
        if len(coordinates) > 1:
            coordinate_text = " ".join(f"{x:.2f},{y:.2f}" for _, x, y in coordinates)
            lines.append(
                f'<polyline points="{coordinate_text}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        for point, x, y in coordinates:
            status = "noisy" if point.noisy else "valid"
            stroke = "#d97706" if point.noisy else color
            lines.extend(
                (
                    (
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" '
                        f'stroke="{stroke}" stroke-width="3" '
                        f'data-point-id="{html.escape(point.point.id)}" '
                        f'data-status="{status}">'
                    ),
                    (
                        f"<title>{html.escape(point.point.id)}: {status}</title>"
                        "</circle>"
                    ),
                )
            )
            if point.noisy:
                lines.append(
                    f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" '
                    'font-family="sans-serif" font-size="11" '
                    'fill="#b45309">N</text>'
                )
        for point in series_points:
            if point.status == "valid":
                continue
            x_value = _x_value(point, comparison["x"])
            if x_value is None:
                continue
            x = _scaled(x_value, x_min, x_max, left, right)
            y = bottom - 8
            lines.extend(
                (
                    (
                        f'<g data-point-id="{html.escape(point.point.id)}" '
                        f'data-status="{html.escape(point.status)}">'
                    ),
                    (
                        f'<line x1="{x - 5:.2f}" y1="{y - 5:.2f}" '
                        f'x2="{x + 5:.2f}" y2="{y + 5:.2f}" stroke="#b91c1c"/>'
                    ),
                    (
                        f'<line x1="{x - 5:.2f}" y1="{y + 5:.2f}" '
                        f'x2="{x + 5:.2f}" y2="{y - 5:.2f}" stroke="#b91c1c"/>'
                    ),
                    (
                        f'<text x="{x + 8:.2f}" y="{y - 5:.2f}" '
                        'font-family="sans-serif" font-size="11" '
                        'fill="#b91c1c">U</text>'
                    ),
                    (
                        f"<title>{html.escape(point.point.id)}: "
                        f"{html.escape(point.status)}</title></g>"
                    ),
                )
            )
        legend_x = left + series_index * 210
        lines.extend(
            (
                (
                    f'<line x1="{legend_x}" y1="{legend_y}" '
                    f'x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" '
                    'stroke-width="2"/>'
                ),
                (
                    f'<text x="{legend_x + 30}" y="{legend_y + 4}" '
                    f'font-family="sans-serif" font-size="12">'
                    f"{html.escape(series['label'])}</text>"
                ),
            )
        )
    lines.extend(
        (
            (
                '<text x="90" y="588" font-family="sans-serif" font-size="11" '
                'fill="#4b5563">Orange outline: noisy (CV &gt; 5%). '
                "Red x: unsupported, failed, or unattempted.</text>"
            ),
            "</svg>",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configuration(
    results_dir: Path,
    points: tuple[_AuditedPoint, ...],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    server_plan = None
    for point in points:
        path = results_dir / "points" / point.point.id / "server-plan.json"
        if path.is_file():
            server_plan = _load_json_object(path)
            break
    if server_plan is None:
        raise ValueError("report requires at least one frozen server plan")
    if server_plan.get("vllm_commit") != manifest.get("vllm_commit"):
        raise ValueError("server plan and run manifest vLLM commits differ")
    if server_plan.get("model_revision") != manifest.get("model_revision"):
        raise ValueError("server plan and run manifest model revisions differ")
    if server_plan.get("tokenizer_revision") != manifest.get("tokenizer_revision"):
        raise ValueError("server plan and run manifest tokenizer revisions differ")
    return server_plan


def _mean_cell(point: _AuditedPoint, metric: str) -> str:
    return _format_number(point.metric_mean(metric))


def _write_report(
    path: Path,
    results_dir: Path,
    metadata: dict[str, Any],
    points: tuple[_AuditedPoint, ...],
    manifest: dict[str, Any],
    configuration: dict[str, Any],
    plot_paths: tuple[Path, ...],
) -> None:
    hardware = metadata["hardware"]
    compatibility = configuration["compatibility"]
    topology = configuration["topology"]
    valid = [point for point in points if point.status == "valid"]
    noisy = [point for point in points if point.noisy]
    incomplete = [point for point in points if point.status != "valid"]
    lines = [
        "# DS4-informed Qwen3.5 1P1D profile",
        "",
        "DS4 supplies input prompts only. This report is a controlled serving "
        "profile, not a model-quality or SWE-bench evaluation.",
        "",
        "## Frozen configuration",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Hardware | {hardware['gpu_count']} × {hardware['gpu_model']} |",
        f"| GPU interconnect | {hardware['interconnect']} |",
        f"| Model | {configuration.get('model', MODEL)} |",
        f"| Model revision | `{manifest['model_revision']}` |",
        f"| Tokenizer revision | `{manifest['tokenizer_revision']}` |",
        f"| vLLM revision | `{manifest['vllm_commit']}` |",
        f"| Precision | {compatibility['dtype']} weights and "
        f"{compatibility['kv_cache_dtype']} KV cache |",
        f"| Topology | P=GPU {topology['prefill']['gpu']}/NUMA "
        f"{topology['prefill']['numa_node']}; D=GPU "
        f"{topology['decode']['gpu']}/NUMA {topology['decode']['numa_node']}; "
        "TP=1 per role |",
        f"| Cache mode | {compatibility['kv_cache_layout']}, Mamba "
        f"{compatibility['mamba_cache_mode']}, prefix caching and chunked "
        "prefill enabled |",
        "| Main execution mode | Default optimized mode |",
        "",
        "## Result inventory",
        "",
        f"- {len(valid)} valid points, {len(noisy)} noisy points, and "
        f"{len(incomplete)} unsupported/failed/unattempted points.",
        "- Each valid row below was recomputed from its preserved official "
        "`bench-result.json` and before/after P/D metrics; the stored "
        "`derived.json` and point summary had to match that recomputation.",
        "- One-token points intentionally omit TPOT. No missing metric is "
        "reported as zero.",
        "",
        "| Point | Status | 1P1D TTFT p50 ms | TPOT p50 ms | Output tok/s | "
        "Observed P hit |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for point in points:
        lines.append(
            f"| `{point.point.id}` | {point.status}"
            f"{' (noisy)' if point.noisy else ''} | "
            f"{_mean_cell(point, 'p50_ttft_ms')} | "
            f"{_mean_cell(point, 'p50_tpot_ms')} | "
            f"{_mean_cell(point, 'output_throughput')} | "
            f"{_mean_cell(point, 'actual_p_hit_ratio')} |"
        )
    lines.extend(("", "## Plots", ""))
    for plot_path in plot_paths:
        relative = plot_path.relative_to(results_dir)
        lines.append(f"- [{plot_path.stem}]({relative.as_posix()})")
    lines.extend(("", "## Unsupported, failed, and noisy points", ""))
    if not incomplete and not noisy:
        lines.append("- None.")
    for point in incomplete:
        lines.append(f"- `{point.point.id}`: **{point.status}** — {point.reason}")
    for point in noisy:
        metrics = ", ".join(point.summary["noisy_metrics"])
        lines.append(f"- `{point.point.id}`: **noisy** — CV > 5% for {metrics}.")
    lines.extend(("", "## Experiment limitations", ""))
    lines.extend(f"- {value}" for value in metadata["limitations"])
    lines.extend(
        (
            "- TTFT is full 1P1D client-observed TTFT; it is not P-only or "
            "isolated transfer latency.",
            "- Unsupported points remain part of the tested matrix and were "
            "not silently reduced.",
            "- No automatic causal or production-traffic claims are added.",
            "",
            "## Gate D human acceptance",
            "",
            "- [x] Valid summary rows were re-audited against official detailed "
            "benchmark JSON and P/D metric deltas.",
            "- [x] Requested, aligned, and observed cache-hit values are retained.",
            "- [x] Unsupported and noisy points are explicitly labeled.",
            "- [ ] Review every plot against `summary.csv` and its raw evidence.",
            "- [ ] Review server logs for OOM, NIXL failure, silent fallback, "
            "compatibility mismatch, and cleanup errors.",
            "- [ ] Confirm the hardware/topology metadata against the captured "
            "target-host preflight.",
            "- [ ] Record Gate D human acceptance and immutable evidence checksums.",
            "",
            "## Legacy retirement checklist",
            "",
            "Perform these only after Gate D human acceptance:",
            "",
            "- [ ] Remove or archive the legacy `gpu_profile.py` and "
            "`profile_spine.py` execution path.",
            "- [ ] Remove Qwen2.5 execution mapping, teacher forcing, and their "
            "legacy result-contract tests/configuration.",
            "- [ ] Replace legacy README/workflow commands with the accepted "
            "Qwen3.5 1P1D workflow.",
            "- [ ] Move retained historical evidence under a clearly labeled "
            "discarded/archive area.",
            "- [ ] Confirm only one documented future workflow remains.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(results_dir: Path, metadata_path: Path) -> ReportArtifacts:
    """Audit a completed Ticket 4 result directory and write report artifacts."""
    results_dir = results_dir.resolve()
    metadata_path = metadata_path.resolve()
    plan_path = results_dir / "plan.json"
    points = load_experiment_plan(plan_path)
    plan_value = _load_json_object(plan_path)
    if plan_value.get("schema_version") != 2:
        raise ValueError("Ticket 4 reporting requires a version 2 experiment plan")
    comparisons = _load_comparisons(plan_value, {point.id for point in points})
    metadata = _load_report_metadata(metadata_path)
    audited, manifest = _audit_results(results_dir, points)
    configuration = _configuration(results_dir, audited, manifest)

    summary_path = results_dir / "summary.csv"
    _write_summary(summary_path, audited)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    audited_map = {point.point.id: point for point in audited}
    plot_paths = []
    for comparison in comparisons:
        plot_path = plots_dir / f"{comparison['id']}.svg"
        _write_plot(plot_path, comparison, audited_map)
        plot_paths.append(plot_path)
    report_path = results_dir / "report.md"
    _write_report(
        report_path,
        results_dir,
        metadata,
        audited,
        manifest,
        configuration,
        tuple(plot_paths),
    )
    copied_metadata = results_dir / "report-metadata.json"
    if metadata_path != copied_metadata:
        copied_metadata.write_bytes(metadata_path.read_bytes())
    return ReportArtifacts(summary_path, report_path, tuple(plot_paths))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the audited Ticket 4 report and return its process status."""
    args = _parser().parse_args(argv)
    build_report(args.results_dir, args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
