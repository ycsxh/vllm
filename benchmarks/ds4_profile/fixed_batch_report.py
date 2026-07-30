# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-audit fixed-batch artifacts and build the node-profile report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.ds4_profile.fixed_batch import (
    FixedBatchPoint,
    batch_observation_from_dict,
    derive_batch_metric_samples,
    load_fixed_batch_plan,
    summarize_point_runs,
    summarize_run_samples,
)

BASE_CSV_FIELDS = (
    "point_id",
    "role",
    "status",
    "status_reason",
    "metric_label",
    "batch_size",
    "hit_ratio",
    "execution_mode",
    "noisy",
    "noisy_metrics",
)
REPORT_METRICS = (
    ("latency_ms", "latency", "ms"),
    ("request_throughput_per_s", "request_throughput", "per_s"),
    (
        "computed_token_throughput_per_s",
        "computed_token_throughput",
        "per_s",
    ),
    (
        "output_token_throughput_per_s",
        "output_token_throughput",
        "per_s",
    ),
)
CSV_FIELDS = BASE_CSV_FIELDS + tuple(
    field
    for _, prefix, unit in REPORT_METRICS
    for percentile in ("p50", "p90", "p95")
    for field in (
        f"{prefix}_{percentile}_mean_{unit}",
        f"{prefix}_{percentile}_cv",
    )
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Files produced by one deterministic report build."""

    summary_csv: Path
    report_md: Path
    plot_paths: tuple[Path, ...]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load JSON object from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _percentile_metric(
    summary: dict[str, Any],
    name: str,
    percentile: str,
    field: str,
) -> float | None:
    metric = summary["metrics"].get(name)
    if metric is None:
        return None
    return float(metric["percentiles"][percentile][field])


def _audit_valid_point(
    point: FixedBatchPoint,
    point_dir: Path,
) -> dict[str, Any]:
    if _load_json_object(point_dir / "point.json") != asdict(point):
        raise ValueError(f"{point.id} artifact differs from the explicit plan")
    status = _load_json_object(point_dir / "status.json")
    if status.get("status") != "valid":
        raise ValueError(f"{point.id} status changed during report audit")
    run_summaries = []
    for run_index in range(1, point.repetitions + 1):
        run_dir = point_dir / f"run-{run_index:02d}"
        samples = []
        for batch_index in range(1, point.measured_batches + 1):
            artifact = _load_json_object(run_dir / f"measured-{batch_index:02d}.json")
            observation = batch_observation_from_dict(artifact.get("observation"))
            samples.extend(derive_batch_metric_samples(point, observation))
        run_summaries.append(summarize_run_samples(samples))
    return summarize_point_runs(run_summaries)


def _row_for_point(
    point: FixedBatchPoint,
    point_dir: Path,
) -> dict[str, Any]:
    status_value = _load_json_object(point_dir / "status.json")
    status = status_value.get("status")
    if status not in {"valid", "failed", "unsupported"}:
        raise ValueError(f"{point.id} has an invalid retained status")
    metric_label = "P-side TTFT proxy" if point.role == "P" else "D-side TPOT proxy"
    row = dict.fromkeys(CSV_FIELDS, "")
    row.update(
        {
            "point_id": point.id,
            "role": point.role,
            "status": status,
            "status_reason": status_value.get("error", status_value.get("reason", "")),
            "metric_label": metric_label,
            "batch_size": point.batch_size,
            "hit_ratio": "" if point.hit_ratio is None else point.hit_ratio,
            "execution_mode": point.execution_mode,
            "noisy": "",
            "noisy_metrics": "",
        }
    )
    if status != "valid":
        return row
    summary = _audit_valid_point(point, point_dir)
    for metric, prefix, unit in REPORT_METRICS:
        for percentile in ("p50", "p90", "p95"):
            row[f"{prefix}_{percentile}_mean_{unit}"] = _percentile_metric(
                summary,
                metric,
                percentile,
                "mean",
            )
            row[f"{prefix}_{percentile}_cv"] = _percentile_metric(
                summary,
                metric,
                percentile,
                "cv",
            )
    row["noisy"] = bool(summary["noisy_metrics"])
    row["noisy_metrics"] = ",".join(summary["noisy_metrics"])
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _series(
    rows: list[dict[str, Any]],
    *,
    role: str,
    metric: str,
) -> list[tuple[str, list[tuple[float, float]]]]:
    selected = [
        row
        for row in rows
        if row["role"] == role
        and row["status"] == "valid"
        and row["execution_mode"] == "primary"
        and row[metric] != ""
    ]
    if role == "P":
        labels = sorted({float(row["hit_ratio"]) for row in selected})
        return [
            (
                f"hit={label:.0%}",
                sorted(
                    (
                        float(row["batch_size"]),
                        float(row[metric]),
                    )
                    for row in selected
                    if float(row["hit_ratio"]) == label
                ),
            )
            for label in labels
        ]
    return [
        (
            "D",
            sorted((float(row["batch_size"]), float(row[metric])) for row in selected),
        )
    ]


def _write_svg(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: list[tuple[str, list[tuple[float, float]]]],
) -> None:
    width = 720
    height = 420
    left, right, top, bottom = 80, 30, 50, 65
    points = [point for _, values in series for point in values]
    x_values = [point[0] for point in points] or [1.0, 16.0]
    y_values = [point[1] for point in points] or [0.0, 1.0]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        y_min, y_max = 0.0, max(1.0, y_max * 1.2)

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    colors = ("#2563eb", "#dc2626", "#059669", "#9333ea")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height - bottom}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">Batch size B</text>',
        f'<text x="18" y="{height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {height / 2})" '
        f'font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>',
    ]
    for series_index, (label, values) in enumerate(series):
        color = colors[series_index % len(colors)]
        coordinates = " ".join(
            f"{x_position(x):.2f},{y_position(y):.2f}" for x, y in values
        )
        if coordinates:
            elements.append(
                f'<polyline points="{coordinates}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
        for x, y in values:
            elements.append(
                f'<circle cx="{x_position(x):.2f}" cy="{y_position(y):.2f}" '
                f'r="4" fill="{color}"><title>B={x:g}, {y:g}</title></circle>'
            )
        elements.append(
            f'<text x="{width - right - 110}" y="{top + 18 * series_index}" '
            f'font-family="sans-serif" font-size="12" fill="{color}">'
            f"{html.escape(label)}</text>"
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    hardware: dict[str, Any],
) -> None:
    lines = [
        "# DS4 fixed-batch P/D node profile",
        "",
        "P values are P-side TTFT proxies and D values are D-side TPOT "
        "proxies; they are not client-observed 1P1D metrics.",
        "",
        "## Frozen scope",
        "",
        f"- Hardware: {hardware['gpu_count']} x {hardware['gpu_model']} "
        f"({hardware['topology']})",
        "- Model: Qwen/Qwen3.5-4B BF16, TP=1 per node",
        "- Input: fixed 12,800-token prompts",
        "- P hit ratios: 0%, 75%, and 90%; no D hit-ratio axis",
        "- Conclusions are specific to the frozen dual RTX 3090 system and "
        "runtime revisions.",
        "",
        "## Audited points",
        "",
        "| Point | Role | B | Hit | Status | Latency p50 mean ms | Noisy |",
        "| --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        hit = "-" if row["hit_ratio"] == "" else f"{float(row['hit_ratio']):.0%}"
        latency = (
            "-"
            if row["latency_p50_mean_ms"] == ""
            else f"{float(row['latency_p50_mean_ms']):.4f}"
        )
        lines.append(
            f"| {row['point_id']} | {row['role']} | {row['batch_size']} | "
            f"{hit} | {row['status']} | {latency} | {row['noisy'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![P latency](p-latency.svg)",
            "",
            "![P computed-token throughput](p-computed-token-throughput.svg)",
            "",
            "![D latency](d-latency.svg)",
            "",
            "![D output-token throughput](d-output-token-throughput.svg)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_fixed_batch_report(
    plan_path: Path,
    results_dir: Path,
    report_dir: Path,
    *,
    hardware: dict[str, Any],
) -> ReportArtifacts:
    """Recompute every result from raw observations and build the report."""
    if set(hardware) != {"gpu_count", "gpu_model", "topology"}:
        raise ValueError("hardware must contain gpu_count, gpu_model, and topology")
    if report_dir.exists():
        raise FileExistsError(f"report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True)
    points = load_fixed_batch_plan(plan_path)
    rows = [
        _row_for_point(point, results_dir / "points" / point.id) for point in points
    ]
    summary_csv = report_dir / "summary.csv"
    report_md = report_dir / "report.md"
    _write_csv(summary_csv, rows)
    plot_specs = (
        (
            "p-latency.svg",
            "P-side TTFT proxy by fixed prefill batch",
            "Iteration latency (ms)",
            "P",
            "latency_p50_mean_ms",
        ),
        (
            "p-computed-token-throughput.svg",
            "P computed-token throughput by fixed prefill batch",
            "Computed tokens/s",
            "P",
            "computed_token_throughput_p50_mean_per_s",
        ),
        (
            "d-latency.svg",
            "D-side TPOT proxy by active decode batch",
            "Steady decode iteration latency (ms)",
            "D",
            "latency_p50_mean_ms",
        ),
        (
            "d-output-token-throughput.svg",
            "D output-token throughput by active decode batch",
            "Output tokens/s",
            "D",
            "output_token_throughput_p50_mean_per_s",
        ),
    )
    plot_paths = []
    for filename, title, y_label, role, metric in plot_specs:
        plot_path = report_dir / filename
        _write_svg(
            plot_path,
            title=title,
            y_label=y_label,
            series=_series(rows, role=role, metric=metric),
        )
        plot_paths.append(plot_path)
    _write_report(report_md, rows, hardware)
    return ReportArtifacts(summary_csv, report_md, tuple(plot_paths))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 3090")
    parser.add_argument("--topology", default="1P1D TP=1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_fixed_batch_report(
        args.plan,
        args.results_dir,
        args.report_dir,
        hardware={
            "gpu_count": args.gpu_count,
            "gpu_model": args.gpu_model,
            "topology": args.topology,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
