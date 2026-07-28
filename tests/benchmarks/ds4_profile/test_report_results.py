# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.ds4_profile.report_results import build_report
from benchmarks.ds4_profile.run_points import (
    ExperimentPoint,
    PreparedPoint,
    PreparedRequest,
    derive_run_result,
    summarize_point_runs,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metrics(
    *,
    local_compute: int = 0,
    local_cache_hit: int = 0,
    external_kv_transfer: int = 0,
) -> str:
    return "\n".join(
        [
            (
                'vllm:prompt_tokens_by_source_total{source="local_compute"} '
                f"{local_compute}"
            ),
            (
                'vllm:prompt_tokens_by_source_total{source="local_cache_hit"} '
                f"{local_cache_hit}"
            ),
            (
                "vllm:prompt_tokens_by_source_total"
                '{source="external_kv_transfer"} '
                f"{external_kv_transfer}"
            ),
            "vllm:nixl_num_failed_transfers_total 0",
            "vllm:nixl_num_failed_notifications_total 0",
            "vllm:nixl_num_kv_expired_reqs_total 0",
            "vllm:num_requests_running 0",
            "vllm:num_requests_waiting 0",
        ]
    )


def _point(point_id: str, *, hit_ratio: float = 0.75) -> dict[str, Any]:
    return {
        "id": point_id,
        "source_request_id": "data/no_think/example.traj.json#assistant-0",
        "hit_ratio": hit_ratio,
        "max_num_batched_tokens": 4096,
        "max_concurrency": 1,
        "output_tokens": 128,
        "num_prompts": 20,
        "repetitions": 3,
        "execution_mode": "optimized",
    }


def _prepared(point_value: dict[str, Any]) -> PreparedPoint:
    point = ExperimentPoint(**point_value)
    requests = tuple(
        PreparedRequest(
            request_id=f"request-{index}",
            prompt="",
            prompt_ids=list(range(23)),
            input_tokens=23,
            isolation_ids=list(range(8)),
            planned_cached_tokens=16 if point.hit_ratio else 0,
            warm_prefix=None,
        )
        for index in range(20)
    )
    return PreparedPoint(
        point=point,
        requests=requests,
        isolation_block_tokens=8,
        cache_alignment_tokens=8,
    )


def _official_result(ttft_offset: float) -> dict[str, Any]:
    return {
        "completed": 20,
        "failed": 0,
        "input_lens": [23] * 20,
        "output_lens": [128] * 20,
        "ttfts": [0.1 + ttft_offset + index / 1000 for index in range(20)],
        "itls": [[0.01] * 127 for _ in range(20)],
        "start_times": [float(index) for index in range(20)],
        "generated_texts": ["x"] * 20,
        "errors": [""] * 20,
        "output_throughput": 100.0 + ttft_offset,
    }


def _point_json(prepared: PreparedPoint) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "isolation_block_tokens": prepared.isolation_block_tokens,
        "cache_alignment_tokens": prepared.cache_alignment_tokens,
        "point": {
            "id": prepared.point.id,
            "source_request_id": prepared.point.source_request_id,
            "hit_ratio": prepared.point.hit_ratio,
            "max_num_batched_tokens": prepared.point.max_num_batched_tokens,
            "max_concurrency": prepared.point.max_concurrency,
            "output_tokens": prepared.point.output_tokens,
            "num_prompts": prepared.point.num_prompts,
            "repetitions": prepared.point.repetitions,
            "execution_mode": prepared.point.execution_mode,
        },
        "requests": [
            {
                "request_id": request.request_id,
                "input_tokens": request.input_tokens,
                "prompt_ids": request.prompt_ids,
                "isolation_ids": request.isolation_ids,
                "planned_cached_tokens": request.planned_cached_tokens,
            }
            for request in prepared.requests
        ],
        "dataset_sha256": "a" * 64,
    }


def _write_valid_point(results_dir: Path, point_value: dict[str, Any]) -> None:
    prepared = _prepared(point_value)
    point_dir = results_dir / "points" / prepared.point.id
    _write_json(point_dir / "point.json", _point_json(prepared))
    _write_json(
        point_dir / "server-plan.json",
        {
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "1" * 40,
            "tokenizer_revision": "1" * 40,
            "vllm_commit": "2" * 40,
            "vllm_dirty": False,
            "topology": {
                "prefill": {
                    "gpu": 0,
                    "cpus": "0-7",
                    "numa_node": 0,
                    "tensor_parallel_size": 1,
                },
                "decode": {
                    "gpu": 1,
                    "cpus": "8-15",
                    "numa_node": 1,
                    "tensor_parallel_size": 1,
                },
            },
            "compatibility": {
                "dtype": "bfloat16",
                "kv_cache_dtype": "bfloat16",
                "kv_cache_layout": "HND",
                "mamba_cache_mode": "align",
                "prefix_caching": True,
                "chunked_prefill": True,
                "enforce_eager": False,
            },
        },
    )
    runs = []
    for repetition, offset in enumerate((0.0, 0.01, 0.02), start=1):
        run_dir = point_dir / f"run-{repetition:02d}"
        official = _official_result(offset)
        before = _metrics()
        p_after = _metrics(local_compute=140, local_cache_hit=320)
        d_after = _metrics(external_kv_transfer=460)
        _write_json(run_dir / "bench-result.json", official)
        (run_dir / "p-metrics-before.txt").write_text(before, encoding="utf-8")
        (run_dir / "p-metrics-after.txt").write_text(p_after, encoding="utf-8")
        (run_dir / "d-metrics-before.txt").write_text(before, encoding="utf-8")
        (run_dir / "d-metrics-after.txt").write_text(d_after, encoding="utf-8")
        derived = derive_run_result(
            prepared,
            official,
            p_metrics_before=before,
            p_metrics_after=p_after,
            d_metrics_before=before,
            d_metrics_after=d_after,
            block_size=8,
        )
        _write_json(run_dir / "derived.json", derived)
        runs.append(derived)
    summary = summarize_point_runs(prepared.point.id, runs)
    _write_json(point_dir / "point-summary.json", summary)
    _write_json(
        point_dir / "status.json",
        {
            "status": "valid",
            "completed_repetitions": 3,
            "required_repetitions": 3,
            "noisy_metrics": summary["noisy_metrics"],
        },
    )


def _write_report_fixture(tmp_path: Path) -> tuple[Path, Path]:
    results_dir = tmp_path / "results"
    valid = _point("decode-valid")
    unsupported = _point("decode-unsupported", hit_ratio=0.0)
    _write_json(
        results_dir / "plan.json",
        {
            "schema_version": 2,
            "points": [valid, unsupported],
            "report": {
                "comparisons": [
                    {
                        "id": "hit-tpot",
                        "title": "Hit ratio and request-level TPOT",
                        "x": "requested_hit_ratio",
                        "metric": "p50_tpot_ms",
                        "series": [
                            {
                                "label": "chunk=4096, concurrency=1",
                                "point_ids": ["decode-unsupported", "decode-valid"],
                            }
                        ],
                    }
                ]
            },
        },
    )
    _write_json(
        results_dir / "run-manifest.json",
        {
            "schema_version": 1,
            "status": "failed",
            "vllm_commit": "2" * 40,
            "vllm_dirty": False,
            "model_revision": "1" * 40,
            "tokenizer_revision": "1" * 40,
            "points": [
                {
                    "id": "decode-valid",
                    "status": "valid",
                    "completed_repetitions": 3,
                },
                {
                    "id": "decode-unsupported",
                    "status": "unsupported",
                    "error_type": "RuntimeError",
                    "error": "CUDA out of memory",
                },
            ],
        },
    )
    _write_valid_point(results_dir, valid)
    unsupported_dir = results_dir / "points/decode-unsupported"
    _write_json(unsupported_dir / "point.json", _point_json(_prepared(unsupported)))
    _write_json(
        unsupported_dir / "point-failure.json",
        {
            "status": "unsupported",
            "error_type": "RuntimeError",
            "error": "CUDA out of memory",
        },
    )
    metadata_path = tmp_path / "report-metadata.json"
    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "hardware": {
                "gpu_count": 2,
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "interconnect": "SYS",
            },
            "limitations": [
                "Controlled burst traffic is not a production arrival model.",
                "DS4 supplies input prompts only; this is not a quality evaluation.",
            ],
        },
    )
    return results_dir, metadata_path


def test_report_reaudits_raw_evidence_and_writes_csv_markdown_and_svg(
    tmp_path: Path,
) -> None:
    results_dir, metadata_path = _write_report_fixture(tmp_path)

    artifacts = build_report(results_dir, metadata_path)

    assert artifacts.summary_csv == results_dir / "summary.csv"
    assert artifacts.report_md == results_dir / "report.md"
    assert artifacts.plot_paths == (results_dir / "plots/hit-tpot.svg",)
    with artifacts.summary_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["point_id"] for row in rows] == [
        "decode-valid",
        "decode-unsupported",
    ]
    assert rows[0]["status"] == "valid"
    assert rows[0]["ttft_p50_mean_ms"] == "119.5"
    assert rows[0]["tpot_p50_mean_ms"] == "10"
    assert rows[0]["p_local_compute_tokens_mean"] == "140"
    assert rows[0]["p_local_cache_hit_tokens_mean"] == "320"
    assert rows[0]["p_external_kv_transfer_tokens_mean"] == "0"
    assert rows[0]["d_external_kv_transfer_tokens_mean"] == "460"
    assert rows[0]["nixl_failed_transfers_delta_mean"] == "0"
    assert rows[0]["nixl_failed_notifications_delta_mean"] == "0"
    assert rows[0]["nixl_expired_requests_delta_mean"] == "0"
    assert rows[0]["noisy"] == "true"
    assert rows[1]["status"] == "unsupported"
    assert rows[1]["status_reason"] == "CUDA out of memory"

    report = artifacts.report_md.read_text(encoding="utf-8")
    assert "Qwen/Qwen3.5-4B" in report
    assert "NVIDIA GeForce RTX 3090" in report
    assert "bfloat16" in report
    assert "DS4 supplies input prompts only" in report
    assert "decode-unsupported" in report
    assert "P local compute" in report
    assert "P local cache hit" in report
    assert "P external KV" in report
    assert "D external KV" in report
    assert "NIXL failed transfers" in report
    assert "| 140 | 320 | 0 | 460 | 0 | 0 | 0 |" in report
    assert "Gate D human acceptance" in report
    assert "Legacy retirement checklist" in report
    plot = artifacts.plot_paths[0].read_text(encoding="utf-8")
    assert "decode-unsupported" in plot
    assert "unsupported" in plot
    assert "decode-valid" in plot
    assert "noisy" in plot


def test_report_fails_when_preserved_derived_result_does_not_match_raw_evidence(
    tmp_path: Path,
) -> None:
    results_dir, metadata_path = _write_report_fixture(tmp_path)
    derived_path = results_dir / "points/decode-valid/run-01/derived.json"
    derived = json.loads(derived_path.read_text())
    derived["p50_ttft_ms"] = -1
    _write_json(derived_path, derived)

    with pytest.raises(ValueError, match="does not match raw evidence"):
        build_report(results_dir, metadata_path)
