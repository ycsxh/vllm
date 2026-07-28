# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.ds4_profile import run_pd, run_points
from benchmarks.ds4_profile.run_points import (
    UnsupportedPointError,
    build_benchmark_command,
    derive_run_result,
    execute_point,
    load_experiment_plan,
    main,
    prepare_experiment,
    prepare_point,
    run_repetition,
    summarize_point_runs,
    write_point_inputs,
)

TOKENIZER_REVISION = "1" * 40
MVP_PLAN = (
    Path(__file__).parents[3]
    / "benchmarks/ds4_profile/config/controlled-mvp-points.json"
)
PILOT_PLAN = (
    Path(__file__).parents[3]
    / "benchmarks/ds4_profile/config/selected-pilot-points.json"
)


class FakeTokenizer:
    name_or_path = "Qwen/Qwen3.5-4B"

    def __init__(self, snapshot_path: Path | None = None) -> None:
        self.init_kwargs = {"_commit_hash": TOKENIZER_REVISION}
        if snapshot_path is not None:
            self.init_kwargs["_ds4_snapshot_path"] = str(snapshot_path)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode())

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return bytes(token_ids).decode()


def _write_prepared(prepared_dir: Path) -> None:
    prepared_dir.mkdir()
    (prepared_dir / "dataset.jsonl").write_text(
        json.dumps({"prompt": "original prompt"}) + "\n", encoding="utf-8"
    )
    (prepared_dir / "rows.jsonl").write_text(
        json.dumps(
            {
                "input_tokens": len("original prompt"),
                "prompt_ids": list(b"original prompt"),
                "request_id": "data/no_think/example.traj.json#assistant-0",
                "source_path": "data/no_think/example.traj.json",
                "source_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (prepared_dir / "provenance.json").write_text(
        json.dumps(
            {
                "row_count": 1,
                "tokenizer": {
                    "model": "Qwen/Qwen3.5-4B",
                    "revision": TOKENIZER_REVISION,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _point(**overrides: Any) -> dict[str, Any]:
    point = {
        "id": "hit-75-chunk-4096-concurrency-1-output-1",
        "source_request_id": "data/no_think/example.traj.json#assistant-0",
        "hit_ratio": 0.75,
        "max_num_batched_tokens": 4096,
        "max_concurrency": 1,
        "output_tokens": 1,
        "num_prompts": 20,
        "repetitions": 3,
    }
    point.update(overrides)
    return point


def test_explicit_plan_prepares_unique_block_aligned_requests(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [_point()]}), encoding="utf-8"
    )

    points = load_experiment_plan(plan_path)
    prepared = prepare_point(
        points[0],
        prepared_dir,
        block_size=8,
        cache_alignment_tokens=16,
        tokenizer=FakeTokenizer(),
    )

    assert len(prepared.requests) == 20
    assert len({request.request_id for request in prepared.requests}) == 20
    assert all(
        request.input_tokens == len(request.prompt_ids) for request in prepared.requests
    )
    assert all(
        request.prompt_ids[:8] == request.isolation_ids for request in prepared.requests
    )
    assert all(len(request.isolation_ids) == 8 for request in prepared.requests)
    assert all(request.planned_cached_tokens % 16 == 0 for request in prepared.requests)
    assert all(
        request.planned_cached_tokens == int(request.input_tokens * 0.75 // 16) * 16
        for request in prepared.requests
    )
    assert all(
        FakeTokenizer().encode(request.warm_prefix, add_special_tokens=False)
        == request.prompt_ids[: request.planned_cached_tokens + 1]
        for request in prepared.requests
    )
    assert all(
        FakeTokenizer().encode(request.prompt, add_special_tokens=False)
        == request.prompt_ids
        for request in prepared.requests
    )


def test_checked_in_mvp_plan_is_the_six_explicit_spec_points() -> None:
    points = load_experiment_plan(MVP_PLAN)

    assert len(points) == 6
    assert points[0].id == "ttft-hit-75-chunk-4096-concurrency-1"
    assert {
        (
            point.hit_ratio,
            point.max_num_batched_tokens,
            point.max_concurrency,
            point.output_tokens,
        )
        for point in points
    } == {
        (0.0, 2048, 1, 1),
        (0.0, 4096, 1, 1),
        (0.75, 2048, 1, 1),
        (0.75, 4096, 1, 1),
        (0.75, 4096, 1, 128),
        (0.75, 4096, 4, 128),
    }
    assert all(point.num_prompts == 20 for point in points)
    assert all(point.repetitions == 3 for point in points)


def test_checked_in_pilot_plan_is_the_explicit_selected_matrix() -> None:
    points = load_experiment_plan(PILOT_PLAN)

    assert len(points) == 30
    assert all(point.num_prompts == 20 for point in points)
    assert all(point.repetitions == 3 for point in points)
    assert all(point.execution_mode == "optimized" for point in points)

    ttft_points = [point for point in points if point.output_tokens == 1]
    decode_points = [point for point in points if point.output_tokens == 128]
    long_source = "data/no_think/astropy__astropy-13236.traj.json#assistant-35"
    medium_source = "data/no_think/astropy__astropy-13236.traj.json#assistant-17"
    baseline_ttft = [
        point
        for point in ttft_points
        if point.source_request_id == long_source and point.max_concurrency == 1
    ]
    assert {
        point.hit_ratio
        for point in baseline_ttft
        if point.max_num_batched_tokens == 4096
    } == {0.0, 0.25, 0.5, 0.75, 0.85, 0.9}
    assert {
        (point.max_num_batched_tokens, point.hit_ratio)
        for point in baseline_ttft
        if point.hit_ratio in {0.0, 0.75, 0.9}
    } == {
        (chunk, hit) for chunk in (1024, 2048, 4096, 8192) for hit in (0.0, 0.75, 0.9)
    }
    assert {
        (point.max_concurrency, point.hit_ratio)
        for point in decode_points
        if point.source_request_id == medium_source
        and point.max_num_batched_tokens == 4096
    } == {
        (concurrency, hit) for concurrency in (1, 2, 4, 8) for hit in (0.0, 0.75, 0.9)
    }
    assert {
        point.source_request_id
        for point in ttft_points
        if point.hit_ratio == 0.75
        and point.max_num_batched_tokens == 4096
        and point.max_concurrency == 1
    } == {
        f"data/no_think/astropy__astropy-13236.traj.json#assistant-{index}"
        for index in (0, 17, 35)
    }
    assert {
        point.output_tokens
        for point in points
        if point.source_request_id == medium_source
        and point.hit_ratio == 0.75
        and point.max_num_batched_tokens == 4096
        and point.max_concurrency == 1
    } == {1, 32, 128}


def test_selected_hit_main_prepares_distinct_aligned_prefixes(
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    source_id = "data/no_think/astropy__astropy-13236.traj.json#assistant-35"
    prompt = "x" * 140
    (prepared_dir / "dataset.jsonl").write_text(
        json.dumps({"prompt": prompt}) + "\n",
        encoding="utf-8",
    )
    (prepared_dir / "rows.jsonl").write_text(
        json.dumps(
            {
                "request_id": source_id,
                "source_path": source_id.partition("#")[0],
                "source_sha256": "a" * 64,
                "input_tokens": len(prompt),
                "prompt_ids": list(prompt.encode()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (prepared_dir / "provenance.json").write_text(
        json.dumps(
            {
                "row_count": 1,
                "tokenizer": {
                    "model": "Qwen/Qwen3.5-4B",
                    "revision": TOKENIZER_REVISION,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hit_main = tuple(
        point
        for point in load_experiment_plan(PILOT_PLAN)
        if point.source_request_id == source_id
        and point.max_num_batched_tokens == 4096
        and point.max_concurrency == 1
        and point.output_tokens == 1
    )

    prepared = prepare_experiment(
        hit_main,
        prepared_dir,
        block_size=8,
        cache_alignment_tokens=8,
        tokenizer=FakeTokenizer(),
    )

    assert len(prepared) == 6
    assert (
        len(
            {
                tuple(request.planned_cached_tokens for request in point.requests)
                for point in prepared
            }
        )
        == 6
    )


def test_plan_allows_only_one_explicit_eager_diagnostic(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "points": [
                    _point(id="eager-a", execution_mode="eager_diagnostic"),
                    _point(id="eager-b", execution_mode="eager_diagnostic"),
                ],
                "report": {"comparisons": []},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at most one eager diagnostic"):
        load_experiment_plan(plan_path)


def test_nonzero_hit_point_must_span_an_effective_cache_page(
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [_point()]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="effective cache page"):
        prepare_point(
            load_experiment_plan(plan_path)[0],
            prepared_dir,
            block_size=8,
            cache_alignment_tokens=64,
            tokenizer=FakeTokenizer(),
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"output_tokens": 0}, "output_tokens"),
        ({"num_prompts": 19}, "num_prompts"),
        ({"num_prompts": 51}, "num_prompts"),
        ({"repetitions": 2}, "repetitions"),
        ({"max_concurrency": 0}, "max_concurrency"),
        ({"hit_ratio": 1.1}, "hit_ratio"),
        ({"execution_mode": "unknown"}, "execution_mode"),
        ({"unknown": True}, "unexpected"),
    ],
)
def test_explicit_plan_rejects_invalid_points(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [_point(**override)]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_experiment_plan(plan_path)


def test_point_preparation_fails_when_isolation_tokens_do_not_round_trip(
    tmp_path: Path,
) -> None:
    class UnstableTokenizer(FakeTokenizer):
        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            return "not-the-same-tokens"

    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [_point()]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="isolation block"):
        prepare_point(
            load_experiment_plan(plan_path)[0],
            prepared_dir,
            block_size=8,
            tokenizer=UnstableTokenizer(),
        )


def test_experiment_rejects_distinct_ratios_with_the_same_aligned_prefix(
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    _point(),
                    _point(
                        id="hit-76-chunk-4096-concurrency-1-output-1",
                        hit_ratio=0.76,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same aligned prefix"):
        prepare_experiment(
            load_experiment_plan(plan_path),
            prepared_dir,
            block_size=8,
            tokenizer=FakeTokenizer(),
        )


def _metrics(
    *,
    local_compute: int = 0,
    local_cache_hit: int = 0,
    external_kv_transfer: int = 0,
    failed_transfers: int = 0,
    failed_notifications: int = 0,
    expired_requests: int = 0,
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
            f"vllm:nixl_num_failed_transfers_total {failed_transfers}",
            f"vllm:nixl_num_failed_notifications_total {failed_notifications}",
            f"vllm:nixl_num_kv_expired_reqs_total {expired_requests}",
            "vllm:num_requests_running 0",
            "vllm:num_requests_waiting 0",
        ]
    )


def _official_result(
    *,
    output_tokens: int,
    num_prompts: int = 20,
    input_tokens: int = 23,
) -> dict[str, Any]:
    itls = (
        [[] for _ in range(num_prompts)]
        if output_tokens == 1
        else [[0.01] * (output_tokens - 1) for _ in range(num_prompts)]
    )
    return {
        "completed": num_prompts,
        "failed": 0,
        "input_lens": [input_tokens] * num_prompts,
        "output_lens": [output_tokens] * num_prompts,
        "ttfts": [0.1 + index / 1000 for index in range(num_prompts)],
        "itls": itls,
        "start_times": [float(index) for index in range(num_prompts)],
        "generated_texts": ["x"] * num_prompts,
        "errors": [""] * num_prompts,
        "output_throughput": 123.0,
    }


def test_derived_ttft_result_omits_tpot_for_one_token_outputs(
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [_point()]}), encoding="utf-8"
    )
    prepared = prepare_point(
        load_experiment_plan(plan_path)[0],
        prepared_dir,
        block_size=8,
        tokenizer=FakeTokenizer(),
    )

    derived = derive_run_result(
        prepared,
        _official_result(output_tokens=1),
        p_metrics_before=_metrics(),
        p_metrics_after=_metrics(
            local_compute=140,
            local_cache_hit=320,
            external_kv_transfer=40,
        ),
        d_metrics_before=_metrics(),
        d_metrics_after=_metrics(external_kv_transfer=460),
        block_size=8,
    )

    assert derived["status"] == "valid"
    assert derived["requested_hit_ratio"] == 0.75
    assert derived["aligned_planned_hit_ratio"] == pytest.approx(320 / 460)
    assert derived["actual_p_hit_ratio"] == pytest.approx(320 / 500)
    assert derived["p_prompt_tokens_by_source"] == {
        "external_kv_transfer": 40,
        "local_cache_hit": 320,
        "local_compute": 140,
    }
    assert derived["p50_ttft_ms"] == pytest.approx(109.5)
    assert derived["p90_ttft_ms"] == pytest.approx(117.1)
    assert derived["p95_ttft_ms"] == pytest.approx(118.05)
    assert not any("tpot" in key for key in derived)


def test_nonzero_planned_hit_rejects_zero_observed_hits(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [_point()]}), encoding="utf-8"
    )
    prepared = prepare_point(
        load_experiment_plan(plan_path)[0],
        prepared_dir,
        block_size=8,
        cache_alignment_tokens=16,
        tokenizer=FakeTokenizer(),
    )

    with pytest.raises(ValueError, match="nonzero planned P cache hit"):
        derive_run_result(
            prepared,
            _official_result(output_tokens=1),
            p_metrics_before=_metrics(),
            p_metrics_after=_metrics(local_compute=460),
            d_metrics_before=_metrics(),
            d_metrics_after=_metrics(external_kv_transfer=460),
            block_size=16,
        )


def test_derived_decode_result_uses_official_request_level_tpot_definition(
    tmp_path: Path,
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [_point(id="decode", output_tokens=128)],
            }
        ),
        encoding="utf-8",
    )
    prepared = prepare_point(
        load_experiment_plan(plan_path)[0],
        prepared_dir,
        block_size=8,
        tokenizer=FakeTokenizer(),
    )

    derived = derive_run_result(
        prepared,
        _official_result(output_tokens=128),
        p_metrics_before=_metrics(),
        p_metrics_after=_metrics(local_compute=140, local_cache_hit=320),
        d_metrics_before=_metrics(),
        d_metrics_after=_metrics(external_kv_transfer=460),
        block_size=8,
    )

    assert derived["p50_tpot_ms"] == pytest.approx(10.0)
    assert derived["p90_tpot_ms"] == pytest.approx(10.0)
    assert derived["p95_tpot_ms"] == pytest.approx(10.0)
    assert derived["p99_itl_ms"] == pytest.approx(10.0)


def test_point_summary_reports_mean_cv_and_noisy_metrics() -> None:
    runs = (
        {
            "actual_p_hit_ratio": 0.7,
            "output_throughput": 90.0,
            "p50_ttft_ms": 90.0,
        },
        {
            "actual_p_hit_ratio": 0.7,
            "output_throughput": 100.0,
            "p50_ttft_ms": 100.0,
        },
        {
            "actual_p_hit_ratio": 0.7,
            "output_throughput": 110.0,
            "p50_ttft_ms": 110.0,
        },
    )

    summary = summarize_point_runs("point-a", runs)

    assert summary["status"] == "valid"
    assert summary["point_id"] == "point-a"
    assert summary["repetitions"] == 3
    assert summary["metrics"]["p50_ttft_ms"] == {
        "values": [90.0, 100.0, 110.0],
        "mean": pytest.approx(100.0),
        "cv": pytest.approx(0.1),
        "noisy": True,
    }
    assert summary["noisy_metrics"] == [
        "output_throughput",
        "p50_ttft_ms",
    ]
    assert not any("tpot" in key for key in summary["metrics"])


@pytest.mark.parametrize(
    ("p_after", "d_after", "message"),
    [
        (_metrics(local_compute=460), _metrics(external_kv_transfer=460), "hit"),
        (
            _metrics(local_compute=140, local_cache_hit=320, failed_transfers=1),
            _metrics(external_kv_transfer=460),
            "failed",
        ),
        (
            _metrics(local_compute=140, local_cache_hit=320),
            _metrics(),
            "external",
        ),
    ],
)
def test_metric_validation_fails_closed(
    tmp_path: Path, p_after: str, d_after: str, message: str
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [_point()]}), encoding="utf-8"
    )
    prepared = prepare_point(
        load_experiment_plan(plan_path)[0],
        prepared_dir,
        block_size=8,
        tokenizer=FakeTokenizer(),
    )

    with pytest.raises(ValueError, match=message):
        derive_run_result(
            prepared,
            _official_result(output_tokens=1),
            p_metrics_before=_metrics(),
            p_metrics_after=p_after,
            d_metrics_before=_metrics(),
            d_metrics_after=d_after,
            block_size=8,
        )


class FakeRuntime:
    def __init__(
        self,
        official_result: dict[str, Any],
        *,
        benchmark_error: BaseException | None = None,
    ) -> None:
        self.official_result = official_result
        self.benchmark_error = benchmark_error
        self.events: list[tuple[str, str]] = []
        self.metric_reads = {"p": 0, "d": 0}

    def get_text(self, url: str, timeout: float) -> str:
        role = "p" if ":8100/" in url else "d"
        self.events.append(("get", role))
        read = self.metric_reads[role]
        self.metric_reads[role] += 1
        if read < 3:
            return _metrics()
        if role == "p":
            return _metrics(local_compute=140, local_cache_hit=320)
        return _metrics(external_kv_transfer=460)

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        if "reset_prefix_cache" in url:
            role = "p" if ":8100/" in url else "d"
            self.events.append(("reset", role))
            return {"success": True}
        self.events.append(("warm", "proxy"))
        return {"choices": [{"text": "x"}]}

    def run(
        self,
        command: tuple[str, ...],
        *,
        environment: dict[str, str],
        cwd: Path,
    ) -> None:
        self.events.append(("run", "benchmark"))
        if self.benchmark_error is not None:
            raise self.benchmark_error
        result_dir = Path(command[command.index("--result-dir") + 1])
        filename = command[command.index("--result-filename") + 1]
        (result_dir / filename).write_text(
            json.dumps(self.official_result), encoding="utf-8"
        )

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", str(seconds)))


class ManagedFakeRuntime(FakeRuntime):
    def __init__(self, official_result: dict[str, Any]) -> None:
        super().__init__(official_result)
        self.metric_reads = {"p": 0, "d": 0}

    def start(self, process: run_pd.ProcessSpec) -> str:
        self.events.append(("start", process.name))
        return f"handle:{process.name}"

    def wait_ready(
        self,
        name: str,
        url: str,
        timeout: float,
        handles: list[Any],
    ) -> None:
        self.events.append(("ready", name))

    def stop(self, handles: list[Any], timeout: float) -> None:
        self.events.append(("stop", ",".join(handles)))

    def get_text(self, url: str, timeout: float) -> str:
        role = "p" if ":8100/" in url else "d"
        read = self.metric_reads[role]
        self.metric_reads[role] += 1
        self.events.append(("get", role))
        if read in (3, 7, 11):
            run_index = (read - 3) // 4
            baseline = run_index * 1000
            if role == "p":
                return _metrics(
                    local_compute=baseline,
                    local_cache_hit=baseline,
                )
            return _metrics(external_kv_transfer=baseline)
        if read in (4, 8, 12):
            run_index = (read - 4) // 4
            baseline = run_index * 1000
            if role == "p":
                return _metrics(
                    local_compute=baseline + 140,
                    local_cache_hit=baseline + 320,
                )
            return _metrics(external_kv_transfer=baseline + 460)
        return _metrics()


def _prepared_point(tmp_path: Path, **point_overrides: Any):
    prepared_dir = tmp_path / "prepared"
    _write_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [_point(**point_overrides)],
            }
        ),
        encoding="utf-8",
    )
    return prepare_point(
        load_experiment_plan(plan_path)[0],
        prepared_dir,
        block_size=8,
        tokenizer=FakeTokenizer(),
    )


def test_benchmark_command_uses_only_the_official_controlled_options(
    tmp_path: Path,
) -> None:
    prepared = _prepared_point(tmp_path)
    command = build_benchmark_command(
        prepared,
        dataset_path=tmp_path / "dataset.jsonl",
        run_dir=tmp_path / "run-01",
        repo_root=tmp_path / "repo",
        tokenizer_path=tmp_path / TOKENIZER_REVISION,
    )

    assert command[:4] == (
        str(tmp_path / "repo/.venv/bin/python"),
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
    )
    assert command[4] == "serve"
    for option, value in (
        ("--dataset-name", "custom"),
        ("--custom-output-len", "1"),
        ("--max-concurrency", "1"),
        ("--num-prompts", "20"),
        ("--num-warmups", "0"),
        ("--ready-check-timeout-sec", "0"),
        ("--request-rate", "inf"),
        ("--percentile-metrics", "ttft,tpot,itl"),
        ("--metric-percentiles", "50,90,95,99"),
        ("--result-filename", "bench-result.json"),
        ("--tokenizer", str(tmp_path / TOKENIZER_REVISION)),
    ):
        assert command[command.index(option) + 1] == value
    for flag in (
        "--disable-shuffle",
        "--ignore-eos",
        "--no-oversample",
        "--save-detailed",
        "--save-result",
        "--skip-chat-template",
    ):
        assert flag in command


def test_repetition_runs_the_controlled_protocol_and_preserves_artifacts(
    tmp_path: Path,
) -> None:
    prepared = _prepared_point(tmp_path)
    point_dir = tmp_path / "point"
    dataset_path = write_point_inputs(prepared, point_dir)
    runtime = FakeRuntime(_official_result(output_tokens=1))
    run_dir = point_dir / "run-01"

    derived = run_repetition(
        prepared,
        run_dir,
        dataset_path=dataset_path,
        repo_root=tmp_path,
        tokenizer_path=tmp_path / TOKENIZER_REVISION,
        runtime_environment={"HF_HUB_OFFLINE": "1"},
        runtime=runtime,
        block_size=8,
        request_timeout=5,
    )

    assert runtime.events[:4] == [
        ("get", "p"),
        ("get", "d"),
        ("reset", "p"),
        ("reset", "d"),
    ]
    assert runtime.events[4:24] == [("warm", "proxy")] * 20
    assert runtime.events[24:] == [
        ("get", "p"),
        ("get", "d"),
        ("reset", "d"),
        ("get", "p"),
        ("get", "d"),
        ("run", "benchmark"),
        ("get", "p"),
        ("get", "d"),
    ]
    assert derived["status"] == "valid"
    assert json.loads((run_dir / "derived.json").read_text()) == derived
    assert json.loads((run_dir / "bench-result.json").read_text()) == (
        _official_result(output_tokens=1)
    )
    assert (run_dir / "p-metrics-before.txt").is_file()
    assert (run_dir / "p-metrics-after.txt").is_file()
    assert (run_dir / "d-metrics-before.txt").is_file()
    assert (run_dir / "d-metrics-after.txt").is_file()


def test_benchmark_failure_retains_partial_artifacts(tmp_path: Path) -> None:
    prepared = _prepared_point(tmp_path)
    point_dir = tmp_path / "point"
    dataset_path = write_point_inputs(prepared, point_dir)
    runtime = FakeRuntime(
        _official_result(output_tokens=1),
        benchmark_error=RuntimeError("benchmark exploded"),
    )
    run_dir = point_dir / "run-01"

    with pytest.raises(RuntimeError, match="benchmark exploded"):
        run_repetition(
            prepared,
            run_dir,
            dataset_path=dataset_path,
            repo_root=tmp_path,
            tokenizer_path=tmp_path / TOKENIZER_REVISION,
            runtime_environment={"HF_HUB_OFFLINE": "1"},
            runtime=runtime,
            block_size=8,
            request_timeout=5,
        )

    assert (run_dir / "p-metrics-before.txt").is_file()
    assert (run_dir / "d-metrics-before.txt").is_file()
    assert json.loads((run_dir / "failure.json").read_text()) == {
        "error": "benchmark exploded",
        "error_type": "RuntimeError",
        "status": "failed",
    }
    assert not (run_dir / "derived.json").exists()


def _launch_config(tmp_path: Path) -> run_pd.LaunchConfig:
    environment = {
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
    return run_pd.LaunchConfig(
        model_revision="2" * 40,
        tokenizer_revision=TOKENIZER_REVISION,
        attention_backend="FLASH_ATTN",
        prefill_cpus="0-7",
        prefill_numa_node=0,
        decode_cpus="8-15",
        decode_numa_node=1,
        run_dir=tmp_path / "unused",
        repo_root=tmp_path,
        vllm_commit="3" * 40,
        vllm_dirty=False,
        runtime_environment=environment,
        request_timeout=5,
    )


def _write_long_prepared(prepared_dir: Path) -> None:
    _write_prepared(prepared_dir)
    long_prompt = "x" * 1000
    (prepared_dir / "dataset.jsonl").write_text(
        json.dumps({"prompt": long_prompt}) + "\n", encoding="utf-8"
    )
    row = json.loads((prepared_dir / "rows.jsonl").read_text())
    row["input_tokens"] = len(long_prompt)
    row["prompt_ids"] = list(long_prompt.encode())
    (prepared_dir / "rows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def _configure_cli_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[run_pd.LaunchConfig, Path]:
    config = _launch_config(tmp_path)
    for name, value in config.runtime_environment.items():
        monkeypatch.setenv(name, value)
    tokenizer_path = tmp_path / TOKENIZER_REVISION
    tokenizer_path.mkdir()
    return config, tokenizer_path


def _point_cli_args(
    prepared_dir: Path,
    plan_path: Path,
    results_dir: Path,
    config: run_pd.LaunchConfig,
    *,
    dry_run: bool = False,
) -> list[str]:
    args = [
        "--prepared-dir",
        str(prepared_dir),
        "--plan",
        str(plan_path),
        "--results-dir",
        str(results_dir),
        "--model-revision",
        config.model_revision,
        "--tokenizer-revision",
        config.tokenizer_revision,
        "--attention-backend",
        config.attention_backend,
        "--prefill-cpus",
        config.prefill_cpus,
        "--prefill-numa-node",
        str(config.prefill_numa_node),
        "--decode-cpus",
        config.decode_cpus,
        "--decode-numa-node",
        str(config.decode_numa_node),
    ]
    if dry_run:
        args.append("--dry-run")
    return args


def test_point_owns_the_fixed_deployment_and_three_repetitions(
    tmp_path: Path,
) -> None:
    prepared = _prepared_point(tmp_path)
    runtime = ManagedFakeRuntime(_official_result(output_tokens=1))

    results = execute_point(
        prepared,
        _launch_config(tmp_path),
        tmp_path / "results",
        runtime=runtime,
        block_size=8,
        tokenizer_path=tmp_path / TOKENIZER_REVISION,
    )

    assert [result["status"] for result in results] == ["valid"] * 3
    assert [event for event in runtime.events if event[0] == "start"] == [
        ("start", "prefill"),
        ("start", "decode"),
        ("start", "proxy"),
    ]
    assert runtime.events[-1] == (
        "stop",
        "handle:prefill,handle:decode,handle:proxy",
    )
    point_dir = tmp_path / "results/points" / prepared.point.id
    server_plan = json.loads((point_dir / "server-plan.json").read_text())
    assert server_plan["compatibility"]["max_num_batched_tokens"] == 4096
    assert "--expected-remote-tokens" not in server_plan["processes"][2]["command"]
    assert (
        json.loads((point_dir / "engine-warmup.json").read_text())["request_count"] == 1
    )
    for run_number in range(1, 4):
        assert (point_dir / f"run-{run_number:02d}/derived.json").is_file()
    summary = json.loads((point_dir / "point-summary.json").read_text())
    assert summary["status"] == "valid"
    assert summary["repetitions"] == 3
    assert summary["noisy_metrics"] == []
    assert summary["metrics"]["p50_ttft_ms"]["mean"] == pytest.approx(109.5)


def test_eager_diagnostic_is_explicit_in_both_server_commands(
    tmp_path: Path,
) -> None:
    prepared = _prepared_point(
        tmp_path,
        id="eager-diagnostic",
        execution_mode="eager_diagnostic",
    )
    runtime = ManagedFakeRuntime(_official_result(output_tokens=1))

    execute_point(
        prepared,
        _launch_config(tmp_path),
        tmp_path / "results",
        runtime=runtime,
        block_size=8,
        tokenizer_path=tmp_path / TOKENIZER_REVISION,
    )

    point_dir = tmp_path / "results/points/eager-diagnostic"
    server_plan = json.loads((point_dir / "server-plan.json").read_text())
    assert server_plan["compatibility"]["enforce_eager"] is True
    assert all(
        "--enforce-eager" in process["command"]
        for process in server_plan["processes"][:2]
    )
    assert "--enforce-eager" not in server_plan["processes"][2]["command"]
    assert (
        json.loads((point_dir / "provenance.json").read_text())["execution_mode"]
        == "eager_diagnostic"
    )


def test_oom_point_is_retained_as_unsupported(tmp_path: Path) -> None:
    class OomRuntime(ManagedFakeRuntime):
        def start(self, process: run_pd.ProcessSpec) -> str:
            process.log_path.parent.mkdir(parents=True, exist_ok=True)
            process.log_path.write_text("torch.OutOfMemoryError: CUDA out of memory\n")
            return super().start(process)

        def wait_ready(
            self,
            name: str,
            url: str,
            timeout: float,
            handles: list[Any],
        ) -> None:
            raise RuntimeError(f"{name} exited before readiness")

    prepared = _prepared_point(tmp_path, id="unsupported")
    point_dir = tmp_path / "results/points/unsupported"

    with pytest.raises(UnsupportedPointError, match="before readiness"):
        execute_point(
            prepared,
            _launch_config(tmp_path),
            tmp_path / "results",
            runtime=OomRuntime(_official_result(output_tokens=1)),
            block_size=8,
            tokenizer_path=tmp_path / TOKENIZER_REVISION,
        )

    assert json.loads((point_dir / "point-failure.json").read_text()) == {
        "error": "prefill exited before readiness",
        "error_type": "RuntimeError",
        "status": "unsupported",
    }
    assert json.loads((point_dir / "status.json").read_text())["status"] == (
        "unsupported"
    )


def test_cli_dry_run_exposes_the_frozen_point_and_server_plans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_long_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [_point()]}), encoding="utf-8"
    )
    config, tokenizer_path = _configure_cli_environment(tmp_path, monkeypatch)

    status = main(
        _point_cli_args(
            prepared_dir,
            plan_path,
            tmp_path / "results",
            config,
            dry_run=True,
        ),
        tokenizer_loader=lambda model, *, revision: FakeTokenizer(tokenizer_path),
    )

    dry_run = json.loads(capsys.readouterr().out)
    assert status == 0
    assert dry_run["schema_version"] == 1
    assert dry_run["points"][0]["id"] == _point()["id"]
    assert dry_run["points"][0]["planned_cached_tokens"] == 12800
    assert (
        dry_run["points"][0]["server_plan"]["compatibility"]["max_num_batched_tokens"]
        == 4096
    )
    assert dry_run["tokenizer_path"] == str(tokenizer_path)
    assert dry_run["points"][0]["benchmark_command"][
        dry_run["points"][0]["benchmark_command"].index("--tokenizer") + 1
    ] == str(tokenizer_path)
    assert not (tmp_path / "results").exists()


def test_first_point_failure_blocks_the_remaining_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_long_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    _point(id="first"),
                    _point(id="second", max_num_batched_tokens=2048),
                ],
            }
        ),
        encoding="utf-8",
    )
    config, tokenizer_path = _configure_cli_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        run_points,
        "_git_state",
        lambda repo_root: ("3" * 40, False),
    )
    attempted = []

    def fail_first(point, *args, **kwargs):
        attempted.append(point.point.id)
        raise RuntimeError("first point failed")

    monkeypatch.setattr(run_points, "execute_point", fail_first)
    results_dir = tmp_path / "results"

    status = main(
        _point_cli_args(prepared_dir, plan_path, results_dir, config),
        tokenizer_loader=lambda model, *, revision: FakeTokenizer(tokenizer_path),
    )

    assert status == 1
    assert attempted == ["first"]
    manifest = json.loads((results_dir / "run-manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["points"] == [
        {
            "id": "first",
            "status": "failed",
            "error_type": "RuntimeError",
            "error": "first point failed",
        }
    ]


def test_later_unsupported_point_is_retained_and_matrix_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_long_prepared(prepared_dir)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    _point(id="first"),
                    _point(id="unsupported", max_num_batched_tokens=2048),
                    _point(id="third", max_num_batched_tokens=8192),
                ],
            }
        ),
        encoding="utf-8",
    )
    config, tokenizer_path = _configure_cli_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(run_points, "_git_state", lambda repo_root: ("3" * 40, False))
    attempted = []

    def execute(point, *args, **kwargs):
        attempted.append(point.point.id)
        if point.point.id == "unsupported":
            raise UnsupportedPointError("CUDA out of memory")
        return ({"status": "valid"},) * 3

    monkeypatch.setattr(run_points, "execute_point", execute)
    results_dir = tmp_path / "results"

    status = main(
        _point_cli_args(prepared_dir, plan_path, results_dir, config),
        tokenizer_loader=lambda model, *, revision: FakeTokenizer(tokenizer_path),
    )

    assert status == 0
    assert attempted == ["first", "unsupported", "third"]
    manifest = json.loads((results_dir / "run-manifest.json").read_text())
    assert manifest["status"] == "complete_with_unsupported"
    assert manifest["points"][1] == {
        "id": "unsupported",
        "status": "unsupported",
        "error_type": "UnsupportedPointError",
        "error": "CUDA out of memory",
    }
