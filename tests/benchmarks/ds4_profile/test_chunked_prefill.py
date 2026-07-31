# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from benchmarks.ds4_profile.chunked_prefill import (
    MODEL_REVISION,
    TOKENIZER_REVISION,
    ChunkedPrefillExecution,
    ChunkedPrefillPoint,
    build_engine_config,
    derive_chunked_prefill_sample,
    group_points_by_token_budget,
    load_chunked_prefill_plan,
    prepare_requests,
    run_chunked_prefill_profile,
)
from benchmarks.ds4_profile.chunked_prefill import (
    main as chunked_prefill_main,
)
from benchmarks.ds4_profile.chunked_prefill_report import (
    build_chunked_prefill_report,
)
from benchmarks.ds4_profile.chunked_prefill_runtime import (
    GroupedSubprocessRuntimeFactory,
)
from benchmarks.ds4_profile.fixed_batch import (
    BatchObservation,
    IterationRecord,
    RequestObservation,
)
from benchmarks.ds4_profile.fixed_batch_runtime import OfflineLLMRuntime

CONFIG_DIR = Path(__file__).parents[3] / "benchmarks/ds4_profile/config"
TEST_MODEL_REVISION = MODEL_REVISION
TEST_TOKENIZER_REVISION = TOKENIZER_REVISION
TEST_VLLM_COMMIT = "3" * 40


def _build_test_report(
    plan_path: Path,
    results_dir: Path,
    report_dir: Path,
    *,
    hardware: dict[str, Any],
):
    return build_chunked_prefill_report(
        plan_path,
        results_dir,
        report_dir,
        hardware=hardware,
        expected_model_revision=TEST_MODEL_REVISION,
        expected_tokenizer_revision=TEST_TOKENIZER_REVISION,
        expected_vllm_commit=TEST_VLLM_COMMIT,
    )


def _write_fake_runtime_invocation(
    engine_config: dict[str, Any],
    engine_dir: Path,
) -> None:
    (engine_dir / "runtime-invocation.json").write_text(
        json.dumps(
            {
                "command": [
                    "numactl",
                    "--physcpubind=0,2,4,6,8,10",
                    "--membind=0",
                    sys.executable,
                    "-m",
                    "benchmarks.ds4_profile.fixed_batch_runtime",
                    "--serve",
                    "--engine-config",
                    str(engine_dir / "engine-config.json"),
                    "--point-dir",
                    str(engine_dir),
                ],
                "environment": {
                    "CUDA_HOME": "/test/cuda",
                    "CUDA_VISIBLE_DEVICES": engine_config["cuda_visible_devices"],
                    "HF_HOME": "/test/hf",
                    "HF_HUB_CACHE": "/test/hf/hub",
                    "HF_HUB_OFFLINE": "1",
                    "LD_LIBRARY_PATH": "/test/cuda/lib",
                    "PATH": "/test/bin",
                    "TRANSFORMERS_OFFLINE": "1",
                    **engine_config["runtime_environment"],
                },
            }
        ),
        encoding="utf-8",
    )


class FakeGroupedRuntime:
    def __init__(
        self,
        engine_config: dict[str, Any],
        engine_dir: Path,
    ) -> None:
        self.engine_config = engine_config
        self.engine_dir = engine_dir
        self.operations: list[tuple[str, int, int]] = []
        self.warm_prefix_lengths: list[int] = []
        _write_fake_runtime_invocation(engine_config, engine_dir)

    def provenance(self) -> dict[str, Any]:
        return {
            "runner_boundary": "vllm.LLM.generate",
            "cache_reset_boundary": "vllm.LLM.reset_prefix_cache",
            "iteration_source": "enable_logging_iteration_details",
            "runtime_versions": {
                "python": "test",
                "torch": "test",
                "vllm": "test",
                "cuda": "test",
            },
            "visible_gpu_count": 1,
            "visible_gpu_model": "NVIDIA GeForce RTX 3090",
            "nvidia_driver": "test-driver",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        }

    def wait_idle(self) -> None:
        self.operations.append(("wait_idle", 0, 0))

    def reset_prefix_cache(self) -> bool:
        self.operations.append(("reset_prefix_cache", 0, 0))
        self.warm_prefix_lengths.clear()
        return True

    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation:
        assert max_tokens == 1
        assert ignore_eos is True
        prompt_length = len(prompt_token_ids[0])
        self.operations.append(("generate", len(prompt_token_ids), prompt_length))
        is_warm = len(prompt_token_ids) == 1 and prompt_length < 13_723
        if is_warm:
            self.warm_prefix_lengths.append(prompt_length)
            cached_tokens = 0
        else:
            cached_tokens = (
                self.warm_prefix_lengths[0] - 1 if self.warm_prefix_lengths else 0
            )
        total_context_tokens = len(prompt_token_ids) * (prompt_length - cached_tokens)
        iterations = []
        while total_context_tokens:
            context_tokens = min(
                self.engine_config["max_num_batched_tokens"],
                total_context_tokens,
            )
            iterations.append(
                IterationRecord(
                    elapsed_ms=1.0,
                    context_requests=len(prompt_token_ids),
                    context_tokens=context_tokens,
                    generation_requests=0,
                    generation_tokens=0,
                )
            )
            total_context_tokens -= context_tokens
        return BatchObservation(
            wall_time_ms=float(len(iterations) + 5),
            requests=tuple(
                RequestObservation(
                    request_id=f"request-{index}",
                    prompt_tokens=prompt_length,
                    cached_tokens=cached_tokens,
                    output_token_ids=(7,),
                )
                for index in range(len(prompt_token_ids))
            ),
            iterations=tuple(iterations),
        )

    def close(self) -> None:
        self.operations.append(("close", 0, 0))


def test_plan_loads_explicit_point_with_aligned_cache_tokens(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": "p-b2-hit75-budget1024",
                        "batch_size": 2,
                        "input_tokens": 13_723,
                        "hit_ratio": 0.75,
                        "output_tokens": 1,
                        "max_num_batched_tokens": 1_024,
                        "repetitions": 3,
                        "warmup_batches": 1,
                        "measured_batches": 3,
                        "execution_mode": "primary",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (point,) = load_chunked_prefill_plan(plan_path)

    assert point.id == "p-b2-hit75-budget1024"
    assert point.cached_tokens == 10_240
    assert point.computed_tokens == 3_483
    assert point.aligned_hit_ratio == 10_240 / 13_723


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_revision", "1" * 40, "model_revision"),
        ("tokenizer_revision", "2" * 40, "tokenizer_revision"),
        ("attention_backend", "FLASHINFER", "FLASH_ATTN"),
        ("p_gpu", "1", "GPU 0"),
        ("p_cpu_affinity", "1,3,5,7,9,11", "CPU affinity"),
        ("p_numa_node", 1, "NUMA node 0"),
    ),
)
def test_execution_rejects_changes_to_frozen_provenance(
    field: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "vllm_commit": "3" * 40,
        "attention_backend": "FLASH_ATTN",
        "p_gpu": "0",
        "p_cpu_affinity": "0,2,4,6,8,10",
        "p_numa_node": 0,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        ChunkedPrefillExecution(**arguments)


def test_checked_in_main_plan_is_the_exact_36_point_matrix() -> None:
    points = load_chunked_prefill_plan(
        CONFIG_DIR / "chunked-prefill-main.json",
    )

    assert len(points) == 36
    assert len({point.id for point in points}) == 36
    assert {
        (
            point.input_tokens,
            point.hit_ratio,
            point.batch_size,
            point.max_num_batched_tokens,
            point.output_tokens,
        )
        for point in points
    } == {
        (13_723, hit, batch_size, budget, 1)
        for budget in (512, 1_024, 2_048, 4_096)
        for hit in (0.0, 0.75, 0.9)
        for batch_size in (1, 2, 4)
    }
    assert all(
        (
            point.repetitions,
            point.warmup_batches,
            point.measured_batches,
            point.execution_mode,
        )
        == (3, 1, 3, "primary")
        for point in points
    )


def test_main_points_form_four_order_preserving_engine_groups() -> None:
    points = load_chunked_prefill_plan(
        CONFIG_DIR / "chunked-prefill-main.json",
    )

    groups = group_points_by_token_budget(points)

    assert [budget for budget, _ in groups] == [512, 1_024, 2_048, 4_096]
    assert [len(group) for _, group in groups] == [9, 9, 9, 9]
    assert [point.id for _, group in groups for point in group] == [
        point.id for point in points
    ]


def test_checked_in_smoke_plan_contains_only_the_required_extremes() -> None:
    points = load_chunked_prefill_plan(
        CONFIG_DIR / "chunked-prefill-smoke.json",
    )

    assert [
        (
            point.batch_size,
            point.hit_ratio,
            point.max_num_batched_tokens,
            point.repetitions,
            point.warmup_batches,
            point.measured_batches,
            point.execution_mode,
        )
        for point in points
    ] == [
        (1, 0.75, 512, 1, 1, 1, "feasibility_smoke"),
        (4, 0.0, 512, 1, 1, 1, "feasibility_smoke"),
    ]


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("duplicate_id", "unique"),
        ("unexpected_field", "unexpected"),
        ("primary_repetitions", "primary points require"),
        ("smoke_batches", "smoke points require"),
    ),
)
def test_plan_rejects_ambiguous_or_weakened_points(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    source = (
        "chunked-prefill-smoke.json"
        if case == "smoke_batches"
        else "chunked-prefill-main.json"
    )
    value = json.loads((CONFIG_DIR / source).read_text(encoding="utf-8"))
    if case == "duplicate_id":
        value["points"].append(dict(value["points"][0]))
    elif case == "unexpected_field":
        value["points"][0]["per_request_chunk_size"] = 512
    elif case == "primary_repetitions":
        value["points"][0]["repetitions"] = 1
    elif case == "smoke_batches":
        value["points"][0]["measured_batches"] = 3
    plan_path = tmp_path / "invalid-plan.json"
    plan_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_chunked_prefill_plan(plan_path)


def test_requests_are_deterministic_isolated_and_warm_exactly_one_extra_token() -> None:
    point = ChunkedPrefillPoint(
        id="p-b4-hit90-budget512",
        batch_size=4,
        input_tokens=13_723,
        hit_ratio=0.9,
        output_tokens=1,
        max_num_batched_tokens=512,
        repetitions=3,
        warmup_batches=1,
        measured_batches=3,
        execution_mode="primary",
    )

    requests = prepare_requests(point, seed=17)

    assert requests == prepare_requests(point, seed=17)
    assert len(requests) == 4
    assert all(len(request.prompt_token_ids) == 13_723 for request in requests)
    assert len({request.prompt_token_ids[:128] for request in requests}) == 4
    assert all(request.planned_cached_tokens == 12_160 for request in requests)
    assert all(
        request.warm_prompt_token_ids == request.prompt_token_ids[:12_161]
        for request in requests
    )


def test_multi_iteration_sample_sums_batch_context_latency_without_dividing_by_b() -> (
    None
):
    point = ChunkedPrefillPoint(
        id="p-b2-hit75-budget1024",
        batch_size=2,
        input_tokens=13_723,
        hit_ratio=0.75,
        output_tokens=1,
        max_num_batched_tokens=1_024,
        repetitions=3,
        warmup_batches=1,
        measured_batches=3,
        execution_mode="primary",
    )
    observation = BatchObservation(
        wall_time_ms=99.0,
        requests=tuple(
            RequestObservation(
                request_id=f"request-{index}",
                prompt_tokens=13_723,
                cached_tokens=10_240,
                output_token_ids=(7,),
            )
            for index in range(2)
        ),
        iterations=(
            IterationRecord(1.0, 2, 1_024, 0, 0),
            IterationRecord(2.0, 2, 1_024, 0, 0),
            IterationRecord(3.0, 2, 1_024, 0, 0),
            IterationRecord(4.0, 2, 1_024, 0, 0),
            IterationRecord(5.0, 2, 1_024, 0, 0),
            IterationRecord(6.0, 2, 1_024, 0, 0),
            IterationRecord(7.0, 1, 822, 0, 0),
        ),
    )

    sample = derive_chunked_prefill_sample(point, observation)

    assert sample == {
        "prefill_completion_latency_ms": 28.0,
        "context_iterations": 7.0,
        "computed_token_throughput_per_s": 248_785.7142857143,
    }


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("request_count", "request count"),
        ("prompt_length", "prompt length"),
        ("output_length", "output length"),
        ("cached_tokens", "cached-token count"),
        ("empty_iterations", "no context iterations"),
        ("context_requests", "context request count"),
        ("budget_overflow", "batch-wide token budget"),
        ("generation_work", "only context work"),
        ("zero_latency", "positive and finite"),
        ("incomplete_work", "total context tokens"),
        ("recomputed_work", "total context tokens"),
    ),
)
def test_sample_rejects_invalid_public_observations(
    case: str,
    message: str,
) -> None:
    point = ChunkedPrefillPoint(
        id="p-b1-hit90-budget512",
        batch_size=1,
        input_tokens=13_723,
        hit_ratio=0.9,
        output_tokens=1,
        max_num_batched_tokens=512,
        repetitions=3,
        warmup_batches=1,
        measured_batches=3,
        execution_mode="primary",
    )
    requests: tuple[RequestObservation, ...] = (
        RequestObservation("request-0", 13_723, 12_160, (7,)),
    )
    iterations: tuple[IterationRecord, ...] = (
        IterationRecord(1.0, 1, 512, 0, 0),
        IterationRecord(2.0, 1, 512, 0, 0),
        IterationRecord(3.0, 1, 512, 0, 0),
        IterationRecord(4.0, 1, 27, 0, 0),
    )
    if case == "request_count":
        requests = ()
    elif case == "prompt_length":
        requests = (replace(requests[0], prompt_tokens=13_722),)
    elif case == "output_length":
        requests = (replace(requests[0], output_token_ids=()),)
    elif case == "cached_tokens":
        requests = (replace(requests[0], cached_tokens=0),)
    elif case == "empty_iterations":
        iterations = ()
    elif case == "context_requests":
        iterations = (replace(iterations[0], context_requests=2), *iterations[1:])
    elif case == "budget_overflow":
        iterations = (
            replace(iterations[0], context_tokens=513),
            replace(iterations[-1], context_tokens=26),
            *iterations[1:-1],
        )
    elif case == "generation_work":
        iterations = (
            replace(iterations[0], generation_requests=1, generation_tokens=1),
            *iterations[1:],
        )
    elif case == "zero_latency":
        iterations = (replace(iterations[0], elapsed_ms=0.0), *iterations[1:])
    elif case == "incomplete_work":
        iterations = (*iterations[:-1], replace(iterations[-1], context_tokens=26))
    elif case == "recomputed_work":
        iterations = (*iterations[:-1], replace(iterations[-1], context_tokens=28))

    with pytest.raises(ValueError, match=message):
        derive_chunked_prefill_sample(
            point,
            BatchObservation(
                wall_time_ms=99.0,
                requests=requests,
                iterations=iterations,
            ),
        )


def test_runner_reuses_one_engine_and_retains_warm_evidence_outside_samples(
    tmp_path: Path,
) -> None:
    runtimes: list[FakeGroupedRuntime] = []

    def runtime_factory(
        _budget: int,
        engine_config: dict[str, Any],
        engine_dir: Path,
    ) -> FakeGroupedRuntime:
        runtime = FakeGroupedRuntime(engine_config, engine_dir)
        runtimes.append(runtime)
        return runtime

    results_dir = tmp_path / "results"
    result = run_chunked_prefill_profile(
        CONFIG_DIR / "chunked-prefill-smoke.json",
        results_dir,
        execution=ChunkedPrefillExecution(
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            vllm_commit="3" * 40,
            attention_backend="FLASH_ATTN",
            p_gpu="0",
            p_cpu_affinity="0,2,4,6,8,10",
            p_numa_node=0,
        ),
        runtime_factory=runtime_factory,
    )

    assert result == {
        "status": "valid",
        "point_statuses": {
            "p-b1-hit75-budget512-smoke": "valid",
            "p-b4-hit0-budget512-smoke": "valid",
        },
    }
    assert len(runtimes) == 1
    runtime = runtimes[0]
    assert runtime.engine_config["enable_chunked_prefill"] is True
    assert runtime.engine_config["max_num_seqs"] == 4
    assert runtime.engine_config["max_num_batched_tokens"] == 512
    assert runtime.engine_config["max_model_len"] == 13_724
    assert runtime.operations.count(("reset_prefix_cache", 0, 0)) == 4
    assert runtime.operations.count(("generate", 1, 10_241)) == 2
    assert runtime.operations.count(("generate", 1, 13_723)) == 2
    assert runtime.operations.count(("generate", 4, 13_723)) == 2
    assert runtime.operations[-1] == ("close", 0, 0)

    hit_point = results_dir / "points/p-b1-hit75-budget512-smoke"
    warmup = json.loads((hit_point / "run-01/warmup-01.json").read_text())
    measured = json.loads((hit_point / "run-01/measured-01.json").read_text())
    summary = json.loads((hit_point / "point-summary.json").read_text())
    assert len(warmup["cache_warm_observations"]) == 1
    assert len(measured["cache_warm_observations"]) == 1
    assert summary["global_sample_count"] == 1
    assert summary["metrics"]["prefill_completion_latency_ms"]["mean"] == 7.0

    cold_point = results_dir / "points/p-b4-hit0-budget512-smoke"
    cold_measured = json.loads((cold_point / "run-01/measured-01.json").read_text())
    assert cold_measured["cache_warm_observations"] == []


def test_runner_launches_exactly_one_engine_for_each_selected_budget(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "four-engines.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": f"p-b1-hit0-budget{budget}-smoke",
                        "batch_size": 1,
                        "input_tokens": 13_723,
                        "hit_ratio": 0.0,
                        "output_tokens": 1,
                        "max_num_batched_tokens": budget,
                        "repetitions": 1,
                        "warmup_batches": 1,
                        "measured_batches": 1,
                        "execution_mode": "feasibility_smoke",
                    }
                    for budget in (512, 1_024, 2_048, 4_096)
                ],
            }
        ),
        encoding="utf-8",
    )
    runtimes: list[FakeGroupedRuntime] = []

    def runtime_factory(
        _budget: int,
        engine_config: dict[str, Any],
        engine_dir: Path,
    ) -> FakeGroupedRuntime:
        runtime = FakeGroupedRuntime(engine_config, engine_dir)
        runtimes.append(runtime)
        return runtime

    run_chunked_prefill_profile(
        plan_path,
        tmp_path / "results",
        execution=ChunkedPrefillExecution(
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            vllm_commit="3" * 40,
            attention_backend="FLASH_ATTN",
            p_gpu="0",
            p_cpu_affinity="0,2,4,6,8,10",
            p_numa_node=0,
        ),
        runtime_factory=runtime_factory,
    )

    assert [
        runtime.engine_config["max_num_batched_tokens"] for runtime in runtimes
    ] == [512, 1_024, 2_048, 4_096]
    assert all(runtime.operations[-1] == ("close", 0, 0) for runtime in runtimes)


def test_runner_retains_group_initialization_capacity_failure_for_every_point(
    tmp_path: Path,
) -> None:
    def runtime_factory(
        _budget: int,
        engine_config: dict[str, Any],
        engine_dir: Path,
    ) -> FakeGroupedRuntime:
        _write_fake_runtime_invocation(engine_config, engine_dir)
        raise RuntimeError("No available memory for the cache blocks.")

    results_dir = tmp_path / "results"
    result = run_chunked_prefill_profile(
        CONFIG_DIR / "chunked-prefill-smoke.json",
        results_dir,
        execution=ChunkedPrefillExecution(
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            vllm_commit="3" * 40,
            attention_backend="FLASH_ATTN",
            p_gpu="0",
            p_cpu_affinity="0,2,4,6,8,10",
            p_numa_node=0,
        ),
        runtime_factory=runtime_factory,
    )

    assert result["status"] == "valid_with_unsupported"
    assert set(result["point_statuses"].values()) == {"unsupported"}
    engine_failure = json.loads(
        (results_dir / "engines/budget-0512/engine-failure.json").read_text()
    )
    assert engine_failure["status"] == "unsupported"
    for point_id in result["point_statuses"]:
        status = json.loads(
            (results_dir / "points" / point_id / "status.json").read_text()
        )
        assert status["status"] == "unsupported"
        assert status["phase"] == "runtime_initialization"

    hardware = {
        "gpu_count": 2,
        "gpu_model": "NVIDIA GeForce RTX 3090",
        "topology": "P=GPU0/NUMA0 on dual RTX 3090",
    }
    _build_test_report(
        CONFIG_DIR / "chunked-prefill-smoke.json",
        results_dir,
        tmp_path / "report",
        hardware=hardware,
    )
    forged_summary = (
        results_dir / "points/p-b1-hit75-budget512-smoke" / "point-summary.json"
    )
    forged_summary.write_text(json.dumps({"status": "valid"}), encoding="utf-8")
    with pytest.raises(ValueError, match="sample or summary evidence"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "forged-initialization-summary-report",
            hardware=hardware,
        )


def test_runner_retains_cache_warm_evidence_when_target_runtime_fails(
    tmp_path: Path,
) -> None:
    class TargetFailureRuntime(FakeGroupedRuntime):
        def generate(
            self,
            prompt_token_ids: tuple[tuple[int, ...], ...],
            *,
            max_tokens: int,
            ignore_eos: bool,
        ) -> BatchObservation:
            if (
                len(prompt_token_ids) == 1
                and len(prompt_token_ids[0]) == 13_723
                and self.warm_prefix_lengths
            ):
                raise RuntimeError("target transport failed")
            return super().generate(
                prompt_token_ids,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
            )

    results_dir = tmp_path / "results"
    result = run_chunked_prefill_profile(
        CONFIG_DIR / "chunked-prefill-smoke.json",
        results_dir,
        execution=ChunkedPrefillExecution(
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            vllm_commit=TEST_VLLM_COMMIT,
            attention_backend="FLASH_ATTN",
            p_gpu="0",
            p_cpu_affinity="0,2,4,6,8,10",
            p_numa_node=0,
        ),
        runtime_factory=lambda _budget, engine_config, engine_dir: TargetFailureRuntime(
            engine_config, engine_dir
        ),
    )

    assert result["status"] == "failed"
    point_dir = results_dir / "points/p-b1-hit75-budget512-smoke"
    partial_path = point_dir / "run-01/partial-sample.json"
    partial = json.loads(partial_path.read_text())
    assert partial["failure_stage"] == "target_runtime"
    assert len(partial["cache_warm_observations"]) == 1
    assert partial["observation"] is None

    hardware = {
        "gpu_count": 2,
        "gpu_model": "NVIDIA GeForce RTX 3090",
        "topology": "P=GPU0/NUMA0 on dual RTX 3090",
    }
    _build_test_report(
        CONFIG_DIR / "chunked-prefill-smoke.json",
        results_dir,
        tmp_path / "report",
        hardware=hardware,
    )

    status_path = point_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["completed_repetitions"] = 1
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds the repetition plan"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "extra-repetition-report",
            hardware=hardware,
        )
    status["completed_repetitions"] = 0
    status_path.write_text(json.dumps(status), encoding="utf-8")

    partial["cache_warm_observations"] = []
    partial_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError, match="cache-warm observation count"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "missing-partial-warm-report",
            hardware=hardware,
        )


def test_grouped_runtime_factory_starts_one_point_independent_engine_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[dict[str, Any], Path]] = []
    sentinel = object()

    def fake_engine_client(
        engine_config: dict[str, Any],
        engine_dir: Path,
    ) -> object:
        captured.append((engine_config, engine_dir))
        return sentinel

    monkeypatch.setattr(
        "benchmarks.ds4_profile.chunked_prefill_runtime.SubprocessOfflineEngine",
        fake_engine_client,
    )
    engine_config = {"max_num_batched_tokens": 1_024}
    engine_dir = tmp_path / "budget-1024"

    runtime = GroupedSubprocessRuntimeFactory()(
        1_024,
        engine_config,
        engine_dir,
    )

    assert runtime is sentinel
    assert captured == [(engine_config, engine_dir)]


def test_report_reaudits_raw_observations_and_fails_closed_on_tampering(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    run_chunked_prefill_profile(
        CONFIG_DIR / "chunked-prefill-smoke.json",
        results_dir,
        execution=ChunkedPrefillExecution(
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            vllm_commit="3" * 40,
            attention_backend="FLASH_ATTN",
            p_gpu="0",
            p_cpu_affinity="0,2,4,6,8,10",
            p_numa_node=0,
        ),
        runtime_factory=lambda _budget, engine_config, engine_dir: FakeGroupedRuntime(
            engine_config, engine_dir
        ),
    )
    hardware = {
        "gpu_count": 2,
        "gpu_model": "NVIDIA GeForce RTX 3090",
        "topology": "P=GPU0/NUMA0 on dual RTX 3090",
    }

    artifacts = _build_test_report(
        CONFIG_DIR / "chunked-prefill-smoke.json",
        results_dir,
        tmp_path / "report",
        hardware=hardware,
    )

    with artifacts.summary_csv.open(newline="", encoding="utf-8") as file:
        rows = {row["point_id"]: row for row in csv.DictReader(file)}
    hit_row = rows["p-b1-hit75-budget512-smoke"]
    assert float(hit_row["planned_cached_tokens"]) == 10_240
    assert float(hit_row["aligned_hit_ratio"]) == 10_240 / 13_723
    assert float(hit_row["prefill_completion_p50_mean_ms"]) == 7.0
    report_text = artifacts.report_md.read_text(encoding="utf-8")
    assert "13,723-token" in report_text
    assert "P-engine-local" in report_text
    assert "client-observed 1P1D TTFT" in report_text
    assert "cache preparation" in report_text
    assert {path.name for path in artifacts.plot_paths} == {
        "p-prefill-completion-latency.svg",
        "p-context-iterations.svg",
        "p-computed-token-throughput.svg",
    }
    with pytest.raises(ValueError, match="dual RTX 3090"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "wrong-hardware-report",
            hardware={
                "gpu_count": 1,
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "topology": "single GPU",
            },
        )

    cold_dir = results_dir / "points/p-b4-hit0-budget512-smoke"
    cold_status_path = cold_dir / "status.json"
    root_status_path = results_dir / "status.json"
    original_cold_status = json.loads(cold_status_path.read_text())
    original_root_status = json.loads(root_status_path.read_text())
    forged_failure = {
        "status": "failed",
        "phase": "point_summary",
        "batch": 99,
        "error_type": "RuntimeError",
        "error": "synthetic report failure",
    }
    (cold_dir / "point-failure.json").write_text(
        json.dumps(forged_failure),
        encoding="utf-8",
    )
    cold_status_path.write_text(
        json.dumps(
            {
                **forged_failure,
                "completed_repetitions": 1,
                "required_repetitions": 1,
            }
        ),
        encoding="utf-8",
    )
    root_status_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "point_statuses": {
                    "p-b1-hit75-budget512-smoke": "valid",
                    "p-b4-hit0-budget512-smoke": "failed",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="point-summary failure position"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "forged-point-summary-position-report",
            hardware=hardware,
        )
    (cold_dir / "point-failure.json").unlink()
    cold_status_path.write_text(json.dumps(original_cold_status), encoding="utf-8")
    root_status_path.write_text(json.dumps(original_root_status), encoding="utf-8")

    point_dir = results_dir / "points/p-b1-hit75-budget512-smoke"
    point_summary_path = point_dir / "point-summary.json"
    point_summary = json.loads(point_summary_path.read_text())
    point_summary["metrics"]["prefill_completion_latency_ms"]["mean"] = 999.0
    point_summary_path.write_text(json.dumps(point_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="stored point summary"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "tampered-summary-report",
            hardware=hardware,
        )

    point_summary["metrics"]["prefill_completion_latency_ms"]["mean"] = 7.0
    point_summary_path.write_text(json.dumps(point_summary), encoding="utf-8")
    provenance_path = results_dir / "engines/budget-0512/provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["runtime"]["runner_boundary"] = "vllm.private.scheduler"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="runner boundary"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "tampered-provenance-report",
            hardware=hardware,
        )
    provenance["runtime"]["runner_boundary"] = "vllm.LLM.generate"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    resolved_path = results_dir / "resolved-plan.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["execution"]["vllm_commit"] = "4" * 40
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    engine_config_path = results_dir / "engines/budget-0512/engine-config.json"
    engine_config = json.loads(engine_config_path.read_text())
    engine_config["vllm_commit"] = "4" * 40
    engine_config_path.write_text(json.dumps(engine_config), encoding="utf-8")
    provenance["execution"]["vllm_commit"] = "4" * 40
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="expected vLLM commit"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "tampered-revision-report",
            hardware=hardware,
        )
    resolved["execution"]["vllm_commit"] = TEST_VLLM_COMMIT
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    engine_config["vllm_commit"] = TEST_VLLM_COMMIT
    engine_config_path.write_text(json.dumps(engine_config), encoding="utf-8")
    provenance["execution"]["vllm_commit"] = TEST_VLLM_COMMIT
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    warm_path = point_dir / "run-01/measured-01.json"
    warm_artifact = json.loads(warm_path.read_text())
    warm_iteration = warm_artifact["cache_warm_observations"][0]["iterations"][0]
    warm_iteration["generation_requests"] = 1
    warm_path.write_text(json.dumps(warm_artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="cache warm.*context-only"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "tampered-warm-report",
            hardware=hardware,
        )
    warm_iteration["generation_requests"] = 0
    warm_path.write_text(json.dumps(warm_artifact), encoding="utf-8")

    measured_path = point_dir / "run-01/measured-01.json"
    measured = json.loads(measured_path.read_text())
    measured["observation"]["requests"][0]["cached_tokens"] = 0
    measured_path.write_text(json.dumps(measured), encoding="utf-8")
    with pytest.raises(ValueError, match="cached-token count"):
        _build_test_report(
            CONFIG_DIR / "chunked-prefill-smoke.json",
            results_dir,
            tmp_path / "tampered-raw-report",
            hardware=hardware,
        )


def test_report_keeps_noisy_and_unsupported_points_visible(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    common = {
        "input_tokens": 13_723,
        "output_tokens": 1,
        "max_num_batched_tokens": 512,
        "repetitions": 3,
        "warmup_batches": 1,
        "measured_batches": 3,
        "execution_mode": "primary",
    }
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        **common,
                        "id": "p-b1-hit75-budget512",
                        "batch_size": 1,
                        "hit_ratio": 0.75,
                    },
                    {
                        **common,
                        "id": "p-b4-hit0-budget512",
                        "batch_size": 4,
                        "hit_ratio": 0.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    class NoisyCapacityRuntime(FakeGroupedRuntime):
        target_index = 0

        def generate(
            self,
            prompt_token_ids: tuple[tuple[int, ...], ...],
            *,
            max_tokens: int,
            ignore_eos: bool,
        ) -> BatchObservation:
            if len(prompt_token_ids) == 4:
                raise RuntimeError("CUDA out of memory")
            observation = super().generate(
                prompt_token_ids,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
            )
            if len(prompt_token_ids[0]) < 13_723:
                return observation
            run_index = self.target_index // 4
            self.target_index += 1
            multiplier = float(run_index + 1)
            return replace(
                observation,
                iterations=tuple(
                    replace(
                        iteration,
                        elapsed_ms=iteration.elapsed_ms * multiplier,
                    )
                    for iteration in observation.iterations
                ),
            )

    results_dir = tmp_path / "results"
    run_chunked_prefill_profile(
        plan_path,
        results_dir,
        execution=ChunkedPrefillExecution(
            model_revision=MODEL_REVISION,
            tokenizer_revision=TOKENIZER_REVISION,
            vllm_commit="3" * 40,
            attention_backend="FLASH_ATTN",
            p_gpu="0",
            p_cpu_affinity="0,2,4,6,8,10",
            p_numa_node=0,
        ),
        runtime_factory=lambda _budget, engine_config, engine_dir: NoisyCapacityRuntime(
            engine_config, engine_dir
        ),
    )

    report = _build_test_report(
        plan_path,
        results_dir,
        tmp_path / "report",
        hardware={
            "gpu_count": 2,
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "topology": "P=GPU0/NUMA0 on dual RTX 3090",
        },
    )

    with report.summary_csv.open(newline="", encoding="utf-8") as file:
        rows = {row["point_id"]: row for row in csv.DictReader(file)}
    assert rows["p-b1-hit75-budget512"]["noisy"] == "True"
    assert rows["p-b1-hit75-budget512"]["global_sample_count"] == "9"
    assert rows["p-b4-hit0-budget512"]["status"] == "unsupported"
    assert "unsupported" in report.report_md.read_text(encoding="utf-8")
    latency_svg = (tmp_path / "report/p-prefill-completion-latency.svg").read_text(
        encoding="utf-8"
    )
    assert "noisy" in latency_svg
    assert "unsupported" in latency_svg

    unsupported_dir = results_dir / "points/p-b4-hit0-budget512"
    failure_paths = (
        unsupported_dir / "point-failure.json",
        unsupported_dir / "status.json",
        unsupported_dir / "run-01/failure.json",
    )
    original_failures = [json.loads(path.read_text()) for path in failure_paths]
    for path, failure in zip(failure_paths, original_failures):
        forged = {**failure, "phase": "measured", "batch": 3}
        path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="sample topology"):
        _build_test_report(
            plan_path,
            results_dir,
            tmp_path / "forged-failure-topology-report",
            hardware={
                "gpu_count": 2,
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "topology": "P=GPU0/NUMA0 on dual RTX 3090",
            },
        )
    for path, failure in zip(failure_paths, original_failures):
        path.write_text(json.dumps(failure), encoding="utf-8")

    (unsupported_dir / "point-failure.json").unlink()
    with pytest.raises(ValueError, match="point-failure"):
        _build_test_report(
            plan_path,
            results_dir,
            tmp_path / "missing-failure-report",
            hardware={
                "gpu_count": 2,
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "topology": "P=GPU0/NUMA0 on dual RTX 3090",
            },
        )


def test_chunked_prefill_cli_dry_run_resolves_four_engines_without_gpu(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_dir = tmp_path / "unused-results"

    returncode = chunked_prefill_main(
        [
            "--plan",
            str(CONFIG_DIR / "chunked-prefill-main.json"),
            "--results-dir",
            str(results_dir),
            "--model-revision",
            MODEL_REVISION,
            "--tokenizer-revision",
            TOKENIZER_REVISION,
            "--attention-backend",
            "FLASH_ATTN",
            "--prefill-cpus",
            "0,2,4,6,8,10",
            "--prefill-numa-node",
            "0",
            "--dry-run",
        ]
    )

    assert returncode == 0
    resolved = json.loads(capsys.readouterr().out)
    assert len(resolved["points"]) == 36
    assert [
        config["max_num_batched_tokens"] for config in resolved["engine_configs"]
    ] == [512, 1_024, 2_048, 4_096]
    assert all(
        config["enable_chunked_prefill"] is True
        and config["max_num_seqs"] == 4
        and config["enable_prefix_caching"] is True
        for config in resolved["engine_configs"]
    )
    assert not results_dir.exists()


def test_chunked_engine_uses_only_supported_public_llm_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_kwargs: dict[str, Any] = {}

    def fake_llm(**kwargs: Any) -> SimpleNamespace:
        llm_kwargs.update(kwargs)
        return SimpleNamespace()

    fake_vllm = ModuleType("vllm")
    fake_vllm.__dict__.update(
        {
            "LLM": fake_llm,
            "SamplingParams": lambda **values: values,
            "TokensPrompt": lambda **values: values,
            "__version__": "test",
        }
    )
    fake_torch = ModuleType("torch")
    fake_torch.__dict__.update(
        {
            "__version__": "test",
            "version": SimpleNamespace(cuda="test"),
        }
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "benchmarks.ds4_profile.fixed_batch_runtime._query_nvidia_gpu",
        lambda _visible_device: {},
    )
    execution = ChunkedPrefillExecution(
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        vllm_commit="3" * 40,
        attention_backend="FLASH_ATTN",
        p_gpu="0",
        p_cpu_affinity="0,2,4,6,8,10",
        p_numa_node=0,
    )

    OfflineLLMRuntime(
        build_engine_config(1_024, execution),
        tmp_path,
    )

    assert llm_kwargs["enable_chunked_prefill"] is True
    assert llm_kwargs["max_num_batched_tokens"] == 1_024
    assert llm_kwargs["max_num_seqs"] == 4
    assert llm_kwargs["enable_prefix_caching"] is True
    assert llm_kwargs["enable_logging_iteration_details"] is True
    assert "device" not in llm_kwargs
