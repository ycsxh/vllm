# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.ds4_profile.fixed_batch import (
    BatchObservation,
    FixedBatchExecution,
    FixedBatchPoint,
    IterationRecord,
    RequestObservation,
    derive_batch_metric_samples,
    load_fixed_batch_plan,
    run_fixed_batch_profile,
)
from benchmarks.ds4_profile.fixed_batch import (
    main as fixed_batch_main,
)
from benchmarks.ds4_profile.fixed_batch_report import build_fixed_batch_report
from benchmarks.ds4_profile.fixed_batch_runtime import OfflineLLMRuntime

CONFIG_DIR = Path(__file__).parents[3] / "benchmarks/ds4_profile/config"


class FakeRuntime:
    def __init__(
        self,
        point: Any,
        engine_config: dict[str, Any],
        point_dir: Path,
    ) -> None:
        self.point = point
        self.engine_config = engine_config
        self.point_dir = point_dir
        self.operations: list[tuple[str, int]] = []
        self.profile_calls: list[str] = []

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime": "fake",
            "hardware": "cpu-contract",
            "runtime_versions": {"vllm": "test"},
            "visible_gpu_count": 1,
            "visible_gpu_model": "NVIDIA GeForce RTX 3090",
        }

    def wait_idle(self) -> None:
        self.operations.append(("wait_idle", 0))

    def reset_prefix_cache(self) -> bool:
        self.operations.append(("reset_prefix_cache", 0))
        return True

    def start_profile(self, profile_prefix: str) -> None:
        self.profile_calls.append(f"start:{profile_prefix}")

    def stop_profile(self) -> None:
        self.profile_calls.append("stop")

    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation:
        prompt_length = len(prompt_token_ids[0])
        self.operations.append(("generate", prompt_length))
        assert max_tokens == 1
        assert ignore_eos is True
        if prompt_length == 9_601:
            return BatchObservation(
                wall_time_ms=3.0,
                requests=(
                    RequestObservation(
                        request_id="warm",
                        prompt_tokens=9_601,
                        cached_tokens=0,
                        output_token_ids=(7,),
                    ),
                ),
                iterations=(
                    IterationRecord(
                        elapsed_ms=2.0,
                        context_requests=1,
                        context_tokens=9_601,
                        generation_requests=0,
                        generation_tokens=0,
                    ),
                ),
            )
        assert len(prompt_token_ids) == 2
        return BatchObservation(
            wall_time_ms=9.0,
            requests=tuple(
                RequestObservation(
                    request_id=f"target-{index}",
                    prompt_tokens=12_800,
                    cached_tokens=9_600,
                    output_token_ids=(7,),
                )
                for index in range(2)
            ),
            iterations=(
                IterationRecord(
                    elapsed_ms=8.0,
                    context_requests=2,
                    context_tokens=6_400,
                    generation_requests=0,
                    generation_tokens=0,
                ),
            ),
        )

    def close(self) -> None:
        self.operations.append(("close", 0))


class DecodeFakeRuntime(FakeRuntime):
    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation:
        self.operations.append(("generate", max_tokens))
        assert len(prompt_token_ids) == 4
        assert all(len(prompt) == 12_800 for prompt in prompt_token_ids)
        assert max_tokens == 128
        assert ignore_eos is True
        return BatchObservation(
            wall_time_ms=132.0,
            requests=tuple(
                RequestObservation(
                    request_id=f"decode-{index}",
                    prompt_tokens=12_800,
                    cached_tokens=0,
                    output_token_ids=tuple(range(128)),
                )
                for index in range(4)
            ),
            iterations=(
                IterationRecord(
                    elapsed_ms=4.0,
                    context_requests=4,
                    context_tokens=51_200,
                    generation_requests=0,
                    generation_tokens=0,
                ),
                IterationRecord(
                    elapsed_ms=2.0,
                    context_requests=0,
                    context_tokens=0,
                    generation_requests=4,
                    generation_tokens=4,
                ),
                *(
                    IterationRecord(
                        elapsed_ms=1.0,
                        context_requests=0,
                        context_tokens=0,
                        generation_requests=4,
                        generation_tokens=4,
                    )
                    for _ in range(126)
                ),
            ),
        )


class CacheMismatchRuntime(FakeRuntime):
    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation:
        observation = super().generate(
            prompt_token_ids,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
        if len(prompt_token_ids[0]) != 12_800:
            return observation
        return BatchObservation(
            wall_time_ms=observation.wall_time_ms,
            requests=tuple(
                RequestObservation(
                    request_id=request.request_id,
                    prompt_tokens=request.prompt_tokens,
                    cached_tokens=0,
                    output_token_ids=request.output_token_ids,
                )
                for request in observation.requests
            ),
            iterations=observation.iterations,
        )


class CapacityFailureRuntime(FakeRuntime):
    def __init__(
        self,
        point: Any,
        engine_config: dict[str, Any],
        point_dir: Path,
        message: str,
    ) -> None:
        super().__init__(point, engine_config, point_dir)
        self.message = message

    def generate(
        self,
        prompt_token_ids: tuple[tuple[int, ...], ...],
        *,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchObservation:
        raise RuntimeError(self.message)


def test_runner_profiles_one_exact_p_batch_and_retains_auditable_artifacts(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": "p-b2-hit75",
                        "role": "P",
                        "batch_size": 2,
                        "input_tokens": 12_800,
                        "hit_ratio": 0.75,
                        "output_tokens": 1,
                        "repetitions": 3,
                        "warmup_batches": 5,
                        "measured_batches": 10,
                        "execution_mode": "primary",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtimes: list[FakeRuntime] = []

    def runtime_factory(
        point: Any,
        engine_config: dict[str, Any],
        point_dir: Path,
    ) -> FakeRuntime:
        runtime = FakeRuntime(point, engine_config, point_dir)
        runtimes.append(runtime)
        return runtime

    results_dir = tmp_path / "results"
    result = run_fixed_batch_profile(
        plan_path,
        results_dir,
        execution=FixedBatchExecution(
            model_revision="1" * 40,
            tokenizer_revision="2" * 40,
            vllm_commit="3" * 40,
            attention_backend="FLASHINFER",
        ),
        runtime_factory=runtime_factory,
    )

    assert result["status"] == "valid"
    assert result["point_statuses"] == {"p-b2-hit75": "valid"}
    assert len(runtimes) == 1
    runtime = runtimes[0]
    assert runtime.engine_config["enable_chunked_prefill"] is False
    assert runtime.engine_config["max_num_seqs"] == 2
    assert runtime.engine_config["max_num_batched_tokens"] == 12_801
    assert runtime.operations.count(("reset_prefix_cache", 0)) == 45
    assert runtime.operations.count(("generate", 9_601)) == 90
    assert runtime.operations.count(("generate", 12_800)) == 45
    assert runtime.operations[-1] == ("close", 0)

    point_dir = results_dir / "points/p-b2-hit75"
    summary = json.loads((point_dir / "point-summary.json").read_text())
    assert summary["metrics"]["latency_ms"]["mean"] == 8.0
    assert summary["metrics"]["latency_ms"]["percentiles"]["p95"]["mean"] == 8.0
    assert summary["metrics"]["request_throughput_per_s"]["mean"] == 250.0
    assert summary["metrics"]["computed_token_throughput_per_s"]["mean"] == 800_000.0
    assert summary["noisy_metrics"] == []

    requests = json.loads((point_dir / "requests.json").read_text())
    assert all(len(request["prompt_token_ids"]) == 12_800 for request in requests)
    assert (
        requests[0]["prompt_token_ids"][:128] != requests[1]["prompt_token_ids"][:128]
    )
    assert all(request["planned_cached_tokens"] == 9_600 for request in requests)
    assert len(tuple(point_dir.glob("run-*/measured-*.json"))) == 30


def test_runner_excludes_d_setup_and_first_step_from_steady_tpot_proxy(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": "d-b4",
                        "role": "D",
                        "batch_size": 4,
                        "input_tokens": 12_800,
                        "output_tokens": 128,
                        "repetitions": 3,
                        "warmup_batches": 5,
                        "measured_batches": 10,
                        "execution_mode": "primary",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtimes: list[DecodeFakeRuntime] = []

    def runtime_factory(
        point: Any,
        engine_config: dict[str, Any],
        point_dir: Path,
    ) -> DecodeFakeRuntime:
        runtime = DecodeFakeRuntime(point, engine_config, point_dir)
        runtimes.append(runtime)
        return runtime

    results_dir = tmp_path / "results"
    result = run_fixed_batch_profile(
        plan_path,
        results_dir,
        execution=FixedBatchExecution(
            model_revision="1" * 40,
            tokenizer_revision="2" * 40,
            vllm_commit="3" * 40,
            attention_backend="FLASHINFER",
        ),
        runtime_factory=runtime_factory,
    )

    assert result["point_statuses"] == {"d-b4": "valid"}
    runtime = runtimes[0]
    assert runtime.engine_config["max_num_batched_tokens"] == 51_200
    assert runtime.operations.count(("reset_prefix_cache", 0)) == 45
    assert runtime.operations.count(("generate", 128)) == 45

    point_dir = results_dir / "points/d-b4"
    summary = json.loads((point_dir / "point-summary.json").read_text())
    assert summary["metric_label"] == "D-side TPOT proxy"
    assert summary["metrics"]["latency_ms"]["mean"] == 1.0
    assert summary["metrics"]["output_token_throughput_per_s"]["mean"] == 4_000.0
    assert summary["metrics"]["first_decode_latency_ms"]["mean"] == 2.0
    assert (
        summary["metrics"]["first_decode_output_token_throughput_per_s"]["mean"]
        == 2_000.0
    )
    measured = json.loads((point_dir / "run-01/measured-01.json").read_text())
    assert measured["setup_iteration"]["context_tokens"] == 51_200
    assert measured["first_decode_iteration"]["elapsed_ms"] == 2.0
    assert len(measured["steady_decode_iterations"]) == 126


def test_plan_requires_a_matching_successful_smoke_before_p_b16_frontier(
    tmp_path: Path,
) -> None:
    primary = {
        "id": "p-b16-hit75",
        "role": "P",
        "batch_size": 16,
        "input_tokens": 12_800,
        "hit_ratio": 0.75,
        "output_tokens": 1,
        "repetitions": 3,
        "warmup_batches": 5,
        "measured_batches": 10,
        "execution_mode": "primary",
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [primary]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matching feasibility smoke"):
        load_fixed_batch_plan(plan_path)

    smoke = {
        **primary,
        "id": "p-b16-hit75-smoke",
        "repetitions": 1,
        "warmup_batches": 1,
        "measured_batches": 1,
        "execution_mode": "feasibility_smoke",
    }
    primary["requires_supported_point"] = smoke["id"]
    plan_path.write_text(
        json.dumps({"schema_version": 1, "points": [smoke, primary]}),
        encoding="utf-8",
    )

    assert [point.id for point in load_fixed_batch_plan(plan_path)] == [
        "p-b16-hit75-smoke",
        "p-b16-hit75",
    ]


def test_runner_retains_partial_evidence_when_cache_validation_fails(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": "p-b2-hit75",
                        "role": "P",
                        "batch_size": 2,
                        "input_tokens": 12_800,
                        "hit_ratio": 0.75,
                        "output_tokens": 1,
                        "repetitions": 3,
                        "warmup_batches": 5,
                        "measured_batches": 10,
                        "execution_mode": "primary",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtimes: list[CacheMismatchRuntime] = []

    def runtime_factory(
        point: Any,
        engine_config: dict[str, Any],
        point_dir: Path,
    ) -> CacheMismatchRuntime:
        runtime = CacheMismatchRuntime(point, engine_config, point_dir)
        runtimes.append(runtime)
        return runtime

    results_dir = tmp_path / "results"
    result = run_fixed_batch_profile(
        plan_path,
        results_dir,
        execution=FixedBatchExecution(
            model_revision="1" * 40,
            tokenizer_revision="2" * 40,
            vllm_commit="3" * 40,
            attention_backend="FLASHINFER",
        ),
        runtime_factory=runtime_factory,
    )

    assert result == {
        "status": "failed",
        "point_statuses": {"p-b2-hit75": "failed"},
    }
    point_dir = results_dir / "points/p-b2-hit75"
    failure = json.loads((point_dir / "point-failure.json").read_text())
    assert failure["error_type"] == "ValueError"
    assert "cached-token count" in failure["error"]
    assert (point_dir / "requests.json").is_file()
    assert (point_dir / "provenance.json").is_file()
    assert (point_dir / "run-01/failure.json").is_file()
    invalid = json.loads((point_dir / "run-01/invalid-observation.json").read_text())
    assert invalid["requests"][0]["cached_tokens"] == 0
    assert runtimes[0].operations[-1] == ("close", 0)


@pytest.mark.parametrize(
    "message",
    (
        "No available memory for the cache blocks.",
        "KV cache is needed, which is larger than the available KV cache memory.",
    ),
)
def test_runner_classifies_verified_vllm_capacity_failures_as_unsupported(
    tmp_path: Path,
    message: str,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": "p-b1-hit0-smoke",
                        "role": "P",
                        "batch_size": 1,
                        "input_tokens": 12_800,
                        "hit_ratio": 0.0,
                        "output_tokens": 1,
                        "repetitions": 1,
                        "warmup_batches": 1,
                        "measured_batches": 1,
                        "execution_mode": "feasibility_smoke",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_fixed_batch_profile(
        plan_path,
        tmp_path / "results",
        execution=FixedBatchExecution(
            model_revision="1" * 40,
            tokenizer_revision="2" * 40,
            vllm_commit="3" * 40,
            attention_backend="FLASHINFER",
        ),
        runtime_factory=lambda point, engine_config, point_dir: CapacityFailureRuntime(
            point, engine_config, point_dir, message
        ),
    )

    assert result == {
        "status": "valid_with_unsupported",
        "point_statuses": {"p-b1-hit0-smoke": "unsupported"},
    }
    failure = json.loads(
        (tmp_path / "results/points/p-b1-hit0-smoke/point-failure.json").read_text()
    )
    assert failure["status"] == "unsupported"


def test_report_reaudits_raw_p_and_d_samples_instead_of_stored_summaries(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": "p-b2-hit75",
                        "role": "P",
                        "batch_size": 2,
                        "input_tokens": 12_800,
                        "hit_ratio": 0.75,
                        "output_tokens": 1,
                        "repetitions": 3,
                        "warmup_batches": 5,
                        "measured_batches": 10,
                        "execution_mode": "primary",
                    },
                    {
                        "id": "d-b4",
                        "role": "D",
                        "batch_size": 4,
                        "input_tokens": 12_800,
                        "output_tokens": 128,
                        "repetitions": 3,
                        "warmup_batches": 5,
                        "measured_batches": 10,
                        "execution_mode": "primary",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def runtime_factory(
        point: Any,
        engine_config: dict[str, Any],
        point_dir: Path,
    ) -> FakeRuntime:
        runtime_type = DecodeFakeRuntime if point.role == "D" else FakeRuntime
        return runtime_type(point, engine_config, point_dir)

    results_dir = tmp_path / "results"
    run_fixed_batch_profile(
        plan_path,
        results_dir,
        execution=FixedBatchExecution(
            model_revision="1" * 40,
            tokenizer_revision="2" * 40,
            vllm_commit="3" * 40,
            attention_backend="FLASHINFER",
        ),
        runtime_factory=runtime_factory,
    )
    p_summary_path = results_dir / "points/p-b2-hit75/point-summary.json"
    p_summary = json.loads(p_summary_path.read_text())
    p_summary["metrics"]["latency_ms"]["mean"] = 999.0
    p_summary_path.write_text(json.dumps(p_summary), encoding="utf-8")

    report = build_fixed_batch_report(
        plan_path,
        results_dir,
        tmp_path / "report",
        hardware={
            "gpu_count": 2,
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "topology": "1P1D TP=1",
        },
    )

    with report.summary_csv.open(newline="", encoding="utf-8") as file:
        rows = {row["point_id"]: row for row in csv.DictReader(file)}
    assert float(rows["p-b2-hit75"]["latency_p50_mean_ms"]) == 8.0
    assert float(rows["p-b2-hit75"]["latency_p95_mean_ms"]) == 8.0
    assert float(rows["d-b4"]["latency_p50_mean_ms"]) == 1.0
    assert float(rows["d-b4"]["first_decode_latency_p50_mean_ms"]) == 2.0
    assert rows["p-b2-hit75"]["metric_label"] == "P-side TTFT proxy"
    assert rows["d-b4"]["metric_label"] == "D-side TPOT proxy"

    report_text = report.report_md.read_text()
    assert "12,800-token" in report_text
    assert "dual RTX 3090" in report_text
    assert "not client-observed" in report_text
    assert "1" * 40 in report_text
    assert "2" * 40 in report_text
    assert "3" * 40 in report_text
    assert "FLASHINFER" in report_text
    assert {path.name for path in report.plot_paths} == {
        "p-latency.svg",
        "p-computed-token-throughput.svg",
        "d-latency.svg",
        "d-output-token-throughput.svg",
    }
    assert all(path.is_file() for path in report.plot_paths)

    for run_index, elapsed_ms in ((2, 16.0), (3, 24.0)):
        for batch_index in range(1, 11):
            measured_path = (
                results_dir
                / f"points/p-b2-hit75/run-{run_index:02d}"
                / f"measured-{batch_index:02d}.json"
            )
            measured = json.loads(measured_path.read_text())
            measured["observation"]["iterations"][0]["elapsed_ms"] = elapsed_ms
            measured_path.write_text(json.dumps(measured), encoding="utf-8")
    d_status_path = results_dir / "points/d-b4/status.json"
    d_status = json.loads(d_status_path.read_text())
    d_status.update({"status": "unsupported", "reason": "capacity frontier"})
    d_status_path.write_text(json.dumps(d_status), encoding="utf-8")

    build_fixed_batch_report(
        plan_path,
        results_dir,
        tmp_path / "annotated-report",
        hardware={
            "gpu_count": 2,
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "topology": "1P1D TP=1",
        },
    )
    assert "noisy" in (tmp_path / "annotated-report/p-latency.svg").read_text().lower()
    assert (
        "unsupported"
        in (tmp_path / "annotated-report/d-latency.svg").read_text().lower()
    )

    resolved_path = results_dir / "resolved-plan.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["execution"]["vllm_commit"] = "4" * 40
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        build_fixed_batch_report(
            plan_path,
            results_dir,
            tmp_path / "invalid-provenance-report",
            hardware={
                "gpu_count": 2,
                "gpu_model": "NVIDIA GeForce RTX 3090",
                "topology": "1P1D TP=1",
            },
        )


def test_offline_runtime_uses_public_llm_results_and_iteration_detail_logs(
    tmp_path: Path,
) -> None:
    iteration_log = tmp_path / "iteration-details.log"

    class FakeLLM:
        def __init__(self) -> None:
            self.profile_calls: list[str] = []

        def reset_prefix_cache(self) -> bool:
            return True

        def start_profile(self, profile_prefix: str | None = None) -> None:
            self.profile_calls.append(f"start:{profile_prefix}")

        def stop_profile(self) -> None:
            self.profile_calls.append("stop")

        def generate(self, prompts, sampling_params, *, use_tqdm):
            assert prompts == [[1, 2, 3], [4, 5, 6]]
            assert sampling_params == {
                "max_tokens": 1,
                "temperature": 0.0,
                "seed": 0,
                "ignore_eos": True,
            }
            assert use_tqdm is False
            with iteration_log.open("a", encoding="utf-8") as file:
                file.write(
                    "INFO Iteration(7): 2 context requests, 6 context tokens, "
                    "0 generation requests, 0 generation tokens, iteration "
                    "elapsed time: 4.25 ms\n"
                )
            return [
                SimpleNamespace(
                    request_id=f"request-{index}",
                    prompt_token_ids=prompt,
                    num_cached_tokens=3,
                    outputs=[SimpleNamespace(token_ids=[9])],
                )
                for index, prompt in enumerate(prompts)
            ]

    llm = FakeLLM()
    runtime = OfflineLLMRuntime(
        {
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "1" * 40,
            "tokenizer_revision": "2" * 40,
        },
        tmp_path,
        llm=llm,
        tokens_prompt_factory=list,
        sampling_params_factory=lambda **values: values,
        runtime_versions={"vllm": "test", "torch": "test"},
    )

    runtime.start_profile("p-b2")
    observation = runtime.generate(
        ((1, 2, 3), (4, 5, 6)),
        max_tokens=1,
        ignore_eos=True,
    )
    runtime.stop_profile()

    assert runtime.reset_prefix_cache() is True
    assert observation.wall_time_ms >= 0
    assert observation.requests[0].cached_tokens == 3
    assert observation.requests[1].output_token_ids == (9,)
    assert observation.iterations == (
        IterationRecord(
            elapsed_ms=4.25,
            context_requests=2,
            context_tokens=6,
            generation_requests=0,
            generation_tokens=0,
        ),
    )
    assert llm.profile_calls == ["start:p-b2", "stop"]
    assert runtime.provenance()["runtime_versions"]["vllm"] == "test"


def test_offline_runtime_passes_only_supported_public_llm_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_kwargs: dict[str, Any] = {}

    def fake_llm(**kwargs: Any) -> SimpleNamespace:
        llm_kwargs.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        "benchmarks.ds4_profile.fixed_batch_runtime._query_nvidia_gpu",
        lambda _visible_device: {},
    )
    monkeypatch.setattr("vllm.LLM", fake_llm)
    monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", "")

    OfflineLLMRuntime(
        {
            "attention_backend": "FLASH_ATTN",
            "block_size": 128,
            "cuda_visible_devices": "0",
            "device": "cuda",
            "dtype": "bfloat16",
            "enable_chunked_prefill": False,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.9,
            "kv_cache_dtype": "bfloat16",
            "language_model_only": True,
            "mamba_cache_mode": "align",
            "max_model_len": 12_801,
            "max_num_batched_tokens": 12_801,
            "max_num_seqs": 1,
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "1" * 40,
            "seed": 17,
            "tensor_parallel_size": 1,
            "tokenizer_revision": "2" * 40,
        },
        tmp_path,
    )

    assert "device" not in llm_kwargs
    assert llm_kwargs["attention_config"] == {"backend": "FLASH_ATTN"}
    assert llm_kwargs["enable_chunked_prefill"] is False


def test_diagnostic_point_profiles_only_the_target_phase_and_is_not_primary(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": "d-b4-diagnostic",
                        "role": "D",
                        "batch_size": 4,
                        "input_tokens": 12_800,
                        "output_tokens": 128,
                        "repetitions": 1,
                        "warmup_batches": 1,
                        "measured_batches": 1,
                        "execution_mode": "diagnostic",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtimes: list[DecodeFakeRuntime] = []

    def runtime_factory(
        point: Any,
        engine_config: dict[str, Any],
        point_dir: Path,
    ) -> DecodeFakeRuntime:
        runtime = DecodeFakeRuntime(point, engine_config, point_dir)
        runtimes.append(runtime)
        return runtime

    results_dir = tmp_path / "results"
    run_fixed_batch_profile(
        plan_path,
        results_dir,
        execution=FixedBatchExecution(
            model_revision="1" * 40,
            tokenizer_revision="2" * 40,
            vllm_commit="3" * 40,
            attention_backend="FLASHINFER",
        ),
        runtime_factory=runtime_factory,
    )

    runtime = runtimes[0]
    assert runtime.profile_calls == ["start:d-b4-diagnostic", "stop"]
    profiler_config = runtime.engine_config["profiler_config"]
    assert profiler_config["delay_iterations"] == 1
    assert profiler_config["max_iterations"] == 127
    assert profiler_config["torch_profiler_dir"].endswith(
        "points/d-b4-diagnostic/traces"
    )


def test_fixed_batch_cli_dry_run_resolves_the_explicit_plan_without_gpu(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {
                        "id": "d-b1-smoke",
                        "role": "D",
                        "batch_size": 1,
                        "input_tokens": 12_800,
                        "output_tokens": 128,
                        "repetitions": 1,
                        "warmup_batches": 1,
                        "measured_batches": 1,
                        "execution_mode": "feasibility_smoke",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    returncode = fixed_batch_main(
        [
            "--plan",
            str(plan_path),
            "--results-dir",
            str(tmp_path / "unused-results"),
            "--model-revision",
            "1" * 40,
            "--tokenizer-revision",
            "2" * 40,
            "--attention-backend",
            "FLASHINFER",
            "--prefill-cpus",
            "0-3",
            "--prefill-numa-node",
            "0",
            "--decode-cpus",
            "4-7",
            "--decode-numa-node",
            "1",
            "--dry-run",
        ]
    )

    assert returncode == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["points"][0]["id"] == "d-b1-smoke"
    assert resolved["engine_configs"][0]["max_num_batched_tokens"] == 12_928
    assert (
        resolved["engine_configs"][0]["max_num_batched_tokens"]
        >= resolved["engine_configs"][0]["max_model_len"]
    )
    assert resolved["engine_configs"][0]["device"] == "cuda"
    assert resolved["engine_configs"][0]["cuda_visible_devices"] == "1"
    assert not (tmp_path / "unused-results").exists()


def test_checked_in_plans_cover_smoke_main_frontier_and_diagnostics() -> None:
    smoke = load_fixed_batch_plan(CONFIG_DIR / "fixed-batch-smoke.json")
    main = load_fixed_batch_plan(CONFIG_DIR / "fixed-batch-main.json")
    frontier = load_fixed_batch_plan(CONFIG_DIR / "fixed-batch-frontier.json")
    diagnostics = load_fixed_batch_plan(CONFIG_DIR / "fixed-batch-diagnostics.json")

    assert {(point.role, point.batch_size) for point in smoke} == {
        ("P", 1),
        ("D", 1),
    }
    assert {
        (point.batch_size, point.hit_ratio) for point in main if point.role == "P"
    } == {
        *((batch_size, hit) for batch_size in (1, 2, 4, 8) for hit in (0, 0.75, 0.9)),
        (16, 0.9),
    }
    assert {point.batch_size for point in main if point.role == "D"} == {1, 2, 4, 8, 16}
    assert len(main) == 18
    assert [point.execution_mode for point in frontier] == [
        "feasibility_smoke",
        "primary",
        "feasibility_smoke",
        "primary",
    ]
    assert all(
        point.requires_supported_point is not None
        for point in frontier
        if point.execution_mode == "primary"
    )
    assert {(point.role, point.batch_size) for point in diagnostics} == {
        (role, batch_size) for role in ("P", "D") for batch_size in (1, 8, 16)
    }


def test_d_aggregation_preserves_every_steady_iteration_as_a_sample() -> None:
    point = FixedBatchPoint(
        id="d-b1",
        role="D",
        batch_size=1,
        input_tokens=12_800,
        hit_ratio=None,
        output_tokens=128,
        repetitions=3,
        warmup_batches=5,
        measured_batches=10,
        execution_mode="primary",
    )
    observation = BatchObservation(
        wall_time_ms=1_000.0,
        requests=(
            RequestObservation(
                request_id="d-0",
                prompt_tokens=12_800,
                cached_tokens=0,
                output_token_ids=tuple(range(128)),
            ),
        ),
        iterations=(
            IterationRecord(5.0, 1, 12_800, 0, 0),
            IterationRecord(4.0, 0, 0, 1, 1),
            *(IterationRecord(float(index), 0, 0, 1, 1) for index in range(1, 127)),
        ),
    )

    samples = derive_batch_metric_samples(point, observation)

    assert len(samples) == 126
    assert samples[0] == {
        "latency_ms": 1.0,
        "output_token_throughput_per_s": 1_000.0,
    }
    assert samples[-1]["latency_ms"] == 126.0
