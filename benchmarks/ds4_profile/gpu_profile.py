# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import time
from dataclasses import dataclass
from typing import Any

from benchmarks.ds4_profile.profile_spine import make_sample_row

_ASYNC_TOKEN_PLACEHOLDER = -1


@dataclass(frozen=True)
class GpuRuntime:
    executor: Any
    worker: Any
    vllm_config: Any
    kv_cache_config: Any
    startup_ms: float
    capture_ms: float


def _create_vllm_config(config: dict[str, Any]):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from vllm.config import CompilationConfig
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs

    profile = config["profile"]
    runtime = config["runtime"]
    compilation = runtime["compilation"]
    engine_args = EngineArgs(
        model=config["model"]["repo_id"],
        tokenizer=config["model"]["tokenizer"],
        revision=config["model"]["revision"],
        dtype=runtime["dtype"],
        kv_cache_dtype=runtime["kv_cache_dtype"],
        tensor_parallel_size=runtime["tensor_parallel_size"],
        max_model_len=runtime["effective_max_model_len"],
        max_num_batched_tokens=profile["max_num_batched_tokens"],
        max_num_seqs=runtime["max_num_seqs"],
        block_size=profile["block_size"],
        gpu_memory_utilization=runtime["gpu_memory_utilization"],
        enable_chunked_prefill=runtime["enable_chunked_prefill"],
        enable_prefix_caching=runtime["enable_prefix_caching"],
        enforce_eager=runtime["enforce_eager"],
        seed=runtime["seed"],
        skip_tokenizer_init=runtime["skip_tokenizer_init"],
        cudagraph_metrics=True,
        compilation_config=CompilationConfig(
            mode=compilation["mode"],
            cudagraph_mode=compilation["cudagraph_mode"],
            cudagraph_capture_sizes=compilation["capture_sizes"],
            compile_sizes=compilation["compile_sizes"],
        ),
    )
    vllm_config = engine_args.create_engine_config()
    if vllm_config.model_config.enforce_eager:
        raise ValueError("Ticket 04 GPU runs may not enable eager execution")
    if vllm_config.use_v2_model_runner:
        raise ValueError("Ticket 04 requires the recorded V1 GPUModelRunner path")
    if vllm_config.compilation_config.mode != CompilationMode.VLLM_COMPILE:
        raise ValueError("Ticket 04 requires VLLM_COMPILE")
    if (
        vllm_config.compilation_config.cudagraph_mode
        != CUDAGraphMode.FULL_AND_PIECEWISE
    ):
        raise ValueError("Ticket 04 requires FULL_AND_PIECEWISE CUDA graphs")
    return vllm_config


def initialize_gpu_runtime(config: dict[str, Any]) -> GpuRuntime:
    """Initialize the low-level executor and its configured KV cache."""
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_capacity,
        get_kv_cache_configs,
    )
    from vllm.v1.core.single_type_kv_cache_manager import (
        register_all_kvcache_specs,
    )
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    vllm_config = _create_vllm_config(config)
    started = time.perf_counter()
    executor = UniProcExecutor(vllm_config)
    register_all_kvcache_specs(vllm_config)
    kv_cache_specs = executor.get_kv_cache_specs()
    available_memory = executor.determine_available_memory()
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_memory
    )
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
    groups = scheduler_kv_cache_config.kv_cache_groups
    if groups:
        vllm_config.cache_config.block_size = min(
            group.kv_cache_spec.block_size for group in groups
        )
        capacity, concurrency = get_kv_cache_capacity(
            vllm_config, scheduler_kv_cache_config
        )
        vllm_config.cache_config.kv_cache_size_tokens = capacity
        vllm_config.cache_config.kv_cache_max_concurrency = concurrency
    vllm_config.validate_block_size()
    startup_ms = (time.perf_counter() - started) * 1000

    try:
        capture_started = time.perf_counter()
        executor.initialize_from_config(kv_cache_configs)
        capture_ms = (time.perf_counter() - capture_started) * 1000
    except Exception:
        executor.shutdown()
        raise
    worker = executor.driver_worker.worker
    return GpuRuntime(
        executor=executor,
        worker=worker,
        vllm_config=vllm_config,
        kv_cache_config=scheduler_kv_cache_config,
        startup_ms=startup_ms,
        capture_ms=capture_ms,
    )


def _initialize_executor(config: dict[str, Any]):
    """Retain the Ticket 04 initialization tuple contract."""
    runtime = initialize_gpu_runtime(config)
    return runtime.executor, runtime.startup_ms, runtime.capture_ms


def _block_ids(worker: Any, token_count: int) -> tuple[list[int], ...]:
    groups = worker.model_runner.kv_cache_config.kv_cache_groups
    return tuple(
        list(range(math.ceil(token_count / group.kv_cache_spec.block_size)))
        for group in groups
    )


def _new_request_output(
    *,
    req_id: str,
    prompt_token_ids: list[int],
    block_ids: tuple[list[int], ...],
    scheduled_tokens: int,
    sampling: dict[str, Any],
    finished_req_ids: set[str] | None = None,
):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                mm_features=[],
                sampling_params=SamplingParams(**sampling),
                pooling_params=None,
                block_ids=block_ids,
                num_computed_tokens=0,
                lora_request=None,
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: scheduled_tokens},
        total_num_scheduled_tokens=scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished_req_ids or set(),
        free_encoder_mm_hashes=[],
    )


def _cached_request_output(
    *,
    req_id: str,
    scheduled_tokens: int,
    num_computed_tokens: int,
    num_output_tokens: int,
):
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        SchedulerOutput,
    )

    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[req_id],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[num_computed_tokens],
            num_output_tokens=[num_output_tokens],
        ),
        num_scheduled_tokens={req_id: scheduled_tokens},
        total_num_scheduled_tokens=scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def execute_worker_step(
    runtime: GpuRuntime, scheduler_output: Any, *, timed: bool
) -> tuple[Any, float | None, float | None]:
    """Execute one worker step, timing only ``execute_model`` when requested."""
    import torch

    torch.accelerator.synchronize()
    if not timed:
        output = runtime.executor.execute_model(scheduler_output)
        if output is None:
            output = runtime.executor.sample_tokens(None)
        torch.accelerator.synchronize()
        return output, None, None

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    output = runtime.executor.execute_model(scheduler_output)
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1000
    if output is None:
        output = runtime.executor.sample_tokens(None)
    torch.accelerator.synchronize()
    return output, wall_ms, start_event.elapsed_time(end_event)


def _execute_timed(executor: Any, scheduler_output: Any):
    worker = executor.driver_worker.worker
    runtime = GpuRuntime(
        executor=executor,
        worker=worker,
        vllm_config=worker.vllm_config,
        kv_cache_config=worker.model_runner.kv_cache_config,
        startup_ms=0.0,
        capture_ms=0.0,
    )
    return execute_worker_step(runtime, scheduler_output, timed=True)


def _sampled_token_id(output: Any) -> int:
    if output is None or not output.sampled_token_ids:
        raise RuntimeError("decode execution did not return a sampled token")
    sampled = output.sampled_token_ids[0]
    if len(sampled) != 1:
        raise RuntimeError(f"expected one sampled token, found {len(sampled)}")
    return sampled[0]


def _cudagraph_observation(output: Any) -> dict[str, int | str]:
    stats = getattr(output, "cudagraph_stats", None)
    if stats is None:
        raise RuntimeError("measured execution did not report CUDA Graph state")
    if stats.runtime_mode not in {"FULL", "PIECEWISE"}:
        raise RuntimeError(
            f"measured execution used unexpected CUDA Graph mode {stats.runtime_mode}"
        )
    return {
        "num_unpadded_tokens": stats.num_unpadded_tokens,
        "num_padded_tokens": stats.num_padded_tokens,
        "num_paddings": stats.num_paddings,
        "runtime_mode": stats.runtime_mode,
    }


def _has_sampled_tokens(output: Any) -> bool:
    return output is not None and any(output.sampled_token_ids)


def _inject_teacher_forced_token(
    worker: Any,
    req_id: str,
    sampled_token_id: int,
    injected_token_id: int,
) -> None:
    runner = worker.model_runner
    request = runner.requests[req_id]
    if not request.output_token_ids:
        raise RuntimeError("cannot replace a sampled token before sampling")
    state_token_id = request.output_token_ids[-1]
    if state_token_id not in {sampled_token_id, _ASYNC_TOKEN_PLACEHOLDER}:
        raise RuntimeError("runner state does not contain the sampled token")
    request.output_token_ids[-1] = injected_token_id
    req_index = runner.input_batch.req_id_to_index[req_id]
    token_index = (
        runner.input_batch.num_prompt_tokens[req_index]
        + len(request.output_token_ids)
        - 1
    )
    runner.input_batch.token_ids_cpu[req_index, token_index] = injected_token_id

    prev_sampled_token_ids = runner.input_batch.prev_sampled_token_ids
    if prev_sampled_token_ids is None:
        return
    if prev_sampled_token_ids.ndim != 2 or req_index >= prev_sampled_token_ids.shape[0]:
        raise RuntimeError("async sampled-token cache has an unexpected shape")
    cached_token_id = int(prev_sampled_token_ids[req_index, 0].item())
    if cached_token_id != sampled_token_id:
        raise RuntimeError(
            "async sampled-token cache does not contain the sampled token"
        )
    import torch

    with torch.inference_mode():
        prev_sampled_token_ids[req_index, 0] = injected_token_id


def _assert_teacher_forced_input(
    worker: Any, req_id: str, expected_token_id: int
) -> None:
    runner = worker.model_runner
    req_id_to_index = runner.input_batch.req_id_to_index
    if len(req_id_to_index) != 1 or req_id_to_index.get(req_id) != 0:
        raise RuntimeError("teacher-forced input verification requires batch size one")
    actual_token_id = int(runner.input_ids.gpu[0].item())
    if actual_token_id != expected_token_id:
        raise RuntimeError(
            "next model step did not consume the injected teacher-forced token"
        )


def _phase(index: int, warmup_repetitions: int) -> tuple[str, int]:
    if index < warmup_repetitions:
        return "warmup", index
    return "steady", index - warmup_repetitions


def _lifecycle_rows(
    config: dict[str, Any],
    plan: dict[str, Any],
    replay: dict[str, Any],
    startup_ms: float,
    capture_ms: float,
) -> list[dict[str, Any]]:
    point_id = (
        "decode-b1-t1"
        if plan["role"] == "decode"
        else f"prefill-b1-t{config['profile']['prefill_chunk_tokens']}"
    )
    common = {
        "run_id": config["run_id"],
        "point_id": point_id,
        "role": plan["role"],
        "scheduled_tokens": plan["scheduled_tokens_per_step"],
        "prompt_tokens": len(replay["execution_prompt_token_ids"]),
        "runner_boundary": plan["runner_boundary"],
    }
    return [
        make_sample_row(
            **common,
            phase="startup",
            ordinal=0,
            elapsed_ms=startup_ms,
        ),
        make_sample_row(
            **common,
            phase="capture",
            ordinal=0,
            elapsed_ms=capture_ms,
        ),
    ]


def _run_prefill(
    executor: Any,
    config: dict[str, Any],
    replay: dict[str, Any],
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, int | str]]:
    worker = executor.driver_worker.worker
    profile = config["profile"]
    prompt_token_ids = replay["execution_prompt_token_ids"]
    scheduled_tokens = profile["prefill_chunk_tokens"]
    point_id = f"prefill-b1-t{scheduled_tokens}"
    total_steps = profile["warmup_repetitions"] + profile["measured_repetitions"]
    cudagraph_stats = []
    previous_req_id = None
    for index in range(total_steps):
        req_id = f"{config['run_id']}:prefill:{index}"
        scheduler_output = _new_request_output(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            block_ids=_block_ids(worker, scheduled_tokens),
            scheduled_tokens=scheduled_tokens,
            sampling=config["runtime"]["sampling"],
            finished_req_ids={previous_req_id} if previous_req_id else set(),
        )
        phase, ordinal = _phase(index, profile["warmup_repetitions"])
        try:
            output, wall_ms, cuda_ms = _execute_timed(executor, scheduler_output)
            if _has_sampled_tokens(output):
                raise RuntimeError("chunked-prefill point unexpectedly sampled a token")
            observation = _cudagraph_observation(output)
        except Exception as error:
            rows.append(
                make_sample_row(
                    run_id=config["run_id"],
                    point_id=point_id,
                    role="prefill",
                    phase=phase,
                    ordinal=ordinal,
                    scheduled_tokens=scheduled_tokens,
                    prompt_tokens=len(prompt_token_ids),
                    context_tokens=scheduled_tokens,
                    elapsed_ms=None,
                    runner_boundary=plan["runner_boundary"],
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        rows.append(
            make_sample_row(
                run_id=config["run_id"],
                point_id=point_id,
                role="prefill",
                phase=phase,
                ordinal=ordinal,
                scheduled_tokens=scheduled_tokens,
                prompt_tokens=len(prompt_token_ids),
                context_tokens=scheduled_tokens,
                elapsed_ms=wall_ms,
                cuda_model_time_ms=cuda_ms,
                cudagraph_runtime_mode=str(observation["runtime_mode"]),
                runner_boundary=plan["runner_boundary"],
            )
        )
        cudagraph_stats.append(observation)
        previous_req_id = req_id
    return cudagraph_stats


def _run_decode(
    executor: Any,
    config: dict[str, Any],
    replay: dict[str, Any],
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, int | str]], int]:
    worker = executor.driver_worker.worker
    profile = config["profile"]
    prompt_token_ids = replay["execution_prompt_token_ids"]
    completion_token_ids = replay["execution_completion_token_ids"]
    req_id = f"{config['run_id']}:decode"
    block_ids = _block_ids(
        worker,
        len(prompt_token_ids)
        + profile["warmup_repetitions"]
        + profile["measured_repetitions"]
        + 1,
    )

    computed_tokens = 0
    setup_output = None
    try:
        for chunk_index, chunk_tokens in enumerate(plan["setup_prefill_chunks"]):
            if chunk_index == 0:
                scheduler_output = _new_request_output(
                    req_id=req_id,
                    prompt_token_ids=prompt_token_ids,
                    block_ids=block_ids,
                    scheduled_tokens=chunk_tokens,
                    sampling=config["runtime"]["sampling"],
                )
            else:
                scheduler_output = _cached_request_output(
                    req_id=req_id,
                    scheduled_tokens=chunk_tokens,
                    num_computed_tokens=computed_tokens,
                    num_output_tokens=0,
                )
            setup_output, _, _ = _execute_timed(executor, scheduler_output)
            computed_tokens += chunk_tokens
        setup_sampled_token_id = _sampled_token_id(setup_output)
        _inject_teacher_forced_token(
            worker,
            req_id,
            setup_sampled_token_id,
            completion_token_ids[0],
        )
    except Exception as error:
        rows.append(
            make_sample_row(
                run_id=config["run_id"],
                point_id="decode-b1-t1",
                role="decode",
                phase="setup",
                ordinal=0,
                scheduled_tokens=0,
                prompt_tokens=len(prompt_token_ids),
                context_tokens=computed_tokens,
                cached_tokens=computed_tokens,
                new_tokens=0,
                elapsed_ms=None,
                runner_boundary=plan["runner_boundary"],
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        )
        raise

    total_steps = profile["warmup_repetitions"] + profile["measured_repetitions"]
    cudagraph_stats = []
    for index in range(total_steps):
        scheduler_output = _cached_request_output(
            req_id=req_id,
            scheduled_tokens=1,
            num_computed_tokens=len(prompt_token_ids) + index,
            num_output_tokens=index + 1,
        )
        injected_token_id = completion_token_ids[index + 1]
        phase, ordinal = _phase(index, profile["warmup_repetitions"])
        cached_tokens = len(prompt_token_ids) + index
        try:
            output, wall_ms, cuda_ms = _execute_timed(executor, scheduler_output)
            _assert_teacher_forced_input(worker, req_id, completion_token_ids[index])
            sampled_token_id = _sampled_token_id(output)
            observation = _cudagraph_observation(output)
            _inject_teacher_forced_token(
                worker,
                req_id,
                sampled_token_id,
                injected_token_id,
            )
        except Exception as error:
            rows.append(
                make_sample_row(
                    run_id=config["run_id"],
                    point_id="decode-b1-t1",
                    role="decode",
                    phase=phase,
                    ordinal=ordinal,
                    scheduled_tokens=1,
                    prompt_tokens=len(prompt_token_ids),
                    context_tokens=cached_tokens + 1,
                    cached_tokens=cached_tokens,
                    elapsed_ms=None,
                    injected_token_id=injected_token_id,
                    runner_boundary=plan["runner_boundary"],
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise
        rows.append(
            make_sample_row(
                run_id=config["run_id"],
                point_id="decode-b1-t1",
                role="decode",
                phase=phase,
                ordinal=ordinal,
                scheduled_tokens=1,
                prompt_tokens=len(prompt_token_ids),
                context_tokens=cached_tokens + 1,
                cached_tokens=cached_tokens,
                elapsed_ms=wall_ms,
                sampled_token_id=sampled_token_id,
                injected_token_id=injected_token_id,
                sampled_token_discarded=True,
                cuda_model_time_ms=cuda_ms,
                cudagraph_runtime_mode=str(observation["runtime_mode"]),
                runner_boundary=plan["runner_boundary"],
            )
        )
        cudagraph_stats.append(observation)
    return cudagraph_stats, setup_sampled_token_id


def run_gpu_worker(
    config: dict[str, Any], replay: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    executor = None
    rows: list[dict[str, Any]] = []
    cudagraph_stats: list[dict[str, int | str]] = []
    compile_enabled = False
    cudagraph_enabled = False
    try:
        executor, startup_ms, capture_ms = _initialize_executor(config)
        worker = executor.driver_worker.worker
        compilation_config = worker.vllm_config.compilation_config
        compile_enabled = compilation_config.mode.name == "VLLM_COMPILE"
        cudagraph_enabled = compilation_config.cudagraph_mode.name != "NONE"
        rows.extend(
            _lifecycle_rows(
                config, plan, replay, startup_ms=startup_ms, capture_ms=capture_ms
            )
        )
        if plan["role"] == "prefill":
            cudagraph_stats = _run_prefill(executor, config, replay, plan, rows)
            setup_sampled_token_id = None
        else:
            cudagraph_stats, setup_sampled_token_id = _run_decode(
                executor, config, replay, plan, rows
            )
        runner = worker.model_runner
        return {
            "schema_version": plan["schema_version"],
            "run_id": config["run_id"],
            "hardware_validated": True,
            "role": plan["role"],
            "runner_boundary": plan["runner_boundary"],
            "model_runner_implementation": (
                f"{type(runner).__module__}.{type(runner).__qualname__}"
            ),
            "compile_enabled": compile_enabled,
            "cudagraph_enabled": cudagraph_enabled,
            "cudagraph_observations": cudagraph_stats,
            "setup_sampled_token_id": setup_sampled_token_id,
            "samples": rows,
            "status": "passed",
        }
    except Exception as error:
        point_id = (
            "decode-b1-t1"
            if plan["role"] == "decode"
            else f"prefill-b1-t{config['profile']['prefill_chunk_tokens']}"
        )
        if not rows or rows[-1]["status"] != "failed":
            failure_ordinal = sum(
                row["point_id"] == point_id and row["phase"] == "startup"
                for row in rows
            )
            rows.append(
                make_sample_row(
                    run_id=config["run_id"],
                    point_id=point_id,
                    role=plan["role"],
                    phase="startup",
                    ordinal=failure_ordinal,
                    scheduled_tokens=plan["scheduled_tokens_per_step"],
                    prompt_tokens=len(replay["execution_prompt_token_ids"]),
                    elapsed_ms=None,
                    runner_boundary=plan["runner_boundary"],
                    compile_enabled=compile_enabled,
                    cudagraph_enabled=cudagraph_enabled,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            )
        failed_row = rows[-1]
        return {
            "schema_version": plan["schema_version"],
            "run_id": config["run_id"],
            "hardware_validated": False,
            "role": plan["role"],
            "runner_boundary": plan["runner_boundary"],
            "compile_enabled": compile_enabled,
            "cudagraph_enabled": cudagraph_enabled,
            "cudagraph_observations": cudagraph_stats,
            "samples": rows,
            "status": "failed",
            "failure": {
                "error": failed_row["error"],
                "ordinal": failed_row["ordinal"],
                "phase": failed_row["phase"],
                "point_id": failed_row["point_id"],
            },
        }
    finally:
        if executor is not None:
            executor.shutdown()
