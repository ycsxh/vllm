# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import time
from typing import Any

from benchmarks.ds4_profile.profile_spine import make_sample_row


def _create_vllm_config(config: dict[str, Any], replay: dict[str, Any]):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from vllm.config import CompilationConfig
    from vllm.engine.arg_utils import EngineArgs

    profile = config["profile"]
    max_model_len = max(
        8192,
        len(replay["execution_prompt_token_ids"])
        + len(replay["execution_completion_token_ids"])
        + 1,
    )
    engine_args = EngineArgs(
        model=config["model"]["repo_id"],
        tokenizer=config["model"]["tokenizer"],
        revision=config["model"]["revision"],
        dtype="half",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=profile["max_num_batched_tokens"],
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=False,
        skip_tokenizer_init=True,
        compilation_config=CompilationConfig(
            mode="VLLM_COMPILE",
            cudagraph_mode="FULL_AND_PIECEWISE",
            cudagraph_capture_sizes=[1],
            compile_sizes=[1, profile["prefill_chunk_tokens"]],
        ),
    )
    vllm_config = engine_args.create_engine_config()
    if vllm_config.model_config.enforce_eager:
        raise ValueError("Ticket 04 GPU runs may not enable eager execution")
    if vllm_config.use_v2_model_runner:
        raise ValueError("Ticket 04 requires the recorded V1 GPUModelRunner path")
    return vllm_config


def _initialize_executor(config: dict[str, Any], replay: dict[str, Any]):
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.core.single_type_kv_cache_manager import (
        register_all_kvcache_specs,
    )
    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    vllm_config = _create_vllm_config(config, replay)
    started = time.perf_counter()
    executor = UniProcExecutor(vllm_config)
    register_all_kvcache_specs(vllm_config)
    kv_cache_specs = executor.get_kv_cache_specs()
    available_memory = executor.determine_available_memory()
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_memory
    )
    startup_ms = (time.perf_counter() - started) * 1000

    capture_started = time.perf_counter()
    executor.initialize_from_config(kv_cache_configs)
    capture_ms = (time.perf_counter() - capture_started) * 1000
    return executor, startup_ms, capture_ms


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
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
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


def _execute_timed(executor: Any, scheduler_output: Any):
    import torch

    torch.accelerator.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    output = executor.execute_model(scheduler_output)
    if output is None:
        output = executor.sample_tokens(None)
    end_event.record()
    torch.accelerator.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1000
    return output, wall_ms, start_event.elapsed_time(end_event)


def _sampled_token_id(output: Any) -> int:
    if output is None or not output.sampled_token_ids:
        raise RuntimeError("decode execution did not return a sampled token")
    sampled = output.sampled_token_ids[0]
    if len(sampled) != 1:
        raise RuntimeError(f"expected one sampled token, found {len(sampled)}")
    return sampled[0]


def _inject_teacher_forced_token(worker: Any, req_id: str, token_id: int) -> int:
    runner = worker.model_runner
    request = runner.requests[req_id]
    if not request.output_token_ids:
        raise RuntimeError("cannot replace a sampled token before sampling")
    sampled_token_id = request.output_token_ids[-1]
    request.output_token_ids[-1] = token_id
    req_index = runner.input_batch.req_id_to_index[req_id]
    token_index = (
        runner.input_batch.num_prompt_tokens[req_index]
        + len(request.output_token_ids)
        - 1
    )
    runner.input_batch.token_ids_cpu[req_index, token_index] = token_id
    return sampled_token_id


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
) -> tuple[list[dict[str, Any]], list[str]]:
    worker = executor.driver_worker.worker
    profile = config["profile"]
    prompt_token_ids = replay["execution_prompt_token_ids"]
    scheduled_tokens = profile["prefill_chunk_tokens"]
    point_id = f"prefill-b1-t{scheduled_tokens}"
    total_steps = profile["warmup_repetitions"] + profile["measured_repetitions"]
    rows = []
    cudagraph_stats = []
    previous_req_id = None
    for index in range(total_steps):
        req_id = f"{config['run_id']}:prefill:{index}"
        scheduler_output = _new_request_output(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            block_ids=_block_ids(worker, scheduled_tokens),
            scheduled_tokens=scheduled_tokens,
            finished_req_ids={previous_req_id} if previous_req_id else set(),
        )
        output, wall_ms, cuda_ms = _execute_timed(executor, scheduler_output)
        if output is not None and output.sampled_token_ids:
            raise RuntimeError("chunked-prefill point unexpectedly sampled a token")
        phase, ordinal = _phase(index, profile["warmup_repetitions"])
        rows.append(
            make_sample_row(
                run_id=config["run_id"],
                point_id=point_id,
                role="prefill",
                phase=phase,
                ordinal=ordinal,
                scheduled_tokens=scheduled_tokens,
                prompt_tokens=len(prompt_token_ids),
                elapsed_ms=wall_ms,
                cuda_model_time_ms=cuda_ms,
                runner_boundary=plan["runner_boundary"],
            )
        )
        stats = getattr(output, "cudagraph_stats", None)
        if stats is not None:
            cudagraph_stats.append(str(stats))
        previous_req_id = req_id
    return rows, cudagraph_stats


def _run_decode(
    executor: Any,
    config: dict[str, Any],
    replay: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], int]:
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
    for chunk_index, chunk_tokens in enumerate(plan["setup_prefill_chunks"]):
        if chunk_index == 0:
            scheduler_output = _new_request_output(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                block_ids=block_ids,
                scheduled_tokens=chunk_tokens,
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
    replaced_token_id = _inject_teacher_forced_token(
        worker, req_id, completion_token_ids[0]
    )
    if replaced_token_id != setup_sampled_token_id:
        raise RuntimeError("runner state does not contain the setup sampled token")

    total_steps = profile["warmup_repetitions"] + profile["measured_repetitions"]
    rows = []
    cudagraph_stats = []
    for index in range(total_steps):
        scheduler_output = _cached_request_output(
            req_id=req_id,
            scheduled_tokens=1,
            num_computed_tokens=len(prompt_token_ids) + index,
            num_output_tokens=index + 1,
        )
        output, wall_ms, cuda_ms = _execute_timed(executor, scheduler_output)
        sampled_token_id = _sampled_token_id(output)
        injected_token_id = completion_token_ids[index + 1]
        replaced_token_id = _inject_teacher_forced_token(
            worker, req_id, injected_token_id
        )
        if replaced_token_id != sampled_token_id:
            raise RuntimeError("runner state does not contain the sampled token")
        phase, ordinal = _phase(index, profile["warmup_repetitions"])
        rows.append(
            make_sample_row(
                run_id=config["run_id"],
                point_id="decode-b1-t1",
                role="decode",
                phase=phase,
                ordinal=ordinal,
                scheduled_tokens=1,
                prompt_tokens=len(prompt_token_ids),
                elapsed_ms=wall_ms,
                sampled_token_id=sampled_token_id,
                injected_token_id=injected_token_id,
                cuda_model_time_ms=cuda_ms,
                runner_boundary=plan["runner_boundary"],
            )
        )
        stats = getattr(output, "cudagraph_stats", None)
        if stats is not None:
            cudagraph_stats.append(str(stats))
    return rows, cudagraph_stats, setup_sampled_token_id


def run_gpu_worker(
    config: dict[str, Any], replay: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    executor = None
    try:
        executor, startup_ms, capture_ms = _initialize_executor(config, replay)
        worker = executor.driver_worker.worker
        rows = _lifecycle_rows(
            config, plan, replay, startup_ms=startup_ms, capture_ms=capture_ms
        )
        if plan["role"] == "prefill":
            measured_rows, cudagraph_stats = _run_prefill(
                executor, config, replay, plan
            )
            setup_sampled_token_id = None
        else:
            measured_rows, cudagraph_stats, setup_sampled_token_id = _run_decode(
                executor, config, replay, plan
            )
        rows.extend(measured_rows)
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
            "compile_enabled": True,
            "cudagraph_enabled": True,
            "cudagraph_observations": cudagraph_stats,
            "setup_sampled_token_id": setup_sampled_token_id,
            "samples": rows,
            "status": "passed",
        }
    finally:
        if executor is not None:
            executor.shutdown()
