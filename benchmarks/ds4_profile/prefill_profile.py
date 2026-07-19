# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic P-side prefill workload planning."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from benchmarks.ds4_profile.profile_spine import (
    V2_SCHEMA_VERSION,
    canonical_payload_json,
    make_comparison_id,
    make_point_id,
)

CacheCondition = Literal["prefix_hit", "full_recompute"]
WorkloadFamily = Literal["homogeneous", "mixed", "exact_replay"]

CHUNK_ALGORITHM = "equal-active-cap-v1"
HOMOGENEOUS_TOKEN_ALGORITHM = "sha256-legal-pool-v1"
CANONICAL_BLOCK_SIZE = 16
CANONICAL_HOMOGENEOUS_PREFIX_TOKENS = 4096


@dataclass(frozen=True)
class PRequestPlan:
    request_key: str
    trajectory_id: str | None
    turn_index: int | None
    reasoning_mode: str | None
    prompt_token_ids: tuple[int, ...]
    context_tokens: int
    cached_tokens: int
    new_tokens: int
    token_digest: str


@dataclass(frozen=True)
class PChunkPlan:
    chunk_index: int
    scheduled_tokens_by_request: dict[str, int]


@dataclass(frozen=True)
class PPointPlan:
    point_id: str
    comparison_id: str
    workload_family: WorkloadFamily
    selector: str
    composition: str
    seed: int
    batch_size: int
    cache_condition: CacheCondition
    planner_digest: str
    requests: tuple[PRequestPlan, ...]
    chunks: tuple[PChunkPlan, ...]
    canonical_payload: dict[str, Any]


@dataclass(frozen=True)
class SchedulerBlockTableEvidence:
    chunk_index: int
    request_key: str
    block_ids_by_group: tuple[tuple[int, ...], ...]
    execution_completed: bool
    gpu_synchronized: bool


@dataclass(frozen=True)
class WorkerKvTensorGroupEvidence:
    group_index: int
    tensor_names: tuple[str, ...]
    tensor_devices: tuple[str, ...]
    tensor_shapes: tuple[tuple[int, ...], ...]
    block_axis: int
    block_dimension: int
    verified_block_ids: tuple[int, ...]
    live_cuda_tensor_proven: bool


@dataclass(frozen=True)
class PrefixPrimeEvidence:
    request_key: str
    intended_cached_tokens: int
    actual_cached_tokens: int
    prime_scheduler_outputs: tuple[SchedulerBlockTableEvidence, ...]
    measured_block_ids_by_group: tuple[tuple[int, ...], ...]
    worker_groups: tuple[WorkerKvTensorGroupEvidence, ...]
    prime_completed: bool
    prime_synchronized: bool
    hardware_validated: bool
    kv_bytes: int


@dataclass(frozen=True)
class AllocationEvidence:
    state: Literal["allocated", "out_of_capacity", "failed"]
    requested_blocks: int
    allocated_blocks: int
    allocatable_blocks: int
    requested_bytes: int
    allocated_bytes: int
    lookup_time_ms: float
    allocation_time_ms: float
    allocator_pressure_proven: bool
    clean_reset_proven: bool


@dataclass(frozen=True)
class ScheduledChunk:
    scheduler_output: Any
    expected_request_ids: tuple[str, ...]
    actual_request_ids: tuple[str, ...]
    expected_tokens_by_request: dict[str, int]
    actual_tokens_by_request: dict[str, int]
    preempted_request_ids: tuple[str, ...]
    unrelated_request_ids: tuple[str, ...]
    allocation: AllocationEvidence


class VllmSchedulerCacheAdapter:
    """Version-specific boundary around Scheduler and live KV cache state."""

    def __init__(
        self,
        scheduler: Any,
        executor: Any,
        worker: Any,
        *,
        request_factory: Callable[[str, list[int]], Any] | None = None,
        synchronize_gpu: Callable[[], None] | None = None,
        block_axes: tuple[int, ...] | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.executor = executor
        self.worker = worker
        self.request_factory = request_factory
        self.synchronize_gpu = synchronize_gpu
        self.block_axes = block_axes
        self._prime_evidence: dict[str, PrefixPrimeEvidence] = {}

    @classmethod
    def from_runtime(
        cls,
        scheduler: Any,
        executor: Any,
        worker: Any,
        *,
        block_axes: tuple[int, ...],
    ) -> "VllmSchedulerCacheAdapter":
        """Create the real adapter while keeping vLLM imports lazy."""
        import torch

        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import Request

        def make_request(request_id: str, tokens: list[int]) -> Any:
            return Request(
                request_id,
                tokens,
                SamplingParams(max_tokens=1, temperature=0.0),
                None,
            )

        return cls(
            scheduler,
            executor,
            worker,
            request_factory=make_request,
            synchronize_gpu=lambda: torch.accelerator.synchronize(
                torch.device("cuda:0")
            ),
            block_axes=block_axes,
        )

    @staticmethod
    def _request_id(request: Any) -> str:
        return str(getattr(request, "request_id", getattr(request, "req_id", request)))

    def _scheduler_request_ids(self) -> set[str]:
        requests = getattr(self.scheduler, "requests", {})
        return (
            set(requests)
            if isinstance(requests, dict)
            else {self._request_id(request) for request in requests}
        )

    def _queue_request_ids(self) -> set[str]:
        return {
            self._request_id(request)
            for name in ("running", "waiting")
            for request in getattr(self.scheduler, name, ())
        }

    @staticmethod
    def _scheduled_vectors(output: Any) -> tuple[dict[str, int], dict[str, Any]]:
        tokens = {
            str(request_id): int(count)
            for request_id, count in getattr(output, "num_scheduled_tokens", {}).items()
        }
        request_data: dict[str, Any] = {}
        for item in getattr(output, "scheduled_new_reqs", ()):
            request_data[str(item.req_id)] = item
        cached = getattr(output, "scheduled_cached_reqs", None)
        if cached is not None:
            for index, request_id in enumerate(getattr(cached, "req_ids", ())):
                request_data[str(request_id)] = (cached, index)
        return tokens, request_data

    @staticmethod
    def _block_ids(request_data: Any) -> tuple[tuple[int, ...], ...]:
        if isinstance(request_data, tuple):
            cached, index = request_data
            return tuple(
                tuple(int(block_id) for block_id in group[index])
                for group in cached.new_block_ids
            )
        return tuple(
            tuple(int(block_id) for block_id in group)
            for group in request_data.block_ids
        )

    @staticmethod
    def _computed_tokens(request_data: Any) -> int:
        if isinstance(request_data, tuple):
            cached, index = request_data
            return int(cached.num_computed_tokens[index])
        return int(request_data.num_computed_tokens)

    def _kv_page_bytes(self) -> tuple[int, ...]:
        config = getattr(self.scheduler, "kv_cache_config", None)
        groups = getattr(config, "kv_cache_groups", ())
        return tuple(int(group.kv_cache_spec.page_size_bytes) for group in groups)

    def _allocation_snapshot(
        self, expected_tokens: dict[str, int], elapsed_ms: float
    ) -> AllocationEvidence:
        manager = self.scheduler.kv_cache_manager
        block_size = int(getattr(manager, "block_size", CANONICAL_BLOCK_SIZE))
        requested = sum(
            (tokens + block_size - 1) // block_size
            for tokens in expected_tokens.values()
        )
        pool = manager.block_pool
        free = int(pool.get_num_free_blocks())
        allocated = max(0, requested - free)
        page_bytes = self._kv_page_bytes()
        block_bytes = (
            sum(page_bytes)
            if page_bytes
            else int(getattr(manager, "kv_block_bytes", 0))
        )
        return AllocationEvidence(
            state="allocated",
            requested_blocks=requested,
            allocated_blocks=allocated,
            allocatable_blocks=free,
            requested_bytes=requested * block_bytes,
            allocated_bytes=allocated * block_bytes,
            lookup_time_ms=elapsed_ms,
            allocation_time_ms=elapsed_ms,
            allocator_pressure_proven=False,
            clean_reset_proven=False,
        )

    def reset_epoch(self) -> None:
        """Abort and flush every request, then prove an empty cache epoch."""
        request_ids = self._scheduler_request_ids() | self._queue_request_ids()
        if request_ids:
            try:
                from vllm.v1.request import RequestStatus

                status = RequestStatus.FINISHED_ABORTED
            except ImportError:
                status = "finished_aborted"
            self.scheduler.finish_requests(sorted(request_ids), status)
            flush = self.scheduler.schedule()
            if getattr(flush, "total_num_scheduled_tokens", 0):
                raise RuntimeError("reset flush scheduled model work")
            model_output = self.executor.execute_model(flush)
            self.scheduler.update_from_output(flush, model_output)
        if self._scheduler_request_ids() or self._queue_request_ids():
            raise RuntimeError("reset left live Scheduler requests")
        if not self.scheduler.reset_prefix_cache():
            raise RuntimeError("prefix cache reset failed")
        pool = self.scheduler.kv_cache_manager.block_pool
        used = int(getattr(pool, "num_gpu_blocks", 0)) - int(pool.get_num_free_blocks())
        if used > 1:
            raise RuntimeError("prefix cache reset left live KV blocks")
        self._prime_evidence.clear()

    def _inspect_worker_groups(
        self,
        block_ids_by_group: tuple[tuple[int, ...], ...],
        *,
        require_hardware: bool,
    ) -> tuple[WorkerKvTensorGroupEvidence, ...]:
        import torch

        config = getattr(self.scheduler, "kv_cache_config", None)
        configured_groups = tuple(getattr(config, "kv_cache_groups", ()))
        tensors = tuple(getattr(self.worker.model_runner, "kv_caches", ()))
        axes = self.block_axes or tuple(0 for _ in configured_groups)
        if len(block_ids_by_group) != len(configured_groups) or len(axes) != len(
            configured_groups
        ):
            raise RuntimeError("KV cache group evidence is incomplete")
        offset = 0
        evidence = []
        for group_index, (group, block_ids, block_axis) in enumerate(
            zip(configured_groups, block_ids_by_group, axes)
        ):
            names = tuple(group.layer_names)
            mapped = tensors[offset : offset + len(names)]
            offset += len(names)
            if len(mapped) != len(names):
                raise RuntimeError("missing configured layer tensor")
            if not mapped or any(
                not isinstance(tensor, torch.Tensor) for tensor in mapped
            ):
                raise RuntimeError("live KV cache entry is not a torch.Tensor")
            if any(block_axis < 0 or block_axis >= tensor.ndim for tensor in mapped):
                raise RuntimeError("invalid KV cache physical block axis")
            dimensions = {int(tensor.shape[block_axis]) for tensor in mapped}
            if len(dimensions) != 1:
                raise RuntimeError("KV cache tensors disagree on block dimension")
            dimension = dimensions.pop()
            if any(block_id < 0 or block_id >= dimension for block_id in block_ids):
                raise RuntimeError("physical block ID is outside the live tensor")
            cuda0 = all(
                tensor.is_cuda and tensor.device == torch.device("cuda:0")
                for tensor in mapped
            )
            if require_hardware and not cuda0:
                raise RuntimeError("live KV cache tensor is not on cuda:0")
            evidence.append(
                WorkerKvTensorGroupEvidence(
                    group_index=group_index,
                    tensor_names=names,
                    tensor_devices=tuple(str(tensor.device) for tensor in mapped),
                    tensor_shapes=tuple(tuple(tensor.shape) for tensor in mapped),
                    block_axis=block_axis,
                    block_dimension=dimension,
                    verified_block_ids=block_ids,
                    live_cuda_tensor_proven=cuda0,
                )
            )
        if offset != len(tensors):
            raise RuntimeError("live KV cache contains unmapped layer tensors")
        return tuple(evidence)

    def prime(
        self, point: PPointPlan, phase: str, ordinal: int
    ) -> tuple[PrefixPrimeEvidence, ...]:
        """Execute and synchronize every reusable prefix outside GPU timing."""
        if point.cache_condition != "prefix_hit":
            return ()
        if self.request_factory is None:
            raise RuntimeError("prefix priming requires a real request factory")
        evidence = []
        for request in point.requests:
            prime_id = f"prime:{phase}:{ordinal}:{request.request_key}"
            prefix = list(request.prompt_token_ids[: request.cached_tokens])
            self.scheduler.add_request(self.request_factory(prime_id, prefix))
            self.scheduler.max_num_scheduled_tokens = request.cached_tokens
            self.scheduler.scheduler_config.long_prefill_token_threshold = (
                request.cached_tokens
            )
            output = self.scheduler.schedule()
            scheduled, request_data = self._scheduled_vectors(output)
            if scheduled != {prime_id: request.cached_tokens}:
                raise RuntimeError("prefix prime SchedulerOutput was partial")
            prime_data = request_data.get(prime_id)
            if prime_data is None:
                raise RuntimeError("prefix prime omitted its Scheduler block table")
            sent_block_ids = self._block_ids(prime_data)
            model_output = self.executor.execute_model(output)
            self.scheduler.update_from_output(output, model_output)
            synchronized = self.synchronize_gpu is not None
            if self.synchronize_gpu is not None:
                self.synchronize_gpu()
            measured = tuple(
                tuple(int(block_id) for block_id in group)
                for group in self.scheduler.kv_cache_manager.get_block_ids(prime_id)
            )
            intended_blocks = request.cached_tokens // CANONICAL_BLOCK_SIZE
            measured = tuple(group[:intended_blocks] for group in measured)
            if tuple(group[:intended_blocks] for group in sent_block_ids) != measured:
                raise RuntimeError("prime block table changed after GPU execution")
            worker_groups = self._inspect_worker_groups(
                measured, require_hardware=synchronized
            )
            page_bytes = self._kv_page_bytes()
            kv_bytes = sum(
                len(group) * page_bytes[index] for index, group in enumerate(measured)
            )
            item = PrefixPrimeEvidence(
                request_key=request.request_key,
                intended_cached_tokens=request.cached_tokens,
                actual_cached_tokens=request.cached_tokens,
                prime_scheduler_outputs=(
                    SchedulerBlockTableEvidence(
                        chunk_index=0,
                        request_key=request.request_key,
                        block_ids_by_group=sent_block_ids,
                        execution_completed=True,
                        gpu_synchronized=synchronized,
                    ),
                ),
                measured_block_ids_by_group=measured,
                worker_groups=worker_groups,
                prime_completed=True,
                prime_synchronized=synchronized,
                hardware_validated=synchronized
                and all(group.live_cuda_tensor_proven for group in worker_groups),
                kv_bytes=kv_bytes,
            )
            if synchronized and not item.hardware_validated:
                raise RuntimeError("prefix prime did not establish GPU0 HBM evidence")
            self._prime_evidence[request.request_key] = item
            evidence.append(item)
            try:
                from vllm.v1.request import RequestStatus

                status = RequestStatus.FINISHED_ABORTED
            except ImportError:
                status = "finished_aborted"
            self.scheduler.finish_requests([prime_id], status)
            flush = self.scheduler.schedule()
            if getattr(flush, "total_num_scheduled_tokens", 0):
                raise RuntimeError("prefix-prime cleanup scheduled model work")
            flush_output = self.executor.execute_model(flush)
            self.scheduler.update_from_output(flush, flush_output)
        return tuple(evidence)

    def schedule_chunk(self, point: PPointPlan, chunk: PChunkPlan) -> ScheduledChunk:
        """Schedule exactly one planned chunk without executing it."""
        expected = dict(chunk.scheduled_tokens_by_request)
        expected_ids = tuple(expected)
        state_ids = self._scheduler_request_ids() | self._queue_request_ids()
        unrelated_state = tuple(sorted(state_ids - set(expected_ids)))
        self.scheduler.max_num_scheduled_tokens = sum(expected.values())
        self.scheduler.scheduler_config.long_prefill_token_threshold = max(
            expected.values()
        )
        started = perf_counter()
        output = self.scheduler.schedule()
        elapsed_ms = (perf_counter() - started) * 1000
        actual, _ = self._scheduled_vectors(output)
        actual_ids = tuple(actual)
        preempted = tuple(
            str(item) for item in getattr(output, "preempted_req_ids", ())
        )
        unrelated = tuple(
            sorted(set(actual_ids) - set(expected_ids) | set(unrelated_state))
        )
        allocation = self._allocation_snapshot(expected, elapsed_ms)
        return ScheduledChunk(
            scheduler_output=output,
            expected_request_ids=expected_ids,
            actual_request_ids=actual_ids,
            expected_tokens_by_request=expected,
            actual_tokens_by_request=actual,
            preempted_request_ids=preempted,
            unrelated_request_ids=unrelated,
            allocation=allocation,
        )

    @staticmethod
    def require_complete_chunk(scheduled: ScheduledChunk) -> None:
        """Reject any SchedulerOutput that must not enter GPU timing."""
        if scheduled.unrelated_request_ids:
            raise RuntimeError(
                "Scheduler state or output contains an unrelated request"
            )
        if scheduled.preempted_request_ids:
            raise RuntimeError("Scheduler preempted an expected request")
        if scheduled.actual_request_ids != scheduled.expected_request_ids:
            raise RuntimeError("Scheduler returned a partial request set")
        if scheduled.actual_tokens_by_request != scheduled.expected_tokens_by_request:
            raise RuntimeError("Scheduler returned a partial token vector")
        if sum(scheduled.actual_tokens_by_request.values()) > 4096:
            raise RuntimeError("Scheduler output exceeds 4096 scheduled tokens")

    def verify_hit(self, point: PPointPlan, scheduler_output: Any) -> None:
        """Prove that measured lookup reuses synchronized resident GPU0 blocks."""
        _, request_data = self._scheduled_vectors(scheduler_output)
        for request in point.requests:
            prime = self._prime_evidence.get(request.request_key)
            if (
                prime is None
                or not prime.prime_completed
                or not prime.prime_synchronized
            ):
                raise RuntimeError("prefix was not executed on GPU0")
            measured = request_data.get(request.request_key)
            if (
                measured is None
                or self._computed_tokens(measured) != request.cached_tokens
            ):
                raise RuntimeError(
                    "measured cached-token count does not match the prime"
                )
            block_ids = self._block_ids(measured)
            prefix_blocks = request.cached_tokens // CANONICAL_BLOCK_SIZE
            prefix_ids = tuple(group[:prefix_blocks] for group in block_ids)
            if prefix_ids != prime.measured_block_ids_by_group:
                raise RuntimeError("measured physical block IDs do not match the prime")
            groups = self._inspect_worker_groups(prefix_ids, require_hardware=True)
            if not prime.hardware_validated or not all(
                group.live_cuda_tensor_proven for group in groups
            ):
                raise RuntimeError("prefix was not proven resident in GPU0 HBM")

    def verify_recompute_miss(self, point: PPointPlan, scheduler_output: Any) -> None:
        """Require a zero-hit lookup for every recompute request."""
        _, request_data = self._scheduled_vectors(scheduler_output)
        for request in point.requests:
            measured = request_data.get(request.request_key)
            if measured is None or self._computed_tokens(measured) != 0:
                raise RuntimeError("full recompute unexpectedly reused cached tokens")

    def classify_out_of_capacity(
        self, point: PPointPlan, chunk: PChunkPlan, scheduled: ScheduledChunk
    ) -> AllocationEvidence | None:
        """Return proven terminal OOC evidence or reject an invalid partial output."""
        complete = (
            not scheduled.unrelated_request_ids
            and not scheduled.preempted_request_ids
            and scheduled.actual_request_ids == scheduled.expected_request_ids
            and scheduled.actual_tokens_by_request
            == scheduled.expected_tokens_by_request
        )
        if complete:
            return None
        expected = set(chunk.scheduled_tokens_by_request)
        state = self._scheduler_request_ids() | self._queue_request_ids()
        actual = set(scheduled.actual_request_ids)
        if (
            state != expected
            or not actual <= expected
            or scheduled.unrelated_request_ids
        ):
            raise RuntimeError(
                "partial SchedulerOutput was not isolated to the expected batch"
            )
        allocation = scheduled.allocation
        if allocation.requested_blocks <= allocation.allocatable_blocks:
            raise RuntimeError("partial SchedulerOutput lacks allocator-pressure proof")
        self.reset_epoch()
        return AllocationEvidence(
            state="out_of_capacity",
            requested_blocks=allocation.requested_blocks,
            allocated_blocks=allocation.allocated_blocks,
            allocatable_blocks=allocation.allocatable_blocks,
            requested_bytes=allocation.requested_bytes,
            allocated_bytes=allocation.allocated_bytes,
            lookup_time_ms=allocation.lookup_time_ms,
            allocation_time_ms=allocation.allocation_time_ms,
            allocator_pressure_proven=True,
            clean_reset_proven=True,
        )


def _request_chunk_accounting(
    point: PPointPlan, chunk: PChunkPlan
) -> tuple[int, int, int, int]:
    context_tokens = 0
    cached_tokens = 0
    new_tokens = 0
    recomputed_tokens = 0
    requests = {request.request_key: request for request in point.requests}
    for request_key, scheduled_tokens in chunk.scheduled_tokens_by_request.items():
        request = requests[request_key]
        prior_tokens = sum(
            prior.scheduled_tokens_by_request.get(request_key, 0)
            for prior in point.chunks[: chunk.chunk_index]
        )
        if point.cache_condition == "prefix_hit":
            cached = request.cached_tokens if prior_tokens == 0 else 0
            context_tokens += cached + scheduled_tokens
            cached_tokens += cached
            new_tokens += scheduled_tokens
            continue
        prefix_tokens = request.context_tokens - request.new_tokens
        recomputed = max(
            0,
            min(prior_tokens + scheduled_tokens, prefix_tokens) - prior_tokens,
        )
        context_tokens += scheduled_tokens
        recomputed_tokens += recomputed
        new_tokens += scheduled_tokens - recomputed
    return context_tokens, cached_tokens, new_tokens, recomputed_tokens


def _request_token_vector(tokens_by_request: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"request_key": request_key, "scheduled_tokens": scheduled_tokens}
        for request_key, scheduled_tokens in tokens_by_request.items()
    ]


def make_prefill_chunk_row(
    *,
    run_id: str,
    point: PPointPlan,
    phase: Literal["warmup", "steady"],
    ordinal: int,
    chunk: PChunkPlan,
    runner_wall_time_ms: float | None,
    cuda_model_time_ms: float | None,
    allocation: dict[str, Any],
    status: str,
    error: str | None,
) -> dict[str, Any]:
    """Build one schema-v2 row from a planned Scheduler chunk."""
    if (
        chunk.chunk_index >= len(point.chunks)
        or point.chunks[chunk.chunk_index] != chunk
    ):
        raise ValueError("chunk does not belong to the point plan")
    actual_tokens = allocation["actual_scheduled_tokens_by_request"]
    actual_chunk = PChunkPlan(chunk.chunk_index, actual_tokens)
    context, cached, new, recomputed = _request_chunk_accounting(point, actual_chunk)
    kv_block_bytes = allocation["kv_block_bytes"]
    requested_blocks = allocation["requested_blocks"]
    allocated_blocks = allocation["allocated_blocks"]
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "run_id": run_id,
        "point_id": point.point_id,
        "comparison_id": point.comparison_id,
        "sample_id": (
            f"{run_id}:{point.point_id}:{phase}:{ordinal}:{chunk.chunk_index}"
        ),
        "role": "prefill",
        "workload_family": point.workload_family,
        "selector": point.selector,
        "composition": point.composition,
        "cache_condition": point.cache_condition,
        "planner_digest": point.planner_digest,
        "phase": phase,
        "ordinal": ordinal,
        "chunk_index": chunk.chunk_index,
        "chunk_count": len(point.chunks),
        "row_kind": "chunk" if status == "passed" else "terminal",
        "status": status,
        "allocation_state": allocation["state"],
        "planned_scheduled_tokens_by_request": _request_token_vector(
            chunk.scheduled_tokens_by_request
        ),
        "actual_scheduled_tokens_by_request": _request_token_vector(actual_tokens),
        "preempted_request_ids": list(allocation["preempted_request_ids"]),
        "unrelated_request_ids": list(allocation["unrelated_request_ids"]),
        "cache_epoch": allocation["cache_epoch"],
        "cache_reset_completed": allocation["cache_reset_completed"],
        "cache_reset_empty": allocation["cache_reset_empty"],
        "allocator_pressure_proven": allocation.get("allocator_pressure_proven", False),
        "clean_reset_proven": allocation.get("clean_reset_proven", False),
        "requested_kv_blocks": requested_blocks,
        "allocatable_kv_blocks": allocation["allocatable_blocks"],
        "allocated_kv_blocks": allocated_blocks,
        "kv_block_bytes": kv_block_bytes,
        "requested_kv_bytes": requested_blocks * kv_block_bytes,
        "allocated_kv_bytes": allocated_blocks * kv_block_bytes,
        "scheduled_tokens": sum(actual_tokens.values()),
        "context_tokens": context,
        "cached_tokens": cached,
        "new_tokens": new,
        "recomputed_tokens": recomputed,
        "lookup_time_ms": allocation["lookup_time_ms"],
        "allocation_time_ms": allocation["allocation_time_ms"],
        "runner_wall_time_ms": runner_wall_time_ms,
        "cuda_model_time_ms": cuda_model_time_ms,
        "runtime_mode": allocation["runtime_mode"],
        "error": error,
    }


def make_planner_digest(
    workload_plan: dict[str, Any],
    rendered_turns: list[dict[str, Any]],
    *,
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
    seed: int,
) -> str:
    """Return the digest binding all inputs to a planner expansion."""
    payload = {
        "planner_schema": "ds4-p-prefill-plan-v1",
        "workload_plan": workload_plan,
        "rendered_turns": rendered_turns,
        "block_size": block_size,
        "token_budget": token_budget,
        "homogeneous_prefix_tokens": homogeneous_prefix_tokens,
        "seed": seed,
        "chunk_algorithm": CHUNK_ALGORITHM,
        "homogeneous_token_algorithm": HOMOGENEOUS_TOKEN_ALGORITHM,
    }
    return hashlib.sha256(canonical_payload_json(payload).encode()).hexdigest()


def _token_digest(token_ids: tuple[int, ...]) -> str:
    payload = canonical_payload_json({"tokens": token_ids})
    return hashlib.sha256(payload.encode()).hexdigest()


def _request_payload(request: PRequestPlan) -> dict[str, Any]:
    return {
        "request_key": request.request_key,
        "trajectory_id": request.trajectory_id,
        "turn_index": request.turn_index,
        "reasoning_mode": request.reasoning_mode,
        "context_tokens": request.context_tokens,
        "cached_tokens": request.cached_tokens,
        "new_tokens": request.new_tokens,
        "token_digest": request.token_digest,
    }


def _planned_chunks_payload(chunks: tuple[PChunkPlan, ...]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_index": chunk.chunk_index,
            "scheduled_tokens_by_request": sorted(
                chunk.scheduled_tokens_by_request.items()
            ),
        }
        for chunk in chunks
    ]


def _point_payload(
    *,
    workload_family: WorkloadFamily,
    selector: str,
    composition: str,
    seed: int,
    cache_condition: CacheCondition,
    planner_digest: str,
    requests: tuple[PRequestPlan, ...],
    chunks: tuple[PChunkPlan, ...],
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
) -> dict[str, Any]:
    return {
        "workload_family": workload_family,
        "selector": selector,
        "requests": [_request_payload(request) for request in requests],
        "composition": composition,
        "seed": seed,
        "batch_size": len(requests),
        "chunk_budget": token_budget,
        "cache_condition": cache_condition,
        "block_size": block_size,
        "homogeneous_prefix_tokens": homogeneous_prefix_tokens,
        "capacity_target": "native",
        "planner_digest": planner_digest,
        "planned_chunks": _planned_chunks_payload(chunks),
    }


def plan_chunks(point: PPointPlan, token_budget: int) -> tuple[PChunkPlan, ...]:
    """Split one point into equal-active-cap chunks in request order."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if not point.requests:
        raise ValueError("point must contain at least one request")
    if len({request.request_key for request in point.requests}) != len(point.requests):
        raise ValueError("point request keys must be unique")

    remaining = {
        request.request_key: (
            request.new_tokens
            if point.cache_condition == "prefix_hit"
            else request.context_tokens
        )
        for request in point.requests
    }
    if any(tokens <= 0 for tokens in remaining.values()):
        raise ValueError("scheduled request tokens must be positive")

    chunks = []
    chunk_index = 0
    while remaining:
        active = [
            request.request_key
            for request in point.requests
            if request.request_key in remaining
        ]
        cap = token_budget // len(active)
        if cap <= 0:
            raise ValueError("token_budget cannot schedule all active requests")
        scheduled = {key: min(remaining[key], cap) for key in active}
        chunks.append(PChunkPlan(chunk_index, scheduled))
        for key, tokens in scheduled.items():
            remaining[key] -= tokens
            if remaining[key] == 0:
                del remaining[key]
        chunk_index += 1
    return tuple(chunks)


def _legal_execution_tokens(rendered_turns: list[dict[str, Any]]) -> tuple[int, ...]:
    legal_tokens = {
        token_id
        for turn in rendered_turns
        for key in ("execution_prompt_token_ids", "execution_completion_token_ids")
        for token_id in (turn.get(key) or [])
    }
    if not legal_tokens:
        raise ValueError("rendered turns contain no execution token IDs")
    return tuple(sorted(legal_tokens))


def _homogeneous_requests(
    *,
    selector: str,
    batch_size: int,
    new_tokens: int,
    prefix_tokens: int,
    legal_tokens: tuple[int, ...],
    seed: int,
    block_size: int,
) -> tuple[PRequestPlan, ...]:
    request_length = prefix_tokens + new_tokens
    token_blocks: set[tuple[int, ...]] = set()
    requests = []
    for request_index in range(batch_size):
        token_ids = tuple(
            legal_tokens[
                int.from_bytes(
                    hashlib.sha256(
                        f"{seed}:{selector}:{request_index}:{position}".encode()
                    ).digest(),
                    "big",
                )
                % len(legal_tokens)
            ]
            for position in range(request_length)
        )
        blocks = [
            token_ids[position : position + block_size]
            for position in range(0, len(token_ids), block_size)
        ]
        if any(block in token_blocks for block in blocks):
            raise ValueError("homogeneous requests share a block-aligned token block")
        token_blocks.update(blocks)
        requests.append(
            PRequestPlan(
                request_key=f"r{request_index}",
                trajectory_id=None,
                turn_index=None,
                reasoning_mode=None,
                prompt_token_ids=token_ids,
                context_tokens=request_length,
                cached_tokens=prefix_tokens,
                new_tokens=new_tokens,
                token_digest=_token_digest(token_ids),
            )
        )
    return tuple(requests)


def _rendered_turns_by_key(
    rendered_turns: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {}
    for turn in rendered_turns:
        key = (turn["trajectory_id"], turn["turn_index"])
        if key in indexed:
            raise ValueError("rendered turns contain duplicate trajectory turns")
        indexed[key] = turn
    return indexed


def _trace_requests(
    references: list[dict[str, Any]],
    rendered_by_key: dict[tuple[str, int], dict[str, Any]],
    block_size: int,
) -> tuple[PRequestPlan, ...]:
    requests = []
    for request_index, reference in enumerate(references):
        key = (reference["trajectory_id"], reference["turn_index"])
        turn = rendered_by_key.get(key)
        if turn is None:
            raise ValueError(f"missing rendered execution prompt for {key}")
        prompt_token_ids = turn.get("execution_prompt_token_ids")
        if prompt_token_ids is None:
            raise ValueError(f"missing rendered execution prompt IDs for {key}")
        token_ids = tuple(prompt_token_ids)
        context_tokens = reference["prompt_tokens"]
        cached_tokens = reference["reusable_prefix_tokens"]
        new_tokens = reference["new_prefill_tokens"]
        if len(token_ids) != context_tokens:
            raise ValueError(f"execution prompt length does not match {key}")
        if cached_tokens % block_size:
            raise ValueError(f"cached prefix for {key} is not block aligned")
        if context_tokens - cached_tokens != new_tokens:
            raise ValueError(f"prefill token accounting does not match {key}")
        requests.append(
            PRequestPlan(
                request_key=f"r{request_index}",
                trajectory_id=reference["trajectory_id"],
                turn_index=reference["turn_index"],
                reasoning_mode=reference["reasoning_mode"],
                prompt_token_ids=token_ids,
                context_tokens=context_tokens,
                cached_tokens=cached_tokens,
                new_tokens=new_tokens,
                token_digest=_token_digest(token_ids),
            )
        )
    return tuple(requests)


def _expand_condition_pair(
    *,
    workload_family: WorkloadFamily,
    selector: str,
    composition: str,
    requests: tuple[PRequestPlan, ...],
    planner_digest: str,
    seed: int,
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
) -> tuple[PPointPlan, PPointPlan]:
    points = []
    for cache_condition in ("prefix_hit", "full_recompute"):
        draft = PPointPlan(
            point_id="",
            comparison_id="",
            workload_family=workload_family,
            selector=selector,
            composition=composition,
            seed=seed,
            batch_size=len(requests),
            cache_condition=cache_condition,
            planner_digest=planner_digest,
            requests=requests,
            chunks=(),
            canonical_payload={},
        )
        chunks = plan_chunks(draft, token_budget)
        payload = _point_payload(
            workload_family=workload_family,
            selector=selector,
            composition=composition,
            seed=seed,
            cache_condition=cache_condition,
            planner_digest=planner_digest,
            requests=requests,
            chunks=chunks,
            block_size=block_size,
            token_budget=token_budget,
            homogeneous_prefix_tokens=homogeneous_prefix_tokens,
        )
        points.append(
            PPointPlan(
                point_id=make_point_id(payload),
                comparison_id=make_comparison_id(payload),
                workload_family=workload_family,
                selector=selector,
                composition=composition,
                seed=seed,
                batch_size=len(requests),
                cache_condition=cache_condition,
                planner_digest=planner_digest,
                requests=requests,
                chunks=chunks,
                canonical_payload=payload,
            )
        )
    return points[0], points[1]


def _validate_inputs(
    workload_plan: dict[str, Any],
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
) -> None:
    if block_size != CANONICAL_BLOCK_SIZE:
        raise ValueError(f"block_size must be {CANONICAL_BLOCK_SIZE}")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if homogeneous_prefix_tokens != CANONICAL_HOMOGENEOUS_PREFIX_TOKENS:
        raise ValueError(
            f"homogeneous prefix must be {CANONICAL_HOMOGENEOUS_PREFIX_TOKENS} tokens"
        )
    if homogeneous_prefix_tokens % CANONICAL_BLOCK_SIZE:
        raise ValueError("homogeneous prefix must be divisible by 16")
    if workload_plan.get("token_budget") != token_budget:
        raise ValueError("workload plan token_budget does not match planner")


def build_prefill_points(
    workload_plan: dict[str, Any],
    rendered_turns: list[dict[str, Any]],
    *,
    block_size: int,
    token_budget: int,
    homogeneous_prefix_tokens: int,
    seed: int,
) -> tuple[PPointPlan, ...]:
    """Expand pinned workload artifacts into paired executable prefill points."""
    _validate_inputs(workload_plan, block_size, token_budget, homogeneous_prefix_tokens)
    planner_digest = make_planner_digest(
        workload_plan,
        rendered_turns,
        block_size=block_size,
        token_budget=token_budget,
        homogeneous_prefix_tokens=homogeneous_prefix_tokens,
        seed=seed,
    )
    legal_tokens = _legal_execution_tokens(rendered_turns)
    rendered_by_key = _rendered_turns_by_key(rendered_turns)
    points: list[PPointPlan] = []

    for item in workload_plan["p_homogeneous"]:
        batch_size = item["batch_size"]
        new_tokens = item["per_request_scheduled_tokens"]
        if batch_size > 8:
            raise ValueError("workload exceeds eight sequences")
        if batch_size <= 0 or new_tokens <= 0:
            raise ValueError("homogeneous workloads must be positive")
        if batch_size * new_tokens > token_budget:
            raise ValueError("homogeneous workload exceeds token budget")
        if item["total_scheduled_tokens"] != batch_size * new_tokens:
            raise ValueError("homogeneous workload has inconsistent token total")
        selector = f"b{batch_size}-t{new_tokens}"
        requests = _homogeneous_requests(
            selector=selector,
            batch_size=batch_size,
            new_tokens=new_tokens,
            prefix_tokens=homogeneous_prefix_tokens,
            legal_tokens=legal_tokens,
            seed=seed,
            block_size=block_size,
        )
        points.extend(
            _expand_condition_pair(
                workload_family="homogeneous",
                selector=selector,
                composition="none",
                requests=requests,
                planner_digest=planner_digest,
                seed=seed,
                block_size=block_size,
                token_budget=token_budget,
                homogeneous_prefix_tokens=homogeneous_prefix_tokens,
            )
        )

    for batch in workload_plan["mixed_batches"]:
        batch_size = batch["batch_size"]
        references = batch["turns"]
        if batch_size > 8:
            raise ValueError("workload exceeds eight sequences")
        if len(references) != batch_size:
            raise ValueError("mixed workload batch size does not match references")
        requests = _trace_requests(references, rendered_by_key, block_size)
        if sum(request.new_tokens for request in requests) > token_budget:
            raise ValueError("mixed workload exceeds token budget")
        selector = f"{batch['composition']}-b{batch_size}"
        points.extend(
            _expand_condition_pair(
                workload_family="mixed",
                selector=selector,
                composition=batch["composition"],
                requests=requests,
                planner_digest=planner_digest,
                seed=seed,
                block_size=block_size,
                token_budget=token_budget,
                homogeneous_prefix_tokens=homogeneous_prefix_tokens,
            )
        )

    for replay in workload_plan["exact_replays"]:
        quantile = int(replay["selection_quantile"] * 100)
        selector = f"{replay['reasoning_mode']}-q{quantile:02d}"
        requests = _trace_requests([replay], rendered_by_key, block_size)
        points.extend(
            _expand_condition_pair(
                workload_family="exact_replay",
                selector=selector,
                composition="quantile",
                requests=requests,
                planner_digest=planner_digest,
                seed=seed,
                block_size=block_size,
                token_budget=token_budget,
                homogeneous_prefix_tokens=homogeneous_prefix_tokens,
            )
        )

    return tuple(points)
