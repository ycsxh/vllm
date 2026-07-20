# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic P-side prefill workload planning."""

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import pyarrow.parquet as pq

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


def _point_record(point: "PPointPlan") -> dict[str, Any]:
    record = {
        "point_id": point.point_id,
        "comparison_id": point.comparison_id,
        "canonical_payload": point.canonical_payload,
    }
    return json.loads(canonical_payload_json(record))


def freeze_expected_manifest(
    config: dict[str, Any],
    points: tuple["PPointPlan", ...],
    run_kind: Literal["full", "smoke"],
) -> dict[str, Any]:
    """Freeze the planner-derived full manifest and selected execution set."""
    if run_kind not in ("full", "smoke"):
        raise ValueError(f"unsupported run kind: {run_kind}")
    if len(points) != 68:
        raise ValueError("P-side planner must produce exactly 68 points")
    point_ids = [point.point_id for point in points]
    if len(set(point_ids)) != 68:
        raise ValueError("P-side planner produced duplicate point IDs")
    selectors: dict[str, list[PPointPlan]] = {}
    for point in points:
        selectors.setdefault(point.selector, []).append(point)
    if len(selectors) != 34 or any(
        len(pair) != 2
        or {point.cache_condition for point in pair} != {"prefix_hit", "full_recompute"}
        or len({point.comparison_id for point in pair}) != 1
        for pair in selectors.values()
    ):
        raise ValueError("P-side planner must produce 34 complete condition pairs")
    planner_digests = {point.planner_digest for point in points}
    if len(planner_digests) != 1:
        raise ValueError("P-side points do not share one planner digest")

    selected = point_ids
    if run_kind == "smoke":
        smoke = config.get("smoke_selectors")
        if not isinstance(smoke, list) or len(smoke) != len(set(smoke)):
            raise ValueError("smoke_selectors must be a unique selector list")
        unknown = sorted(set(smoke) - set(selectors))
        if unknown:
            raise ValueError(f"unknown smoke selectors: {', '.join(unknown)}")
        selected = [point.point_id for point in points if point.selector in smoke]
        if len(selected) != 2 * len(smoke):
            raise ValueError("smoke selection broke a condition pair")

    frozen = copy.deepcopy(config)
    profile = frozen["profile"]
    frozen["run_kind"] = run_kind
    frozen["canonical_planner_inputs"] = {
        "schema_version": V2_SCHEMA_VERSION,
        "workload_selectors": list(selectors),
        "kv_cache_groups": ["0"],
        "seed": profile["seed"],
        "block_size": profile["block_size"],
        "chunk_budget": profile["max_num_batched_tokens"],
        "planner_digest": planner_digests.pop(),
    }
    frozen["points"] = [_point_record(point) for point in points]
    frozen["canonical_full_manifest"] = point_ids
    frozen["expected_manifest"] = selected
    return frozen


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


@dataclass(frozen=True)
class _AllocatorSnapshot:
    requested_blocks: int
    allocated_blocks: int
    allocatable_blocks: int
    requested_bytes: int
    allocated_bytes: int
    refusal_proven: bool


def _make_request_factory(scheduler: Any) -> Callable[[str, list[int]], Any]:
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.request import Request

    caching_hash_fn = get_hash_fn_by_name(
        scheduler.cache_config.prefix_caching_hash_algo
    )
    init_none_hash(caching_hash_fn)
    block_hasher = get_request_block_hasher(scheduler.hash_block_size, caching_hash_fn)

    def make_request(request_id: str, tokens: list[int]) -> Any:
        request_key = request_id.rsplit(":", 1)[-1]
        return Request(
            request_id,
            tokens,
            SamplingParams(max_tokens=1, temperature=0.0),
            None,
            cache_salt=request_key,
            block_hasher=block_hasher,
        )

    return make_request


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
        layer_order: tuple[str, ...] | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.executor = executor
        self.worker = worker
        self.request_factory = request_factory
        self.synchronize_gpu = synchronize_gpu
        self.block_axes = block_axes
        self.layer_order = layer_order
        self._prime_evidence: dict[str, PrefixPrimeEvidence] = {}

    @classmethod
    def from_runtime(
        cls,
        scheduler: Any,
        executor: Any,
        worker: Any,
        *,
        block_axes: tuple[int, ...] | None = None,
    ) -> "VllmSchedulerCacheAdapter":
        """Create the real adapter while keeping vLLM imports lazy."""
        import torch

        from vllm.model_executor.models.utils import extract_layer_index

        insertion_order = tuple(
            layer_name
            for tensor in scheduler.kv_cache_config.kv_cache_tensors
            for layer_name in tensor.shared_by
        )
        layer_order = tuple(sorted(insertion_order, key=extract_layer_index))
        if block_axes is None:
            tensors = dict(zip(layer_order, worker.model_runner.kv_caches))
            resolved_axes = []
            for group in scheduler.kv_cache_config.kv_cache_groups:
                mapped = tuple(tensors[name] for name in group.layer_names)
                candidates = {
                    axis
                    for axis in range(mapped[0].ndim)
                    if all(
                        int(tensor.shape[axis]) == scheduler.kv_cache_config.num_blocks
                        for tensor in mapped
                    )
                }
                if len(candidates) != 1:
                    raise RuntimeError("could not resolve one live physical block axis")
                resolved_axes.append(candidates.pop())
            block_axes = tuple(resolved_axes)
        return cls(
            scheduler,
            executor,
            worker,
            request_factory=_make_request_factory(scheduler),
            synchronize_gpu=lambda: torch.accelerator.synchronize(
                torch.device("cuda:0")
            ),
            block_axes=block_axes,
            layer_order=layer_order,
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
            entry = cached.new_block_ids[index]
            if entry is None:
                raise RuntimeError("cached request omitted its Scheduler block table")
            return tuple(tuple(int(block_id) for block_id in group) for group in entry)
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

    @classmethod
    def _updated_block_table(
        cls,
        current: tuple[tuple[int, ...], ...],
        request_id: str,
        request_data: Any,
    ) -> tuple[tuple[int, ...], ...]:
        sent = cls._block_ids(request_data)
        if not isinstance(request_data, tuple) or not current:
            return sent
        cached, _ = request_data
        if request_id in cached.resumed_req_ids:
            return sent
        if len(current) != len(sent):
            raise RuntimeError("prime Scheduler block-table groups changed")
        return tuple(existing + new for existing, new in zip(current, sent))

    def _kv_page_bytes(self) -> tuple[int, ...]:
        config = getattr(self.scheduler, "kv_cache_config", None)
        groups = getattr(config, "kv_cache_groups", ())
        return tuple(int(group.kv_cache_spec.page_size_bytes) for group in groups)

    def _allocation_snapshot(
        self,
        allocator: _AllocatorSnapshot,
        elapsed_ms: float,
    ) -> AllocationEvidence:
        return AllocationEvidence(
            state="allocated",
            requested_blocks=allocator.requested_blocks,
            allocated_blocks=allocator.allocated_blocks,
            allocatable_blocks=allocator.allocatable_blocks,
            requested_bytes=allocator.requested_bytes,
            allocated_bytes=allocator.allocated_bytes,
            lookup_time_ms=elapsed_ms,
            allocation_time_ms=elapsed_ms,
            allocator_pressure_proven=allocator.refusal_proven,
            clean_reset_proven=False,
        )

    def _schedule_with_allocator_evidence(
        self,
    ) -> tuple[Any, _AllocatorSnapshot]:
        """Capture exact manager values when ``allocate_slots`` refuses."""
        manager = self.scheduler.kv_cache_manager
        original_allocate = manager.allocate_slots
        page_bytes = self._kv_page_bytes()
        allocated_blocks = 0
        allocated_bytes = 0
        requested_blocks = 0
        requested_bytes = 0
        refusal: tuple[int, int, int] | None = None

        def tracked_allocate(*args: Any, **kwargs: Any) -> Any:
            coordinator = manager.coordinator
            original_required = coordinator.get_num_blocks_to_allocate
            required_calls: list[int] = []
            required_group_calls: list[tuple[int, ...]] = []

            def tracked_required(*call_args: Any, **call_kwargs: Any) -> int:
                group_counts: list[int] = []
                originals = []
                for single_manager in coordinator.single_type_managers:
                    original_group = single_manager.get_num_blocks_to_allocate
                    originals.append((single_manager, original_group))

                    def tracked_group(
                        *group_args: Any,
                        _original: Callable[..., Any] = original_group,
                        **group_kwargs: Any,
                    ) -> int:
                        count = int(_original(*group_args, **group_kwargs))
                        group_counts.append(count)
                        return count

                    single_manager.get_num_blocks_to_allocate = tracked_group
                try:
                    required = int(original_required(*call_args, **call_kwargs))
                finally:
                    for single_manager, original_group in originals:
                        single_manager.get_num_blocks_to_allocate = original_group
                required_calls.append(required)
                required_group_calls.append(tuple(group_counts))
                return required

            coordinator.get_num_blocks_to_allocate = tracked_required
            try:
                allocation_result = original_allocate(*args, **kwargs)
            finally:
                coordinator.get_num_blocks_to_allocate = original_required
            nonlocal allocated_blocks, allocated_bytes
            nonlocal requested_blocks, requested_bytes, refusal
            if allocation_result is None:
                if not required_calls:
                    raise RuntimeError(
                        "allocator refusal omitted its required-block calculation"
                    )
                request = args[0]
                has_scheduled = bool(kwargs.get("has_scheduled_reqs", True))
                try:
                    from vllm.v1.request import RequestStatus

                    waiting = request.status in (
                        RequestStatus.WAITING,
                        RequestStatus.PREEMPTED,
                    )
                except ImportError:
                    waiting = False
                watermark = (
                    int(manager.watermark_blocks) if has_scheduled and waiting else 0
                )
                full_gate_refusal = bool(
                    kwargs.get("full_sequence_must_fit", False)
                    and len(required_calls) == 1
                )
                reserved = (
                    0 if full_gate_refusal else int(kwargs.get("reserved_blocks", 0))
                )
                available = max(
                    0, int(manager.block_pool.get_num_free_blocks()) - reserved
                )
                group_required = required_group_calls[-1]
                group_bytes = sum(
                    count * page_bytes[index]
                    for index, count in enumerate(group_required)
                )
                refusal = (
                    required_calls[-1] + watermark,
                    available,
                    group_bytes,
                )
            else:
                groups = getattr(allocation_result, "blocks", allocation_result)
                group_allocated = tuple(len(group) for group in groups)
                allocated_count = sum(group_allocated)
                allocation_bytes = sum(
                    count * page_bytes[index]
                    for index, count in enumerate(group_allocated)
                )
                required = required_calls[-1]
                required_for_groups = required_group_calls[-1]
                allocated_blocks += allocated_count
                allocated_bytes += allocation_bytes
                requested_blocks += required
                requested_bytes += sum(
                    count * page_bytes[index]
                    for index, count in enumerate(required_for_groups)
                )
            return allocation_result

        manager.allocate_slots = tracked_allocate
        try:
            output = self.scheduler.schedule()
        finally:
            manager.allocate_slots = original_allocate
        free_blocks = int(manager.block_pool.get_num_free_blocks())
        if refusal is None:
            return output, _AllocatorSnapshot(
                requested_blocks=requested_blocks,
                allocated_blocks=allocated_blocks,
                allocatable_blocks=allocated_blocks + free_blocks,
                requested_bytes=requested_bytes,
                allocated_bytes=allocated_bytes,
                refusal_proven=False,
            )
        refused_blocks, available_blocks, refused_bytes = refusal
        return output, _AllocatorSnapshot(
            requested_blocks=allocated_blocks + refused_blocks,
            allocated_blocks=allocated_blocks,
            allocatable_blocks=allocated_blocks + available_blocks,
            requested_bytes=allocated_bytes + refused_bytes,
            allocated_bytes=allocated_bytes,
            refusal_proven=refused_blocks > available_blocks,
        )

    def reset_epoch(self) -> None:
        """Abort and flush every request, then prove an empty cache epoch."""
        request_ids = self._scheduler_request_ids() | self._queue_request_ids()
        finished_ids_pending = bool(getattr(self.scheduler, "finished_req_ids", set()))
        if request_ids:
            try:
                from vllm.v1.request import RequestStatus

                status = RequestStatus.FINISHED_ABORTED
            except ImportError:
                status = "finished_aborted"
            self.scheduler.finish_requests(sorted(request_ids), status)
        if request_ids or finished_ids_pending:
            flush = self.scheduler.schedule()
            if getattr(flush, "total_num_scheduled_tokens", 0):
                raise RuntimeError("reset flush scheduled model work")
            model_output = self._execute_untimed(flush)
            self.scheduler.update_from_output(flush, model_output)
            if self.synchronize_gpu is not None:
                self.synchronize_gpu()
        if self._scheduler_request_ids() or self._queue_request_ids():
            raise RuntimeError("reset left live Scheduler requests")
        if not self.scheduler.reset_prefix_cache():
            raise RuntimeError("prefix cache reset failed")
        pool = self.scheduler.kv_cache_manager.block_pool
        used = int(getattr(pool, "num_gpu_blocks", 0)) - int(pool.get_num_free_blocks())
        if used != 1:
            raise RuntimeError("prefix cache reset left live KV blocks")
        self._prime_evidence.clear()

    def add_measurement_requests(self, point: PPointPlan) -> None:
        """Add the exact measured batch to an otherwise empty Scheduler."""
        if self.request_factory is None:
            raise RuntimeError("measurement requires a real request factory")
        if self._scheduler_request_ids() or self._queue_request_ids():
            raise RuntimeError("measurement epoch contains unrelated requests")
        for request in point.requests:
            scheduler_request = self.request_factory(
                request.request_key, list(request.prompt_token_ids)
            )
            scheduler_request.skip_reading_prefix_cache = (
                point.cache_condition == "full_recompute"
            )
            self.scheduler.add_request(scheduler_request)

    def probe_out_of_capacity(self, point: PPointPlan) -> AllocationEvidence | None:
        """Prove full-batch admission failure before partial scheduling."""
        if self.request_factory is None:
            raise RuntimeError("capacity probe requires a real request factory")
        manager = self.scheduler.kv_cache_manager
        empty_blocks = manager.empty_kv_cache_blocks.blocks
        requested_sequence_blocks = 0
        for request in point.requests:
            scheduler_request = self.request_factory(
                request.request_key, list(request.prompt_token_ids)
            )
            full_num_tokens = min(scheduler_request.num_tokens, manager.max_model_len)
            requested_sequence_blocks += int(
                manager.coordinator.get_num_blocks_to_allocate(
                    request_id=scheduler_request.request_id,
                    num_tokens=full_num_tokens,
                    new_computed_blocks=empty_blocks,
                    num_encoder_tokens=0,
                    total_computed_tokens=0,
                    num_local_computed_tokens=0,
                    num_tokens_main_model=full_num_tokens,
                    apply_admission_cap=True,
                )
            )
        capacity = int(self.scheduler.kv_cache_config.num_blocks)
        if requested_sequence_blocks <= capacity:
            return None

        self.add_measurement_requests(point)
        allocated_blocks = 0
        started = perf_counter()
        for request in point.requests:
            scheduler_request = self.scheduler.requests[request.request_key]
            required_calls: list[int] = []
            original_required = manager.coordinator.get_num_blocks_to_allocate

            def tracked_required(
                *args: Any,
                _original: Callable[..., Any] = original_required,
                _calls: list[int] = required_calls,
                **kwargs: Any,
            ) -> int:
                required = int(_original(*args, **kwargs))
                _calls.append(required)
                return required

            manager.coordinator.get_num_blocks_to_allocate = tracked_required
            free_blocks = int(manager.block_pool.get_num_free_blocks())
            has_scheduled = allocated_blocks > 0
            try:
                blocks = manager.allocate_slots(
                    scheduler_request,
                    scheduler_request.num_tokens,
                    full_sequence_must_fit=True,
                    has_scheduled_reqs=has_scheduled,
                )
            finally:
                manager.coordinator.get_num_blocks_to_allocate = original_required
            if blocks is not None:
                groups = getattr(blocks, "blocks", blocks)
                allocated_blocks += sum(len(group) for group in groups)
                continue
            if not required_calls:
                raise RuntimeError("capacity probe refusal omitted block demand")
            watermark = int(manager.watermark_blocks) if has_scheduled else 0
            required_blocks = required_calls[0] + watermark
            if required_blocks <= free_blocks:
                raise RuntimeError("capacity probe refusal lacked allocator pressure")
            elapsed_ms = (perf_counter() - started) * 1000
            allocation = AllocationEvidence(
                state="out_of_capacity",
                requested_blocks=allocated_blocks + required_blocks,
                allocated_blocks=allocated_blocks,
                allocatable_blocks=allocated_blocks + free_blocks,
                requested_bytes=0,
                allocated_bytes=0,
                lookup_time_ms=elapsed_ms,
                allocation_time_ms=elapsed_ms,
                allocator_pressure_proven=True,
                clean_reset_proven=False,
            )
            self.reset_epoch()
            return replace(allocation, clean_reset_proven=True)
        self.reset_epoch()
        raise RuntimeError("capacity probe did not produce an allocator refusal")

    def update_after_execute(self, scheduler_output: Any, model_output: Any) -> None:
        """Apply one completed worker step to Scheduler state."""
        self.scheduler.update_from_output(scheduler_output, model_output)

    def completed_prime_evidence(self) -> tuple[PrefixPrimeEvidence, ...]:
        """Return every request prime completed in the current epoch."""
        return tuple(self._prime_evidence.values())

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
        configured_names = tuple(
            name for group in configured_groups for name in group.layer_names
        )
        layer_order = self.layer_order or configured_names
        if len(layer_order) != len(tensors) or set(layer_order) != set(
            configured_names
        ):
            raise RuntimeError("live KV cache layer ordering is incomplete")
        tensor_by_name = dict(zip(layer_order, tensors))
        evidence = []
        for group_index, (group, block_ids, block_axis) in enumerate(
            zip(configured_groups, block_ids_by_group, axes)
        ):
            names = tuple(group.layer_names)
            try:
                mapped = tuple(tensor_by_name[name] for name in names)
            except KeyError as error:
                raise RuntimeError("missing configured layer tensor") from error
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
        return tuple(evidence)

    def _execute_untimed(self, scheduler_output: Any) -> Any:
        output = self.executor.execute_model(scheduler_output)
        if output is None:
            output = self.executor.sample_tokens(None)
        if output is None:
            raise RuntimeError("executor returned no Scheduler update output")
        return output

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
            if request.cached_tokens == 0:
                continue
            prime_id = f"prime:{phase}:{ordinal}:{request.request_key}"
            prefix = list(request.prompt_token_ids[: request.cached_tokens])
            self.scheduler.add_request(self.request_factory(prime_id, prefix))
            primed_tokens = 0
            prime_outputs = []
            sent_table: tuple[tuple[int, ...], ...] = ()
            measured: tuple[tuple[int, ...], ...] = ()
            worker_groups: tuple[WorkerKvTensorGroupEvidence, ...] = ()
            synchronized = self.synchronize_gpu is not None
            chunk_index = 0
            while primed_tokens < request.cached_tokens:
                budget = min(4096, request.cached_tokens - primed_tokens)
                self.scheduler.max_num_scheduled_tokens = budget
                self.scheduler.scheduler_config.long_prefill_token_threshold = budget
                output = self.scheduler.schedule()
                scheduled, request_data = self._scheduled_vectors(output)
                if set(scheduled) != {prime_id} or not (
                    0 < scheduled[prime_id] <= budget
                ):
                    raise RuntimeError("prefix prime SchedulerOutput was partial")
                prime_data = request_data.get(prime_id)
                if prime_data is None:
                    raise RuntimeError("prefix prime omitted its Scheduler block table")
                sent_block_ids = self._block_ids(prime_data)
                sent_table = self._updated_block_table(sent_table, prime_id, prime_data)
                model_output = self._execute_untimed(output)
                if self.synchronize_gpu is not None:
                    self.synchronize_gpu()
                measured = tuple(
                    tuple(int(block_id) for block_id in group)
                    for group in self.scheduler.kv_cache_manager.get_block_ids(prime_id)
                )
                worker_groups = self._inspect_worker_groups(
                    measured, require_hardware=synchronized
                )
                prime_outputs.append(
                    SchedulerBlockTableEvidence(
                        chunk_index=chunk_index,
                        request_key=request.request_key,
                        block_ids_by_group=sent_block_ids,
                        execution_completed=True,
                        gpu_synchronized=synchronized,
                    )
                )
                completed_tokens = (
                    self._computed_tokens(prime_data) + scheduled[prime_id]
                )
                if not primed_tokens < completed_tokens <= request.cached_tokens:
                    raise RuntimeError("prefix prime made invalid token progress")
                self.scheduler.update_from_output(output, model_output)
                primed_tokens = completed_tokens
                chunk_index += 1
            intended_blocks = request.cached_tokens // CANONICAL_BLOCK_SIZE
            measured = tuple(group[:intended_blocks] for group in measured)
            if tuple(group[:intended_blocks] for group in sent_table) != measured:
                raise RuntimeError("prime block table changed after GPU execution")
            page_bytes = self._kv_page_bytes()
            kv_bytes = sum(
                len(group) * page_bytes[index] for index, group in enumerate(measured)
            )
            item = PrefixPrimeEvidence(
                request_key=request.request_key,
                intended_cached_tokens=request.cached_tokens,
                actual_cached_tokens=request.cached_tokens,
                prime_scheduler_outputs=tuple(prime_outputs),
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
            flush_output = self._execute_untimed(flush)
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
        output, allocator = self._schedule_with_allocator_evidence()
        elapsed_ms = (perf_counter() - started) * 1000
        actual, _ = self._scheduled_vectors(output)
        actual_ids = tuple(actual)
        preempted = tuple(
            str(item) for item in getattr(output, "preempted_req_ids", ())
        )
        unrelated = tuple(
            sorted(set(actual_ids) - set(expected_ids) | set(unrelated_state))
        )
        allocation = self._allocation_snapshot(allocator, elapsed_ms)
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
            measured = request_data.get(request.request_key)
            if request.cached_tokens == 0:
                if measured is None or self._computed_tokens(measured) != 0:
                    raise RuntimeError("zero-cached request unexpectedly reused blocks")
                continue
            prime = self._prime_evidence.get(request.request_key)
            if (
                prime is None
                or not prime.prime_completed
                or not prime.prime_synchronized
            ):
                raise RuntimeError("prefix was not executed on GPU0")
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
        if not allocation.allocator_pressure_proven:
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
        request = requests.get(request_key)
        if request is None:
            continue
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


def _allocation_payload(
    adapter: VllmSchedulerCacheAdapter,
    scheduled: ScheduledChunk,
    allocation: AllocationEvidence,
    *,
    cache_epoch: int,
    runtime_mode: str | None,
) -> dict[str, Any]:
    page_bytes = adapter._kv_page_bytes()
    if len(set(page_bytes)) != 1:
        raise RuntimeError("raw chunk schema requires one uniform KV block size")
    return {
        "state": allocation.state,
        "actual_scheduled_tokens_by_request": dict(scheduled.actual_tokens_by_request),
        "preempted_request_ids": scheduled.preempted_request_ids,
        "unrelated_request_ids": scheduled.unrelated_request_ids,
        "cache_epoch": cache_epoch,
        "cache_reset_completed": True,
        "cache_reset_empty": True,
        "allocator_pressure_proven": allocation.allocator_pressure_proven,
        "clean_reset_proven": allocation.clean_reset_proven,
        "requested_blocks": allocation.requested_blocks,
        "allocatable_blocks": allocation.allocatable_blocks,
        "allocated_blocks": allocation.allocated_blocks,
        "kv_block_bytes": page_bytes[0],
        "lookup_time_ms": allocation.lookup_time_ms,
        "allocation_time_ms": allocation.allocation_time_ms,
        "runtime_mode": runtime_mode,
    }


def _runtime_mode(model_output: Any) -> str:
    stats = getattr(model_output, "cudagraph_stats", None)
    mode = str(getattr(stats, "runtime_mode", ""))
    if mode not in {"FULL", "PIECEWISE"}:
        raise RuntimeError(f"measured execution used unexpected CUDA Graph mode {mode}")
    return mode


def _run_point_repetition_impl(
    point: PPointPlan,
    *,
    phase: Literal["warmup", "steady"],
    ordinal: int,
    adapter: Any,
    execute_timed: Callable[[Any], tuple[Any, float | None, float | None]],
    failure_coordinate: list[PChunkPlan],
    completed_rows: list[dict[str, Any]],
    completed_evidence: list[PrefixPrimeEvidence],
    failure_scheduled: list[ScheduledChunk | None],
    run_id: str = "unit-run",
) -> dict[str, Any]:
    """Run one reset-isolated point repetition in fail-closed order."""
    epoch_payload = f"{point.point_id}:{phase}:{ordinal}".encode()
    cache_epoch = int.from_bytes(hashlib.sha256(epoch_payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    adapter.reset_epoch()
    prime_evidence = ()
    try:
        prime_evidence = adapter.prime(point, phase, ordinal)
    finally:
        snapshot = getattr(
            adapter, "completed_prime_evidence", lambda: prime_evidence
        )()
        completed_evidence.extend(snapshot)
    capacity_failure = getattr(adapter, "probe_out_of_capacity", lambda _point: None)(
        point
    )
    if capacity_failure is not None:
        chunk = point.chunks[0]
        scheduled = ScheduledChunk(
            scheduler_output=None,
            expected_request_ids=tuple(chunk.scheduled_tokens_by_request),
            actual_request_ids=(),
            expected_tokens_by_request=dict(chunk.scheduled_tokens_by_request),
            actual_tokens_by_request={},
            preempted_request_ids=(),
            unrelated_request_ids=(),
            allocation=capacity_failure,
        )
        allocation = _allocation_payload(
            adapter,
            scheduled,
            capacity_failure,
            cache_epoch=cache_epoch,
            runtime_mode=None,
        )
        completed_rows.append(
            make_prefill_chunk_row(
                run_id=run_id,
                point=point,
                phase=phase,
                ordinal=ordinal,
                chunk=chunk,
                runner_wall_time_ms=None,
                cuda_model_time_ms=None,
                allocation=allocation,
                status="out_of_capacity",
                error=None,
            )
        )
        return {
            "status": "out_of_capacity",
            "rows": completed_rows,
            "prefix_evidence": tuple(completed_evidence),
        }
    adapter.add_measurement_requests(point)
    rows = completed_rows
    for chunk in point.chunks:
        failure_coordinate[0] = chunk
        failure_scheduled[0] = None
        scheduled = adapter.schedule_chunk(point, chunk)
        failure_scheduled[0] = scheduled
        out_of_capacity = adapter.classify_out_of_capacity(point, chunk, scheduled)
        if out_of_capacity is not None:
            allocation = _allocation_payload(
                adapter,
                scheduled,
                out_of_capacity,
                cache_epoch=cache_epoch,
                runtime_mode=None,
            )
            rows.append(
                make_prefill_chunk_row(
                    run_id=run_id,
                    point=point,
                    phase=phase,
                    ordinal=ordinal,
                    chunk=chunk,
                    runner_wall_time_ms=None,
                    cuda_model_time_ms=None,
                    allocation=allocation,
                    status="out_of_capacity",
                    error=None,
                )
            )
            return {
                "status": "out_of_capacity",
                "rows": rows,
                "prefix_evidence": prime_evidence,
            }
        adapter.require_complete_chunk(scheduled)
        if chunk.chunk_index == 0:
            if point.cache_condition == "prefix_hit":
                adapter.verify_hit(point, scheduled.scheduler_output)
            else:
                adapter.verify_recompute_miss(point, scheduled.scheduler_output)
        model_output, wall_ms, cuda_ms = execute_timed(scheduled.scheduler_output)
        if wall_ms is None or cuda_ms is None:
            raise RuntimeError("timed worker execution omitted GPU timings")
        mode = _runtime_mode(model_output)
        adapter.update_after_execute(scheduled.scheduler_output, model_output)
        allocation = _allocation_payload(
            adapter,
            scheduled,
            scheduled.allocation,
            cache_epoch=cache_epoch,
            runtime_mode=mode,
        )
        rows.append(
            make_prefill_chunk_row(
                run_id=run_id,
                point=point,
                phase=phase,
                ordinal=ordinal,
                chunk=chunk,
                runner_wall_time_ms=wall_ms,
                cuda_model_time_ms=cuda_ms,
                allocation=allocation,
                status="passed",
                error=None,
            )
        )
    return {"status": "passed", "rows": rows, "prefix_evidence": prime_evidence}


def run_point_repetition(
    point: PPointPlan,
    *,
    phase: Literal["warmup", "steady"],
    ordinal: int,
    adapter: Any,
    execute_timed: Callable[[Any], tuple[Any, float | None, float | None]],
    run_id: str = "unit-run",
) -> dict[str, Any]:
    """Run one repetition and preserve a structured failure coordinate."""
    failure_coordinate = [point.chunks[0]]
    completed_rows: list[dict[str, Any]] = []
    completed_evidence: list[PrefixPrimeEvidence] = []
    failure_scheduled: list[ScheduledChunk | None] = [None]
    try:
        return _run_point_repetition_impl(
            point,
            phase=phase,
            ordinal=ordinal,
            adapter=adapter,
            execute_timed=execute_timed,
            run_id=run_id,
            failure_coordinate=failure_coordinate,
            completed_rows=completed_rows,
            completed_evidence=completed_evidence,
            failure_scheduled=failure_scheduled,
        )
    except Exception as error:
        chunk = failure_coordinate[0]
        epoch_payload = f"{point.point_id}:{phase}:{ordinal}".encode()
        cache_epoch = int.from_bytes(
            hashlib.sha256(epoch_payload).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        page_bytes = adapter._kv_page_bytes()
        block_bytes = page_bytes[0] if page_bytes else 0
        scheduled = failure_scheduled[0]
        actual_tokens = {} if scheduled is None else scheduled.actual_tokens_by_request
        allocation = {
            "state": "failed",
            "actual_scheduled_tokens_by_request": actual_tokens,
            "preempted_request_ids": (
                () if scheduled is None else scheduled.preempted_request_ids
            ),
            "unrelated_request_ids": (
                () if scheduled is None else scheduled.unrelated_request_ids
            ),
            "cache_epoch": cache_epoch,
            "cache_reset_completed": False,
            "cache_reset_empty": False,
            "allocator_pressure_proven": False,
            "clean_reset_proven": False,
            "requested_blocks": (
                0 if scheduled is None else scheduled.allocation.requested_blocks
            ),
            "allocatable_blocks": (
                0 if scheduled is None else scheduled.allocation.allocatable_blocks
            ),
            "allocated_blocks": (
                0 if scheduled is None else scheduled.allocation.allocated_blocks
            ),
            "kv_block_bytes": block_bytes,
            "lookup_time_ms": (
                0.0 if scheduled is None else scheduled.allocation.lookup_time_ms
            ),
            "allocation_time_ms": (
                0.0 if scheduled is None else scheduled.allocation.allocation_time_ms
            ),
            "runtime_mode": None,
        }
        error_text = f"{type(error).__name__}: {error}"
        row = make_prefill_chunk_row(
            run_id=run_id,
            point=point,
            phase=phase,
            ordinal=ordinal,
            chunk=chunk,
            runner_wall_time_ms=None,
            cuda_model_time_ms=None,
            allocation=allocation,
            status="failed",
            error=error_text,
        )
        return {
            "status": "failed",
            "rows": [*completed_rows, row],
            "prefix_evidence": tuple(completed_evidence),
            "error": error_text,
        }


def _flatten_prefix_evidence(
    run_id: str,
    point_id: str,
    phase: str,
    ordinal: int,
    evidence: PrefixPrimeEvidence,
) -> list[dict[str, Any]]:
    rows = []
    for group_index, (block_ids, worker_group) in enumerate(
        zip(evidence.measured_block_ids_by_group, evidence.worker_groups)
    ):
        prime_block_ids = tuple(
            block_id
            for output in evidence.prime_scheduler_outputs
            for block_id in output.block_ids_by_group[group_index]
        )
        if prime_block_ids != block_ids:
            raise RuntimeError(
                "serialized prime block tables do not match measured IDs"
            )
        rows.append(
            {
                "schema_version": V2_SCHEMA_VERSION,
                "run_id": run_id,
                "point_id": point_id,
                "phase": phase,
                "ordinal": ordinal,
                "request_key": evidence.request_key,
                "kv_cache_group": str(group_index),
                "prime_scheduler_block_ids": list(prime_block_ids),
                "measured_scheduler_block_ids": list(block_ids),
                "live_kv_tensor_names": list(worker_group.tensor_names),
                "live_kv_tensor_devices": list(worker_group.tensor_devices),
                "live_kv_tensor_shapes": [
                    list(shape) for shape in worker_group.tensor_shapes
                ],
                "block_axis": worker_group.block_axis,
                "block_dimension": worker_group.block_dimension,
                "verified_physical_block_ids": list(worker_group.verified_block_ids),
                "intended_cached_tokens": evidence.intended_cached_tokens,
                "actual_cached_tokens": evidence.actual_cached_tokens,
                "prime_completed": evidence.prime_completed,
                "prime_synchronized": evidence.prime_synchronized,
                "live_cuda_tensor_proven": (worker_group.live_cuda_tensor_proven),
                "hardware_validated": evidence.hardware_validated,
            }
        )
    if len(rows) != len(evidence.measured_block_ids_by_group):
        raise RuntimeError("prefix evidence omitted a live KV cache group")
    return rows


def run_prefill_matrix(
    config: dict[str, Any], points: tuple[PPointPlan, ...]
) -> dict[str, Any]:
    """Run the selected P-side matrix through Scheduler and GPUWorker."""
    from benchmarks.ds4_profile.gpu_profile import (
        execute_worker_step,
        initialize_gpu_runtime,
    )

    runtime = None
    rows: list[dict[str, Any]] = []
    prefix_evidence: list[dict[str, Any]] = []
    status = "passed"
    error = None
    failure_point = points[0] if points else None
    failure_phase: Literal["warmup", "steady"] = "warmup"
    failure_ordinal = 0
    try:
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
        from vllm.v1.structured_output import StructuredOutputManager

        runtime = initialize_gpu_runtime(config)
        scheduler_class = runtime.vllm_config.scheduler_config.get_scheduler_cls()
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            runtime.kv_cache_config, runtime.vllm_config
        )
        scheduler = scheduler_class(
            vllm_config=runtime.vllm_config,
            kv_cache_config=runtime.kv_cache_config,
            structured_output_manager=StructuredOutputManager(runtime.vllm_config),
            include_finished_set=False,
            log_stats=False,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )
        adapter = VllmSchedulerCacheAdapter.from_runtime(
            scheduler, runtime.executor, runtime.worker
        )

        def execute_timed(output: Any) -> tuple[Any, float | None, float | None]:
            return execute_worker_step(runtime, output, timed=True)

        profile = config["profile"]
        repetitions = tuple(
            ("warmup", ordinal) for ordinal in range(profile["warmup_repetitions"])
        )
        steady = tuple(
            ("steady", ordinal) for ordinal in range(profile["measured_repetitions"])
        )
        for point in points:
            for phase, ordinal in (*repetitions, *steady):
                failure_point = point
                failure_phase = phase
                failure_ordinal = ordinal
                result = run_point_repetition(
                    point,
                    phase=phase,
                    ordinal=ordinal,
                    adapter=adapter,
                    execute_timed=execute_timed,
                    run_id=config["run_id"],
                )
                rows.extend(result["rows"])
                for item in result["prefix_evidence"]:
                    prefix_evidence.extend(
                        _flatten_prefix_evidence(
                            config["run_id"],
                            point.point_id,
                            phase,
                            ordinal,
                            item,
                        )
                    )
                if result["status"] == "failed":
                    raise RuntimeError(result["error"])
                if result["status"] == "out_of_capacity":
                    status = "out_of_capacity"
                    break
    except Exception as caught:
        status = "failed"
        error = f"{type(caught).__name__}: {caught}"
        if failure_point is not None and not (rows and rows[-1]["status"] == "failed"):
            chunk = failure_point.chunks[0]
            epoch_payload = (
                f"{failure_point.point_id}:{failure_phase}:{failure_ordinal}"
            ).encode()
            cache_epoch = int.from_bytes(
                hashlib.sha256(epoch_payload).digest()[:8], "big"
            ) & ((1 << 63) - 1)
            rows.append(
                make_prefill_chunk_row(
                    run_id=config.get("run_id") or "unknown-run",
                    point=failure_point,
                    phase=failure_phase,
                    ordinal=failure_ordinal,
                    chunk=chunk,
                    runner_wall_time_ms=None,
                    cuda_model_time_ms=None,
                    allocation={
                        "state": "failed",
                        "actual_scheduled_tokens_by_request": {},
                        "preempted_request_ids": (),
                        "unrelated_request_ids": (),
                        "cache_epoch": cache_epoch,
                        "cache_reset_completed": False,
                        "cache_reset_empty": False,
                        "requested_blocks": 0,
                        "allocatable_blocks": 0,
                        "allocated_blocks": 0,
                        "kv_block_bytes": 0,
                        "lookup_time_ms": 0.0,
                        "allocation_time_ms": 0.0,
                        "runtime_mode": None,
                    },
                    status="failed",
                    error=error,
                )
            )
    finally:
        if runtime is not None:
            runtime.executor.shutdown()
    hardware_validated = (
        status != "failed"
        and bool(prefix_evidence)
        and all(item["hardware_validated"] for item in prefix_evidence)
    )
    return {
        "schema_version": V2_SCHEMA_VERSION,
        "run_id": config.get("run_id"),
        "role": "prefill",
        "runner_boundary": "GPUWorker.execute_model",
        "point_manifest": [point.canonical_payload for point in points],
        "prefix_evidence": prefix_evidence,
        "samples": rows,
        "startup_ms": None if runtime is None else runtime.startup_ms,
        "capture_ms": None if runtime is None else runtime.capture_ms,
        "compile_enabled": bool(
            runtime is not None
            and runtime.vllm_config.compilation_config.mode.name == "VLLM_COMPILE"
        ),
        "cudagraph_enabled": bool(
            runtime is not None
            and runtime.vllm_config.compilation_config.cudagraph_mode.name != "NONE"
        ),
        "hardware_validated": hardware_validated,
        "status": status,
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
                composition="none",
                requests=requests,
                planner_digest=planner_digest,
                seed=seed,
                block_size=block_size,
                token_budget=token_budget,
                homogeneous_prefix_tokens=homogeneous_prefix_tokens,
            )
        )

    return tuple(points)


def load_prefill_points(config: dict[str, Any]) -> tuple[PPointPlan, ...]:
    """Recompute the canonical P-side points from the pinned Ticket 02 inputs."""
    workload_plan = json.loads(Path(config["artifacts"]["workload_plan"]).read_text())
    rendered_turns = pq.read_table(config["artifacts"]["rendered_turns"]).to_pylist()
    profile = config["profile"]
    points = build_prefill_points(
        workload_plan,
        rendered_turns,
        block_size=profile["block_size"],
        token_budget=profile["max_num_batched_tokens"],
        homogeneous_prefix_tokens=profile["homogeneous_prefix_tokens"],
        seed=profile["seed"],
    )
    frozen_records = config.get("points")
    if frozen_records is not None and frozen_records != [
        _point_record(point) for point in points
    ]:
        raise ValueError("frozen P-side points differ from the pinned inputs")
    return points


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _replace_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(value)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _assemble_result(
    config_path: Path,
    preflight_path: Path,
    worker_path: Path,
    output_dir: Path,
) -> bool:
    from benchmarks.ds4_profile.profile_spine import write_v2_result_artifacts

    config = json.loads(config_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    worker = json.loads(worker_path.read_text())
    malformed_fields = [
        name
        for name in ("point_manifest", "samples", "prefix_evidence")
        if not isinstance(worker.get(name), list)
    ]
    point_manifest = worker.get("point_manifest", [])
    if malformed_fields or not all(
        isinstance(payload, dict) for payload in point_manifest
    ):
        malformed_fields.append("point_manifest payload")
        point_manifest = []
    samples = worker.get("samples", []) if not malformed_fields else []
    evidence = worker.get("prefix_evidence", []) if not malformed_fields else []
    observed = {make_point_id(payload) for payload in point_manifest}
    expected = set(config["expected_manifest"])
    worker_passed = (
        preflight.get("status") == "ready"
        and worker.get("status") in {"passed", "out_of_capacity"}
        and worker.get("returncode") == 0
        and worker.get("hardware_validated") is True
        and observed == expected
        and len(point_manifest) == len(expected)
        and not malformed_fields
    )
    provenance = {
        "schema_version": V2_SCHEMA_VERSION,
        "run_id": config["run_id"],
        "validation_state": "remote_pending" if worker_passed else "remote_failed",
        "hardware_validated": worker_passed,
        "preflight": preflight,
        "image": {"id": worker.get("image_id", "unknown")},
        "model": config.get("model"),
        "runtime": config.get("runtime"),
        "source": config.get("source", {}),
        "worker": {
            "command": worker.get("command"),
            "returncode": worker.get("returncode"),
            "status": worker.get("status"),
            "error": worker.get("error"),
            "startup_ms": worker.get("startup_ms"),
            "capture_ms": worker.get("capture_ms"),
        },
    }
    if observed != expected:
        provenance["validation_error"] = "worker manifest differs from frozen manifest"
    if malformed_fields:
        provenance["validation_error"] = "malformed worker fields: " + ", ".join(
            malformed_fields
        )
    try:
        write_v2_result_artifacts(
            config,
            samples,
            evidence,
            provenance,
            output_dir,
        )
    except ValueError as error:
        if not worker_passed:
            raise
        worker_passed = False
        provenance["validation_state"] = "remote_failed"
        provenance["hardware_validated"] = False
        provenance["validation_error"] = str(error)
        write_v2_result_artifacts(
            config,
            samples,
            evidence,
            provenance,
            output_dir,
        )
    return worker_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DS4 P-side profile.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "fixture"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--run-kind", choices=("full", "smoke"), default="full")
        command.add_argument("--output", type=Path)
    worker = subparsers.add_parser("gpu-worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--config", type=Path, required=True)
    assemble.add_argument("--preflight", type=Path, required=True)
    assemble.add_argument("--worker-result", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command in {"plan", "fixture"}:
            config = json.loads(args.config.read_text())
            frozen = freeze_expected_manifest(
                config, load_prefill_points(config), args.run_kind
            )
            if args.output is None:
                print(json.dumps(frozen, sort_keys=True))
            else:
                _write_json(args.output, frozen)
            return
        if args.command == "gpu-worker":
            config = json.loads(args.config.read_text())
            points = load_prefill_points(config)
            expected = set(config["expected_manifest"])
            selected = tuple(point for point in points if point.point_id in expected)
            if {point.point_id for point in selected} != expected:
                raise ValueError("expected_manifest references an unknown point")
            result = run_prefill_matrix(config, selected)
            result["image_id"] = os.environ.get("DS4_IMAGE_ID", "unknown")
            _write_json(args.output, result)
            if result["status"] == "failed":
                raise SystemExit(2)
            return
        if args.command == "assemble":
            passed = _assemble_result(
                args.config,
                args.preflight,
                args.worker_result,
                args.output_dir,
            )
            if not passed:
                raise SystemExit(2)
            return
        from benchmarks.ds4_profile.profile_spine import _validate_result_dir

        _validate_result_dir(args.result_dir)
        provenance_path = args.result_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        if (
            provenance.get("validation_state") == "remote_pending"
            and provenance.get("hardware_validated") is True
        ):
            provenance["validation_state"] = "remote_verified"
            _replace_text(
                provenance_path,
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            )
        if provenance.get("validation_state") != "remote_verified":
            raise ValueError("result is not hardware verified")
        print(args.result_dir)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        if args.command == "gpu-worker":
            config = json.loads(args.config.read_text())
            _write_json(
                args.output,
                {
                    "schema_version": V2_SCHEMA_VERSION,
                    "run_id": config.get("run_id"),
                    "role": "prefill",
                    "runner_boundary": "GPUWorker.execute_model",
                    "point_manifest": [],
                    "prefix_evidence": [],
                    "samples": [],
                    "hardware_validated": False,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        print(f"P-side profile failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
