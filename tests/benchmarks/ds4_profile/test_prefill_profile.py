# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import benchmarks.ds4_profile.prefill_profile as prefill_profile
from benchmarks.ds4_profile import profile_spine
from benchmarks.ds4_profile.prefill_profile import build_prefill_points
from benchmarks.ds4_profile.profile_spine import make_comparison_id, make_point_id

HOMOGENEOUS_CASES = (
    (1, 128),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (2, 128),
    (2, 512),
    (2, 1024),
    (2, 2048),
    (4, 128),
    (4, 512),
    (4, 1024),
    (8, 128),
    (8, 256),
    (8, 512),
)


def _fixture_turn(index: int, new_tokens: int, reasoning_mode: str) -> dict:
    cached_tokens = 16
    prompt_tokens = cached_tokens + new_tokens
    token_ids = [1 + (index * 37 + position) % 512 for position in range(prompt_tokens)]
    return {
        "trajectory_id": f"trajectory-{index}",
        "turn_index": index,
        "reasoning_mode": reasoning_mode,
        "prompt_tokens": prompt_tokens,
        "reusable_prefix_tokens": cached_tokens,
        "new_prefill_tokens": new_tokens,
        "execution_prompt_token_ids": token_ids,
        "execution_completion_token_ids": [1 + index],
    }


def _pinned_inputs() -> tuple[dict, list[dict]]:
    """Return a compact, deterministic Ticket 02-shaped planner fixture."""
    lengths = (16, 32, 48, 64, 80, 96, 112, 128, 256, 512)
    turns = [
        _fixture_turn(
            index,
            length,
            "no_think" if index % 2 == 0 else "think_high",
        )
        for index, length in enumerate(lengths)
    ]

    def reference(index: int) -> dict:
        turn = turns[index]
        return {
            key: turn[key]
            for key in (
                "trajectory_id",
                "turn_index",
                "reasoning_mode",
                "prompt_tokens",
                "reusable_prefix_tokens",
                "new_prefill_tokens",
            )
        }

    mixed_batches = []
    batch_indexes = {
        "similar": {2: (0, 1), 4: (0, 1, 2, 3), 8: tuple(range(8))},
        "random": {2: (8, 9), 4: (8, 3, 7, 1), 8: tuple(range(8))},
        "high_skew": {
            2: (0, 9),
            4: (0, 1, 2, 9),
            8: (0, 1, 2, 3, 4, 5, 6, 9),
        },
    }
    for composition, by_batch_size in batch_indexes.items():
        for batch_size, indexes in by_batch_size.items():
            references = [reference(index) for index in indexes]
            mixed_batches.append(
                {
                    "composition": composition,
                    "batch_size": batch_size,
                    "turns": references,
                    "total_scheduled_tokens": sum(
                        item["new_prefill_tokens"] for item in references
                    ),
                }
            )

    exact_replays = []
    for offset, reasoning_mode in enumerate(("no_think", "think_high")):
        for quantile, index in zip((0.0, 0.25, 0.5, 0.75, 1.0), range(offset, 10, 2)):
            exact_replays.append(
                {
                    **reference(index),
                    "selection_quantile": quantile,
                }
            )
    return {
        "token_budget": 4096,
        "p_homogeneous": [
            {
                "batch_size": batch_size,
                "per_request_scheduled_tokens": new_tokens,
                "total_scheduled_tokens": batch_size * new_tokens,
            }
            for batch_size, new_tokens in HOMOGENEOUS_CASES
        ],
        "mixed_batches": mixed_batches,
        "exact_replays": exact_replays,
    }, turns


def _build_pinned_points():
    plan, turns = _pinned_inputs()
    return build_prefill_points(
        plan,
        turns,
        block_size=16,
        token_budget=4096,
        homogeneous_prefix_tokens=4096,
        seed=20260715,
    )


class _FakeBlockPool:
    num_gpu_blocks = 128

    def __init__(self, free_blocks: int = 120) -> None:
        self.free_blocks = free_blocks

    def get_num_free_blocks(self) -> int:
        return self.free_blocks


class _FakeSingleTypeManager:
    def __init__(self, required_blocks: int = 1) -> None:
        self.required_blocks = required_blocks

    def get_num_blocks_to_allocate(self, *_args, **_kwargs) -> int:
        return self.required_blocks


class _FakeCoordinator:
    def __init__(self, required_blocks: int = 1) -> None:
        self.single_type_managers = [_FakeSingleTypeManager(required_blocks)]

    @property
    def required_blocks(self) -> int:
        return self.single_type_managers[0].required_blocks

    @required_blocks.setter
    def required_blocks(self, value: int) -> None:
        self.single_type_managers[0].required_blocks = value

    def get_num_blocks_to_allocate(self, *_args, **_kwargs) -> int:
        return sum(
            manager.get_num_blocks_to_allocate()
            for manager in self.single_type_managers
        )


class _FakeKvCacheManager:
    block_size = 16
    watermark_blocks = 0

    def __init__(self, free_blocks: int) -> None:
        self.block_pool = _FakeBlockPool(free_blocks)
        self.coordinator = _FakeCoordinator()
        self.owner: Any = None

    def allocate_slots(self, request, _num_new_tokens, **_kwargs):
        required = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id
        )
        if required > self.block_pool.get_num_free_blocks():
            return None
        if self.owner is not None and self.owner.allocate_during_schedule:
            self.block_pool.free_blocks -= required
        return ([0],)

    def get_block_ids(self, _request_id):
        assert self.owner is not None
        return (list(range((self.owner.prime_progress + 15) // 16)),)


class _FakeScheduler:
    def __init__(self, output, *, free_blocks: int = 120) -> None:
        self.output = output
        self.requests = {request_id: object() for request_id in output.active_ids}
        self.running = [SimpleNamespace(request_id=item) for item in output.active_ids]
        self.waiting: list[SimpleNamespace] = []
        self.scheduler_config = SimpleNamespace(long_prefill_token_threshold=0)
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=(
                SimpleNamespace(
                    layer_names=("layer.0",),
                    kv_cache_spec=SimpleNamespace(page_size_bytes=64),
                ),
            )
        )
        self.kv_cache_manager = _FakeKvCacheManager(free_blocks)
        self.kv_cache_manager.owner = self
        self.reset_succeeds = True
        self.prime_request = None
        self.prime_progress = 0
        self.prime_initial_computed = 0
        self.allocate_during_schedule = False

    def schedule(self):
        if self.prime_request is not None:
            remaining = len(self.prime_request.prompt_token_ids) - self.prime_progress
            if remaining:
                scheduled = min(self.max_num_scheduled_tokens, remaining)
                self.prime_progress += scheduled
                blocks = tuple(range((self.prime_progress + 15) // 16))
                return _fake_output(
                    {self.prime_request.request_id: scheduled},
                    computed_tokens=self.prime_progress - scheduled,
                    block_ids=blocks,
                )
        expected_total = getattr(self, "max_num_scheduled_tokens", 0)
        if expected_total:
            self.kv_cache_manager.allocate_slots(
                SimpleNamespace(request_id="r0", status="waiting"), expected_total
            )
        return self.output

    def add_request(self, request) -> None:
        self.requests[request.request_id] = request
        self.running = [request]
        self.prime_request = request
        self.prime_progress = self.prime_initial_computed

    def finish_requests(self, request_ids, _status) -> None:
        for request_id in request_ids:
            self.requests.pop(request_id, None)
        self.running = []
        self.waiting = []
        self.prime_request = None
        self.output = _fake_output({}, active_ids=())

    def update_from_output(self, _scheduler_output, _model_output) -> None:
        pass

    def reset_prefix_cache(self) -> bool:
        if self.reset_succeeds:
            self.kv_cache_manager.block_pool.free_blocks = 127
        return self.reset_succeeds


class _FakeExecutor:
    def __init__(self, *, execute_returns_none: bool = False) -> None:
        self.execute_calls = 0
        self.execute_returns_none = execute_returns_none
        self.sample_calls = 0

    def execute_model(self, _scheduler_output):
        self.execute_calls += 1
        if self.execute_returns_none:
            return None
        return SimpleNamespace()

    def sample_tokens(self, _hidden_states):
        self.sample_calls += 1
        return SimpleNamespace()


class _Cuda1Tensor(torch.Tensor):
    @property
    def is_cuda(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:1")


def _fake_output(
    tokens: dict[str, int],
    *,
    active_ids: tuple[str, ...] | None = None,
    computed_tokens: int = 0,
    block_ids: tuple[int, ...] = (10,),
    preempted: tuple[str, ...] = (),
):
    new_requests = [
        SimpleNamespace(
            req_id=request_id,
            num_computed_tokens=computed_tokens,
            block_ids=(list(block_ids),),
        )
        for request_id in tokens
    ]
    return SimpleNamespace(
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_new_reqs=new_requests,
        scheduled_cached_reqs=SimpleNamespace(req_ids=()),
        preempted_req_ids=preempted,
        active_ids=tuple(tokens) if active_ids is None else active_ids,
    )


def _adapter(output, *, tensor=None, free_blocks: int = 120, block_axes=(0,)):
    scheduler = _FakeScheduler(output, free_blocks=free_blocks)
    executor = _FakeExecutor()
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            kv_caches=[torch.empty((128, 2)) if tensor is None else tensor]
        )
    )
    return (
        prefill_profile.VllmSchedulerCacheAdapter(
            scheduler,
            executor,
            worker,
            block_axes=block_axes,
        ),
        executor,
    )


def _adapter_point(cache_condition: str = "prefix_hit"):
    request = prefill_profile.PRequestPlan(
        request_key="r0",
        trajectory_id=None,
        turn_index=None,
        reasoning_mode=None,
        prompt_token_ids=tuple(range(32)),
        context_tokens=32,
        cached_tokens=16,
        new_tokens=16,
        token_digest="d" * 64,
    )
    chunk = prefill_profile.PChunkPlan(
        0, {"r0": 16 if cache_condition == "prefix_hit" else 32}
    )
    return prefill_profile.PPointPlan(
        point_id="point",
        comparison_id="comparison",
        workload_family="homogeneous",
        selector="adapter",
        composition="none",
        seed=1,
        batch_size=1,
        cache_condition=cache_condition,
        planner_digest="e" * 64,
        requests=(request,),
        chunks=(chunk,),
        canonical_payload={},
    )


def _prime_evidence(
    *, completed=True, synchronized=True, hardware=True, block_ids=(10,)
):
    return prefill_profile.PrefixPrimeEvidence(
        request_key="r0",
        intended_cached_tokens=16,
        actual_cached_tokens=16,
        prime_scheduler_outputs=(
            prefill_profile.SchedulerBlockTableEvidence(
                chunk_index=0,
                request_key="r0",
                block_ids_by_group=(block_ids,),
                execution_completed=completed,
                gpu_synchronized=synchronized,
            ),
        ),
        measured_block_ids_by_group=(block_ids,),
        worker_groups=(),
        prime_completed=completed,
        prime_synchronized=synchronized,
        hardware_validated=hardware,
        kv_bytes=64,
    )


def test_schedule_chunk_preserves_exact_planned_request_vector() -> None:
    point = _adapter_point()
    adapter, executor = _adapter(_fake_output({"r0": 16}))

    scheduled = adapter.schedule_chunk(point, point.chunks[0])
    adapter.require_complete_chunk(scheduled)

    assert scheduled.actual_tokens_by_request == {"r0": 16}
    assert executor.execute_calls == 0


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (_fake_output({}, active_ids=("r0",)), "partial request set"),
        (_fake_output({"r0": 8}), "partial token vector"),
        (_fake_output({"r0": 16}, preempted=("r0",)), "preempted"),
        (_fake_output({"r0": 16, "other": 1}), "unrelated"),
    ],
)
def test_incomplete_scheduler_output_never_reaches_gpu_timing(output, message) -> None:
    point = _adapter_point()
    adapter, executor = _adapter(output)

    scheduled = adapter.schedule_chunk(point, point.chunks[0])
    with pytest.raises(RuntimeError, match=message):
        adapter.require_complete_chunk(scheduled)

    assert executor.execute_calls == 0


def test_hit_requires_an_executed_synchronized_prime() -> None:
    point = _adapter_point()
    output = _fake_output({"r0": 16}, computed_tokens=16)
    adapter, executor = _adapter(output)

    with pytest.raises(RuntimeError, match="prefix was not executed on GPU0"):
        adapter.verify_hit(point, output)

    assert executor.execute_calls == 0


def test_cpu_prime_executes_but_never_claims_hardware_validation() -> None:
    point = _adapter_point()
    adapter, executor = _adapter(_fake_output({}, active_ids=()))
    adapter.request_factory = lambda request_id, tokens: SimpleNamespace(
        request_id=request_id, prompt_token_ids=tokens
    )

    evidence = adapter.prime(point, "warmup", 0)

    assert len(evidence) == 1
    assert evidence[0].prime_completed
    assert not evidence[0].prime_synchronized
    assert not evidence[0].hardware_validated
    assert not evidence[0].worker_groups[0].live_cuda_tensor_proven
    assert executor.execute_calls == 2


def test_prime_chunks_a_prefix_larger_than_the_scheduler_budget() -> None:
    point = _adapter_point()
    request = replace(
        point.requests[0],
        prompt_token_ids=tuple(range(4128)),
        context_tokens=4128,
        cached_tokens=4112,
    )
    point = replace(point, requests=(request,))
    adapter, _ = _adapter(_fake_output({}, active_ids=()), tensor=torch.empty((512, 2)))
    adapter.request_factory = lambda request_id, tokens: SimpleNamespace(
        request_id=request_id, prompt_token_ids=tokens
    )

    evidence = adapter.prime(point, "steady", 0)

    assert [item.chunk_index for item in evidence[0].prime_scheduler_outputs] == [0, 1]
    assert len(evidence[0].measured_block_ids_by_group[0]) == 257


def test_prime_accepts_blocks_computed_by_an_earlier_shared_prefix() -> None:
    point = _adapter_point()
    request = replace(
        point.requests[0],
        prompt_token_ids=tuple(range(48)),
        context_tokens=48,
        cached_tokens=32,
    )
    point = replace(point, requests=(request,))
    adapter, _ = _adapter(_fake_output({}, active_ids=()))
    adapter.scheduler.prime_initial_computed = 16
    adapter.request_factory = lambda request_id, tokens: SimpleNamespace(
        request_id=request_id, prompt_token_ids=tokens
    )

    evidence = adapter.prime(point, "steady", 0)

    prime = evidence[0]
    assert len(prime.prime_scheduler_outputs) == 1
    assert len(prime.measured_block_ids_by_group[0]) == 2


def test_prime_handles_executor_none_with_sampling_fallback() -> None:
    point = _adapter_point()
    adapter, _ = _adapter(_fake_output({}, active_ids=()))
    executor = _FakeExecutor(execute_returns_none=True)
    adapter.executor = executor
    adapter.request_factory = lambda request_id, tokens: SimpleNamespace(
        request_id=request_id, prompt_token_ids=tokens
    )

    evidence = adapter.prime(point, "warmup", 0)

    assert evidence[0].prime_completed
    assert executor.sample_calls == 2


def test_cached_request_block_tables_use_request_major_api_shape() -> None:
    cached = SimpleNamespace(
        new_block_ids=[([1, 2], [3]), ([4], [5, 6])],
        num_computed_tokens=[16, 32],
    )

    assert prefill_profile.VllmSchedulerCacheAdapter._block_ids((cached, 1)) == (
        (4,),
        (5, 6),
    )


def test_worker_tensors_follow_bind_kv_cache_layer_order() -> None:
    adapter, _ = _adapter(_fake_output({"r0": 16}))
    adapter.scheduler.kv_cache_config.kv_cache_groups = (
        SimpleNamespace(layer_names=("layer.1",)),
        SimpleNamespace(layer_names=("layer.0",)),
    )
    adapter.worker.model_runner.kv_caches = [
        torch.empty((4, 2)),
        torch.empty((8, 2)),
    ]
    adapter.block_axes = (0, 0)
    adapter.layer_order = ("layer.0", "layer.1")

    groups = adapter._inspect_worker_groups(((7,), (3,)), require_hardware=False)

    assert [group.block_dimension for group in groups] == [8, 4]


def test_hit_rejects_different_physical_block_ids_before_timing() -> None:
    point = _adapter_point()
    output = _fake_output({"r0": 16}, computed_tokens=16, block_ids=(11,))
    adapter, executor = _adapter(output)
    adapter._prime_evidence["r0"] = _prime_evidence()

    with pytest.raises(RuntimeError, match="physical block IDs"):
        adapter.verify_hit(point, output)

    assert executor.execute_calls == 0


def test_cpu_tensor_cannot_validate_resident_gpu0_blocks() -> None:
    point = _adapter_point()
    output = _fake_output({"r0": 16}, computed_tokens=16)
    adapter, executor = _adapter(output)
    adapter._prime_evidence["r0"] = _prime_evidence()

    with pytest.raises(RuntimeError, match="not on cuda:0"):
        adapter.verify_hit(point, output)

    assert executor.execute_calls == 0


@pytest.mark.parametrize(
    ("tensor", "block_axes", "block_ids", "message"),
    [
        (object(), (0,), (10,), "not a torch.Tensor"),
        (torch.empty((4, 2)), (2,), (1,), "invalid.*block axis"),
        (torch.empty((4, 2)), (0,), (4,), "outside the live tensor"),
    ],
)
def test_live_tensor_inspection_fails_closed(
    tensor, block_axes, block_ids, message
) -> None:
    adapter, _ = _adapter(
        _fake_output({"r0": 16}), tensor=tensor, block_axes=block_axes
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._inspect_worker_groups((block_ids,), require_hardware=False)


def test_live_tensor_inspection_rejects_cuda1() -> None:
    tensor = torch.Tensor._make_subclass(
        _Cuda1Tensor, torch.empty((128, 2)), require_grad=False
    )
    adapter, _ = _adapter(_fake_output({"r0": 16}), tensor=tensor)

    with pytest.raises(RuntimeError, match="not on cuda:0"):
        adapter._inspect_worker_groups(((10,),), require_hardware=True)


def test_full_recompute_requires_zero_cached_tokens() -> None:
    point = _adapter_point("full_recompute")
    output = _fake_output({"r0": 32}, computed_tokens=16)
    adapter, executor = _adapter(output)

    with pytest.raises(RuntimeError, match="unexpectedly reused"):
        adapter.verify_recompute_miss(point, output)

    assert executor.execute_calls == 0


def test_reset_epoch_rejects_failed_prefix_cache_reset() -> None:
    adapter, _ = _adapter(_fake_output({"r0": 16}))
    adapter.scheduler.reset_succeeds = False

    with pytest.raises(RuntimeError, match="prefix cache reset failed"):
        adapter.reset_epoch()


def test_reset_epoch_synchronizes_the_gpu_flush() -> None:
    adapter, _ = _adapter(_fake_output({"r0": 16}))
    synchronizations = []
    adapter.synchronize_gpu = lambda: synchronizations.append("sync")

    adapter.reset_epoch()

    assert synchronizations == ["sync"]


def test_allocator_ooc_requires_pressure_and_clean_reset() -> None:
    point = _adapter_point()
    adapter, executor = _adapter(_fake_output({}, active_ids=("r0",)), free_blocks=0)
    scheduled = adapter.schedule_chunk(point, point.chunks[0])

    allocation = adapter.classify_out_of_capacity(point, point.chunks[0], scheduled)

    assert allocation is not None
    assert allocation.state == "out_of_capacity"
    assert allocation.allocator_pressure_proven
    assert allocation.clean_reset_proven


def test_allocator_ooc_records_authoritative_manager_requirement() -> None:
    point = _adapter_point()
    adapter, _ = _adapter(_fake_output({}, active_ids=("r0",)), free_blocks=2)
    adapter.scheduler.kv_cache_manager.coordinator.required_blocks = 3

    scheduled = adapter.schedule_chunk(point, point.chunks[0])
    allocation = adapter.classify_out_of_capacity(point, point.chunks[0], scheduled)

    assert allocation is not None
    assert allocation.requested_blocks == 3
    assert allocation.allocatable_blocks == 2
    assert allocation.allocated_blocks == 0


def test_successful_allocation_records_blocks_and_group_bytes() -> None:
    point = _adapter_point()
    adapter, _ = _adapter(_fake_output({"r0": 16}), free_blocks=2)
    adapter.scheduler.allocate_during_schedule = True

    scheduled = adapter.schedule_chunk(point, point.chunks[0])

    assert scheduled.allocation.requested_blocks == 1
    assert scheduled.allocation.allocated_blocks == 1
    assert scheduled.allocation.allocatable_blocks == 2
    assert scheduled.allocation.requested_bytes == 64
    assert scheduled.allocation.allocated_bytes == 64


def test_partial_output_without_allocator_pressure_is_invalid() -> None:
    point = _adapter_point()
    adapter, executor = _adapter(_fake_output({}, active_ids=("r0",)), free_blocks=120)
    scheduled = adapter.schedule_chunk(point, point.chunks[0])

    with pytest.raises(RuntimeError, match="allocator-pressure proof"):
        adapter.classify_out_of_capacity(point, point.chunks[0], scheduled)

    assert executor.execute_calls == 0


class _OrchestrationAdapter:
    def __init__(self, events: list[str], outcome: str = "complete") -> None:
        self.events = events
        self.outcome = outcome

    def reset_epoch(self) -> None:
        self.events.append("reset")

    def prime(self, _point, _phase, _ordinal):
        self.events.append("prime_gpu0")
        return ()

    def add_measurement_requests(self, _point) -> None:
        pass

    def schedule_chunk(self, point, chunk):
        self.last_chunk_index = chunk.chunk_index
        expected = dict(chunk.scheduled_tokens_by_request)
        actual = dict(expected)
        preempted: tuple[str, ...] = ()
        unrelated: tuple[str, ...] = ()
        if self.outcome in {"empty", "ooc"} or (
            self.outcome in {"late_ooc", "late_invalid"} and chunk.chunk_index == 1
        ):
            actual = {}
        elif self.outcome == "partial_tokens":
            actual["r0"] -= 1
        elif self.outcome == "preempted":
            preempted = ("r0",)
        elif self.outcome == "unrelated":
            actual["other"] = 1
            unrelated = ("other",)
        allocation = prefill_profile.AllocationEvidence(
            state="allocated",
            requested_blocks=1,
            allocated_blocks=1,
            allocatable_blocks=8,
            requested_bytes=64,
            allocated_bytes=64,
            lookup_time_ms=0.1,
            allocation_time_ms=0.1,
            allocator_pressure_proven=False,
            clean_reset_proven=False,
        )
        return prefill_profile.ScheduledChunk(
            scheduler_output=SimpleNamespace(),
            expected_request_ids=tuple(expected),
            actual_request_ids=tuple(actual),
            expected_tokens_by_request=expected,
            actual_tokens_by_request=actual,
            preempted_request_ids=preempted,
            unrelated_request_ids=unrelated,
            allocation=allocation,
        )

    def classify_out_of_capacity(self, _point, _chunk, scheduled):
        if self.outcome not in {"ooc", "late_ooc"} or (
            self.outcome == "late_ooc" and self.last_chunk_index != 1
        ):
            return None
        self.events.append("clean_reset")
        return replace(
            scheduled.allocation,
            state="out_of_capacity",
            requested_blocks=9,
            allocated_blocks=0,
            allocatable_blocks=8,
            requested_bytes=576,
            allocated_bytes=0,
            allocator_pressure_proven=True,
            clean_reset_proven=True,
        )

    def require_complete_chunk(self, scheduled) -> None:
        prefill_profile.VllmSchedulerCacheAdapter.require_complete_chunk(scheduled)

    def verify_hit(self, _point, _output) -> None:
        self.events.append("verify_resident")

    def verify_recompute_miss(self, _point, _output) -> None:
        self.events.append("verify_miss")

    def update_after_execute(self, _scheduler_output, _model_output) -> None:
        self.events.append("update")

    def _kv_page_bytes(self):
        return (64,)


def _timed_executor(events: list[str]):
    def execute(_scheduler_output):
        events.append("timed:0")
        output = SimpleNamespace(cudagraph_stats=SimpleNamespace(runtime_mode="FULL"))
        return output, 1.0, 0.5

    return execute


def test_hit_repetition_primes_and_verifies_before_any_timed_chunk() -> None:
    events: list[str] = []
    result = prefill_profile.run_point_repetition(
        _adapter_point(),
        phase="steady",
        ordinal=0,
        adapter=_OrchestrationAdapter(events),
        execute_timed=_timed_executor(events),
    )

    assert events[:4] == ["reset", "prime_gpu0", "verify_resident", "timed:0"]
    assert result["status"] == "passed"


@pytest.mark.parametrize(
    "outcome", ["empty", "partial_tokens", "preempted", "unrelated"]
)
def test_incomplete_repetition_output_is_never_timed(outcome: str) -> None:
    events: list[str] = []

    result = prefill_profile.run_point_repetition(
        _adapter_point(),
        phase="steady",
        ordinal=0,
        adapter=_OrchestrationAdapter(events, outcome),
        execute_timed=_timed_executor(events),
    )

    assert result["status"] == "failed"
    assert result["rows"][0]["status"] == "failed"
    assert not any(event.startswith("timed:") for event in events)


def test_allocator_pressure_emits_one_untimed_terminal_row() -> None:
    events: list[str] = []
    result = prefill_profile.run_point_repetition(
        _adapter_point(),
        phase="warmup",
        ordinal=0,
        adapter=_OrchestrationAdapter(events, "ooc"),
        execute_timed=_timed_executor(events),
    )

    assert result["status"] == "out_of_capacity"
    assert len(result["rows"]) == 1
    assert result["rows"][0]["row_kind"] == "terminal"
    assert result["rows"][0]["runner_wall_time_ms"] is None
    assert result["rows"][0]["runtime_mode"] is None
    assert isinstance(result["rows"][0]["cache_epoch"], int)
    assert events == ["reset", "prime_gpu0", "clean_reset"]


def test_later_allocator_pressure_preserves_only_completed_chunk_timing() -> None:
    events: list[str] = []
    result = prefill_profile.run_point_repetition(
        _artifact_point("prefix_hit"),
        phase="warmup",
        ordinal=0,
        adapter=_OrchestrationAdapter(events, "late_ooc"),
        execute_timed=_timed_executor(events),
    )

    assert result["status"] == "out_of_capacity"
    assert [row["status"] for row in result["rows"]] == [
        "passed",
        "out_of_capacity",
    ]
    assert result["rows"][0]["runner_wall_time_ms"] == 1.0
    assert result["rows"][1]["runner_wall_time_ms"] is None
    assert events.count("timed:0") == 1


def test_later_invalid_output_preserves_completed_and_failed_coordinates() -> None:
    events: list[str] = []
    result = prefill_profile.run_point_repetition(
        _artifact_point("prefix_hit"),
        phase="steady",
        ordinal=0,
        adapter=_OrchestrationAdapter(events, "late_invalid"),
        execute_timed=_timed_executor(events),
    )

    assert result["status"] == "failed"
    assert [row["status"] for row in result["rows"]] == ["passed", "failed"]
    assert [row["chunk_index"] for row in result["rows"]] == [0, 1]
    assert events.count("timed:0") == 1


def test_reset_flushes_pending_finished_ids_without_live_requests() -> None:
    adapter, executor = _adapter(_fake_output({}, active_ids=()))
    adapter.scheduler.finished_req_ids = {"r0"}

    adapter.reset_epoch()

    assert executor.execute_calls == 1


def test_prefix_evidence_is_flattened_to_the_v2_group_schema() -> None:
    worker_group = prefill_profile.WorkerKvTensorGroupEvidence(
        group_index=0,
        tensor_names=("layer.0",),
        tensor_devices=("cuda:0",),
        tensor_shapes=((128, 2),),
        block_axis=0,
        block_dimension=128,
        verified_block_ids=(10,),
        live_cuda_tensor_proven=True,
    )
    evidence = replace(
        _prime_evidence(),
        worker_groups=(worker_group,),
    )

    rows = prefill_profile._flatten_prefix_evidence(
        "run", "point", "steady", 0, evidence
    )

    assert rows == [
        {
            "schema_version": "2.0.0",
            "run_id": "run",
            "point_id": "point",
            "phase": "steady",
            "ordinal": 0,
            "request_key": "r0",
            "kv_cache_group": "group-0",
            "prime_scheduler_block_ids": [10],
            "measured_scheduler_block_ids": [10],
            "live_kv_tensor_names": ["layer.0"],
            "live_kv_tensor_devices": ["cuda:0"],
            "live_kv_tensor_shapes": [[128, 2]],
            "block_axis": 0,
            "block_dimension": 128,
            "verified_physical_block_ids": [10],
            "intended_cached_tokens": 16,
            "actual_cached_tokens": 16,
            "prime_completed": True,
            "prime_synchronized": True,
            "live_cuda_tensor_proven": True,
            "hardware_validated": True,
        }
    ]


def test_execute_worker_step_times_only_execute_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.ds4_profile import gpu_profile

    events = []

    class FakeEvent:
        def __init__(self, index: int) -> None:
            self.index = index

        def record(self) -> None:
            events.append(f"record:{self.index}")

        def synchronize(self) -> None:
            events.append(f"event_sync:{self.index}")

        def elapsed_time(self, _other) -> float:
            return 0.5

    event_count = 0

    def make_event(**_kwargs):
        nonlocal event_count
        event = FakeEvent(event_count)
        event_count += 1
        return event

    class Executor:
        def execute_model(self, _output):
            events.append("execute")
            return None

        def sample_tokens(self, _hidden):
            events.append("sample")
            return SimpleNamespace()

    monkeypatch.setattr(torch, "Event", make_event)
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda *_args, **_kwargs: events.append("accelerator_sync"),
    )
    runtime = gpu_profile.GpuRuntime(
        Executor(), None, None, None, startup_ms=0.0, capture_ms=0.0
    )

    _, wall_ms, cuda_ms = gpu_profile.execute_worker_step(
        runtime, SimpleNamespace(), timed=True
    )

    assert events == [
        "accelerator_sync",
        "record:0",
        "execute",
        "record:1",
        "event_sync:1",
        "sample",
        "accelerator_sync",
    ]
    assert wall_ms is not None
    assert cuda_ms == 0.5


def test_run_prefill_matrix_uses_public_runtime_and_always_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks.ds4_profile import gpu_profile
    from vllm.v1 import structured_output
    from vllm.v1.core import kv_cache_utils

    events: list[str] = []

    class Executor:
        def shutdown(self) -> None:
            events.append("shutdown")

    scheduler_calls = []

    def make_scheduler(**kwargs):
        scheduler_calls.append(kwargs)
        return SimpleNamespace()

    scheduler_config = SimpleNamespace(get_scheduler_cls=lambda: make_scheduler)
    compilation_config = SimpleNamespace(
        mode=SimpleNamespace(name="VLLM_COMPILE"),
        cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE"),
    )
    runtime = gpu_profile.GpuRuntime(
        Executor(),
        SimpleNamespace(),
        SimpleNamespace(
            scheduler_config=scheduler_config,
            compilation_config=compilation_config,
        ),
        SimpleNamespace(),
        startup_ms=10.0,
        capture_ms=20.0,
    )
    adapter = _OrchestrationAdapter(events)
    monkeypatch.setattr(gpu_profile, "initialize_gpu_runtime", lambda _config: runtime)
    monkeypatch.setattr(
        gpu_profile,
        "execute_worker_step",
        lambda _runtime, _output, *, timed: _timed_executor(events)(_output),
    )
    monkeypatch.setattr(
        kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (16, 16)
    )
    monkeypatch.setattr(
        structured_output, "StructuredOutputManager", lambda _config: object()
    )
    monkeypatch.setattr(
        prefill_profile.VllmSchedulerCacheAdapter,
        "from_runtime",
        classmethod(lambda _cls, *_args, **_kwargs: adapter),
    )
    config = {
        "run_id": "matrix-run",
        "profile": {"warmup_repetitions": 0, "measured_repetitions": 1},
    }

    result = prefill_profile.run_prefill_matrix(
        config, (_adapter_point("full_recompute"),)
    )

    assert result["status"] == "passed"
    assert result["error"] is None
    assert len(result["samples"]) == 1
    assert result["startup_ms"] == 10.0
    assert set(scheduler_calls[0]) == {
        "vllm_config",
        "kv_cache_config",
        "structured_output_manager",
        "include_finished_set",
        "log_stats",
        "block_size",
        "hash_block_size",
    }
    assert events[-1] == "shutdown"


def test_planner_expands_the_pinned_matrix_to_68_points() -> None:
    points = _build_pinned_points()

    assert len(points) == 68
    assert Counter(point.workload_family for point in points) == {
        "homogeneous": 30,
        "mixed": 18,
        "exact_replay": 20,
    }
    assert all(point.batch_size <= 8 for point in points)
    assert all(
        sum(chunk.scheduled_tokens_by_request.values()) <= 4096
        for point in points
        for chunk in point.chunks
    )
    assert len({point.planner_digest for point in points}) == 1
    assert all(
        point.canonical_payload["planned_chunks"]
        == [
            {
                "chunk_index": chunk.chunk_index,
                "scheduled_tokens_by_request": sorted(
                    chunk.scheduled_tokens_by_request.items()
                ),
            }
            for chunk in point.chunks
        ]
        for point in points
    )


def test_homogeneous_prefixes_are_4096_tokens_and_request_distinct() -> None:
    point = next(
        point
        for point in _build_pinned_points()
        if point.selector == "b8-t512" and point.cache_condition == "prefix_hit"
    )

    assert {request.cached_tokens for request in point.requests} == {4096}
    assert len({request.token_digest for request in point.requests}) == 8
    assert point.chunks[0].scheduled_tokens_by_request == {
        request.request_key: 512 for request in point.requests
    }


def test_planner_preserves_requests_and_pairs_conditions() -> None:
    plan, _ = _pinned_inputs()
    points = _build_pinned_points()

    assert len({point.point_id for point in points}) == len(points)
    assert [point.point_id for point in points] == [
        point.point_id for point in _build_pinned_points()
    ]
    assert set(Counter(point.comparison_id for point in points).values()) == {2}

    mixed = next(
        point
        for point in points
        if point.workload_family == "mixed"
        and point.selector == "random-b4"
        and point.cache_condition == "prefix_hit"
    )
    expected_mixed = next(
        batch
        for batch in plan["mixed_batches"]
        if batch["composition"] == "random" and batch["batch_size"] == 4
    )
    assert [request.trajectory_id for request in mixed.requests] == [
        turn["trajectory_id"] for turn in expected_mixed["turns"]
    ]
    assert [request.turn_index for request in mixed.requests] == [
        turn["turn_index"] for turn in expected_mixed["turns"]
    ]

    replay = next(
        point
        for point in points
        if point.workload_family == "exact_replay"
        and point.selector == "no_think-q00"
        and point.cache_condition == "prefix_hit"
    )
    selected = next(
        item
        for item in plan["exact_replays"]
        if item["reasoning_mode"] == "no_think" and item["selection_quantile"] == 0.0
    )
    assert (replay.requests[0].trajectory_id, replay.requests[0].turn_index) == (
        selected["trajectory_id"],
        selected["turn_index"],
    )


def test_full_recompute_chunks_context_and_removes_completed_requests() -> None:
    points = _build_pinned_points()
    point = next(
        point
        for point in points
        if point.selector == "high_skew-b8"
        and point.cache_condition == "full_recompute"
    )

    scheduled: Counter[str] = Counter()
    active_sizes = []
    for chunk in point.chunks:
        active_sizes.append(len(chunk.scheduled_tokens_by_request))
        scheduled.update(chunk.scheduled_tokens_by_request)
    assert active_sizes == sorted(active_sizes, reverse=True)
    assert active_sizes[0] == 8
    assert active_sizes[-1] < 8
    assert scheduled == {
        request.request_key: request.context_tokens for request in point.requests
    }


def test_planner_rejects_invalid_capacity_or_prefix_alignment() -> None:
    plan, turns = _pinned_inputs()
    too_many = copy.deepcopy(plan)
    too_many["p_homogeneous"].append(
        {
            "batch_size": 9,
            "per_request_scheduled_tokens": 1,
            "total_scheduled_tokens": 9,
        }
    )

    with pytest.raises(ValueError, match="eight sequences"):
        build_prefill_points(
            too_many,
            turns,
            block_size=16,
            token_budget=4096,
            homogeneous_prefix_tokens=4096,
            seed=20260715,
        )
    with pytest.raises(ValueError, match="block_size must be 16"):
        build_prefill_points(
            plan,
            turns,
            block_size=8,
            token_budget=4096,
            homogeneous_prefix_tokens=4096,
            seed=20260715,
        )
    with pytest.raises(ValueError, match="homogeneous prefix must be 4096"):
        build_prefill_points(
            plan,
            turns,
            block_size=16,
            token_budget=4096,
            homogeneous_prefix_tokens=4080,
            seed=20260715,
        )


def test_point_id_covers_planned_chunks_and_planner_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = _build_pinned_points()[0]
    changed_chunk = copy.deepcopy(point.canonical_payload)
    changed_chunk["planned_chunks"][0]["scheduled_tokens_by_request"][0] = (
        "r0",
        changed_chunk["planned_chunks"][0]["scheduled_tokens_by_request"][0][1] - 1,
    )
    assert make_point_id(changed_chunk) != point.point_id

    monkeypatch.setattr(prefill_profile, "CHUNK_ALGORITHM", "changed-chunk-v2")
    changed_chunk_algorithm = _build_pinned_points()[0]
    assert changed_chunk_algorithm.planner_digest != point.planner_digest
    assert changed_chunk_algorithm.point_id != point.point_id

    monkeypatch.setattr(
        prefill_profile,
        "HOMOGENEOUS_TOKEN_ALGORITHM",
        "changed-homogeneous-token-v2",
    )
    changed_algorithms = _build_pinned_points()[0]
    assert changed_algorithms.planner_digest != changed_chunk_algorithm.planner_digest
    assert changed_algorithms.point_id != changed_chunk_algorithm.point_id


def _artifact_point(cache_condition: str) -> prefill_profile.PPointPlan:
    request = prefill_profile.PRequestPlan(
        request_key="r0",
        trajectory_id=None,
        turn_index=None,
        reasoning_mode=None,
        prompt_token_ids=tuple(range(200)),
        context_tokens=200,
        cached_tokens=100,
        new_tokens=100,
        token_digest="a" * 64,
    )
    chunks = (
        prefill_profile.PChunkPlan(0, {"r0": 40}),
        prefill_profile.PChunkPlan(1, {"r0": 60}),
    )
    payload = {
        "workload_family": "homogeneous",
        "selector": "artifact-b1-t100",
        "requests": [
            {
                "request_key": request.request_key,
                "trajectory_id": None,
                "turn_index": None,
                "reasoning_mode": None,
                "context_tokens": request.context_tokens,
                "cached_tokens": request.cached_tokens,
                "new_tokens": request.new_tokens,
                "token_digest": request.token_digest,
            }
        ],
        "composition": "none",
        "seed": 20260715,
        "batch_size": 1,
        "chunk_budget": 4096,
        "cache_condition": cache_condition,
        "block_size": 16,
        "homogeneous_prefix_tokens": 4096,
        "capacity_target": "native",
        "planner_digest": "b" * 64,
        "planned_chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "scheduled_tokens_by_request": sorted(
                    chunk.scheduled_tokens_by_request.items()
                ),
            }
            for chunk in chunks
        ],
    }
    return prefill_profile.PPointPlan(
        point_id=make_point_id(payload),
        comparison_id=make_comparison_id(payload),
        workload_family="homogeneous",
        selector=payload["selector"],
        composition="none",
        seed=payload["seed"],
        batch_size=1,
        cache_condition=cache_condition,
        planner_digest=payload["planner_digest"],
        requests=(request,),
        chunks=chunks,
        canonical_payload=payload,
    )


def _passed_chunk_row(
    point: prefill_profile.PPointPlan,
    ordinal: int,
    chunk: prefill_profile.PChunkPlan,
    wall_time_ms: float,
) -> dict:
    return prefill_profile.make_prefill_chunk_row(
        run_id="run-task-3",
        point=point,
        phase="steady",
        ordinal=ordinal,
        chunk=chunk,
        runner_wall_time_ms=wall_time_ms,
        cuda_model_time_ms=wall_time_ms - 0.5,
        allocation={
            "state": "allocated",
            "actual_scheduled_tokens_by_request": (chunk.scheduled_tokens_by_request),
            "preempted_request_ids": (),
            "unrelated_request_ids": (),
            "cache_epoch": ordinal,
            "cache_reset_completed": True,
            "cache_reset_empty": True,
            "requested_blocks": 8,
            "allocatable_blocks": 128,
            "allocated_blocks": 8,
            "kv_block_bytes": 1024,
            "lookup_time_ms": 0.2,
            "allocation_time_ms": 0.3,
            "runtime_mode": "FULL",
        },
        status="passed",
        error=None,
    )


def _artifact_rows() -> tuple[
    prefill_profile.PPointPlan,
    prefill_profile.PPointPlan,
    list[dict],
]:
    hit = _artifact_point("prefix_hit")
    recompute = _artifact_point("full_recompute")
    raw_rows: list[dict] = []
    for ordinal in range(10):
        raw_rows.extend(
            _passed_chunk_row(hit, ordinal, chunk, wall_time)
            for chunk, wall_time in zip(hit.chunks, (4.0, 6.0), strict=True)
        )
        raw_rows.extend(
            _passed_chunk_row(recompute, ordinal, chunk, wall_time)
            for chunk, wall_time in zip(recompute.chunks, (6.0, 10.0), strict=True)
        )
    return hit, recompute, raw_rows


def test_turn_statistics_are_recomputed_from_chunks() -> None:
    hit, recompute, raw_rows = _artifact_rows()

    turns = profile_spine.summarize_turn_samples(raw_rows, (hit, recompute))

    hit_turn = next(row for row in turns if row["cache_condition"] == "prefix_hit")
    assert hit_turn["runner_wall_time_ms"] == 10.0
    assert hit_turn["throughput_tokens_per_s"] == 10_000.0


def test_aggregates_use_exactly_ten_steady_turns() -> None:
    hit, recompute, raw_rows = _artifact_rows()
    turns = profile_spine.summarize_turn_samples(raw_rows, (hit, recompute))

    aggregates = profile_spine.aggregate_turn_samples(turns, 0.05)

    assert {row["sample_count"] for row in aggregates} == {10}


def test_comparison_statistics_are_recomputed_from_aggregates() -> None:
    hit, recompute, raw_rows = _artifact_rows()
    turns = profile_spine.summarize_turn_samples(raw_rows, (hit, recompute))
    aggregates = profile_spine.aggregate_turn_samples(turns, 0.05)

    comparisons = profile_spine.compare_conditions(aggregates, (hit, recompute), [])

    assert comparisons[0]["recompute_penalty_ms"] == 6.0


def test_comparison_rejects_unvalidated_terminal_ooc_claim() -> None:
    hit, recompute, raw_rows = _artifact_rows()
    turns = profile_spine.summarize_turn_samples(raw_rows, (hit, recompute))
    aggregates = profile_spine.aggregate_turn_samples(turns, 0.05)
    terminal = _passed_chunk_row(hit, 0, hit.chunks[0], 4.0)
    terminal.update(
        row_kind="terminal",
        status="out_of_capacity",
        allocation_state="out_of_capacity",
        requested_kv_blocks=129,
        runner_wall_time_ms=None,
        cuda_model_time_ms=None,
        runtime_mode=None,
    )

    with pytest.raises(ValueError, match="unvalidated terminal OOC"):
        profile_spine.compare_conditions(
            [row for row in aggregates if row["point_id"] != hit.point_id],
            (hit, recompute),
            [terminal],
        )


@pytest.mark.parametrize(
    "mutation",
    (
        {"chunk_index": -1},
        {"chunk_count": 1},
        {"phase": "steady", "ordinal": 10},
        {"sample_id": "wrong-coordinate"},
    ),
)
def test_comparison_requires_canonical_terminal_coordinates(
    mutation: dict[str, object],
) -> None:
    hit, recompute, raw_rows = _artifact_rows()
    turns = profile_spine.summarize_turn_samples(raw_rows, (hit, recompute))
    aggregates = profile_spine.aggregate_turn_samples(turns, 0.05)
    terminal = _passed_chunk_row(hit, 0, hit.chunks[0], 4.0)
    terminal.update(
        row_kind="terminal",
        status="out_of_capacity",
        allocation_state="out_of_capacity",
        requested_kv_blocks=129,
        allocatable_kv_blocks=128,
        requested_kv_bytes=129 * 1024,
        allocator_pressure_proven=True,
        clean_reset_proven=True,
        runner_wall_time_ms=None,
        cuda_model_time_ms=None,
        runtime_mode=None,
    )
    terminal.update(mutation)

    with pytest.raises(ValueError, match="unvalidated terminal OOC"):
        profile_spine.compare_conditions(
            [row for row in aggregates if row["point_id"] != hit.point_id],
            (hit, recompute),
            [terminal],
        )
