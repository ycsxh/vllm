# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math

import pytest

from tda_forward.contracts import (
    AdmissionOutcome,
    AdmissionResult,
    CacheAction,
    CacheActionAck,
    CacheActionStatus,
    PlannedPath,
)
from tda_forward.coordinator import ReentryCoordinator, RequestValidationError
from tda_forward.fakes import FakeDecodeAdapter, FakePrefillAdapter
from tda_forward.mirror import DecodePrefixMirror


def payload(
    session_id: object = "s",
    t_pred: object = 5.0,
    *,
    tokens: list[int] | None = None,
    stream: bool = False,
) -> dict[str, object]:
    return {
        "model": "fake",
        "prompt": tokens or list(range(8)),
        "session_id": session_id,
        "t_pred": t_pred,
        "stream": stream,
    }


async def run_turn(
    coordinator: ReentryCoordinator, request: dict[str, object]
) -> list[dict[str, object]]:
    return [chunk async for chunk in coordinator.start_turn(request)]


def make_coordinator(
    *, g: float = 10.0
) -> tuple[ReentryCoordinator, FakePrefillAdapter, FakeDecodeAdapter]:
    prefill = FakePrefillAdapter()
    decode = FakeDecodeAdapter()
    coordinator = ReentryCoordinator(
        g=g,
        block_size=4,
        prefill=prefill,
        decode=decode,
        mirror=DecodePrefixMirror(block_size=4),
    )
    return coordinator, prefill, decode


@pytest.mark.parametrize(
    ("body", "g"),
    [
        (payload(session_id=None), 10.0),
        (payload(session_id=""), 10.0),
        (payload(session_id=math.inf), 10.0),
        (payload(t_pred="soon"), 10.0),
        (payload(t_pred=math.nan), 10.0),
        (payload(), math.inf),
    ],
)
def test_invalid_policy_inputs_fail_before_dispatch(
    body: dict[str, object], g: float
) -> None:
    prefill = FakePrefillAdapter()
    decode = FakeDecodeAdapter()

    if math.isfinite(g):
        coordinator = ReentryCoordinator(
            g=g,
            block_size=4,
            prefill=prefill,
            decode=decode,
            mirror=DecodePrefixMirror(block_size=4),
        )
        with pytest.raises(RequestValidationError):
            coordinator.start_turn(body)
    else:
        with pytest.raises(ValueError):
            ReentryCoordinator(
                g=g,
                block_size=4,
                prefill=prefill,
                decode=decode,
                mirror=DecodePrefixMirror(block_size=4),
            )
    assert prefill.calls == []
    assert decode.calls == []


@pytest.mark.asyncio
async def test_boundary_rule_drives_the_next_turn() -> None:
    coordinator, prefill, decode = make_coordinator()

    await run_turn(coordinator, payload(t_pred=10.0))
    assert coordinator.plan_for("s").action is CacheAction.RETAIN_D
    assert coordinator.plan_for("s").path is PlannedPath.D_LOCAL_AP

    decode.queue_admission(
        "s",
        AdmissionResult(
            outcome=AdmissionOutcome.D_HIT,
            actual_local_cached_tokens=8,
            prompt_tokens=8,
            locally_computed_tokens=0,
        ),
    )
    await run_turn(coordinator, payload(t_pred=10.01))
    assert coordinator.plan_for("s").action is CacheAction.EVICT_D
    assert coordinator.plan_for("s").path is PlannedPath.P_SIDE_AP
    assert len(prefill.calls) == 1
    assert [call.kind for call in decode.calls].count("admit_local") == 1


@pytest.mark.asyncio
async def test_first_and_p_bound_turn_use_prefill_side_path() -> None:
    coordinator, prefill, decode = make_coordinator()

    chunks = await run_turn(coordinator, payload(t_pred=50.0))

    assert chunks[-1]["choices"] == [{"text": "prefill"}]
    assert [call.kind for call in prefill.calls] == ["prefill"]
    assert [call.kind for call in decode.calls] == ["stream_from_prefill", "finish"]
    record = coordinator.records[-1]
    assert record.g == 10.0
    assert record.planned_path is PlannedPath.P_SIDE_AP
    assert record.engine_outcome is None
    assert record.eviction_status is CacheActionStatus.EVICTED


@pytest.mark.asyncio
@pytest.mark.parametrize(("cached", "computed"), [(8, 0), (4, 4)])
async def test_full_and_partial_d_hit_stay_on_decode(
    cached: int, computed: int
) -> None:
    coordinator, prefill, decode = make_coordinator()
    await run_turn(coordinator, payload(t_pred=1.0))
    decode.queue_admission(
        "s",
        AdmissionResult(
            outcome=AdmissionOutcome.D_HIT,
            actual_local_cached_tokens=cached,
            prompt_tokens=8,
            locally_computed_tokens=computed,
        ),
    )

    chunks = await run_turn(coordinator, payload(t_pred=1.0))

    assert chunks[-1]["choices"] == [{"text": "local"}]
    assert len(prefill.calls) == 1
    admission_call = next(call for call in decode.calls if call.kind == "admit_local")
    assert admission_call.prompt_token_ids == tuple(range(8))
    record = coordinator.records[-1]
    assert record.engine_outcome is AdmissionOutcome.D_HIT
    assert record.actual_local_cached_tokens == cached
    assert record.locally_computed_tokens == computed
    assert record.hit_ratio == cached / 8


@pytest.mark.asyncio
async def test_d_miss_falls_back_before_client_visible_output() -> None:
    coordinator, prefill, decode = make_coordinator()
    await run_turn(coordinator, payload(t_pred=1.0))
    decode.queue_admission(
        "s",
        AdmissionResult(
            outcome=AdmissionOutcome.D_MISS,
            actual_local_cached_tokens=0,
            prompt_tokens=8,
            locally_computed_tokens=0,
        ),
    )

    chunks = await run_turn(coordinator, payload(t_pred=1.0, stream=True))

    assert chunks == [{"choices": [{"text": "prefill"}]}]
    assert [call.kind for call in decode.calls[-3:]] == [
        "admit_local",
        "stream_from_prefill",
        "finish",
    ]
    record = coordinator.records[-1]
    assert record.actual_path == "P_FALLBACK"
    assert record.fallback_reason == "D_MISS"


@pytest.mark.asyncio
async def test_capacity_deferred_does_not_fall_back() -> None:
    coordinator, prefill, decode = make_coordinator()
    await run_turn(coordinator, payload(t_pred=1.0))
    decode.queue_admission(
        "s",
        AdmissionResult(
            outcome=AdmissionOutcome.CAPACITY_DEFERRED,
            actual_local_cached_tokens=4,
            prompt_tokens=8,
            locally_computed_tokens=4,
            capacity_delay_ms=12.5,
        ),
    )

    await run_turn(coordinator, payload(t_pred=1.0))

    assert len(prefill.calls) == 1
    record = coordinator.records[-1]
    assert record.engine_outcome is AdmissionOutcome.CAPACITY_DEFERRED
    assert record.capacity_delay_ms == 12.5
    assert record.fallback_reason is None


@pytest.mark.asyncio
async def test_evict_ack_changes_intent_without_fabricating_mirror_removes() -> None:
    coordinator, _, decode = make_coordinator()
    coordinator.mirror.register_session("s", list(range(8)))
    coordinator.mirror.apply_batch(
        1,
        [
            1.0,
            [
                {
                    "type": "BlockStored",
                    "block_hashes": [101, 102],
                    "parent_block_hash": None,
                    "token_ids": list(range(8)),
                    "block_size": 4,
                    "medium": "GPU",
                    "lora_id": None,
                    "lora_name": None,
                    "extra_keys": None,
                }
            ],
            0,
        ],
    )
    decode.queue_ack(
        "s",
        CacheActionAck(
            status=CacheActionStatus.DEFERRED,
            invalidated_blocks=2,
            immediately_reusable_blocks=1,
            deferred_active_blocks=1,
            estimated_reusable_bytes=4096,
        ),
    )

    await run_turn(coordinator, payload(t_pred=50.0))

    assert coordinator.plan_for("s").path is PlannedPath.P_SIDE_AP
    assert coordinator.mirror.estimated_cached_tokens("s") == 8
    record = coordinator.records[-1]
    assert record.invalidated_blocks == 2
    assert record.immediately_reusable_blocks == 1
    assert record.deferred_active_blocks == 1
    assert record.as_dict().keys() == {
        "request_id",
        "session_id",
        "turn_sequence",
        "t_pred",
        "g",
        "action",
        "planned_path",
        "actual_path",
        "proxy_estimated_cached_tokens",
        "engine_outcome",
        "actual_local_cached_tokens",
        "prompt_tokens",
        "locally_computed_tokens",
        "hit_ratio",
        "estimate_error",
        "capacity_delay_ms",
        "eviction_status",
        "invalidated_blocks",
        "immediately_reusable_blocks",
        "deferred_active_blocks",
        "estimated_reusable_bytes",
        "fallback_reason",
    }


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_independent_plans_and_turn_sequences() -> None:
    coordinator, _, _ = make_coordinator()

    await asyncio.gather(
        run_turn(coordinator, payload("retain", 1.0)),
        run_turn(coordinator, payload("evict", 50.0)),
    )

    assert coordinator.plan_for("retain").path is PlannedPath.D_LOCAL_AP
    assert coordinator.plan_for("evict").path is PlannedPath.P_SIDE_AP
    records = {record.session_id: record for record in coordinator.records}
    assert records["retain"].turn_sequence == 1
    assert records["evict"].turn_sequence == 1


@pytest.mark.asyncio
async def test_turns_within_one_session_are_serialized() -> None:
    coordinator, prefill, _ = make_coordinator()
    prefill.gate = asyncio.Event()

    first = asyncio.create_task(run_turn(coordinator, payload("s", 50.0)))
    while not prefill.calls:
        await asyncio.sleep(0)
    second = asyncio.create_task(run_turn(coordinator, payload("s", 50.0)))
    await asyncio.sleep(0)

    assert len(prefill.calls) == 1
    prefill.gate.set()
    await asyncio.gather(first, second)
    assert [record.turn_sequence for record in coordinator.records] == [1, 2]
