# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Portable contracts shared by the TDAforward coordinator and adapters."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from enum import StrEnum


class CacheAction(StrEnum):
    RETAIN_D = "RETAIN_D"
    EVICT_D = "EVICT_D"


class CacheActionStatus(StrEnum):
    RETAINED = "RETAINED"
    EVICTED = "EVICTED"
    DEFERRED = "DEFERRED"


class ExecutionPath(StrEnum):
    D_LOCAL_AP = "D_LOCAL_AP"
    P_SIDE_AP = "P_SIDE_AP"


class LocalCacheStatus(StrEnum):
    D_HIT = "D_HIT"
    D_MISS = "D_MISS"


@dataclass(frozen=True)
class DecodeExecutionObservation:
    cache_status: LocalCacheStatus
    actual_local_cached_tokens: int
    prompt_tokens: int
    locally_computed_tokens: int
    capacity_delay_ms: float = 0.0


def validate_decode_execution_observation(
    observation: DecodeExecutionObservation,
) -> None:
    if not isinstance(observation.cache_status, LocalCacheStatus):
        raise RuntimeError("Decode returned a malformed local cache status")
    numeric = (
        observation.actual_local_cached_tokens,
        observation.prompt_tokens,
        observation.locally_computed_tokens,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in numeric
    ):
        raise RuntimeError("Decode returned malformed token counts")
    if (
        not math.isfinite(observation.capacity_delay_ms)
        or observation.capacity_delay_ms < 0
    ):
        raise RuntimeError("Decode returned malformed capacity delay")
    if (
        observation.cache_status is LocalCacheStatus.D_MISS
        and observation.actual_local_cached_tokens != 0
    ):
        raise RuntimeError("D_MISS must report zero local cached tokens")
    if (
        observation.cache_status is LocalCacheStatus.D_HIT
        and observation.actual_local_cached_tokens == 0
    ):
        raise RuntimeError("D_HIT requires a positive local hit")


@dataclass(frozen=True)
class DecodeExecution:
    observation: DecodeExecutionObservation
    stream: AsyncIterator[dict[str, object]]


@dataclass(frozen=True)
class CacheActionAck:
    status: CacheActionStatus
    invalidated_blocks: int = 0
    immediately_reusable_blocks: int = 0
    deferred_active_blocks: int = 0
    estimated_reusable_bytes: int = 0


@dataclass(frozen=True)
class PreparedTurn:
    request_id: str
    api_path: str
    session_id: str
    t_pred: float
    payload: dict[str, object]
    prompt_token_ids: tuple[int, ...]
    lora_name: str | None = None
    cache_namespace: str | None = None
    block_extra_keys: tuple[object | None, ...] | None = None
    cache_action: CacheAction = CacheAction.EVICT_D


@dataclass(frozen=True)
class PrefillResult:
    prompt_tokens: int
    transfer: object | None = None


@dataclass(frozen=True)
class TurnRecord:
    request_id: str
    session_id: str
    turn_sequence: int
    t_pred: float
    g: float
    cache_action: CacheAction
    cache_action_ack: CacheActionAck
    used_d_binding: bool
    execution_path: ExecutionPath
    proxy_estimated_cached_tokens: int
    local_cache_status: LocalCacheStatus | None
    actual_local_cached_tokens: int | None
    prompt_tokens: int
    locally_computed_tokens: int | None
    hit_ratio: float | None
    estimate_error: int | None
    capacity_delay_ms: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
