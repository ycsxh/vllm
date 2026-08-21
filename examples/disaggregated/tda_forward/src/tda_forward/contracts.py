# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Portable contracts shared by the TDAforward coordinator and adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class CacheAction(StrEnum):
    RETAIN_D = "RETAIN_D"
    EVICT_D = "EVICT_D"


class CacheActionStatus(StrEnum):
    RETAINED = "RETAINED"
    EVICTED = "EVICTED"
    DEFERRED = "DEFERRED"


class PlannedPath(StrEnum):
    D_LOCAL_AP = "D_LOCAL_AP"
    P_SIDE_AP = "P_SIDE_AP"


class ActualPath(StrEnum):
    D_LOCAL_AP = "D_LOCAL_AP"
    P_SIDE_AP = "P_SIDE_AP"
    P_FALLBACK = "P_FALLBACK"


class AdmissionOutcome(StrEnum):
    D_HIT = "D_HIT"
    D_MISS = "D_MISS"
    CAPACITY_DEFERRED = "CAPACITY_DEFERRED"


@dataclass(frozen=True)
class ReentryPlan:
    action: CacheAction
    path: PlannedPath


@dataclass(frozen=True)
class AdmissionResult:
    outcome: AdmissionOutcome
    actual_local_cached_tokens: int
    prompt_tokens: int
    locally_computed_tokens: int
    capacity_delay_ms: float = 0.0


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
    action: CacheAction
    planned_path: PlannedPath
    actual_path: ActualPath
    proxy_estimated_cached_tokens: int
    engine_outcome: AdmissionOutcome | None
    actual_local_cached_tokens: int | None
    prompt_tokens: int
    locally_computed_tokens: int | None
    hit_ratio: float | None
    estimate_error: int | None
    capacity_delay_ms: float | None
    eviction_status: CacheActionStatus | None
    invalidated_blocks: int | None
    immediately_reusable_blocks: int | None
    deferred_active_blocks: int | None
    estimated_reusable_bytes: int | None
    fallback_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
