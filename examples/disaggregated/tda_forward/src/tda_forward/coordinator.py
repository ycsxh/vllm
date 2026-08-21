# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deep TDAforward ReentryCoordinator policy and orchestration module."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from tda_forward.contracts import (
    ActualPath,
    AdmissionOutcome,
    AdmissionResult,
    CacheAction,
    CacheActionAck,
    CacheActionStatus,
    PlannedPath,
    PrefillResult,
    PreparedTurn,
    ReentryPlan,
    TurnRecord,
)
from tda_forward.mirror import DecodePrefixMirror


class RequestValidationError(ValueError):
    """Raised before worker dispatch when a request violates the contract."""


class PrefillAdapter(Protocol):
    async def prefill(self, turn: PreparedTurn) -> PrefillResult: ...


class DecodeAdapter(Protocol):
    async def admit_local(
        self, turn: PreparedTurn, proxy_estimated_cached_tokens: int
    ) -> AdmissionResult: ...

    def stream_from_prefill(
        self, turn: PreparedTurn, prefill: PrefillResult
    ) -> AsyncIterator[dict[str, object]]: ...

    def stream_local(
        self, turn: PreparedTurn, admission: AdmissionResult
    ) -> AsyncIterator[dict[str, object]]: ...

    async def finish(
        self, turn: PreparedTurn, action: CacheAction
    ) -> CacheActionAck: ...


class TokenizerAdapter(Protocol):
    def token_ids(self, payload: Mapping[str, object]) -> Sequence[int]: ...


class PortableTokenizer:
    """Deterministic tokenizer for the fake harness, replaceable by vLLM."""

    def token_ids(self, payload: Mapping[str, object]) -> Sequence[int]:
        explicit = payload.get("prompt_token_ids")
        if explicit is not None:
            return self._integer_tokens(explicit)
        prompt = payload.get("prompt")
        if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
            return self._integer_tokens(prompt)
        if isinstance(prompt, str):
            return list(prompt.encode())
        messages = payload.get("messages")
        if messages is not None:
            encoded = json.dumps(
                messages, sort_keys=True, separators=(",", ":")
            ).encode()
            return list(encoded)
        raise RequestValidationError(
            "request must contain prompt_token_ids, integer prompt tokens, "
            "a text prompt, or messages"
        )

    @staticmethod
    def _integer_tokens(value: object) -> list[int]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise RequestValidationError("prompt tokens must be a sequence")
        tokens = list(value)
        if any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < (1 << 32)
            for token in tokens
        ):
            raise RequestValidationError(
                "prompt token IDs must be unsigned 32-bit integers"
            )
        return tokens


@dataclass
class _SessionState:
    turn_sequence: int
    plan: ReentryPlan


_INITIAL_PLAN = ReentryPlan(CacheAction.EVICT_D, PlannedPath.P_SIDE_AP)


class ReentryCoordinator:
    """Own routing intent, advisory prefix state, fallback, and records."""

    def __init__(
        self,
        *,
        g: float,
        block_size: int,
        prefill: PrefillAdapter,
        decode: DecodeAdapter,
        mirror: DecodePrefixMirror,
        tokenizer: TokenizerAdapter | None = None,
    ) -> None:
        self.g = self._finite_nonnegative(g, "configured g", ValueError)
        if isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if mirror.block_size != block_size:
            raise ValueError("coordinator and mirror block sizes must match")
        self.block_size = block_size
        self.prefill = prefill
        self.decode = decode
        self.mirror = mirror
        self.tokenizer = tokenizer or PortableTokenizer()
        self.records: list[TurnRecord] = []
        self._states: dict[str, _SessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def plan_for(self, session_id: str) -> ReentryPlan:
        state = self._states.get(session_id)
        return _INITIAL_PLAN if state is None else state.plan

    def start_turn(
        self,
        payload: Mapping[str, object],
        *,
        api_path: str = "/v1/completions",
        request_id: str | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        """Validate synchronously, then return the orchestrated output stream."""
        prepared = self._prepare(
            payload,
            api_path=api_path,
            request_id=request_id,
        )
        return self._run_turn(prepared)

    async def consume_events(
        self,
        event_queue: asyncio.Queue[tuple[int, object, float | None] | None],
    ) -> None:
        """Consume mirror events independently from request forwarding."""
        while True:
            envelope = await event_queue.get()
            if envelope is None:
                return
            sequence, batch, received_at = envelope
            self.mirror.apply_batch(
                sequence,
                batch,
                received_at=received_at,
            )

    def _prepare(
        self,
        payload: Mapping[str, object],
        *,
        api_path: str,
        request_id: str | None,
    ) -> PreparedTurn:
        if not isinstance(payload, Mapping):
            raise RequestValidationError("request body must be an object")
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise RequestValidationError("session_id must be a nonempty string")
        t_pred = self._finite_nonnegative(
            payload.get("t_pred"), "t_pred", RequestValidationError
        )
        request_g = payload.get("g")
        if request_g is not None:
            parsed_g = self._finite_nonnegative(request_g, "g", RequestValidationError)
            if parsed_g != self.g:
                raise RequestValidationError(
                    f"request g={parsed_g} does not match configured g={self.g}"
                )
        try:
            prompt_token_ids = tuple(self.tokenizer.token_ids(payload))
        except RequestValidationError:
            raise
        except ValueError as error:
            raise RequestValidationError(str(error)) from error
        lora_name = self._optional_string(payload.get("lora_name"), "lora_name")
        cache_namespace = self._optional_string(
            payload.get("cache_salt", payload.get("cache_namespace")),
            "cache_salt",
        )
        block_extra_keys = payload.get("block_extra_keys")
        if block_extra_keys is not None:
            if not isinstance(block_extra_keys, Sequence) or isinstance(
                block_extra_keys, (str, bytes)
            ):
                raise RequestValidationError("block_extra_keys must be a sequence")
            block_extra_keys = tuple(block_extra_keys)

        forwarded = dict(payload)
        for extension in (
            "session_id",
            "t_pred",
            "g",
            "prompt_token_ids",
            "block_extra_keys",
            "cache_namespace",
        ):
            forwarded.pop(extension, None)
        return PreparedTurn(
            request_id=request_id or str(uuid.uuid4()),
            api_path=api_path,
            session_id=session_id,
            t_pred=t_pred,
            payload=forwarded,
            prompt_token_ids=prompt_token_ids,
            lora_name=lora_name,
            cache_namespace=cache_namespace,
            block_extra_keys=block_extra_keys,
            cache_action=(
                CacheAction.RETAIN_D if t_pred <= self.g else CacheAction.EVICT_D
            ),
        )

    async def _run_turn(self, turn: PreparedTurn) -> AsyncIterator[dict[str, object]]:
        lock = self._locks.setdefault(turn.session_id, asyncio.Lock())
        async with lock:
            state = self._states.setdefault(
                turn.session_id,
                _SessionState(turn_sequence=0, plan=_INITIAL_PLAN),
            )
            turn_sequence = state.turn_sequence + 1
            self.mirror.register_session(
                turn.session_id,
                turn.prompt_token_ids,
                lora_name=turn.lora_name,
                cache_namespace=turn.cache_namespace,
                block_extra_keys=turn.block_extra_keys,
            )
            estimate = self.mirror.estimated_cached_tokens(turn.session_id)
            planned_path = state.plan.path
            admission: AdmissionResult | None = None
            fallback_reason: str | None = None

            if planned_path is PlannedPath.P_SIDE_AP:
                actual_path = ActualPath.P_SIDE_AP
                stream = self._prefill_stream(turn)
            else:
                admission = await self.decode.admit_local(turn, estimate)
                self._validate_admission(admission)
                if admission.outcome is AdmissionOutcome.D_MISS:
                    actual_path = ActualPath.P_FALLBACK
                    fallback_reason = AdmissionOutcome.D_MISS.value
                    stream = self._prefill_stream(turn)
                else:
                    actual_path = ActualPath.D_LOCAL_AP
                    stream = self.decode.stream_local(turn, admission)

            async for chunk in stream:
                yield chunk

            action = turn.cache_action
            ack = await self.decode.finish(turn, action)
            self._validate_ack(ack, action)
            state.turn_sequence = turn_sequence
            state.plan = ReentryPlan(
                action=action,
                path=(
                    PlannedPath.D_LOCAL_AP
                    if action is CacheAction.RETAIN_D
                    else PlannedPath.P_SIDE_AP
                ),
            )
            self.records.append(
                self._make_record(
                    turn,
                    turn_sequence=turn_sequence,
                    action=action,
                    planned_path=planned_path,
                    actual_path=actual_path,
                    estimate=estimate,
                    g=self.g,
                    admission=admission,
                    ack=ack,
                    fallback_reason=fallback_reason,
                )
            )

    async def _prefill_stream(
        self, turn: PreparedTurn
    ) -> AsyncIterator[dict[str, object]]:
        prefill = await self.prefill.prefill(turn)
        async for chunk in self.decode.stream_from_prefill(turn, prefill):
            yield chunk

    @staticmethod
    def _make_record(
        turn: PreparedTurn,
        *,
        turn_sequence: int,
        action: CacheAction,
        planned_path: PlannedPath,
        actual_path: ActualPath,
        estimate: int,
        g: float,
        admission: AdmissionResult | None,
        ack: CacheActionAck,
        fallback_reason: str | None,
    ) -> TurnRecord:
        actual_cached = (
            None if admission is None else admission.actual_local_cached_tokens
        )
        prompt_tokens = (
            len(turn.prompt_token_ids) if admission is None else admission.prompt_tokens
        )
        eviction = action is CacheAction.EVICT_D
        return TurnRecord(
            request_id=turn.request_id,
            session_id=turn.session_id,
            turn_sequence=turn_sequence,
            t_pred=turn.t_pred,
            g=g,
            action=action,
            planned_path=planned_path,
            actual_path=actual_path,
            proxy_estimated_cached_tokens=estimate,
            engine_outcome=None if admission is None else admission.outcome,
            actual_local_cached_tokens=actual_cached,
            prompt_tokens=prompt_tokens,
            locally_computed_tokens=(
                None if admission is None else admission.locally_computed_tokens
            ),
            hit_ratio=(
                None
                if admission is None or admission.prompt_tokens == 0
                else admission.actual_local_cached_tokens / admission.prompt_tokens
            ),
            estimate_error=(
                None if actual_cached is None else actual_cached - estimate
            ),
            capacity_delay_ms=(
                None if admission is None else admission.capacity_delay_ms
            ),
            eviction_status=ack.status if eviction else None,
            invalidated_blocks=ack.invalidated_blocks if eviction else None,
            immediately_reusable_blocks=(
                ack.immediately_reusable_blocks if eviction else None
            ),
            deferred_active_blocks=ack.deferred_active_blocks if eviction else None,
            estimated_reusable_bytes=(
                ack.estimated_reusable_bytes if eviction else None
            ),
            fallback_reason=fallback_reason,
        )

    @staticmethod
    def _validate_admission(result: AdmissionResult) -> None:
        numeric = (
            result.actual_local_cached_tokens,
            result.prompt_tokens,
            result.locally_computed_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in numeric
        ):
            raise RuntimeError("Decode returned malformed token counts")
        if not math.isfinite(result.capacity_delay_ms) or result.capacity_delay_ms < 0:
            raise RuntimeError("Decode returned malformed capacity delay")
        if (
            result.outcome is AdmissionOutcome.D_MISS
            and result.actual_local_cached_tokens != 0
        ):
            raise RuntimeError("D_MISS must report zero local cached tokens")
        if (
            result.outcome
            in (AdmissionOutcome.D_HIT, AdmissionOutcome.CAPACITY_DEFERRED)
            and result.actual_local_cached_tokens == 0
        ):
            raise RuntimeError(f"{result.outcome.value} requires a positive local hit")

    @staticmethod
    def _validate_ack(ack: CacheActionAck, action: CacheAction) -> None:
        if not isinstance(ack.status, CacheActionStatus):
            raise RuntimeError("Decode returned a malformed Cache Action ACK")
        expected_statuses = (
            {CacheActionStatus.RETAINED}
            if action is CacheAction.RETAIN_D
            else {CacheActionStatus.EVICTED, CacheActionStatus.DEFERRED}
        )
        if ack.status not in expected_statuses:
            raise RuntimeError("Decode returned an ACK inconsistent with its action")
        counts = (
            ack.invalidated_blocks,
            ack.immediately_reusable_blocks,
            ack.deferred_active_blocks,
            ack.estimated_reusable_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise RuntimeError("Decode returned malformed eviction aggregates")

    @staticmethod
    def _optional_string(value: object, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise RequestValidationError(f"{name} must be a string")
        return value or None

    @staticmethod
    def _finite_nonnegative(
        value: object,
        name: str,
        error_type: type[ValueError],
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise error_type(f"{name} must be a finite nonnegative number")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise error_type(f"{name} must be a finite nonnegative number")
        return parsed
