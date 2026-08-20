# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic fake Prefill and Decode adapters for portable contract tests."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

from tda_forward.contracts import (
    AdmissionOutcome,
    AdmissionResult,
    CacheAction,
    CacheActionAck,
    PrefillResult,
    PreparedTurn,
)


@dataclass(frozen=True)
class AdapterCall:
    kind: str
    session_id: str
    action: CacheAction | None = None
    proxy_estimated_cached_tokens: int | None = None
    prompt_token_ids: tuple[int, ...] = ()


class FakePrefillAdapter:
    def __init__(self) -> None:
        self.calls: list[AdapterCall] = []
        self.gate: asyncio.Event | None = None

    async def prefill(self, turn: PreparedTurn) -> PrefillResult:
        self.calls.append(
            AdapterCall(
                "prefill",
                turn.session_id,
                prompt_token_ids=turn.prompt_token_ids,
            )
        )
        if self.gate is not None:
            await self.gate.wait()
        return PrefillResult(prompt_tokens=len(turn.prompt_token_ids))


class FakeDecodeAdapter:
    def __init__(self) -> None:
        self.calls: list[AdapterCall] = []
        self._admissions: dict[str, deque[AdmissionResult]] = defaultdict(deque)
        self._acks: dict[str, deque[CacheActionAck]] = defaultdict(deque)

    def queue_admission(self, session_id: str, result: AdmissionResult) -> None:
        self._admissions[session_id].append(result)

    def queue_ack(self, session_id: str, ack: CacheActionAck) -> None:
        self._acks[session_id].append(ack)

    async def admit_local(
        self, turn: PreparedTurn, proxy_estimated_cached_tokens: int
    ) -> AdmissionResult:
        self.calls.append(
            AdapterCall(
                "admit_local",
                turn.session_id,
                proxy_estimated_cached_tokens=proxy_estimated_cached_tokens,
                prompt_token_ids=turn.prompt_token_ids,
            )
        )
        queued = self._admissions[turn.session_id]
        if queued:
            return queued.popleft()
        return AdmissionResult(
            outcome=AdmissionOutcome.D_MISS,
            actual_local_cached_tokens=0,
            prompt_tokens=len(turn.prompt_token_ids),
            locally_computed_tokens=0,
        )

    async def stream_from_prefill(
        self, turn: PreparedTurn, prefill: PrefillResult
    ) -> AsyncIterator[dict[str, object]]:
        del prefill
        self.calls.append(
            AdapterCall(
                "stream_from_prefill",
                turn.session_id,
                prompt_token_ids=turn.prompt_token_ids,
            )
        )
        yield self._response_chunk(turn, "prefill")

    async def stream_local(
        self, turn: PreparedTurn, admission: AdmissionResult
    ) -> AsyncIterator[dict[str, object]]:
        del admission
        self.calls.append(
            AdapterCall(
                "stream_local",
                turn.session_id,
                prompt_token_ids=turn.prompt_token_ids,
            )
        )
        yield self._response_chunk(turn, "local")

    async def finish(self, turn: PreparedTurn, action: CacheAction) -> CacheActionAck:
        self.calls.append(
            AdapterCall(
                "finish",
                turn.session_id,
                action=action,
                prompt_token_ids=turn.prompt_token_ids,
            )
        )
        queued = self._acks[turn.session_id]
        if queued:
            return queued.popleft()
        return CacheActionAck(
            status="retained" if action is CacheAction.RETAIN_D else "evicted"
        )

    @staticmethod
    def _response_chunk(turn: PreparedTurn, text: str) -> dict[str, object]:
        if turn.api_path == "/v1/chat/completions":
            if turn.payload.get("stream", False):
                return {"choices": [{"delta": {"content": text}}]}
            return {"choices": [{"message": {"role": "assistant", "content": text}}]}
        return {"choices": [{"text": text}]}
