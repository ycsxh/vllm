# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Production adapters for native vLLM HTTP and KV event contracts."""

from __future__ import annotations

import asyncio
import importlib
import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from tda_forward.contracts import (
    AdmissionOutcome,
    AdmissionResult,
    CacheAction,
    CacheActionAck,
    CacheActionStatus,
    PrefillResult,
    PreparedTurn,
)


def _headers(request_id: str) -> dict[str, str]:
    return {"X-Request-Id": request_id}


class VllmTokenizerAdapter:
    """Use the model tokenizer loaded through vLLM's native tokenizer helper."""

    def __init__(self, model: str, *, trust_remote_code: bool = False) -> None:
        tokenizer_module = importlib.import_module("vllm.transformers_utils.tokenizer")
        self._tokenizer = tokenizer_module.get_tokenizer(
            model,
            trust_remote_code=trust_remote_code,
        )

    def token_ids(self, payload: Mapping[str, object]) -> Sequence[int]:
        explicit = payload.get("prompt_token_ids")
        if explicit is not None:
            return self._integer_tokens(explicit)
        prompt = payload.get("prompt")
        if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
            return self._integer_tokens(prompt)
        if isinstance(prompt, str):
            return self._tokenizer.encode(prompt)
        messages = payload.get("messages")
        if messages is not None:
            tokens = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return self._integer_tokens(tokens)
        raise ValueError("request has no tokenizable prompt")

    @staticmethod
    def _integer_tokens(value: object) -> list[int]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("prompt tokens must be a sequence")
        tokens = list(value)
        if any(
            isinstance(token, bool) or not isinstance(token, int) for token in tokens
        ):
            raise ValueError("prompt tokens must be integers")
        return tokens


class VllmPrefillAdapter:
    """Issue native non-streaming remote-prefill requests to one P worker."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        remote_host: str,
    ) -> None:
        self._client = client
        self._remote_host = remote_host

    async def prefill(self, turn: PreparedTurn) -> PrefillResult:
        payload = dict(turn.payload)
        payload["stream"] = False
        payload["max_tokens"] = 1
        payload.pop("max_completion_tokens", None)
        payload.pop("min_tokens", None)
        payload.pop("stream_options", None)
        payload.setdefault(
            "kv_transfer_params",
            {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        )
        response = await self._client.post(
            turn.api_path,
            json=payload,
            headers=_headers(turn.request_id),
        )
        response.raise_for_status()
        body = response.json()
        transfer = body.get("kv_transfer_params")
        if isinstance(transfer, dict):
            transfer = dict(transfer)
            transfer["remote_host"] = self._remote_host
        usage = body.get("usage")
        prompt_tokens = (
            usage.get("prompt_tokens", len(turn.prompt_token_ids))
            if isinstance(usage, dict)
            else len(turn.prompt_token_ids)
        )
        return PrefillResult(prompt_tokens=prompt_tokens, transfer=transfer)


@dataclass
class _DecodeSession:
    stream_context: Any
    response: httpx.Response
    lines: AsyncIterator[str]
    cache_action_ack: CacheActionAck | None = None
    pending_chunks: list[dict[str, object]] = field(default_factory=list)
    closed: bool = False

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            await self.stream_context.__aexit__(None, None, None)


class VllmDecodeAdapter:
    """Fuse Scheduler admission and generation over vLLM's native SSE output."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self._sessions: dict[str, _DecodeSession] = {}
        self._completed_acks: dict[str, CacheActionAck] = {}

    async def admit_local(
        self,
        turn: PreparedTurn,
        proxy_estimated_cached_tokens: int,
    ) -> AdmissionResult:
        transfer = {
            "tda_forward": {
                "admission": {
                    "path": "D_LOCAL_AP",
                    "proxy_estimated_cached_tokens": proxy_estimated_cached_tokens,
                },
                "cache_action": turn.cache_action.value,
            }
        }
        session = await self._open(turn, transfer)
        try:
            async for line in session.lines:
                chunk = self._decode_sse(line)
                if chunk is None:
                    continue
                result = self._admission_result(chunk)
                if result is None:
                    continue
                if result.outcome is AdmissionOutcome.D_MISS:
                    await session.close()
                else:
                    if chunk.get("choices"):
                        session.pending_chunks.append(chunk)
                    self._sessions[turn.request_id] = session
                return result
        except BaseException:
            await session.close()
            raise
        await session.close()
        raise RuntimeError("Decode stream ended without an admission result")

    def stream_local(
        self,
        turn: PreparedTurn,
        admission: AdmissionResult,
    ) -> AsyncIterator[dict[str, object]]:
        del admission
        session = self._sessions.get(turn.request_id)
        if session is None:
            raise RuntimeError("Decode admission did not retain a live request")
        return self._stream_session(turn, session)

    def stream_from_prefill(
        self,
        turn: PreparedTurn,
        prefill: PrefillResult,
    ) -> AsyncIterator[dict[str, object]]:
        return self._stream_prefilled(turn, prefill)

    async def finish(
        self,
        turn: PreparedTurn,
        action: CacheAction,
    ) -> CacheActionAck:
        self._sessions.pop(turn.request_id, None)
        ack = self._completed_acks.pop(turn.request_id, None)
        if ack is None:
            raise RuntimeError("Decode stream ended without a Cache Action ACK")
        if action is not turn.cache_action:
            raise RuntimeError("Cache Action changed after Decode dispatch")
        return ack

    async def _stream_prefilled(
        self,
        turn: PreparedTurn,
        prefill: PrefillResult,
    ) -> AsyncIterator[dict[str, object]]:
        transfer = dict(prefill.transfer) if isinstance(prefill.transfer, dict) else {}
        tda_forward = transfer.get("tda_forward")
        tda_forward = dict(tda_forward) if isinstance(tda_forward, dict) else {}
        tda_forward["cache_action"] = turn.cache_action.value
        transfer["tda_forward"] = tda_forward
        session = await self._open(turn, transfer)
        self._sessions[turn.request_id] = session
        async for chunk in self._stream_session(turn, session):
            yield chunk

    async def _stream_session(
        self,
        turn: PreparedTurn,
        session: _DecodeSession,
    ) -> AsyncIterator[dict[str, object]]:
        exhausted = False
        try:
            while session.pending_chunks:
                yield session.pending_chunks.pop(0)
            async for line in session.lines:
                chunk = self._decode_sse(line)
                if chunk is None:
                    continue
                ack = self._cache_action_ack(chunk)
                if ack is not None:
                    session.cache_action_ack = ack
                if chunk.get("choices") or chunk.get("usage"):
                    yield chunk
            exhausted = True
        finally:
            closed = False
            try:
                await session.close()
                closed = True
            finally:
                self._sessions.pop(turn.request_id, None)
                if exhausted and closed and session.cache_action_ack is not None:
                    self._completed_acks[turn.request_id] = session.cache_action_ack

    async def _open(
        self,
        turn: PreparedTurn,
        transfer: dict[str, Any],
    ) -> _DecodeSession:
        payload = dict(turn.payload)
        payload["stream"] = True
        if not turn.payload.get("stream", False):
            payload["stream_options"] = {"include_usage": True}
        payload["kv_transfer_params"] = transfer
        stream_context = self._client.stream(
            "POST",
            turn.api_path,
            json=payload,
            headers=_headers(turn.request_id),
        )
        response = await stream_context.__aenter__()
        try:
            response.raise_for_status()
        except BaseException:
            await stream_context.__aexit__(None, None, None)
            raise
        return _DecodeSession(stream_context, response, response.aiter_lines())

    @staticmethod
    def _decode_sse(line: str) -> dict[str, object] | None:
        if not line.startswith("data: ") or line == "data: [DONE]":
            return None
        decoded = json.loads(line[6:])
        if not isinstance(decoded, dict):
            raise RuntimeError("Decode returned a non-object SSE chunk")
        return decoded

    @classmethod
    def _admission_result(cls, chunk: Mapping[str, object]) -> AdmissionResult | None:
        value = cls._tda_value(chunk, "admission_result")
        if value is None:
            return None
        outcome = cls._required_string(value, "outcome")
        return AdmissionResult(
            outcome=AdmissionOutcome(outcome),
            actual_local_cached_tokens=cls._required_int(
                value, "actual_local_cached_tokens"
            ),
            prompt_tokens=cls._required_int(value, "prompt_tokens"),
            locally_computed_tokens=cls._required_int(value, "locally_computed_tokens"),
            capacity_delay_ms=cls._optional_number(value, "capacity_delay_ms", 0.0),
        )

    @classmethod
    def _cache_action_ack(cls, chunk: Mapping[str, object]) -> CacheActionAck | None:
        value = cls._tda_value(chunk, "cache_action_ack")
        if value is None:
            return None
        status = cls._required_string(value, "status")
        return CacheActionAck(
            status=CacheActionStatus(status),
            invalidated_blocks=cls._optional_int(value, "invalidated_blocks"),
            immediately_reusable_blocks=cls._optional_int(
                value, "immediately_reusable_blocks"
            ),
            deferred_active_blocks=cls._optional_int(value, "deferred_active_blocks"),
            estimated_reusable_bytes=cls._optional_int(
                value, "estimated_reusable_bytes"
            ),
        )

    @staticmethod
    def _required_string(value: Mapping[str, object], key: str) -> str:
        parsed = value.get(key)
        if not isinstance(parsed, str):
            raise RuntimeError(f"Decode returned non-string {key}")
        return parsed

    @staticmethod
    def _required_int(value: Mapping[str, object], key: str) -> int:
        parsed = value.get(key)
        if isinstance(parsed, bool) or not isinstance(parsed, int):
            raise RuntimeError(f"Decode returned non-integer {key}")
        return parsed

    @classmethod
    def _optional_int(cls, value: Mapping[str, object], key: str) -> int:
        return 0 if key not in value else cls._required_int(value, key)

    @staticmethod
    def _optional_number(
        value: Mapping[str, object], key: str, default: float
    ) -> float:
        parsed = value.get(key, default)
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            raise RuntimeError(f"Decode returned non-numeric {key}")
        return float(parsed)

    @staticmethod
    def _tda_value(
        chunk: Mapping[str, object],
        key: str,
    ) -> dict[str, object] | None:
        transfer = chunk.get("kv_transfer_params")
        if not isinstance(transfer, dict):
            return None
        tda_forward = transfer.get("tda_forward")
        if not isinstance(tda_forward, dict):
            return None
        value = tda_forward.get(key)
        return value if isinstance(value, dict) else None


class NativeEventPump:
    """Feed native publisher batches into the Proxy's independent event queue."""

    def __init__(
        self,
        subscriber_factory: Callable[[], Any],
        *,
        on_failure: Callable[[str], None] | None = None,
    ) -> None:
        self._subscriber_factory = subscriber_factory
        self._on_failure = on_failure
        self._subscriber: Any | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._failure: str | None = None

    @classmethod
    def from_endpoints(
        cls,
        endpoints: str | list[str],
        *,
        topic: str = "",
        on_failure: Callable[[str], None] | None = None,
    ) -> NativeEventPump:
        event_module = importlib.import_module("vllm.distributed.kv_events")
        return cls(
            lambda: event_module.ZmqEventSubscriber(endpoints, topic=topic),
            on_failure=on_failure,
        )

    async def run(
        self,
        event_queue: asyncio.Queue[tuple[int, object, float | None] | None],
    ) -> None:
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()

        def finish(error: BaseException | None = None) -> None:
            if completed.done():
                return
            if error is None:
                completed.set_result(None)
            else:
                completed.set_exception(error)

        def receive() -> None:
            try:
                subscriber = self._subscriber_factory()
                self._subscriber = subscriber
                while not self._stop.is_set():
                    envelope = subscriber.receive_one(100)
                    if envelope is not None:
                        sequence, batch = envelope
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait,
                            (sequence, batch, time.time()),
                        )
            except BaseException as error:
                loop.call_soon_threadsafe(finish, error)
            finally:
                subscriber = self._subscriber
                if subscriber is not None:
                    subscriber.close()
                loop.call_soon_threadsafe(finish)

        self._stop.clear()
        self._thread = threading.Thread(
            target=receive,
            name="tda-native-event-pump",
            daemon=True,
        )
        self._thread.start()
        try:
            await completed
        except asyncio.CancelledError:
            self.close()
            raise
        except BaseException as error:
            self._failure = f"native event pump failed: {error}"
            if self._on_failure is not None:
                self._on_failure(self._failure)
            self.close()
            raise

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("native event subscriber did not stop")

    @property
    def status(self) -> dict[str, object]:
        subscriber = self._subscriber
        stats = getattr(subscriber, "stats", None) if subscriber is not None else None
        return {
            "alive": self._thread is not None and self._thread.is_alive(),
            "failure": self._failure,
            "subscriber": asdict(stats) if stats is not None else None,
        }
