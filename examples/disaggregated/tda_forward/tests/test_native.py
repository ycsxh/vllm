# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from tda_forward.contracts import (
    AdmissionOutcome,
    CacheAction,
    CacheActionStatus,
    PrefillResult,
    PreparedTurn,
)
from tda_forward.mirror import DecodePrefixMirror
from tda_forward.native import (
    NativeEventPump,
    VllmDecodeAdapter,
    VllmPrefillAdapter,
    VllmTokenizerAdapter,
)


def test_vllm_tokenizer_adapter_uses_public_tokenizer_api(monkeypatch):
    class Tokenizer:
        def encode(self, prompt: str) -> list[int]:
            assert prompt == "hello"
            return [11, 12]

    def get_tokenizer(model: str, *, trust_remote_code: bool):
        assert model == "test-model"
        assert trust_remote_code is True
        return Tokenizer()

    def import_module(name: str):
        if name != "vllm.tokenizers":
            raise ModuleNotFoundError(name)
        return SimpleNamespace(get_tokenizer=get_tokenizer)

    monkeypatch.setattr("tda_forward.native.importlib.import_module", import_module)

    adapter = VllmTokenizerAdapter("test-model", trust_remote_code=True)

    assert adapter.token_ids({"prompt": "hello"}) == [11, 12]


def _turn(cache_action: CacheAction = CacheAction.RETAIN_D) -> PreparedTurn:
    return PreparedTurn(
        request_id="req-1",
        api_path="/v1/completions",
        session_id="session-1",
        t_pred=1.0,
        payload={"model": "test", "prompt": [1, 2, 3, 4, 5]},
        prompt_token_ids=(1, 2, 3, 4, 5),
        cache_action=cache_action,
    )


async def test_vllm_decode_adapter_uses_native_scheduler_control_outputs():
    seen_payload = None
    admission = {
        "outcome": "D_HIT",
        "actual_local_cached_tokens": 4,
        "prompt_tokens": 5,
        "locally_computed_tokens": 1,
        "capacity_delay_ms": 0.0,
    }
    ack = {
        "status": "RETAINED",
        "invalidated_blocks": 0,
        "immediately_reusable_blocks": 0,
        "deferred_active_blocks": 0,
        "estimated_reusable_bytes": 0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_payload
        seen_payload = json.loads(request.content)
        chunks = [
            {
                "choices": [{"index": 0, "text": "", "finish_reason": None}],
                "kv_transfer_params": {"tda_forward": {"admission_result": admission}},
            },
            {"choices": [{"index": 0, "text": "ok", "finish_reason": None}]},
            {
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "kv_transfer_params": {"tda_forward": {"cache_action_ack": ack}},
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)
    turn = _turn()

    result = await adapter.admit_local(turn, proxy_estimated_cached_tokens=4)
    chunks = [chunk async for chunk in adapter.stream_local(turn, result)]
    finish = await adapter.finish(turn, CacheAction.RETAIN_D)
    await client.aclose()

    assert seen_payload is not None
    assert seen_payload["stream"] is True
    assert seen_payload["stream_options"] == {"include_usage": True}
    assert seen_payload["kv_transfer_params"]["tda_forward"] == {
        "admission": {
            "path": "D_LOCAL_AP",
            "proxy_estimated_cached_tokens": 4,
        },
        "cache_action": "RETAIN_D",
    }
    assert result.outcome is AdmissionOutcome.D_HIT
    assert result.actual_local_cached_tokens == 4
    first_choices = []
    for chunk in chunks:
        choices = chunk["choices"]
        assert isinstance(choices, list)
        choice = choices[0]
        assert isinstance(choice, dict)
        first_choices.append(choice)
    assert [choice["text"] for choice in first_choices] == ["", "ok", ""]
    assert first_choices[-1]["finish_reason"] == "stop"
    assert finish.status is CacheActionStatus.RETAINED
    assert adapter._sessions == {}
    assert adapter._completed_acks == {}


async def test_vllm_prefill_adapter_returns_native_transfer_metadata():
    seen_payload = None
    transfer = {
        "remote_engine_id": "p-engine",
        "remote_request_id": "req-1",
        "remote_block_ids": [[1, 2]],
        "remote_port": 14579,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_payload
        seen_payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"kv_transfer_params": transfer, "usage": {"prompt_tokens": 5}},
        )

    client = httpx.AsyncClient(
        base_url="http://prefill",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmPrefillAdapter(client, remote_host="10.0.0.1")

    result = await adapter.prefill(_turn(CacheAction.EVICT_D))
    await client.aclose()

    assert seen_payload is not None
    assert seen_payload["stream"] is False
    assert seen_payload["max_tokens"] == 1
    assert result.prompt_tokens == 5
    assert result.transfer == {**transfer, "remote_host": "10.0.0.1"}


async def test_vllm_decode_adapter_falls_back_after_terminal_miss():
    payloads = []
    miss = {
        "outcome": "D_MISS",
        "actual_local_cached_tokens": 0,
        "prompt_tokens": 5,
        "locally_computed_tokens": 0,
        "capacity_delay_ms": 0.0,
    }
    ack = {
        "status": "EVICTED",
        "invalidated_blocks": 1,
        "immediately_reusable_blocks": 1,
        "deferred_active_blocks": 0,
        "estimated_reusable_bytes": 32,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if len(payloads) == 1:
            chunks = [
                {"kv_transfer_params": {"tda_forward": {"admission_result": miss}}}
            ]
        else:
            chunks = [
                {
                    "choices": [
                        {"index": 0, "text": "fallback", "finish_reason": "stop"}
                    ]
                },
                {"kv_transfer_params": {"tda_forward": {"cache_action_ack": ack}}},
            ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, content=body + "data: [DONE]\n\n")

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)
    turn = _turn(CacheAction.EVICT_D)

    admission = await adapter.admit_local(turn, proxy_estimated_cached_tokens=4)
    chunks = [
        chunk
        async for chunk in adapter.stream_from_prefill(
            turn,
            PrefillResult(
                prompt_tokens=5,
                transfer={"do_remote_prefill": True, "remote_block_ids": [[7]]},
            ),
        )
    ]
    finish = await adapter.finish(turn, CacheAction.EVICT_D)
    await client.aclose()

    assert admission.outcome is AdmissionOutcome.D_MISS
    choices = chunks[0]["choices"]
    assert isinstance(choices, list)
    first_choice = choices[0]
    assert isinstance(first_choice, dict)
    assert first_choice["text"] == "fallback"
    assert "admission" not in payloads[1]["kv_transfer_params"]["tda_forward"]
    assert payloads[1]["kv_transfer_params"]["tda_forward"]["cache_action"] == (
        "EVICT_D"
    )
    assert finish.status is CacheActionStatus.EVICTED
    assert finish.immediately_reusable_blocks == 1


async def test_native_event_pump_fails_closed_on_malformed_native_frame():
    thread_ids = []

    class MalformedSubscriber:
        def __init__(self) -> None:
            thread_ids.append(threading.get_ident())

        def receive_one(self, timeout: int):
            del timeout
            thread_ids.append(threading.get_ident())
            raise ValueError("invalid KV event publisher envelope")

        def close(self) -> None:
            thread_ids.append(threading.get_ident())

    mirror = DecodePrefixMirror(block_size=4)
    pump = NativeEventPump(MalformedSubscriber, on_failure=mirror.invalidate)

    with pytest.raises(ValueError, match="invalid KV event publisher envelope"):
        await pump.run(asyncio.Queue())

    assert len(set(thread_ids)) == 1
    assert thread_ids[0] != threading.get_ident()
    assert not mirror.is_valid
    assert mirror.invalid_reason == (
        "native event pump failed: invalid KV event publisher envelope"
    )
    assert pump.status["alive"] is False


async def test_decode_stream_without_ack_does_not_leak_session():
    admission = {
        "outcome": "D_HIT",
        "actual_local_cached_tokens": 4,
        "prompt_tokens": 5,
        "locally_computed_tokens": 1,
        "capacity_delay_ms": 0.0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        chunks = [
            {"kv_transfer_params": {"tda_forward": {"admission_result": admission}}},
            {"choices": [{"index": 0, "text": "partial"}]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, content=body + "data: [DONE]\n\n")

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)
    turn = _turn()

    result = await adapter.admit_local(turn, proxy_estimated_cached_tokens=4)
    assert [chunk async for chunk in adapter.stream_local(turn, result)]
    with pytest.raises(RuntimeError, match="without a Cache Action ACK"):
        await adapter.finish(turn, CacheAction.RETAIN_D)

    assert adapter._sessions == {}
    assert adapter._completed_acks == {}
    await client.aclose()


async def test_ack_followed_by_malformed_stream_is_not_retained():
    admission = {
        "outcome": "D_HIT",
        "actual_local_cached_tokens": 4,
        "prompt_tokens": 5,
        "locally_computed_tokens": 1,
        "capacity_delay_ms": 0.0,
    }
    ack = {
        "status": "RETAINED",
        "invalidated_blocks": 0,
        "immediately_reusable_blocks": 0,
        "deferred_active_blocks": 0,
        "estimated_reusable_bytes": 0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        chunks = [
            {"kv_transfer_params": {"tda_forward": {"admission_result": admission}}},
            {"kv_transfer_params": {"tda_forward": {"cache_action_ack": ack}}},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, content=body + "data: {malformed\n\n")

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)
    turn = _turn()

    result = await adapter.admit_local(turn, proxy_estimated_cached_tokens=4)
    with pytest.raises(json.JSONDecodeError):
        [chunk async for chunk in adapter.stream_local(turn, result)]

    assert adapter._sessions == {}
    assert adapter._completed_acks == {}
    with pytest.raises(RuntimeError, match="without a Cache Action ACK"):
        await adapter.finish(turn, CacheAction.RETAIN_D)
    await client.aclose()


async def test_ack_is_not_retained_when_stream_close_fails(monkeypatch):
    admission = {
        "outcome": "D_HIT",
        "actual_local_cached_tokens": 4,
        "prompt_tokens": 5,
        "locally_computed_tokens": 1,
        "capacity_delay_ms": 0.0,
    }
    ack = {
        "status": "RETAINED",
        "invalidated_blocks": 0,
        "immediately_reusable_blocks": 0,
        "deferred_active_blocks": 0,
        "estimated_reusable_bytes": 0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        chunks = [
            {"kv_transfer_params": {"tda_forward": {"admission_result": admission}}},
            {"kv_transfer_params": {"tda_forward": {"cache_action_ack": ack}}},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, content=body + "data: [DONE]\n\n")

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)
    turn = _turn()
    result = await adapter.admit_local(turn, proxy_estimated_cached_tokens=4)
    session = adapter._sessions[turn.request_id]

    async def fail_close(self) -> None:
        del self
        raise RuntimeError("stream close failed")

    monkeypatch.setattr(type(session), "close", fail_close)
    with pytest.raises(RuntimeError, match="stream close failed"):
        [chunk async for chunk in adapter.stream_local(turn, result)]

    assert adapter._sessions == {}
    assert adapter._completed_acks == {}
    await client.aclose()
