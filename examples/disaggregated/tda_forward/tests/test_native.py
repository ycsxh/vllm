# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from tda_forward.contracts import (
    CacheAction,
    CacheActionStatus,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "cached", "computed"),
    [("D_HIT", 4, 1), ("D_MISS", 0, 5)],
)
async def test_bound_decode_streams_after_cache_observation(
    status: str, cached: int, computed: int
) -> None:
    seen_payload = None
    observation = {
        "cache_status": status,
        "actual_local_cached_tokens": cached,
        "prompt_tokens": 5,
        "locally_computed_tokens": computed,
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
                "kv_transfer_params": {
                    "tda_forward": {"execution_observation": observation}
                },
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

    execution = await adapter.start_bound(turn)
    chunks = [chunk async for chunk in execution.stream]
    ack = await adapter.finish(turn, CacheAction.RETAIN_D)
    await client.aclose()

    assert seen_payload is not None
    assert seen_payload["stream"] is True
    assert seen_payload["stream_options"] == {"include_usage": True}
    assert seen_payload["kv_transfer_params"]["tda_forward"] == {
        "decode_bound": True,
        "cache_action": "RETAIN_D",
    }
    assert execution.observation.cache_status.value == status
    assert [choice["text"] for chunk in chunks for choice in chunk["choices"]] == [
        "ok",
        "",
    ]
    assert ack.status is CacheActionStatus.RETAINED


@pytest.mark.asyncio
async def test_bound_decode_preserves_chunks_before_execution_observation():
    observation = {
        "cache_status": "D_HIT",
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
            {"choices": [{"index": 0, "text": "before", "finish_reason": None}]},
            {
                "kv_transfer_params": {
                    "tda_forward": {"execution_observation": observation}
                }
            },
            {"choices": [{"index": 0, "text": "after", "finish_reason": None}]},
            {
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "kv_transfer_params": {"tda_forward": {"cache_action_ack": ack}},
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, content=body + "data: [DONE]\n\n")

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)
    turn = _turn()

    execution = await adapter.start_bound(turn)
    chunks = [chunk async for chunk in execution.stream]
    ack = await adapter.finish(turn, CacheAction.RETAIN_D)

    assert [choice["text"] for chunk in chunks for choice in chunk["choices"]] == [
        "before",
        "after",
        "",
    ]
    assert ack.status is CacheActionStatus.RETAINED
    await client.aclose()


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            {
                "cache_status": "D_MISS",
                "actual_local_cached_tokens": 1,
                "prompt_tokens": 5,
                "locally_computed_tokens": 4,
                "capacity_delay_ms": 0.0,
            },
            "D_MISS must report zero local cached tokens",
        ),
        (
            {
                "cache_status": "D_HIT",
                "actual_local_cached_tokens": 0,
                "prompt_tokens": 5,
                "locally_computed_tokens": 5,
                "capacity_delay_ms": 0.0,
            },
            "D_HIT requires a positive local hit",
        ),
        (
            {
                "cache_status": "D_HIT",
                "actual_local_cached_tokens": 4,
                "prompt_tokens": 5,
                "locally_computed_tokens": -1,
                "capacity_delay_ms": 0.0,
            },
            "malformed token counts",
        ),
        (
            {
                "cache_status": "D_HIT",
                "actual_local_cached_tokens": 4,
                "prompt_tokens": 5,
                "locally_computed_tokens": 1,
                "capacity_delay_ms": float("nan"),
            },
            "malformed capacity delay",
        ),
    ],
)
@pytest.mark.asyncio
async def test_bound_decode_closes_on_semantically_malformed_observation(
    observation: dict[str, object], message: str
) -> None:
    stream_closed = False

    class TrackingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunk = {
                "kv_transfer_params": {
                    "tda_forward": {"execution_observation": observation}
                }
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        async def aclose(self) -> None:
            nonlocal stream_closed
            stream_closed = True

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=TrackingStream())

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)

    with pytest.raises(RuntimeError, match=message):
        await adapter.start_bound(_turn())

    assert stream_closed
    await client.aclose()


@pytest.mark.asyncio
async def test_vllm_prefill_adapter_replaces_caller_transfer_metadata():
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
    turn = _turn(CacheAction.EVICT_D)
    turn = replace(
        turn,
        payload={
            **turn.payload,
            "kv_transfer_params": {
                "tda_forward": {"decode_bound": True},
                "caller_owned": True,
            },
        },
    )

    result = await adapter.prefill(turn)
    await client.aclose()

    assert seen_payload is not None
    assert seen_payload["stream"] is False
    assert seen_payload["max_tokens"] == 1
    assert seen_payload["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    assert result.prompt_tokens == 5
    assert result.transfer == {**transfer, "remote_host": "10.0.0.1"}


@pytest.mark.asyncio
async def test_bound_decode_rejects_a_missing_execution_observation():
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            content='data: {"choices": [{"text": "partial"}]}\n\ndata: [DONE]\n\n',
        )

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)

    with pytest.raises(RuntimeError, match="without an execution observation"):
        await adapter.start_bound(_turn())

    await client.aclose()


@pytest.mark.asyncio
async def test_bound_decode_rejects_a_malformed_execution_observation():
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        chunk = {
            "kv_transfer_params": {
                "tda_forward": {"execution_observation": {"cache_status": "D_HIT"}}
            }
        }
        return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\n")

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)

    with pytest.raises(RuntimeError, match="non-integer actual_local_cached_tokens"):
        await adapter.start_bound(_turn())

    await client.aclose()


@pytest.mark.asyncio
async def test_bound_decode_rejects_a_duplicate_execution_observation():
    observation = {
        "cache_status": "D_HIT",
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
            {
                "kv_transfer_params": {
                    "tda_forward": {"execution_observation": observation}
                }
            },
            {
                "kv_transfer_params": {
                    "tda_forward": {"execution_observation": observation}
                }
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
    turn = _turn()
    execution = await adapter.start_bound(turn)

    with pytest.raises(RuntimeError, match="duplicate execution observation"):
        [chunk async for chunk in execution.stream]
    with pytest.raises(RuntimeError, match="without a Cache Action ACK"):
        await adapter.finish(turn, CacheAction.RETAIN_D)

    await client.aclose()


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_decode_stream_without_ack_fails_finish():
    observation = {
        "cache_status": "D_HIT",
        "actual_local_cached_tokens": 4,
        "prompt_tokens": 5,
        "locally_computed_tokens": 1,
        "capacity_delay_ms": 0.0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        chunks = [
            {
                "kv_transfer_params": {
                    "tda_forward": {"execution_observation": observation}
                }
            },
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

    execution = await adapter.start_bound(turn)
    assert [chunk async for chunk in execution.stream]
    with pytest.raises(RuntimeError, match="without a Cache Action ACK"):
        await adapter.finish(turn, CacheAction.RETAIN_D)

    await client.aclose()


@pytest.mark.asyncio
async def test_ack_followed_by_malformed_stream_is_not_retained():
    observation = {
        "cache_status": "D_HIT",
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
            {
                "kv_transfer_params": {
                    "tda_forward": {"execution_observation": observation}
                }
            },
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

    execution = await adapter.start_bound(turn)
    with pytest.raises(json.JSONDecodeError):
        [chunk async for chunk in execution.stream]

    with pytest.raises(RuntimeError, match="without a Cache Action ACK"):
        await adapter.finish(turn, CacheAction.RETAIN_D)
    await client.aclose()


@pytest.mark.asyncio
async def test_ack_is_not_retained_when_stream_close_fails():
    observation = {
        "cache_status": "D_HIT",
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

    class CloseFailingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunks = [
                {
                    "kv_transfer_params": {
                        "tda_forward": {"execution_observation": observation}
                    }
                },
                {"kv_transfer_params": {"tda_forward": {"cache_action_ack": ack}}},
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            yield (body + "data: [DONE]\n\n").encode()

        async def aclose(self) -> None:
            raise RuntimeError("stream close failed")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=CloseFailingStream())

    client = httpx.AsyncClient(
        base_url="http://decode",
        transport=httpx.MockTransport(handler),
    )
    adapter = VllmDecodeAdapter(client)
    turn = _turn()
    execution = await adapter.start_bound(turn)
    with pytest.raises(RuntimeError, match="stream close failed"):
        [chunk async for chunk in execution.stream]

    with pytest.raises(RuntimeError, match="without a Cache Action ACK"):
        await adapter.finish(turn, CacheAction.RETAIN_D)
    await client.aclose()
