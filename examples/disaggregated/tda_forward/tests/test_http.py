# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import httpx
import pytest

from tda_forward.coordinator import ReentryCoordinator
from tda_forward.fakes import FakeDecodeAdapter, FakePrefillAdapter
from tda_forward.http import create_app
from tda_forward.mirror import DecodePrefixMirror
from tda_forward.native import NativeEventPump


def make_app(
    event_queue: asyncio.Queue[tuple[int, object, float | None] | None] | None = None,
    decode: FakeDecodeAdapter | None = None,
):
    coordinator = ReentryCoordinator(
        g=10.0,
        block_size=4,
        prefill=FakePrefillAdapter(),
        decode=decode or FakeDecodeAdapter(),
        mirror=DecodePrefixMirror(block_size=4),
    )
    return create_app(coordinator, event_queue=event_queue), coordinator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "input_field", "expected_choice"),
    [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
            {"message": {"role": "assistant", "content": "prefill"}},
        ),
        (
            "/v1/completions",
            {"prompt": [1, 2, 3, 4]},
            {"text": "prefill"},
        ),
    ],
)
async def test_openai_endpoints_route_through_the_coordinator(
    endpoint: str,
    input_field: dict[str, object],
    expected_choice: dict[str, object],
) -> None:
    app, coordinator = make_app()
    transport = httpx.ASGITransport(app=app)
    request = {
        "model": "fake",
        "session_id": "s",
        "t_pred": 1.0,
        "stream": False,
    }
    request.update(input_field)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(endpoint, json=request)

    assert response.status_code == 200
    assert response.json()["choices"] == [expected_choice]
    assert coordinator.records[-1].session_id == "s"


@pytest.mark.asyncio
async def test_http_validation_happens_before_worker_dispatch() -> None:
    app, coordinator = make_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/completions",
            json={"model": "fake", "prompt": [1], "t_pred": 1.0},
        )

    assert response.status_code == 422
    assert coordinator.prefill.calls == []
    assert coordinator.decode.calls == []


@pytest.mark.asyncio
async def test_delayed_event_producer_does_not_block_streaming() -> None:
    event_queue: asyncio.Queue[tuple[int, object, float | None] | None] = (
        asyncio.Queue()
    )
    app, _ = make_app(event_queue)
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        response = await asyncio.wait_for(
            client.post(
                "/v1/completions",
                json={
                    "model": "fake",
                    "prompt": [1, 2, 3, 4],
                    "session_id": "s",
                    "t_pred": 1.0,
                    "stream": True,
                },
            ),
            timeout=0.5,
        )

    assert response.status_code == 200
    assert 'data: {"choices":[{"text":"prefill"}]}' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_non_stream_response_aggregates_native_chat_deltas() -> None:
    class ChatDeltaDecodeAdapter(FakeDecodeAdapter):
        async def stream_from_prefill(self, turn, prefill):
            del turn, prefill
            yield {
                "id": "chatcmpl-1",
                "created": 1,
                "model": "test",
                "kv_transfer_params": {
                    "tda_forward": {
                        "execution_observation": {
                            "cache_status": "D_HIT",
                            "actual_local_cached_tokens": 16,
                            "prompt_tokens": 17,
                            "locally_computed_tokens": 1,
                            "capacity_delay_ms": 0.0,
                        }
                    }
                },
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "hello"},
                        "finish_reason": None,
                    }
                ],
            }
            yield {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield {"id": "chatcmpl-1", "choices": [], "usage": {"total_tokens": 7}}

    app, _ = make_app(decode=ChatDeltaDecodeAdapter())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "fake",
                "messages": [{"role": "user", "content": "hello"}],
                "session_id": "s",
                "t_pred": 50.0,
                "stream": False,
            },
        )

    response_body = response.json()
    assert response_body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello world",
    }
    assert response_body["usage"] == {"total_tokens": 7}
    assert "kv_transfer_params" not in response_body


@pytest.mark.asyncio
async def test_health_fails_closed_when_native_event_task_dies() -> None:
    class BrokenSubscriber:
        def receive_one(self, timeout: int):
            del timeout
            raise ValueError("malformed native frame")

        def close(self) -> None:
            return

    event_queue: asyncio.Queue[tuple[int, object, float | None] | None] = (
        asyncio.Queue()
    )
    _, coordinator = make_app(event_queue)
    pump = NativeEventPump(
        BrokenSubscriber,
        on_failure=coordinator.mirror.invalidate,
    )
    app = create_app(coordinator, event_queue=event_queue, event_pump=pump)
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        for _ in range(100):
            if not coordinator.mirror.is_valid:
                break
            await asyncio.sleep(0)
        response = await client.get("/health")

    assert response.json()["mirror_valid"] is False
    assert response.json()["event_pump"]["alive"] is False
    assert "malformed native frame" in response.json()["mirror_invalid_reason"]
