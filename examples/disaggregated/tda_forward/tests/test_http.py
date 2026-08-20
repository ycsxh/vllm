# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import httpx
import pytest

from tda_forward.coordinator import ReentryCoordinator
from tda_forward.fakes import FakeDecodeAdapter, FakePrefillAdapter
from tda_forward.http import create_app
from tda_forward.mirror import DecodePrefixMirror


def make_app(
    event_queue: asyncio.Queue[tuple[int, object, float | None] | None] | None = None,
):
    coordinator = ReentryCoordinator(
        g=10.0,
        block_size=4,
        prefill=FakePrefillAdapter(),
        decode=FakeDecodeAdapter(),
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
