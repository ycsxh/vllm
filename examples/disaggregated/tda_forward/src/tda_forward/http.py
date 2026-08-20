# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thin OpenAI-compatible HTTP shell for the ReentryCoordinator."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tda_forward.coordinator import ReentryCoordinator, RequestValidationError

EventQueue = asyncio.Queue[tuple[int, object, float | None] | None]


def create_app(
    coordinator: ReentryCoordinator,
    *,
    event_queue: EventQueue | None = None,
) -> FastAPI:
    """Create the Proxy app without moving policy into the HTTP layer."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        consumer: asyncio.Task[None] | None = None
        if event_queue is not None:
            consumer = asyncio.create_task(coordinator.consume_events(event_queue))
        try:
            yield
        finally:
            if consumer is not None:
                consumer.cancel()
                with suppress(asyncio.CancelledError):
                    await consumer

    app = FastAPI(title="TDAforward Proxy", lifespan=lifespan)
    app.state.coordinator = coordinator

    async def handle(request: Request):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise HTTPException(status_code=400, detail="invalid JSON body") from error
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=422,
                detail="request body must be an object",
            )
        try:
            chunks = coordinator.start_turn(
                payload,
                api_path=request.url.path,
                request_id=request.headers.get("x-request-id"),
            )
        except RequestValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        if payload.get("stream", False):
            return StreamingResponse(
                _as_sse(chunks),
                media_type="text/event-stream",
            )
        collected = [chunk async for chunk in chunks]
        response = collected[-1] if collected else {"choices": []}
        return JSONResponse(response)

    app.add_api_route(
        "/v1/chat/completions",
        handle,
        methods=["POST"],
    )
    app.add_api_route(
        "/v1/completions",
        handle,
        methods=["POST"],
    )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "mirror_valid": coordinator.mirror.is_valid,
            "mirror_invalid_reason": coordinator.mirror.invalid_reason,
        }

    return app


async def _as_sse(
    chunks: AsyncIterator[dict[str, object]],
) -> AsyncIterator[str]:
    async for chunk in chunks:
        yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"
