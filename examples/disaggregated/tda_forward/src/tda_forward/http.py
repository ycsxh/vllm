# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Thin OpenAI-compatible HTTP shell for the ReentryCoordinator."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tda_forward.coordinator import ReentryCoordinator, RequestValidationError

EventQueue = asyncio.Queue[tuple[int, object, float | None] | None]


class EventPump(Protocol):
    async def run(self, event_queue: EventQueue) -> None: ...

    def close(self) -> None: ...

    @property
    def status(self) -> dict[str, object]: ...


def create_app(
    coordinator: ReentryCoordinator,
    *,
    event_queue: EventQueue | None = None,
    event_pump: EventPump | None = None,
    shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Create the Proxy app without moving policy into the HTTP layer."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        consumer: asyncio.Task[None] | None = None
        producer: asyncio.Task[None] | None = None
        if event_queue is not None:
            consumer = asyncio.create_task(coordinator.consume_events(event_queue))
            if event_pump is not None:
                producer = asyncio.create_task(event_pump.run(event_queue))
        try:
            yield
        finally:
            if producer is not None:
                producer.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await producer
                if event_pump is not None:
                    event_pump.close()
            if consumer is not None:
                consumer.cancel()
                with suppress(asyncio.CancelledError):
                    await consumer
            if shutdown is not None:
                await shutdown()

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
        response = _aggregate_stream(collected, request.url.path)
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
            "event_pump": event_pump.status if event_pump is not None else None,
        }

    @app.get("/acceptance/state")
    async def acceptance_state() -> dict[str, object]:
        """Expose a read-only evidence snapshot without changing routing state."""
        return {
            "records": [record.as_dict() for record in coordinator.records],
            "mirror": {
                "valid": coordinator.mirror.is_valid,
                "invalid_reason": coordinator.mirror.invalid_reason,
                "last_sequence": coordinator.mirror.last_sequence,
                "gap_count": coordinator.mirror.gap_count,
                "first_gap_at": coordinator.mirror.first_gap_at,
                "metrics": asdict(coordinator.mirror.metrics),
            },
            "event_pump": event_pump.status if event_pump is not None else None,
        }

    return app


async def _as_sse(
    chunks: AsyncIterator[dict[str, object]],
) -> AsyncIterator[str]:
    async for chunk in chunks:
        yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def _aggregate_stream(
    chunks: list[dict[str, object]],
    api_path: str,
) -> dict[str, object]:
    if api_path == "/v1/chat/completions":
        response = _aggregate_chat_stream(chunks)
    else:
        response = _aggregate_completion_stream(chunks)
    if not response["choices"] and chunks:
        return chunks[-1]
    return response


def _response_metadata(chunks: list[dict[str, object]], object_type: str) -> dict:
    response: dict[str, Any] = {"object": object_type, "choices": []}
    for chunk in chunks:
        for key in ("id", "created", "model", "system_fingerprint", "usage"):
            value = chunk.get(key)
            if value is not None:
                response[key] = value
    return response


def _aggregate_completion_stream(
    chunks: list[dict[str, object]],
) -> dict[str, object]:
    response = _response_metadata(chunks, "text_completion")
    choices: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        raw_choices = chunk.get("choices")
        if not isinstance(raw_choices, list):
            continue
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                continue
            index = raw_choice.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            choice = choices.setdefault(
                index,
                {"index": index, "text": "", "logprobs": None, "finish_reason": None},
            )
            text = raw_choice.get("text")
            if isinstance(text, str):
                choice["text"] += text
            for key in ("finish_reason", "stop_reason"):
                if raw_choice.get(key) is not None:
                    choice[key] = raw_choice[key]
    response["choices"] = [choices[index] for index in sorted(choices)]
    return response


def _aggregate_chat_stream(
    chunks: list[dict[str, object]],
) -> dict[str, object]:
    response = _response_metadata(chunks, "chat.completion")
    choices: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        raw_choices = chunk.get("choices")
        if not isinstance(raw_choices, list):
            continue
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                continue
            index = raw_choice.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            choice = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                },
            )
            delta = raw_choice.get("delta")
            if isinstance(delta, dict):
                message = choice["message"]
                role = delta.get("role")
                if isinstance(role, str):
                    message["role"] = role
                for key in ("content", "reasoning"):
                    value = delta.get(key)
                    if isinstance(value, str):
                        message[key] = message.get(key, "") + value
                if isinstance(delta.get("tool_calls"), list):
                    _merge_tool_call_deltas(message, delta["tool_calls"])
            for key in ("finish_reason", "stop_reason"):
                if raw_choice.get(key) is not None:
                    choice[key] = raw_choice[key]
    response["choices"] = [choices[index] for index in sorted(choices)]
    return response


def _merge_tool_call_deltas(
    message: dict[str, Any],
    deltas: list[object],
) -> None:
    calls = message.setdefault("tool_calls", [])
    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        index = delta.get("index", len(calls))
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            continue
        while len(calls) <= index:
            calls.append({"index": len(calls), "function": {}})
        call = calls[index]
        for key in ("id", "type"):
            if delta.get(key) is not None:
                call[key] = delta[key]
        function_delta = delta.get("function")
        if not isinstance(function_delta, dict):
            continue
        function = call.setdefault("function", {})
        name = function_delta.get("name")
        if isinstance(name, str):
            function["name"] = name
        arguments = function_delta.get("arguments")
        if isinstance(arguments, str):
            function["arguments"] = function.get("arguments", "") + arguments
