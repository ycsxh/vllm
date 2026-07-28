# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated 1P1D pull proxy for the DS4 Qwen3.5 experiments."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

PREFILL_KV_TRANSFER_PARAMS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


def _invalid_metadata(detail: str) -> None:
    raise HTTPException(
        status_code=502,
        detail=f"invalid prefill kv_transfer_params: {detail}",
    )


def _validate_kv_transfer_params(
    params: dict[str, Any], expected_remote_tokens: int | None
) -> None:
    if params.get("do_remote_prefill") is not True:
        _invalid_metadata("do_remote_prefill must be true")
    if params.get("do_remote_decode") is not False:
        _invalid_metadata("do_remote_decode must be false")

    remote_block_ids = params.get("remote_block_ids")
    valid_block_ids = (
        isinstance(remote_block_ids, list)
        and bool(remote_block_ids)
        and all(isinstance(group, list) for group in remote_block_ids)
        and any(remote_block_ids)
        and all(
            type(block_id) is int and block_id >= 0
            for group in remote_block_ids
            for block_id in group
        )
    )
    if not valid_block_ids:
        _invalid_metadata("remote_block_ids must contain block IDs")

    for field in ("remote_engine_id", "remote_request_id"):
        value = params.get(field)
        if not isinstance(value, str) or not value.strip():
            _invalid_metadata(f"{field} must be a nonempty string")
    if params.get("remote_host") != "127.0.0.1":
        _invalid_metadata("remote_host must be 127.0.0.1")
    if type(params.get("remote_port")) is not int or params["remote_port"] != 5600:
        _invalid_metadata("remote_port must be 5600")
    if type(params.get("tp_size")) is not int or params["tp_size"] != 1:
        _invalid_metadata("tp_size must be 1")
    remote_num_tokens = params.get("remote_num_tokens")
    if type(remote_num_tokens) is not int or remote_num_tokens <= 0:
        _invalid_metadata("remote_num_tokens must be a positive integer")
    if (
        expected_remote_tokens is not None
        and remote_num_tokens != expected_remote_tokens
    ):
        _invalid_metadata(f"remote_num_tokens must be {expected_remote_tokens}")


async def _post_json(
    url: str, payload: dict[str, Any], request_timeout: float
) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(url, json=payload) as response,
    ):
        response.raise_for_status()
        return await response.json(content_type=None)


async def _post_stream(
    url: str, payload: dict[str, Any], request_timeout: float
) -> StreamingResponse:
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        response = await session.post(url, json=payload)
        response.raise_for_status()
    except BaseException:
        await session.close()
        raise

    async def generate() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.content.iter_any():
                yield chunk
        finally:
            response.release()
            await session.close()

    return StreamingResponse(generate(), media_type=response.content_type)


def create_app(
    prefill_url: str,
    decode_url: str,
    request_timeout: float,
    expected_remote_tokens: int | None,
) -> FastAPI:
    """Create the validated proxy application."""
    app = FastAPI()
    prefill_endpoint = f"{prefill_url.rstrip('/')}/v1/completions"
    decode_endpoint = f"{decode_url.rstrip('/')}/v1/completions"

    @app.get("/status")
    async def status() -> dict[str, str]:
        return {"prefill": prefill_url, "decode": decode_url}

    @app.post("/v1/completions")
    async def create_completion(request: Request) -> Response:
        request_payload = await request.json()
        prefill_payload = {
            **request_payload,
            "max_tokens": 1,
            "stream": False,
            "kv_transfer_params": PREFILL_KV_TRANSFER_PARAMS,
        }
        prefill_payload.pop("stream_options", None)
        prefill_response = await _post_json(
            prefill_endpoint, prefill_payload, request_timeout
        )
        if "kv_transfer_params" not in prefill_response:
            raise HTTPException(
                status_code=502,
                detail="prefill response lacks kv_transfer_params",
            )
        kv_transfer_params = prefill_response["kv_transfer_params"]
        if not isinstance(kv_transfer_params, dict):
            raise HTTPException(
                status_code=502,
                detail="prefill kv_transfer_params must be an object",
            )
        _validate_kv_transfer_params(kv_transfer_params, expected_remote_tokens)
        decode_payload = {
            **request_payload,
            "kv_transfer_params": kv_transfer_params,
        }
        if decode_payload.get("stream") is True:
            return await _post_stream(decode_endpoint, decode_payload, request_timeout)
        decode_response = await _post_json(
            decode_endpoint, decode_payload, request_timeout
        )
        return JSONResponse(decode_response)

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--expected-remote-tokens", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    uvicorn.run(
        create_app(
            args.prefill_url,
            args.decode_url,
            args.request_timeout,
            args.expected_remote_tokens,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
