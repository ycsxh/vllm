# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Locally executable TDAforward Proxy backed by deterministic fake workers."""

from __future__ import annotations

import argparse
import asyncio

import httpx
import uvicorn
from fastapi import FastAPI

from tda_forward.coordinator import ReentryCoordinator
from tda_forward.fakes import FakeDecodeAdapter, FakePrefillAdapter
from tda_forward.http import create_app
from tda_forward.mirror import DecodePrefixMirror
from tda_forward.native import (
    NativeEventPump,
    VllmDecodeAdapter,
    VllmPrefillAdapter,
    VllmTokenizerAdapter,
)


def build_local_app(*, g: float, block_size: int) -> FastAPI:
    mirror = DecodePrefixMirror(block_size=block_size)
    coordinator = ReentryCoordinator(
        g=g,
        block_size=block_size,
        prefill=FakePrefillAdapter(),
        decode=FakeDecodeAdapter(),
        mirror=mirror,
    )
    return create_app(coordinator)


def build_native_app(
    *,
    g: float,
    block_size: int,
    model: str,
    tokenizer_revision: str | None,
    prefill_url: str,
    decode_url: str,
    event_endpoints: list[str] | None,
    event_topic: str,
    trust_remote_code: bool,
    event_consumer_delay: float = 0.0,
    event_subscriber_hwm: int = 100_000,
) -> FastAPI:
    """Build the fixed 1P1D Proxy against native vLLM process boundaries."""
    prefill_client = httpx.AsyncClient(base_url=prefill_url, timeout=None)
    decode_client = httpx.AsyncClient(base_url=decode_url, timeout=None)
    mirror = DecodePrefixMirror(block_size=block_size)
    coordinator = ReentryCoordinator(
        g=g,
        block_size=block_size,
        prefill=VllmPrefillAdapter(
            prefill_client,
            remote_host=httpx.URL(prefill_url).host or "",
        ),
        decode=VllmDecodeAdapter(decode_client),
        mirror=mirror,
        tokenizer=VllmTokenizerAdapter(
            model,
            revision=tokenizer_revision,
            trust_remote_code=trust_remote_code,
        ),
    )
    event_queue = asyncio.Queue() if event_endpoints else None
    event_pump = (
        NativeEventPump.from_endpoints(
            event_endpoints,
            topic=event_topic,
            subscriber_hwm=event_subscriber_hwm,
            on_failure=mirror.invalidate,
            consumer_delay_seconds=event_consumer_delay,
        )
        if event_endpoints
        else None
    )

    async def shutdown() -> None:
        await prefill_client.aclose()
        await decode_client.aclose()

    return create_app(
        coordinator,
        event_queue=event_queue,
        event_pump=event_pump,
        shutdown=shutdown,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the portable TDAforward Proxy with fake workers"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--g", type=float, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--prefill-url")
    parser.add_argument("--decode-url")
    parser.add_argument("--event-endpoint", action="append", default=[])
    parser.add_argument("--event-topic", default="")
    parser.add_argument("--disable-events", action="store_true")
    parser.add_argument("--event-consumer-delay", type=float, default=0.0)
    parser.add_argument("--event-subscriber-hwm", type=int, default=100_000)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.native:
        missing = [
            name
            for name in (
                "model",
                "tokenizer_revision",
                "prefill_url",
                "decode_url",
            )
            if getattr(args, name) is None
        ]
        if not args.disable_events and not args.event_endpoint:
            missing.append("event_endpoint")
        if missing:
            required = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise SystemExit("--native requires " + required)
        if args.disable_events and args.event_endpoint:
            raise SystemExit("--disable-events conflicts with --event-endpoint")
        if not args.disable_events and len(args.event_endpoint) != 1:
            raise SystemExit("--native requires exactly one --event-endpoint")
        app = build_native_app(
            g=args.g,
            block_size=args.block_size,
            model=args.model,
            tokenizer_revision=args.tokenizer_revision,
            prefill_url=args.prefill_url,
            decode_url=args.decode_url,
            event_endpoints=None if args.disable_events else args.event_endpoint,
            event_topic=args.event_topic,
            trust_remote_code=args.trust_remote_code,
            event_consumer_delay=args.event_consumer_delay,
            event_subscriber_hwm=args.event_subscriber_hwm,
        )
    else:
        app = build_local_app(g=args.g, block_size=args.block_size)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
