# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Locally executable TDAforward Proxy backed by deterministic fake workers."""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI

from tda_forward.coordinator import ReentryCoordinator
from tda_forward.fakes import FakeDecodeAdapter, FakePrefillAdapter
from tda_forward.http import create_app
from tda_forward.mirror import DecodePrefixMirror


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the portable TDAforward Proxy with fake workers"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--g", type=float, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_local_app(g=args.g, block_size=args.block_size)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
