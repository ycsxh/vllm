# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dynamo-compatible public block and sequence hashing.

This is a Python port of the public XXH3 path at Dynamo revision
``0226d2cf15af8b4a79b098a7ea24af168168c8c2``. The local Proxy needs the
same content identity behavior but must not depend on the complete Dynamo
runtime.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import xxhash

CHAIN_XXH3_SEED = 1337
_U64_MASK = (1 << 64) - 1
_EXTRA_KEY_DOMAIN = b"tdaforward/vllm-extra-keys/v1\0"


def _xxh3(data: bytes, seed: int) -> int:
    return xxhash.xxh3_64_intdigest(data, seed=seed)


def _salt_hash(cache_namespace: str | None, lora_name: str | None) -> int:
    seed = CHAIN_XXH3_SEED
    if lora_name:
        seed = (seed + _xxh3(lora_name.encode(), 0)) & _U64_MASK
    if cache_namespace:
        seed = (seed + _xxh3(cache_namespace.encode(), 1)) & _U64_MASK
    return seed


def _encode_extra(value: object) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b\x01" if value else b"b\x00"
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 64):
            raise ValueError("extra-key integers must fit signed or unsigned 64-bit")
        return b"i" + (value & _U64_MASK).to_bytes(8, "little")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("extra-key floats must be finite")
        return b"f" + struct.pack("<d", value)
    if isinstance(value, str):
        encoded = value.encode()
        return b"s" + len(encoded).to_bytes(8, "little") + encoded
    if isinstance(value, bytes):
        return b"y" + len(value).to_bytes(8, "little") + value
    if isinstance(value, (tuple, list)):
        encoded_items = [_encode_extra(item) for item in value]
        return (
            b"q"
            + len(encoded_items).to_bytes(8, "little")
            + b"".join(len(item).to_bytes(8, "little") + item for item in encoded_items)
        )
    raise TypeError(f"unsupported vLLM extra-key value: {type(value).__name__}")


def _validate_tokens(tokens: Sequence[int]) -> None:
    if any(isinstance(token, bool) or not 0 <= token < (1 << 32) for token in tokens):
        raise ValueError("token IDs must be unsigned 32-bit integers")


def block_hashes(
    tokens: Sequence[int],
    block_size: int,
    *,
    lora_name: str | None = None,
    cache_namespace: str | None = None,
    block_extra_keys: Sequence[object | None] | None = None,
) -> list[int]:
    """Hash every complete token block using Dynamo's public XXH3 domain."""
    if isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    _validate_tokens(tokens)
    complete_blocks = len(tokens) // block_size
    if block_extra_keys is not None and len(block_extra_keys) < complete_blocks:
        raise ValueError("block_extra_keys must cover every complete token block")

    seed = _salt_hash(cache_namespace, lora_name)
    hashes: list[int] = []
    for index in range(complete_blocks):
        start = index * block_size
        chunk = tokens[start : start + block_size]
        block_bytes = struct.pack(f"<{block_size}I", *chunk)
        extra = block_extra_keys[index] if block_extra_keys is not None else None
        if extra is not None:
            encoded = _encode_extra(extra)
            block_bytes += (
                _EXTRA_KEY_DOMAIN + len(encoded).to_bytes(8, "little") + encoded
            )
        hashes.append(_xxh3(block_bytes, seed))
    return hashes


def sequence_hashes(
    tokens: Sequence[int],
    block_size: int,
    *,
    lora_name: str | None = None,
    cache_namespace: str | None = None,
    block_extra_keys: Sequence[object | None] | None = None,
) -> list[int]:
    """Return Dynamo's parent-chained hashes for all complete blocks."""
    blocks = block_hashes(
        tokens,
        block_size,
        lora_name=lora_name,
        cache_namespace=cache_namespace,
        block_extra_keys=block_extra_keys,
    )
    if not blocks:
        return []
    chained = [blocks[0]]
    for block_hash in blocks[1:]:
        chained.append(
            _xxh3(struct.pack("<QQ", chained[-1], block_hash), CHAIN_XXH3_SEED)
        )
    return chained
