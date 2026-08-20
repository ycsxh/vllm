# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Portable Decode Prefix Mirror over vLLM's native KV event vocabulary."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from tda_forward.cursor import CursorKind, SequenceCursor
from tda_forward.hashing import block_hashes

_MISSING = object()
_U64_MASK = (1 << 64) - 1


@dataclass
class LineageEntry:
    """One ordered content block in a session lineage."""

    local_hash: int
    external_hash: int | None = None
    present: bool = False


@dataclass
class MirrorMetrics:
    event_batches: int = 0
    stored_blocks: int = 0
    removed_blocks: int = 0
    clears: int = 0
    last_event_lag_seconds: float | None = None


def _field(value: object, name: str, default: object = _MISSING) -> object:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is _MISSING:
        raise ValueError(f"missing event field {name!r}")
    return default


def _event_type(event: object) -> str:
    tag = _field(event, "type", None)
    if isinstance(tag, str):
        return tag
    return type(event).__name__


def _hash_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("block hashes must be integers")
    if not -(1 << 63) <= value < (1 << 64):
        raise ValueError("block hashes must fit signed or unsigned 64-bit")
    return value & _U64_MASK


class DecodePrefixMirror:
    """Advisory Decode-residency view for linear per-session lineages."""

    def __init__(self, block_size: int) -> None:
        if isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self._sessions: dict[str, list[LineageEntry]] = {}
        self._local_refs: dict[int, set[tuple[str, int]]] = defaultdict(set)
        self._external_refs: dict[int, set[tuple[str, int]]] = defaultdict(set)
        self._cursor = SequenceCursor()
        self._invalid_reason: str | None = None
        self.gap_count = 0
        self.metrics = MirrorMetrics()

    @property
    def is_valid(self) -> bool:
        return self._invalid_reason is None

    @property
    def invalid_reason(self) -> str | None:
        return self._invalid_reason

    @property
    def last_sequence(self) -> int | None:
        return self._cursor.last_applied

    def register_session(
        self,
        session_id: str,
        token_ids: Sequence[int],
        *,
        lora_name: str | None = None,
        cache_namespace: str | None = None,
        block_extra_keys: Sequence[object | None] | None = None,
    ) -> None:
        """Register the current ordered prompt lineage for one session."""
        local_hashes = block_hashes(
            token_ids,
            self.block_size,
            lora_name=lora_name,
            cache_namespace=cache_namespace,
            block_extra_keys=block_extra_keys,
        )
        previous = self._sessions.get(session_id, [])
        preserved: list[LineageEntry] = []
        prefix_matches = True
        for index, local_hash in enumerate(local_hashes):
            if (
                prefix_matches
                and index < len(previous)
                and previous[index].local_hash == local_hash
            ):
                preserved.append(replace(previous[index]))
            else:
                prefix_matches = False
                preserved.append(LineageEntry(local_hash=local_hash))

        self._detach_session(session_id)
        self._sessions[session_id] = preserved
        for index, entry in enumerate(preserved):
            ref = (session_id, index)
            self._local_refs[entry.local_hash].add(ref)
            if entry.external_hash is not None:
                self._external_refs[entry.external_hash].add(ref)

    def _detach_session(self, session_id: str) -> None:
        for index, entry in enumerate(self._sessions.get(session_id, [])):
            ref = (session_id, index)
            self._discard_ref(self._local_refs, entry.local_hash, ref)
            if entry.external_hash is not None:
                self._discard_ref(self._external_refs, entry.external_hash, ref)

    @staticmethod
    def _discard_ref(
        index: dict[int, set[tuple[str, int]]],
        key: int,
        ref: tuple[str, int],
    ) -> None:
        refs = index.get(key)
        if refs is None:
            return
        refs.discard(ref)
        if not refs:
            index.pop(key, None)

    def estimated_cached_tokens(self, session_id: str) -> int:
        count = 0
        for entry in self._sessions.get(session_id, []):
            if not entry.present or entry.external_hash is None:
                break
            count += 1
        return count * self.block_size

    def snapshot(self, session_id: str) -> list[LineageEntry]:
        return [replace(entry) for entry in self._sessions.get(session_id, [])]

    def apply_batch(
        self,
        sequence: int,
        batch: object,
        *,
        received_at: float | None = None,
    ) -> str:
        """Apply one native vLLM batch with its transport sequence number."""
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            return self._invalidate("malformed event sequence", "malformed")
        if not self.is_valid:
            return "invalid"
        observation = self._cursor.observe(sequence)
        if observation.kind is CursorKind.STALE:
            return "stale"
        if observation.kind is CursorKind.GAP:
            self.gap_count += 1
            self._cursor = self._cursor.advance_to(sequence)
            self._clear_presence()
            return self._invalidate(
                f"event sequence gap: expected {observation.expected}, got {sequence}",
                "gap",
            )

        try:
            timestamp, events = self._batch_parts(batch)
            now = time.time() if received_at is None else received_at
            if not math.isfinite(now):
                raise ValueError("received_at must be finite")
            for event in events:
                self._apply_event(event)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            self._cursor = self._cursor.advance_to(sequence)
            self._clear_presence()
            return self._invalidate(f"malformed KV event batch: {error}", "malformed")

        self._cursor = self._cursor.advance_to(sequence)
        self.metrics.event_batches += 1
        self.metrics.last_event_lag_seconds = max(0.0, now - timestamp)
        return "applied"

    @staticmethod
    def _batch_parts(batch: object) -> tuple[float, Sequence[object]]:
        if isinstance(batch, Mapping) or hasattr(batch, "events"):
            timestamp = _field(batch, "ts")
            events = _field(batch, "events")
        elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            if len(batch) not in (2, 3):
                raise ValueError("native KVEventBatch must have two or three fields")
            timestamp, events = batch[0], batch[1]
        else:
            raise ValueError("unsupported native KVEventBatch representation")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError("batch timestamp must be numeric")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("batch timestamp must be finite")
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            raise ValueError("batch events must be a sequence")
        return timestamp, events

    def _apply_event(self, event: object) -> None:
        kind = _event_type(event)
        if kind == "BlockStored":
            self._apply_stored(event)
        elif kind == "BlockRemoved":
            self._apply_removed(event)
        elif kind == "AllBlocksCleared":
            self.metrics.clears += 1
            self._clear_presence()
        else:
            raise ValueError(f"unknown KV event type {kind!r}")

    def _apply_stored(self, event: object) -> None:
        raw_hashes = _field(event, "block_hashes")
        token_ids = _field(event, "token_ids")
        block_size = _field(event, "block_size")
        parent_raw = _field(event, "parent_block_hash")
        lora_name = _field(event, "lora_name", None)
        cache_namespace = _field(event, "cache_salt", None)
        extra_keys = _field(event, "extra_keys", None)
        medium = _field(event, "medium", None)
        if medium not in (None, "GPU"):
            return
        if block_size != self.block_size:
            raise ValueError(
                f"event block_size {block_size!r} does not match {self.block_size}"
            )
        if not isinstance(raw_hashes, Sequence) or isinstance(raw_hashes, (str, bytes)):
            raise ValueError("block_hashes must be a sequence")
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
            raise ValueError("token_ids must be a sequence")
        external_hashes = [_hash_value(value) for value in raw_hashes]
        if len(token_ids) < len(external_hashes) * self.block_size:
            raise ValueError("token_ids do not cover every stored block")
        if extra_keys is not None and (
            not isinstance(extra_keys, Sequence)
            or isinstance(extra_keys, (str, bytes))
            or len(extra_keys) < len(external_hashes)
        ):
            raise ValueError("extra_keys must cover every stored block")
        local_hashes = block_hashes(
            token_ids[: len(external_hashes) * self.block_size],
            self.block_size,
            lora_name=lora_name if isinstance(lora_name, str) else None,
            cache_namespace=(
                cache_namespace if isinstance(cache_namespace, str) else None
            ),
            block_extra_keys=extra_keys,
        )
        parent = None if parent_raw is None else _hash_value(parent_raw)
        for local_hash, external_hash in zip(
            local_hashes, external_hashes, strict=True
        ):
            self._correlate_store(local_hash, external_hash, parent)
            parent = external_hash
            self.metrics.stored_blocks += 1

    def _correlate_store(
        self, local_hash: int, external_hash: int, parent: int | None
    ) -> None:
        for session_id, index in tuple(self._local_refs.get(local_hash, ())):
            lineage = self._sessions[session_id]
            parent_matches = (index == 0 and parent is None) or (
                index > 0
                and parent is not None
                and lineage[index - 1].external_hash == parent
            )
            if not parent_matches:
                continue
            entry = lineage[index]
            ref = (session_id, index)
            if entry.external_hash is not None and entry.external_hash != external_hash:
                self._discard_ref(self._external_refs, entry.external_hash, ref)
            entry.external_hash = external_hash
            entry.present = True
            self._external_refs[external_hash].add(ref)

    def _apply_removed(self, event: object) -> None:
        raw_hashes = _field(event, "block_hashes")
        medium = _field(event, "medium", None)
        if medium not in (None, "GPU"):
            return
        if not isinstance(raw_hashes, Sequence) or isinstance(raw_hashes, (str, bytes)):
            raise ValueError("block_hashes must be a sequence")
        for raw_hash in raw_hashes:
            external_hash = _hash_value(raw_hash)
            for session_id, index in tuple(self._external_refs.get(external_hash, ())):
                self._sessions[session_id][index].present = False
            self.metrics.removed_blocks += 1

    def _clear_presence(self) -> None:
        for lineage in self._sessions.values():
            for entry in lineage:
                entry.present = False

    def _invalidate(self, reason: str, result: str) -> str:
        if self._invalid_reason is None:
            self._invalid_reason = reason
        return result
