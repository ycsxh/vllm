# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Portable Decode Prefix Mirror over vLLM's native KV event vocabulary."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from tda_forward.cursor import CursorKind, SequenceCursor
from tda_forward.hashing import block_hashes

_MISSING = object()
_U64_MASK = (1 << 64) - 1
_IndexKey = TypeVar("_IndexKey")


class BatchApplyResult(StrEnum):
    APPLIED = "applied"
    STALE = "stale"
    GAP = "gap"
    MALFORMED = "malformed"
    INVALID = "invalid"


@dataclass(frozen=True)
class LineageRef:
    session_id: str
    index: int


@dataclass(frozen=True)
class ContentKey:
    token_ids: tuple[int, ...]
    extra_keys: object | None


@dataclass(frozen=True)
class ExternalKey:
    group_idx: int
    block_hash: int


@dataclass
class LineageEntry:
    """One ordered content block in a session lineage."""

    content_key: ContentKey
    local_hash: int
    external_hashes: dict[int, int]


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


def _freeze_extra(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("extra-key floats must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_extra(item) for item in value)
    raise ValueError(f"unsupported vLLM extra-key value: {type(value).__name__}")


def _group_index(event: object) -> int:
    value = _field(event, "group_idx", None)
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("group_idx must be a nonnegative integer")
    return value


class DecodePrefixMirror:
    """Advisory Decode-residency view for linear per-session lineages."""

    def __init__(self, block_size: int) -> None:
        if isinstance(block_size, bool) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self._sessions: dict[str, list[LineageEntry]] = {}
        self._session_groups: dict[str, set[int]] = {}
        self._content_refs: dict[ContentKey, set[LineageRef]] = defaultdict(set)
        self._external_refs: dict[ExternalKey, set[LineageRef]] = defaultdict(set)
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

    def invalidate(self, reason: str) -> None:
        """Fail closed when the native event transport is no longer trustworthy."""
        if self._invalid_reason is None:
            self._invalid_reason = reason

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
        )
        complete_blocks = len(local_hashes)
        if block_extra_keys is not None and len(block_extra_keys) < complete_blocks:
            raise ValueError("block_extra_keys must cover every complete token block")
        content_keys = self._content_keys(
            token_ids,
            complete_blocks,
            block_extra_keys=block_extra_keys,
            lora_name=lora_name,
            cache_namespace=cache_namespace,
        )
        previous = self._sessions.get(session_id, [])
        preserved: list[LineageEntry] = []
        prefix_matches = True
        for index, (local_hash, content_key) in enumerate(
            zip(local_hashes, content_keys, strict=True)
        ):
            if (
                prefix_matches
                and index < len(previous)
                and previous[index].content_key == content_key
            ):
                preserved.append(
                    LineageEntry(
                        content_key=content_key,
                        local_hash=local_hash,
                        external_hashes=dict(previous[index].external_hashes),
                    )
                )
            else:
                prefix_matches = False
                preserved.append(LineageEntry(content_key, local_hash, {}))

        self._detach_session(session_id)
        self._sessions[session_id] = preserved
        self._session_groups.setdefault(session_id, set())
        for index, entry in enumerate(preserved):
            ref = LineageRef(session_id, index)
            self._content_refs[entry.content_key].add(ref)
            for group_idx, external_hash in entry.external_hashes.items():
                self._external_refs[ExternalKey(group_idx, external_hash)].add(ref)

    def _content_keys(
        self,
        token_ids: Sequence[int],
        complete_blocks: int,
        *,
        block_extra_keys: Sequence[object | None] | None,
        lora_name: str | None = None,
        cache_namespace: str | None = None,
    ) -> list[ContentKey]:
        keys: list[ContentKey] = []
        for index in range(complete_blocks):
            start = index * self.block_size
            if block_extra_keys is not None:
                extra = block_extra_keys[index]
            else:
                parts = ([lora_name] if lora_name else []) + (
                    [cache_namespace] if index == 0 and cache_namespace else []
                )
                extra = tuple(parts) if parts else None
            keys.append(
                ContentKey(
                    tuple(token_ids[start : start + self.block_size]),
                    _freeze_extra(extra),
                )
            )
        return keys

    def _detach_session(self, session_id: str) -> None:
        for index, entry in enumerate(self._sessions.get(session_id, [])):
            ref = LineageRef(session_id, index)
            self._discard_ref(self._content_refs, entry.content_key, ref)
            for group_idx, external_hash in entry.external_hashes.items():
                self._discard_ref(
                    self._external_refs,
                    ExternalKey(group_idx, external_hash),
                    ref,
                )

    @staticmethod
    def _discard_ref(
        index: dict[_IndexKey, set[LineageRef]],
        key: _IndexKey,
        ref: LineageRef,
    ) -> None:
        refs = index.get(key)
        if refs is None:
            return
        refs.discard(ref)
        if not refs:
            index.pop(key, None)

    def estimated_cached_tokens(self, session_id: str) -> int:
        required_groups = self._session_groups.get(session_id, set())
        if not required_groups:
            return 0
        count = 0
        for entry in self._sessions.get(session_id, []):
            if not required_groups.issubset(entry.external_hashes):
                break
            count += 1
        return count * self.block_size

    def snapshot(self, session_id: str) -> list[LineageEntry]:
        return [
            LineageEntry(
                entry.content_key,
                entry.local_hash,
                dict(entry.external_hashes),
            )
            for entry in self._sessions.get(session_id, [])
        ]

    def apply_batch(
        self,
        sequence: int,
        batch: object,
        *,
        received_at: float | None = None,
    ) -> BatchApplyResult:
        """Apply one native vLLM batch with its transport sequence number."""
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            return self._invalidate(
                "malformed event sequence", BatchApplyResult.MALFORMED
            )
        if not self.is_valid:
            return BatchApplyResult.INVALID
        observation = self._cursor.observe(sequence)
        if observation.kind is CursorKind.STALE:
            return BatchApplyResult.STALE
        if observation.kind is CursorKind.GAP:
            self.gap_count += 1
            self._cursor = self._cursor.advance_to(sequence)
            self._clear_presence()
            return self._invalidate(
                f"event sequence gap: expected {observation.expected}, got {sequence}",
                BatchApplyResult.GAP,
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
            return self._invalidate(
                f"malformed KV event batch: {error}", BatchApplyResult.MALFORMED
            )

        self._cursor = self._cursor.advance_to(sequence)
        self.metrics.event_batches += 1
        self.metrics.last_event_lag_seconds = max(0.0, now - timestamp)
        return BatchApplyResult.APPLIED

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
        extra_keys = _field(event, "extra_keys", None)
        group_idx = _group_index(event)
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
        content_keys = self._content_keys(
            token_ids[: len(external_hashes) * self.block_size],
            len(external_hashes),
            block_extra_keys=extra_keys,
            lora_name=lora_name if isinstance(lora_name, str) else None,
        )
        parent = None if parent_raw is None else _hash_value(parent_raw)
        for content_key, external_hash in zip(
            content_keys, external_hashes, strict=True
        ):
            self._correlate_store(content_key, external_hash, parent, group_idx)
            parent = external_hash
            self.metrics.stored_blocks += 1

    def _correlate_store(
        self,
        content_key: ContentKey,
        external_hash: int,
        parent: int | None,
        group_idx: int,
    ) -> None:
        for ref in tuple(self._content_refs.get(content_key, ())):
            lineage = self._sessions[ref.session_id]
            parent_matches = (ref.index == 0 and parent is None) or (
                ref.index > 0
                and parent is not None
                and lineage[ref.index - 1].external_hashes.get(group_idx) == parent
            )
            if not parent_matches:
                continue
            entry = lineage[ref.index]
            previous_hash = entry.external_hashes.get(group_idx)
            if previous_hash is not None and previous_hash != external_hash:
                self._discard_ref(
                    self._external_refs,
                    ExternalKey(group_idx, previous_hash),
                    ref,
                )
            entry.external_hashes[group_idx] = external_hash
            self._session_groups[ref.session_id].add(group_idx)
            self._external_refs[ExternalKey(group_idx, external_hash)].add(ref)

    def _apply_removed(self, event: object) -> None:
        raw_hashes = _field(event, "block_hashes")
        medium = _field(event, "medium", None)
        group_idx = _group_index(event)
        if medium not in (None, "GPU"):
            return
        if not isinstance(raw_hashes, Sequence) or isinstance(raw_hashes, (str, bytes)):
            raise ValueError("block_hashes must be a sequence")
        for raw_hash in raw_hashes:
            external_hash = _hash_value(raw_hash)
            external_key = ExternalKey(group_idx, external_hash)
            for ref in tuple(self._external_refs.pop(external_key, ())):
                self._sessions[ref.session_id][ref.index].external_hashes.pop(
                    group_idx, None
                )
            self.metrics.removed_blocks += 1

    def _clear_presence(self) -> None:
        self._external_refs.clear()
        for lineage in self._sessions.values():
            for entry in lineage:
                entry.external_hashes.clear()

    def _invalidate(self, reason: str, result: BatchApplyResult) -> BatchApplyResult:
        if self._invalid_reason is None:
            self._invalid_reason = reason
        return result
