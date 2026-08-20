# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Python port of Dynamo's monotonic event cursor.

Source: ``lib/kv-router/src/recovery/cursor.rs`` at revision
``0226d2cf15af8b4a79b098a7ea24af168168c8c2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CursorKind(StrEnum):
    INITIAL = "initial"
    CONTIGUOUS = "contiguous"
    GAP = "gap"
    STALE = "stale"


@dataclass(frozen=True)
class CursorObservation:
    kind: CursorKind
    got: int
    expected: int | None = None
    last_applied: int | None = None


@dataclass(frozen=True)
class SequenceCursor:
    last_applied: int | None = None

    def observe(self, got: int) -> CursorObservation:
        if self.last_applied is None:
            return CursorObservation(CursorKind.INITIAL, got)
        if got <= self.last_applied:
            return CursorObservation(
                CursorKind.STALE,
                got,
                last_applied=self.last_applied,
            )
        expected = self.last_applied + 1
        if got == expected:
            return CursorObservation(CursorKind.CONTIGUOUS, got)
        return CursorObservation(CursorKind.GAP, got, expected=expected)

    def advance_to(self, got: int) -> SequenceCursor:
        return SequenceCursor(last_applied=got)
