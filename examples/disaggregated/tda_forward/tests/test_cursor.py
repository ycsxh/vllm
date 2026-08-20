# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tda_forward.cursor import CursorKind, SequenceCursor


def test_pinned_dynamo_cursor_equivalence_vectors() -> None:
    initial = SequenceCursor().observe(5)
    assert (initial.kind, initial.got, initial.expected) == (
        CursorKind.INITIAL,
        5,
        None,
    )

    cursor = SequenceCursor(last_applied=10)
    contiguous = cursor.observe(11)
    gap = cursor.observe(15)
    duplicate = cursor.observe(10)
    older = cursor.observe(9)

    assert contiguous.kind is CursorKind.CONTIGUOUS
    assert (gap.kind, gap.expected, gap.got) == (CursorKind.GAP, 11, 15)
    assert (duplicate.kind, duplicate.last_applied) == (CursorKind.STALE, 10)
    assert (older.kind, older.last_applied) == (CursorKind.STALE, 10)
