# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tda_forward.mirror import BatchApplyResult, DecodePrefixMirror


def stored(
    hashes: list[int],
    tokens: list[int],
    *,
    parent: int | None = None,
    block_size: int = 4,
    lora_name: str | None = None,
    extra_keys: list[tuple[object, ...] | None] | None = None,
    group_idx: int | None = None,
) -> dict[str, object]:
    return {
        "type": "BlockStored",
        "block_hashes": hashes,
        "parent_block_hash": parent,
        "token_ids": tokens,
        "block_size": block_size,
        "lora_id": None,
        "medium": "GPU",
        "lora_name": lora_name,
        "extra_keys": extra_keys,
        "group_idx": group_idx,
    }


def batch(ts: float, *events: dict[str, object]) -> list[object]:
    return [ts, list(events), 0]


def test_longest_contiguous_prefix_and_shared_content_fan_out() -> None:
    mirror = DecodePrefixMirror(block_size=4)
    prompt = list(range(12))
    mirror.register_session("alpha", prompt)
    mirror.register_session("beta", prompt[:8])

    assert (
        mirror.apply_batch(
            10,
            batch(1.0, stored([101, 102], prompt[:8])),
            received_at=2.5,
        )
        is BatchApplyResult.APPLIED
    )
    assert mirror.metrics.last_event_lag_seconds == 1.5
    assert mirror.metrics.stored_blocks == 2
    assert mirror.estimated_cached_tokens("alpha") == 8
    assert mirror.estimated_cached_tokens("beta") == 8

    mirror.apply_batch(
        11,
        batch(2.0, {"type": "BlockRemoved", "block_hashes": [101]}),
    )
    assert mirror.estimated_cached_tokens("alpha") == 0
    assert mirror.estimated_cached_tokens("beta") == 0


def test_remove_preserves_ordered_lineage() -> None:
    mirror = DecodePrefixMirror(block_size=4)
    prompt = list(range(12))
    mirror.register_session("s", prompt)
    mirror.apply_batch(1, batch(1.0, stored([11, 12, 13], prompt)))

    mirror.apply_batch(
        2,
        batch(2.0, {"type": "BlockRemoved", "block_hashes": [12]}),
    )
    assert mirror.estimated_cached_tokens("s") == 4


def test_clear_preserves_ordered_lineage() -> None:
    mirror = DecodePrefixMirror(block_size=4)
    prompt = list(range(12))
    mirror.register_session("s", prompt)
    mirror.apply_batch(1, batch(1.0, stored([11, 12, 13], prompt)))

    mirror.apply_batch(3, batch(3.0, {"type": "AllBlocksCleared"}))
    assert mirror.estimated_cached_tokens("s") == 0
    assert [entry.external_hashes for entry in mirror.snapshot("s")] == [{}, {}, {}]


def test_lora_and_multimodal_identity_do_not_cross_correlate() -> None:
    mirror = DecodePrefixMirror(block_size=4)
    tokens = [1, 2, 3, 4]
    mirror.register_session("base", tokens)
    mirror.register_session("lora", tokens, lora_name="adapter-a")
    mirror.register_session("image", tokens, block_extra_keys=[("image", 42)])

    mirror.apply_batch(1, batch(1.0, stored([1], tokens)))
    assert mirror.estimated_cached_tokens("base") == 4
    assert mirror.estimated_cached_tokens("lora") == 0
    assert mirror.estimated_cached_tokens("image") == 0

    mirror.apply_batch(
        2,
        batch(
            2.0,
            stored(
                [2],
                tokens,
                lora_name="adapter-a",
                extra_keys=[("adapter-a",)],
            ),
        ),
    )
    mirror.apply_batch(
        3,
        batch(3.0, stored([3], tokens, extra_keys=[("image", 42)])),
    )
    assert mirror.estimated_cached_tokens("lora") == 4
    assert mirror.estimated_cached_tokens("image") == 4


def test_native_extra_keys_match_lora_and_cache_salt_registration() -> None:
    mirror = DecodePrefixMirror(block_size=4)
    tokens = list(range(8))
    mirror.register_session(
        "s",
        tokens,
        lora_name="adapter-a",
        cache_namespace="tenant-a",
    )

    mirror.apply_batch(
        1,
        batch(
            1.0,
            stored(
                [11, 12],
                tokens,
                lora_name="adapter-a",
                extra_keys=[("adapter-a", "tenant-a"), ("adapter-a",)],
            ),
        ),
    )

    assert mirror.estimated_cached_tokens("s") == 8


def test_group_specific_remove_requires_every_known_group_to_be_present() -> None:
    mirror = DecodePrefixMirror(block_size=4)
    tokens = list(range(8))
    mirror.register_session("s", tokens)

    mirror.apply_batch(1, batch(1.0, stored([11, 12], tokens, group_idx=0)))
    mirror.apply_batch(2, batch(2.0, stored([21, 22], tokens, group_idx=1)))
    mirror.apply_batch(
        3,
        batch(
            3.0,
            {
                "type": "BlockRemoved",
                "block_hashes": [11],
                "medium": "GPU",
                "group_idx": 0,
            },
        ),
    )

    assert mirror.estimated_cached_tokens("s") == 0


def test_duplicate_and_out_of_order_sequences_are_idempotent() -> None:
    mirror = DecodePrefixMirror(block_size=4)
    mirror.register_session("s", [1, 2, 3, 4])
    event_batch = batch(1.0, stored([99], [1, 2, 3, 4]))

    assert mirror.apply_batch(7, event_batch) is BatchApplyResult.APPLIED
    first = mirror.snapshot("s")
    assert mirror.apply_batch(7, event_batch) is BatchApplyResult.STALE
    assert mirror.apply_batch(6, event_batch) is BatchApplyResult.STALE
    assert mirror.snapshot("s") == first
    assert mirror.is_valid


def test_sequence_gap_invalidates_run_and_clears_presence() -> None:
    mirror = DecodePrefixMirror(block_size=4)
    mirror.register_session("s", [1, 2, 3, 4])
    mirror.apply_batch(20, batch(1.0, stored([99], [1, 2, 3, 4])))

    assert mirror.apply_batch(22, batch(2.0)) is BatchApplyResult.GAP
    assert not mirror.is_valid
    assert mirror.gap_count == 1
    assert mirror.estimated_cached_tokens("s") == 0
    assert (
        mirror.apply_batch(23, batch(3.0, stored([100], [1, 2, 3, 4])))
        is BatchApplyResult.INVALID
    )
    assert mirror.estimated_cached_tokens("s") == 0


def test_malformed_input_invalidates_run() -> None:
    mirror = DecodePrefixMirror(block_size=4)

    assert (
        mirror.apply_batch(1, [1.0, [{"type": "BlockStored"}], 0])
        is BatchApplyResult.MALFORMED
    )
    assert not mirror.is_valid
