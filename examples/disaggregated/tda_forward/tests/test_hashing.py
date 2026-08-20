# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tda_forward.hashing import block_hashes, sequence_hashes


def test_public_xxh3_vectors_match_pinned_dynamo_revision() -> None:
    tokens = list(range(12))

    assert block_hashes(tokens, 4) == [
        9_143_094_415_614_847_814,
        12_124_091_508_212_882_195,
        1_088_361_764_653_601_668,
    ]
    assert sequence_hashes(tokens, 4) == [
        9_143_094_415_614_847_814,
        9_510_318_385_840_434_533,
        16_998_148_715_716_822_651,
    ]


def test_lora_and_cache_namespace_match_dynamo_salt_domains() -> None:
    tokens = list(range(12))

    assert sequence_hashes(tokens, 4, lora_name="adapter-a") == [
        6_665_157_333_201_000_639,
        13_351_519_314_849_170_528,
        9_730_552_557_794_562_145,
    ]
    assert sequence_hashes(
        tokens,
        4,
        lora_name="adapter-a",
        cache_namespace="tenant-a",
    ) == [
        8_269_515_957_840_517_240,
        16_913_744_751_721_653_295,
        5_821_023_414_475_969_460,
    ]


def test_hashing_ignores_partial_tail() -> None:
    tokens = list(range(10))

    assert len(block_hashes(tokens, 4)) == 2
