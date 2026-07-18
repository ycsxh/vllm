# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

SCHEMA_VERSION = "1.0.0"

ReasoningMode = Literal["no_think", "think_high"]

SCALAR_TURN_FIELDS = (
    "trajectory_id",
    "task_id",
    "reasoning_mode",
    "turn_index",
    "prompt_tokens",
    "exact_lcp_tokens",
    "reusable_prefix_tokens",
    "global_prefix_tokens",
    "task_prefix_tokens",
)


@dataclass(frozen=True)
class ReplayTurn:
    trajectory_id: str
    task_id: str
    reasoning_mode: ReasoningMode
    turn_index: int
    prompt_token_ids: tuple[int, ...]
    prompt_tokens: int
    exact_lcp_tokens: int
    reusable_prefix_tokens: int
    global_prefix_tokens: int
    task_prefix_tokens: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: bytes | int) -> str:
    if isinstance(value, bytes):
        return f"sha256:{value.hex()}"
    raise ValueError("Ticket 07 requires byte-valued SHA-256 KV event hashes")


def load_full_turns(config: dict[str, Any]) -> list[ReplayTurn]:
    """Reconstruct full prompt-only turns and validate Ticket 02 scalars."""
    from benchmarks.ds4_profile import workloads

    artifacts = config["artifacts"]
    rendered = workloads.render_turns(
        manifest_path=Path(artifacts["manifest"]),
        normalized_turns_path=Path(artifacts["normalized_turns"]),
        tokenizer_path=Path(config["tokenizer"]["path"]),
        block_size=config["replay"]["block_size"],
        include_token_ids=True,
    )
    ticket_02 = pq.read_table(artifacts["rendered_turns"]).to_pylist()
    expected = {
        (row["trajectory_id"], row["turn_index"]): row for row in ticket_02
    }
    turns: list[ReplayTurn] = []
    for row in rendered:
        key = (row["trajectory_id"], row["turn_index"])
        if key not in expected or any(
            row[field] != expected[key][field] for field in SCALAR_TURN_FIELDS
        ):
            raise ValueError(f"Ticket 02 scalar mismatch for {key}")
        prompt_token_ids = tuple(row["_prompt_token_ids"])
        if len(prompt_token_ids) != row["prompt_tokens"]:
            raise ValueError(f"prompt token mismatch for {key}")
        turns.append(
            ReplayTurn(
                trajectory_id=row["trajectory_id"],
                task_id=row["task_id"],
                reasoning_mode=row["reasoning_mode"],
                turn_index=row["turn_index"],
                prompt_token_ids=prompt_token_ids,
                prompt_tokens=row["prompt_tokens"],
                exact_lcp_tokens=row["exact_lcp_tokens"],
                reusable_prefix_tokens=row["reusable_prefix_tokens"],
                global_prefix_tokens=row["global_prefix_tokens"],
                task_prefix_tokens=row["task_prefix_tokens"],
            )
        )
    return sorted(turns, key=lambda turn: (turn.trajectory_id, turn.turn_index))
