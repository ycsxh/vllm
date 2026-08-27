# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Deterministic concurrent agentic-session workload."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.tda_forward.schema import WorkloadConfig

SCENARIOS = (
    "full_reuse",
    "partial_reuse",
    "zero_reuse",
    "evict_d",
    "capacity_recovery",
    "shared_prefix",
)


@dataclass(frozen=True)
class Turn:
    session_id: str
    turn_index: int
    scenario: str
    prompt: str
    t_pred: float
    max_tokens: int
    seed: int
    warmup: bool = False

    def payload(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "prompt": self.prompt,
            "session_id": self.session_id,
            "t_pred": self.t_pred,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "seed": self.seed,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }


@dataclass(frozen=True)
class Session:
    session_id: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    turn_index: int
    scenario: str
    status: str
    ttft_ms: float | None
    tpot_ms: float | None
    output_tokens: int
    elapsed_ms: float | None
    output_sha256: str | None = None
    error: str | None = None

    @classmethod
    def success(
        cls,
        turn: Turn,
        *,
        ttft_ms: float,
        tpot_ms: float | None,
        output_tokens: int = 1,
        elapsed_ms: float | None = None,
        output_sha256: str | None = None,
    ) -> TurnResult:
        return cls(
            turn.session_id,
            turn.turn_index,
            turn.scenario,
            "completed",
            ttft_ms,
            tpot_ms,
            output_tokens,
            elapsed_ms,
            output_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _words(label: str, count: int) -> str:
    phrase = f"{label} deterministic cache context "
    return (phrase * ((count + 3) // 4)).strip()


def build_workload(config: WorkloadConfig, seed: int) -> tuple[Session, ...]:
    """Build stable prompts whose intent is verified against engine observations."""
    rng = random.Random(seed)
    shared = _words("shared", config.shared_prefix_words)
    sessions = []
    for session_index in range(config.session_count):
        session_id = f"seed-{seed}-session-{session_index:02d}"
        scenario = SCENARIOS[session_index % len(SCENARIOS)]
        unique = _words(f"unique-{seed}-{session_index}", 192)
        base = f"{shared}\n{unique}" if scenario == "shared_prefix" else unique
        turns = []
        for turn_index in range(config.turns_per_session):
            current = scenario if turn_index else "warmup"
            prompt = base
            if scenario == "partial_reuse":
                prompt += _words(f" append-{turn_index}", turn_index * 64)
            elif scenario == "capacity_recovery":
                if turn_index in (1, 2):
                    prompt += _words(
                        f" capacity-{seed}-{session_index}-{turn_index}",
                        config.pressure_words,
                    )
                else:
                    prompt += _words(f" recovery-{turn_index}", turn_index * 32)
            elif scenario == "zero_reuse" and turn_index >= 2:
                prompt = _words(
                    f"pressure-{seed}-{session_index}-{turn_index}",
                    config.pressure_words,
                )
            elif scenario == "full_reuse":
                prompt = base
            elif scenario == "shared_prefix":
                prompt = f"{shared}\n{unique}\nturn {turn_index}"
            t_pred = (
                config.g_seconds + 1
                if scenario == "evict_d" and turn_index % 2 == 1
                else config.g_seconds
            )
            turns.append(
                Turn(
                    session_id=session_id,
                    turn_index=turn_index,
                    scenario=current,
                    prompt=prompt,
                    t_pred=t_pred,
                    max_tokens=config.max_tokens,
                    seed=rng.randrange(0, 2**31),
                    warmup=turn_index < config.warmup_turns,
                )
            )
        sessions.append(Session(session_id, tuple(turns)))
    return tuple(sessions)


async def run_sessions(
    sessions: tuple[Session, ...] | list[Session],
    send: Callable[[Turn], Awaitable[TurnResult]],
) -> list[TurnResult]:
    """Run one coroutine per session, serial inside and concurrent across sessions."""

    async def run_one(session: Session) -> list[TurnResult]:
        results = []
        for turn in session.turns:
            results.append(await send(turn))
        return results

    grouped = await asyncio.gather(*(run_one(session) for session in sessions))
    return [result for results in grouped for result in results]
