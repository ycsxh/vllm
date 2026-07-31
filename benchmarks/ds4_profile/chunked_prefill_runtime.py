# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent public-offline runtime for chunked-prefill engine groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.ds4_profile.fixed_batch_runtime import SubprocessOfflineEngine


class GroupedSubprocessRuntimeFactory:
    """Create one persistent offline engine for a batch-wide token budget."""

    def __call__(
        self,
        token_budget: int,
        engine_config: dict[str, Any],
        engine_dir: Path,
    ) -> SubprocessOfflineEngine:
        if engine_config.get("max_num_batched_tokens") != token_budget:
            raise ValueError("engine token budget differs from its group")
        return SubprocessOfflineEngine(engine_config, engine_dir)
