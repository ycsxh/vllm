# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dependency-free metric parsing and aggregation helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from typing import Any

import regex as re

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})? "
    r"(?P<value>[-+0-9.eE]+)$"
)


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a deterministic nearest-rank percentile."""
    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    index = max(0, math.ceil(quantile * len(samples)) - 1)
    return samples[index]


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize preserved raw samples without replacing them."""
    samples = [float(value) for value in values]
    return {
        "count": len(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "mean": statistics.fmean(samples) if samples else None,
    }


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    """Parse scalar Prometheus exposition samples used by the evidence runner."""
    samples = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed Prometheus sample on line {line_number}")
        labels = {}
        raw_labels = match.group("labels")
        if raw_labels:
            for part in raw_labels.split(","):
                key, separator, value = part.partition("=")
                if (
                    not separator
                    or not value.startswith('"')
                    or not value.endswith('"')
                ):
                    raise ValueError(
                        f"malformed Prometheus labels on line {line_number}"
                    )
                labels[key] = value[1:-1]
        samples.append(
            {
                "name": match.group("name"),
                "labels": labels,
                "value": float(match.group("value")),
            }
        )
    return samples


def metric_value(
    samples: list[dict[str, Any]],
    name: str,
    *,
    labels: dict[str, str] | None = None,
) -> float:
    """Require exactly one matching scalar."""
    labels = labels or {}
    matches = [
        sample["value"]
        for sample in samples
        if sample["name"] == name
        and all(sample["labels"].get(key) == value for key, value in labels.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {name} sample, found {len(matches)}")
    return matches[0]
