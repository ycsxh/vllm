# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Acceptance-only vLLM CLI wrapper for scheduler and publisher timing."""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.tda_forward.metrics import distribution

_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_SCHEDULER_STEP_MS: list[float] = []
_EVENT_CONSTRUCTION_MS: list[float] = []
_EVENT_ENQUEUE_MS: list[float] = []
_PUBLISHERS: list[Any] = []
_PUBLISHER_SNAPSHOTS: list[dict[str, Any]] = []
_FLUSH_STOP = threading.Event()
_INSTALLED = False
_WRITER_STARTED = False
_WINDOW_STARTED_AT: float | None = None
_PUBLISHER_BASELINES: list[dict[str, Any]] = []
_WINDOW_QUEUE_HIGH_WATERMARK = 0
_FIRST_PUBLISHER_OVERFLOW_AT: float | None = None


class _DelayedGetQueue:
    def __init__(self, queue: Any, delay_seconds: float) -> None:
        self._queue = queue
        self._delay_seconds = delay_seconds

    def get(self, *args, **kwargs):
        item = self._queue.get(*args, **kwargs)
        if item is not None and _measurement_window_open():
            time.sleep(self._delay_seconds)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self._queue, name)


def _record(values: list[float], value: float) -> None:
    with _LOCK:
        values.append(value)


def _publisher_snapshots() -> list[dict[str, Any]]:
    snapshots = []
    for publisher in _PUBLISHERS:
        try:
            snapshots.append(asdict(publisher.stats))
        except Exception as error:
            snapshots.append({"stats_error": repr(error)})
    return snapshots


def _measurement_window_open() -> bool:
    marker = os.environ.get("TDA_ACCEPTANCE_WINDOW_PATH")
    return bool(marker and Path(marker).is_file())


def _start_writer() -> None:
    global _WRITER_STARTED
    with _LOCK:
        if _WRITER_STARTED:
            return
        _WRITER_STARTED = True
    atexit.register(_finish)
    threading.Thread(
        target=_periodic_flush,
        name="tda-acceptance-stats",
        daemon=True,
    ).start()


def _start_measurement_window() -> bool:
    global _PUBLISHER_BASELINES, _WINDOW_STARTED_AT
    if not _measurement_window_open():
        return False
    with _LOCK:
        if _WINDOW_STARTED_AT is None:
            _WINDOW_STARTED_AT = time.time()
            _PUBLISHER_BASELINES = _publisher_snapshots()
            _PUBLISHER_SNAPSHOTS.clear()
    return True


def _install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.distributed.kv_events import ZmqEventPublisher
    from vllm.v1.core.sched.scheduler import Scheduler

    original_schedule = Scheduler.schedule
    original_scheduler_shutdown = Scheduler.shutdown
    original_init = ZmqEventPublisher.__init__
    original_publish = ZmqEventPublisher.publish
    original_observe = ZmqEventPublisher.observe_event_construction_time

    def schedule(self, *args, **kwargs):
        _start_writer()
        measured = _start_measurement_window()
        started = time.perf_counter()
        try:
            return original_schedule(self, *args, **kwargs)
        finally:
            if measured:
                _record(_SCHEDULER_STEP_MS, (time.perf_counter() - started) * 1000)

    def publisher_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        drain_delay = float(
            os.environ.get("TDA_ACCEPTANCE_PUBLISHER_DRAIN_DELAY_SECONDS", "0")
        )
        if drain_delay > 0:
            self._event_queue = _DelayedGetQueue(self._event_queue, drain_delay)
        _PUBLISHERS.append(self)
        _start_writer()

    def scheduler_shutdown(self, *args, **kwargs):
        try:
            return original_scheduler_shutdown(self, *args, **kwargs)
        finally:
            _finish()

    def publish(self, events):
        global _FIRST_PUBLISHER_OVERFLOW_AT, _WINDOW_QUEUE_HIGH_WATERMARK
        _start_writer()
        measured = _start_measurement_window()
        dropped_before = self.stats.dropped_batches
        started = time.perf_counter()
        try:
            return original_publish(self, events)
        finally:
            if measured:
                _record(_EVENT_ENQUEUE_MS, (time.perf_counter() - started) * 1000)
                try:
                    depth = int(self.stats.queue_depth)
                except Exception:
                    depth = 0
                with _LOCK:
                    _WINDOW_QUEUE_HIGH_WATERMARK = max(
                        _WINDOW_QUEUE_HIGH_WATERMARK, depth
                    )
                    if (
                        self.stats.dropped_batches > dropped_before
                        and _FIRST_PUBLISHER_OVERFLOW_AT is None
                    ):
                        _FIRST_PUBLISHER_OVERFLOW_AT = time.time()

    def observe(self, duration_seconds):
        if _start_measurement_window():
            _record(_EVENT_CONSTRUCTION_MS, duration_seconds * 1000)
        return original_observe(self, duration_seconds)

    Scheduler.schedule = schedule
    Scheduler.shutdown = scheduler_shutdown
    ZmqEventPublisher.__init__ = publisher_init
    ZmqEventPublisher.publish = publish
    ZmqEventPublisher.observe_event_construction_time = observe
    _INSTALLED = True


def _window_publisher_stats(
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    additive = {
        "enqueued_batches",
        "enqueued_events",
        "enqueued_blocks",
        "published_batches",
        "published_events",
        "published_blocks",
        "published_bytes",
        "dropped_batches",
        "dropped_events",
        "dropped_blocks",
        "total_event_construction_time_seconds",
        "total_enqueue_time_seconds",
        "total_publish_lag_seconds",
    }
    adjusted = []
    for index, stats in enumerate(current):
        if "stats_error" in stats or index >= len(_PUBLISHER_BASELINES):
            adjusted.append(stats)
            continue
        baseline = _PUBLISHER_BASELINES[index]
        value = dict(stats)
        for name in additive:
            if name in value and name in baseline:
                value[name] -= baseline[name]
        value["queue_high_watermark"] = _WINDOW_QUEUE_HIGH_WATERMARK
        adjusted.append(value)
    return adjusted


def _write() -> None:
    with _WRITE_LOCK:
        destination = os.environ.get("TDA_ACCEPTANCE_STATS_PATH")
        if not destination:
            return
        with _LOCK:
            window_started_at = _WINDOW_STARTED_AT
            publishers = (
                _window_publisher_stats(_publisher_snapshots())
                if window_started_at is not None
                else []
            )
            if publishers:
                _PUBLISHER_SNAPSHOTS.append(
                    {"timestamp": time.time(), "publishers": publishers}
                )
            value = {
                "schema_version": 1,
                "measurement_window_started_at": window_started_at,
                "first_publisher_overflow_at": _FIRST_PUBLISHER_OVERFLOW_AT,
                "scheduler_step_ms": {
                    "raw": list(_SCHEDULER_STEP_MS),
                    "summary": distribution(_SCHEDULER_STEP_MS),
                },
                "event_construction_ms": {
                    "raw": list(_EVENT_CONSTRUCTION_MS),
                    "summary": distribution(_EVENT_CONSTRUCTION_MS),
                },
                "event_enqueue_ms": {
                    "raw": list(_EVENT_ENQUEUE_MS),
                    "summary": distribution(_EVENT_ENQUEUE_MS),
                },
                "publishers": publishers,
                "publisher_samples": list(_PUBLISHER_SNAPSHOTS),
            }
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)


def _finish() -> None:
    _FLUSH_STOP.set()
    _write()


def _periodic_flush() -> None:
    while not _FLUSH_STOP.wait(1.0):
        _write()


def main() -> None:
    """Install observers before invoking the regular vLLM CLI."""
    from vllm.entrypoints.cli.main import main as cli_main

    cli_main()


if os.environ.get("TDA_ACCEPTANCE_STATS_PATH"):
    # multiprocessing spawn re-executes this module as __mp_main__. Installing
    # at import time is what places the observers inside EngineCore.
    _install()


if __name__ == "__main__":
    main()
