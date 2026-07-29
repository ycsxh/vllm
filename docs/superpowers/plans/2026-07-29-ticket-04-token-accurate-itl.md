# Ticket 4 Token-Accurate ITL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the controlled DS4 completion benchmark record exactly one ITL
sample per output-token transition when an SSE delta contains multiple token
IDs.

**Architecture:** The DS4 runner will request the existing vLLM
`return_token_ids` extension through `--extra-body`. The generic OpenAI
completion client will use explicit per-delta token cardinality when present,
preserve its one-token-per-choice fallback when absent or null, and reject
malformed non-null cardinality instead of weakening downstream validation.

**Tech Stack:** Python 3.12, asyncio, aiohttp-compatible test doubles, pytest,
Ruff, vLLM benchmark CLI.

## Global Constraints

- Operate only on the personal fork `ycsxh/vllm`; treat
  `vllm-project/vllm` as read-only.
- Keep commit `24e2ef4fc1b886a88e792b0d5a833b7e1638dae5` and every
  `attempt-02` artifact immutable.
- Never reuse a diagnostic or validation result directory.
- Do not replace request-level evidence with aggregate console metrics.
- Do not change `derive_run_result`, scheduling, batching, chunked prefill,
  CUDA graph behavior, or the P/D proxy protocol.
- Preserve compatibility with OpenAI-compatible endpoints that do not return
  token IDs.
- Request token IDs only for the controlled DS4 `/v1/completions` benchmark.
- Preserve TTFT as arrival of the first output token.
- Record `0.0` between tokens observed in the same SSE delta; do not invent
  server-side generation timestamps.
- Treat a non-null, non-list `token_ids` value as malformed response data.
- Use `.venv/bin/python` for Python commands; never use system `python3` or
  bare `pip`.
- Observe all regression tests failing for the intended reasons before
  changing implementation code.

---

### Task 1: Add the Complete Red Regression Surface

**Files:**

- Create: `tests/benchmarks/test_endpoint_request_func.py`
- Modify:
  `tests/benchmarks/ds4_profile/test_run_points.py:810`

**Interfaces:**

- Consumes:
  `vllm.benchmarks.lib.endpoint_request_func.RequestFuncInput` and
  `async_request_openai_completions(request_func_input, session, pbar=None)`.
- Produces: deterministic behavioral tests for token-aware timing and an exact
  DS4 command assertion for
  `--extra-body '{"return_token_ids":true}'`.

- [ ] **Step 1: Create deterministic completion-stream test doubles**

Create `tests/benchmarks/test_endpoint_request_func.py` with these imports and
helpers:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest

import vllm.benchmarks.lib.endpoint_request_func as request_func_module
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

pytestmark = pytest.mark.skip_global_cleanup


class _FakeContent:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads

    async def iter_any(self):
        for payload in self.payloads:
            yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"


class _FakeResponse:
    status = 200
    reason = "OK"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.content = _FakeContent(payloads)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeSession:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.response = _FakeResponse(payloads)
        self.request_json: dict[str, Any] | None = None

    def post(
        self,
        *,
        url: str,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> _FakeResponse:
        del url, headers
        self.request_json = json
        return self.response


def _run_completion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payloads: list[dict[str, Any]],
    timestamps: list[float],
):
    clock: Iterator[float] = iter(timestamps)
    monkeypatch.setattr(
        request_func_module.time,
        "perf_counter",
        lambda: next(clock),
    )
    request = RequestFuncInput(
        prompt="prompt",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=3,
        model="test-model",
    )
    return asyncio.run(
        request_func_module.async_request_openai_completions(
            request,
            _FakeSession(payloads),
        )
    )
```

- [ ] **Step 2: Test expansion of a multi-token delta**

Append:

```python
def test_completion_uses_delta_token_ids_for_itl_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _run_completion(
        monkeypatch,
        payloads=[
            {"choices": [{"text": "ab", "token_ids": [101, 102]}]},
            {"choices": [{"text": "c", "token_ids": [103]}]},
            {"usage": {"prompt_tokens": 1, "completion_tokens": 3}},
        ],
        timestamps=[10.0, 11.0, 14.0],
    )

    assert output.success is True
    assert output.generated_text == "abc"
    assert output.output_tokens == 3
    assert output.ttft == 1.0
    assert output.itl == [0.0, 3.0]
    assert output.latency == 4.0
```

This proves that the client creates one transition for the second token in the
first delta and one transition for the later delta.

- [ ] **Step 3: Test that a zero-token terminal choice does not affect timing**

Append:

```python
def test_completion_ignores_zero_token_terminal_choice_for_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _run_completion(
        monkeypatch,
        payloads=[
            {"choices": [{"text": "a", "token_ids": [101]}]},
            {
                "choices": [
                    {
                        "text": "",
                        "token_ids": [],
                        "finish_reason": "length",
                    }
                ]
            },
            {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ],
        timestamps=[20.0, 21.0, 30.0],
    )

    assert output.success is True
    assert output.ttft == 1.0
    assert output.itl == []
    assert output.latency == 1.0
```

- [ ] **Step 4: Test generic OpenAI fallback without token IDs**

Append:

```python
def test_completion_preserves_one_token_per_choice_without_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _run_completion(
        monkeypatch,
        payloads=[
            {"choices": [{"text": "a"}]},
            {"choices": [{"text": "b"}]},
            {"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
        ],
        timestamps=[30.0, 31.0, 34.0],
    )

    assert output.success is True
    assert output.generated_text == "ab"
    assert output.output_tokens == 2
    assert output.ttft == 1.0
    assert output.itl == [3.0]
    assert output.latency == 4.0
```

- [ ] **Step 5: Test fail-closed handling of malformed token IDs**

Append:

```python
def test_completion_rejects_non_list_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _run_completion(
        monkeypatch,
        payloads=[
            {"choices": [{"text": "a", "token_ids": 101}]},
            {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ],
        timestamps=[40.0],
    )

    assert output.success is False
    assert "token_ids" in output.error
    assert "list" in output.error
```

- [ ] **Step 6: Characterize vLLM's null-token-ID fallback**

Append:

```python
def test_completion_treats_null_token_ids_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _run_completion(
        monkeypatch,
        payloads=[
            {"choices": [{"text": "a", "token_ids": None}]},
            {"choices": [{"text": "b", "token_ids": None}]},
            {"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
        ],
        timestamps=[50.0, 51.0, 54.0],
    )

    assert output.success is True
    assert output.generated_text == "ab"
    assert output.output_tokens == 2
    assert output.ttft == 1.0
    assert output.itl == [3.0]
    assert output.latency == 4.0
```

- [ ] **Step 7: Add the exact DS4 request-contract assertion**

In
`test_benchmark_command_uses_only_the_official_controlled_options`, add after
the valued-option loop:

```python
    assert command[command.index("--extra-body") + 1] == (
        '{"return_token_ids":true}'
    )
```

- [ ] **Step 8: Run all regression checks and capture the intended red**

Run:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_endpoint_request_func.py \
  tests/benchmarks/ds4_profile/test_run_points.py::test_benchmark_command_uses_only_the_official_controlled_options \
  -q
```

Expected: the multi-token, zero-token, malformed-token-ID, and DS4 command
tests fail for their intended behavioral reasons. The absent and null
token-ID compatibility characterizations remain green. Confirm no failure is
an import, fixture, collection, or socket-permission error.

---

### Task 2: Implement Token-Aware Completion Timing

**Files:**

- Modify:
  `vllm/benchmarks/lib/endpoint_request_func.py:198-246`
- Test: `tests/benchmarks/test_endpoint_request_func.py`

**Interfaces:**

- Consumes: `choices[0]`, where optional `token_ids` is absent, null, or a
  `list[int]`.
- Produces: `RequestFuncOutput.ttft`, `itl`, and `latency` based on
  token-bearing delta arrivals, while preserving `generated_text` and
  usage-provided `output_tokens`.

- [ ] **Step 1: Derive exact token cardinality for every choice**

Replace the current unconditional one-choice/one-token timing block inside
`if choices := data.get("choices"):` with:

```python
                                choice = choices[0]
                                text = choice.get("text")
                                token_ids = choice.get("token_ids")
                                if token_ids is None:
                                    num_delta_tokens = 1
                                elif isinstance(token_ids, list):
                                    num_delta_tokens = len(token_ids)
                                else:
                                    raise TypeError(
                                        "choices[0].token_ids must be a list or null"
                                    )
```

- [ ] **Step 2: Record one client-observed timing per represented token**

Immediately after the cardinality block, add:

```python
                                if num_delta_tokens:
                                    timestamp = time.perf_counter()
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        output.ttft = timestamp - st
                                        output.itl.extend(
                                            [0.0] * (num_delta_tokens - 1)
                                        )
                                    else:
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )
                                        output.itl.extend(
                                            [0.0] * (num_delta_tokens - 1)
                                        )
                                    most_recent_timestamp = timestamp
                                generated_text += text or ""
```

Delete the superseded unconditional timestamp, TTFT, ITL, and
`most_recent_timestamp` updates. A zero-token choice still contributes text
but cannot establish TTFT, append ITL, or change the final token-arrival time.

- [ ] **Step 3: Run the endpoint regression tests green**

Run:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_endpoint_request_func.py \
  -q
```

Expected: `5 passed`.

---

### Task 3: Request Token IDs from the Controlled DS4 Command

**Files:**

- Modify: `benchmarks/ds4_profile/run_points.py:806-851`
- Test:
  `tests/benchmarks/ds4_profile/test_run_points.py::test_benchmark_command_uses_only_the_official_controlled_options`

**Interfaces:**

- Consumes: the controlled DS4 `/v1/completions` benchmark command tuple.
- Produces: the exact adjacent argument pair
  `("--extra-body", '{"return_token_ids":true}')`.

- [ ] **Step 1: Add the scoped extra request body**

In `build_benchmark_command`, add these tuple entries immediately after the
completion endpoint:

```python
        "--extra-body",
        '{"return_token_ids":true}',
```

Do not add the option to generic benchmark commands or modify proxy request
forwarding.

- [ ] **Step 2: Run the DS4 command regression green**

Run:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_run_points.py::test_benchmark_command_uses_only_the_official_controlled_options \
  -q
```

Expected: `1 passed`.

- [ ] **Step 3: Run the complete combined focused regression surface**

Run:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/test_endpoint_request_func.py \
  tests/benchmarks/ds4_profile/test_run_points.py \
  -q
```

Expected: both files pass with no collection, request, or socket-permission
failures.

- [ ] **Step 4: Commit the token-accurate implementation**

Run:

```bash
git add \
  vllm/benchmarks/lib/endpoint_request_func.py \
  benchmarks/ds4_profile/run_points.py \
  tests/benchmarks/test_endpoint_request_func.py \
  tests/benchmarks/ds4_profile/test_run_points.py
git commit -m "fix: account for multi-token completion deltas"
```

Before committing, confirm that the diff contains no changes to the validator,
proxy, server, scheduler, or any result artifact.

---

### Task 4: Verify the B-Class Fix Before A-Class Diagnosis

**Files:**

- Verify:
  `vllm/benchmarks/lib/endpoint_request_func.py`
- Verify: `benchmarks/ds4_profile/run_points.py`
- Verify: `tests/benchmarks/test_endpoint_request_func.py`
- Verify: `tests/benchmarks/ds4_profile/test_run_points.py`

**Interfaces:**

- Consumes: the committed B-class diff.
- Produces: CPU and static-analysis evidence that the approved design is
  implemented without weakening DS4 validation.

- [ ] **Step 1: Run the focused 88-test DS4 replacement suite**

Run with target socket permissions:

```bash
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py \
  tests/benchmarks/ds4_profile/test_report_results.py \
  -q
```

Expected: the prior 88 tests plus the new DS4 command assertion pass. Record
the exact updated count rather than assuming it remains 88.

- [ ] **Step 2: Run Ruff on every changed Python file**

Run:

```bash
.venv/bin/ruff check \
  vllm/benchmarks/lib/endpoint_request_func.py \
  benchmarks/ds4_profile/run_points.py \
  tests/benchmarks/test_endpoint_request_func.py \
  tests/benchmarks/ds4_profile/test_run_points.py
.venv/bin/ruff format --check \
  vllm/benchmarks/lib/endpoint_request_func.py \
  benchmarks/ds4_profile/run_points.py \
  tests/benchmarks/test_endpoint_request_func.py \
  tests/benchmarks/ds4_profile/test_run_points.py
```

Expected: both commands exit zero without modifying files.

- [ ] **Step 3: Audit the diff against the approved design**

Run:

```bash
git diff 24e2ef4fc1b886a88e792b0d5a833b7e1638dae5 -- \
  vllm/benchmarks/lib/endpoint_request_func.py \
  benchmarks/ds4_profile/run_points.py \
  tests/benchmarks/test_endpoint_request_func.py \
  tests/benchmarks/ds4_profile/test_run_points.py
rg -n "derive_run_result|aggregate|console" \
  vllm/benchmarks/lib/endpoint_request_func.py \
  benchmarks/ds4_profile/run_points.py
```

Confirm the implementation exactly covers token cardinality, zero-token
choices, fallback compatibility, malformed data, and the DS4 request flag.
Confirm it does not touch or bypass `derive_run_result`.

- [ ] **Step 4: Run independent Standards + Spec review**

Invoke `/code-review` against frozen base
`24e2ef4fc1b886a88e792b0d5a833b7e1638dae5`. The Standards review must check
the repository `AGENTS.md` and local test conventions. The Spec review must
use
`docs/superpowers/specs/2026-07-29-ticket-04-token-accurate-itl-design.md`.
Resolve every blocking finding and rerun the affected checks before moving to
the A-class GPU probe.

- [ ] **Step 5: Record the B-class handoff state**

Record:

- exact commit;
- focused test commands and counts;
- Ruff results;
- Standards + Spec verdicts;
- confirmation that no live result directory was created;
- confirmation that `attempt-02` remains untouched.

Do not run the B-class GPU target validation yet. The Gate D sequence requires
the class A root cause and fix first, followed by fresh one-point validations
for both fixes.
