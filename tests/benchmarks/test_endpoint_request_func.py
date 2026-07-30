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
        timestamps=[20.0, 21.0, 30.0, 30.0],
    )

    assert output.success is True
    assert output.ttft == 1.0
    assert output.itl == []
    assert output.latency == 1.0


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


def test_completion_rejects_non_list_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _run_completion(
        monkeypatch,
        payloads=[
            {"choices": [{"text": "a", "token_ids": 101}]},
            {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        ],
        timestamps=[40.0, 41.0],
    )

    assert output.success is False
    assert "token_ids" in output.error
    assert "list" in output.error


@pytest.mark.parametrize(
    "token_id_field",
    [{}, {"token_ids": None}],
    ids=["absent", "null"],
)
def test_completion_preserves_fallback_without_token_ids(
    monkeypatch: pytest.MonkeyPatch,
    token_id_field: dict[str, Any],
) -> None:
    output = _run_completion(
        monkeypatch,
        payloads=[
            {"choices": [{"text": "a", **token_id_field}]},
            {"choices": [{"text": "b", **token_id_field}]},
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
