# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@contextlib.contextmanager
def _fake_completion_server(response_payload):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(content_length)))
            body = json.dumps(response_payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@contextlib.contextmanager
def _proxy_process(prefill_port, decode_port, proxy_port):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmarks.ds4_profile.pd_proxy",
            "--prefill-url",
            f"http://127.0.0.1:{prefill_port}",
            "--decode-url",
            f"http://127.0.0.1:{decode_port}",
            "--port",
            str(proxy_port),
            "--request-timeout",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    status_url = f"http://127.0.0.1:{proxy_port}/status"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"proxy exited before readiness:\n{output}")
            try:
                with opener.open(status_url, timeout=0.2):
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.02)
        else:
            raise AssertionError("proxy did not become ready")
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_proxy_forwards_validated_prefill_metadata_to_decode(free_tcp_port):
    kv_transfer_params = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": [[11, 12], [21]],
        "remote_engine_id": "prefill-engine",
        "remote_request_id": "prefill-request",
        "remote_host": "127.0.0.1",
        "remote_port": 5600,
        "tp_size": 1,
        "remote_num_tokens": 641,
    }
    prefill_response = {
        "choices": [{"text": ""}],
        "kv_transfer_params": kv_transfer_params,
    }
    decode_response = {"choices": [{"text": "deterministic output"}]}
    request_payload = {
        "model": "Qwen/Qwen3.5-4B",
        "prompt": "fixed smoke prompt",
        "max_tokens": 16,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": False,
    }

    with (
        _fake_completion_server(prefill_response) as (prefill_port, prefill_requests),
        _fake_completion_server(decode_response) as (decode_port, decode_requests),
        _proxy_process(prefill_port, decode_port, free_tcp_port),
    ):
        status, response = _post_json(
            f"http://127.0.0.1:{free_tcp_port}/v1/completions",
            request_payload,
        )

    assert status == 200
    assert response == decode_response
    assert prefill_requests == [
        {
            **request_payload,
            "max_tokens": 1,
            "stream": False,
            "kv_transfer_params": {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            },
        }
    ]
    assert decode_requests == [
        {**request_payload, "kv_transfer_params": kv_transfer_params}
    ]


def test_proxy_rejects_missing_prefill_metadata_without_calling_decode(
    free_tcp_port,
):
    request_payload = {
        "model": "Qwen/Qwen3.5-4B",
        "prompt": "fixed smoke prompt",
        "max_tokens": 16,
        "stream": False,
    }

    with (
        _fake_completion_server({"choices": [{"text": ""}]}) as (
            prefill_port,
            prefill_requests,
        ),
        _fake_completion_server({"choices": [{"text": "must not run"}]}) as (
            decode_port,
            decode_requests,
        ),
        _proxy_process(prefill_port, decode_port, free_tcp_port),
    ):
        status, response = _post_json(
            f"http://127.0.0.1:{free_tcp_port}/v1/completions",
            request_payload,
        )

    assert status == 502
    assert response["detail"] == "prefill response lacks kv_transfer_params"
    assert len(prefill_requests) == 1
    assert decode_requests == []


def test_proxy_rejects_non_object_prefill_metadata_without_calling_decode(
    free_tcp_port,
):
    request_payload = {
        "model": "Qwen/Qwen3.5-4B",
        "prompt": "fixed smoke prompt",
        "max_tokens": 16,
        "stream": False,
    }
    prefill_response = {
        "choices": [{"text": ""}],
        "kv_transfer_params": [],
    }

    with (
        _fake_completion_server(prefill_response) as (
            prefill_port,
            prefill_requests,
        ),
        _fake_completion_server({"choices": [{"text": "must not run"}]}) as (
            decode_port,
            decode_requests,
        ),
        _proxy_process(prefill_port, decode_port, free_tcp_port),
    ):
        status, response = _post_json(
            f"http://127.0.0.1:{free_tcp_port}/v1/completions",
            request_payload,
        )

    assert status == 502
    assert response["detail"] == "prefill kv_transfer_params must be an object"
    assert len(prefill_requests) == 1
    assert decode_requests == []


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("do_remote_prefill", False, "do_remote_prefill must be true"),
        ("do_remote_decode", True, "do_remote_decode must be false"),
        ("remote_block_ids", [], "remote_block_ids must contain block IDs"),
        (
            "remote_block_ids",
            [[11, "12"]],
            "remote_block_ids must contain block IDs",
        ),
        ("remote_engine_id", "", "remote_engine_id must be a nonempty string"),
        ("remote_request_id", None, "remote_request_id must be a nonempty string"),
        ("remote_host", "prefill", "remote_host must be 127.0.0.1"),
        ("remote_port", 5601, "remote_port must be 5600"),
        ("tp_size", 2, "tp_size must be 1"),
        ("remote_num_tokens", 640, "remote_num_tokens must be 641"),
    ],
)
def test_proxy_rejects_invalid_pull_metadata_without_calling_decode(
    free_tcp_port,
    field,
    value,
    detail,
):
    kv_transfer_params = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_block_ids": [[11, 12], [21]],
        "remote_engine_id": "prefill-engine",
        "remote_request_id": "prefill-request",
        "remote_host": "127.0.0.1",
        "remote_port": 5600,
        "tp_size": 1,
        "remote_num_tokens": 641,
    }
    kv_transfer_params[field] = value
    prefill_response = {
        "choices": [{"text": ""}],
        "kv_transfer_params": kv_transfer_params,
    }
    request_payload = {
        "model": "Qwen/Qwen3.5-4B",
        "prompt": "fixed smoke prompt",
        "max_tokens": 16,
        "stream": False,
    }

    with (
        _fake_completion_server(prefill_response) as (
            prefill_port,
            prefill_requests,
        ),
        _fake_completion_server({"choices": [{"text": "must not run"}]}) as (
            decode_port,
            decode_requests,
        ),
        _proxy_process(prefill_port, decode_port, free_tcp_port),
    ):
        status, response = _post_json(
            f"http://127.0.0.1:{free_tcp_port}/v1/completions",
            request_payload,
        )

    assert status == 502
    assert response["detail"] == f"invalid prefill kv_transfer_params: {detail}"
    assert len(prefill_requests) == 1
    assert decode_requests == []
