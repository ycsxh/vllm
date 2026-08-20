# TDAforward portable Proxy slice

This directory is the locally executable part of TDAforward issue #16. It runs
on ARM64 macOS without CUDA, a model, NIXL, or a live vLLM engine. A thin
OpenAI-compatible FastAPI adapter delegates all policy and state transitions to
`ReentryCoordinator`; deterministic fake Prefill and Decode adapters exercise
the Engine Reentry Contract.

## Local environment and tests

From this directory:

```bash
uv sync
.venv/bin/python -m pytest
.venv/bin/ty check src tests
.venv/bin/ruff check src tests
```

The committed `uv.lock` makes this a repository-managed environment. Do not use
system `python3` or bare `pip` for this slice.

Start the fake-worker Proxy on the specified MVP port:

```bash
uv run tda-forward-proxy --g 1000 --port 8000
```

Then send a completion turn:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"fake","prompt":[1,2,3,4],"session_id":"demo","t_pred":500}'
```

`session_id` must be a nonempty string. `t_pred` and the configured `g` must be
finite, nonnegative numbers. A request may repeat `g`, but it must exactly match
the configured value. Policy fields are consumed by the Proxy and are not sent
to workers.

## Contract boundaries

- `ReentryCoordinator` owns per-session sequential turns, the exact
  `t_pred <= g` rule, routing intent, Decode-miss fallback, Cache Action ACK
  handling, and structured records.
- `DecodePrefixMirror` consumes vLLM's native `KVEventBatch`, `BlockStored`,
  `BlockRemoved`, and `AllBlocksCleared` field vocabulary. It deliberately does
  not define another production event schema. The transport sequence remains a
  separate envelope value, as it is in vLLM's ZMQ publisher.
- `FakePrefillAdapter` and `FakeDecodeAdapter` model first/P-bound execution,
  full or partial `D_HIT`, `D_MISS`, `CAPACITY_DEFERRED`, streaming, and
  aggregate `EVICT_D` acknowledgements.
- `create_app` starts event consumption as a task independent from request
  forwarding and response streaming. Event delay cannot block client output.

The default portable tokenizer exists only for the fake harness. A native vLLM
adapter supplies the engine tokenizer through the `TokenizerAdapter` seam; the
coordinator's policy and session state do not change.

## Provenance and deliberate scope

The source baseline for vLLM event vocabulary and disaggregated request shape is
vLLM revision `1dab44455972017a97366cd6bd645ae014a9db45`:

- `vllm/distributed/kv_events.py`
- `examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py`

The following small primitives are Python ports from NVIDIA Dynamo revision
`0226d2cf15af8b4a79b098a7ea24af168168c8c2`, licensed under Apache-2.0:

| Local primitive | Pinned Dynamo source | Equivalence coverage |
| --- | --- | --- |
| Public XXH3 block salt and parent chain | `lib/kv-hashing/src/salt.rs`, `lib/tokens/src/lib.rs`, `lib/kv-router/src/protocols.rs` | `tests/test_hashing.py` pins block, LoRA, namespace, and chained numeric vectors. |
| Monotonic cursor observations | `lib/kv-router/src/recovery/cursor.rs` | `tests/test_cursor.py` ports initial, contiguous, gap, duplicate, and stale vectors. |
| Store/remove/clear and longest-prefix invariants | `lib/kv-router/src/indexer/tests.rs` | `tests/test_mirror.py` covers ordered lineage, shared content, removal, clear, duplicate delivery, and gaps. |

There was no dependency-light Dynamo Python primitive for these required
operations at the pinned revision, so the portable slice ports the minimal Rust
behavior instead of taking a runtime dependency on Dynamo.

One TDAforward-specific structural divergence is intentional and tested:

1. A fixed 1P1D deployment uses ordered per-session lineages and reverse shared
   references instead of Dynamo's multi-worker radix index and recovery service.
   Any source gap or malformed batch invalidates the run and clears advisory
   presence; it does not replay or rebuild online.

The mirror compares native vLLM `extra_keys` structurally when correlating
events; it does not add another hash domain. Presence is tracked independently
per native `group_idx`, and a prefix is considered resident only when every
group observed for that session is present.

The slice makes no claim about CUDA behavior, real model output, live vLLM
scheduler admission, NIXL transfer, physical block eviction, or performance on
the target two-GPU server.
