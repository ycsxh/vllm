# TDAforward Proxy

This directory contains the portable contract harness and native 1P1D process
adapters. The ARM64 mode needs no CUDA, model, NIXL, or live vLLM engine.
Native mode uses the same coordinator while binding its tokenizer, Prefill and
Decode requests, and cache-event stream to vLLM's production contracts.

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

On the target server, start native mode after the P and D vLLM instances are
healthy and the D event publisher is reachable:

```bash
uv run tda-forward-proxy \
  --native \
  --model <Qwen3.5-4B-model-path> \
  --g 1000 \
  --block-size 16 \
  --prefill-url http://127.0.0.1:8100 \
  --decode-url http://127.0.0.1:8200 \
  --event-endpoint tcp://127.0.0.1:5557 \
  --port 8000
```

The URL arguments are service roots; the Proxy preserves the incoming
`/v1/completions` or `/v1/chat/completions` path. Fixed 1P1D native mode accepts
exactly one Decode `--event-endpoint`. Native mode loads the tokenizer through vLLM,
decodes `KVEventBatch` directly, and keeps event consumption in an independent
task so publisher lag cannot block response streaming.

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
  `t_pred <= g` rule, routing intent, Cache Action ACK handling, and structured
  records. With no D binding, it sends a Prefill-side AP request and uses NIXL
  transfer. With a D binding, it sends Decode-local append prefill, including
  when Decode has zero local reuse.
- `DecodePrefixMirror` consumes vLLM's native `KVEventBatch`, `BlockStored`,
  `BlockRemoved`, and `AllBlocksCleared` field vocabulary. It deliberately does
  not define another production event schema. The transport sequence remains a
  separate envelope value, as it is in vLLM's ZMQ publisher.
- `FakePrefillAdapter` and `FakeDecodeAdapter` model initial unbound execution,
  full or partial `D_HIT`, `D_MISS`, capacity delay, streaming, and aggregate
  `EVICT_D` acknowledgements. `D_HIT` and `D_MISS` report final Decode-local
  reuse; they never select the worker. `capacity_delay_ms` reports Decode
  waiting and never causes rerouting.
- `create_app` starts event consumption as a task independent from request
  forwarding and response streaming. Event delay cannot block client output.
- `VllmPrefillAdapter` and `VllmDecodeAdapter` use the existing OpenAI
  `kv_transfer_params` extension. A D-local request carries its Scheduler
  admission attempt and final Cache Action in one live request. The zero-token
  `D_HIT` control output protects admitted blocks before model work. The
  next-turn D binding changes only after the stream completes and a valid Cache
  Action ACK arrives. `RETAIN_D` is soft retention: ordinary LRU-eligible
  retention with neither a pin nor a TTL.
- `ZmqEventSubscriber` decodes the publisher's native sequence envelope and
  `KVEventBatch`; the Proxy defines no parallel wire schema.

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

The ARM64 tests establish contract and Scheduler behavior but do not establish
CUDA, NIXL, model-output, or performance results. Those claims require the
Qwen3.5-4B BF16 1P1D run on the target dual-RTX-3090 server.
