# TDAforward D-binding execution

Status: approved by the repository operator on 2026-08-25.

## Purpose

Make the previously committed D-binding signal the sole authority for choosing
where a turn performs append prefill. Decode-local cache reuse and capacity
waiting remain observable execution facts, but neither may create, clear, or
override the binding.

This design supersedes the current zero-hit Prefill fallback. A D-bound turn
with no reusable Decode-local prefix performs the full append prefill on Decode.
An unbound turn uses the existing Prefill-side path and NIXL transfer.

## Invariants

- The first turn of a session is unbound and therefore uses Prefill-side AP.
- A session has either one committed next-turn D binding or no binding. There
  is no P-binding state; Prefill is the default when the D binding is absent.
- The current turn's `t_pred` and configured `g` determine only whether a D
  binding will be committed for the next turn.
- `t_pred <= g` selects `RETAIN_D`; after a successful terminal ACK it commits
  the next-turn D binding.
- `t_pred > g` selects `EVICT_D`; after a successful terminal ACK it clears the
  next-turn D binding.
- A D-bound turn always performs append prefill on Decode. Full, partial, and
  zero local cache hits do not change the execution path.
- Decode capacity pressure keeps a D-bound request in Decode's normal waiting
  and LRU scheduling. It never reroutes the request to Prefill.
- `RETAIN_D` does not pin blocks and creates no lease, TTL, ownership record, or
  nonstandard eviction priority. Retained cache remains ordinary LRU-eligible
  state.
- Proxy cache estimates are advisory observations. They do not enter the
  engine's routing or scheduling decision.
- A turn commits its next-turn binding only after the complete output stream
  and a valid Cache Action ACK. Failure or cancellation preserves the last
  committed binding.

## Domain model

The coordinator's per-session state is:

```text
SessionState
├── turn_sequence
└── next_turn_d_bound: bool
```

The state intentionally does not encode `P_BOUND`. An absent D binding is
sufficient to select the Prefill-side path.

Cache status and capacity history are orthogonal observations:

```text
DecodeExecutionObservation
├── cache_status: D_HIT | D_MISS
├── actual_local_cached_tokens
├── prompt_tokens
├── locally_computed_tokens
└── capacity_delay_ms
```

`D_MISS` means that the request reused zero Decode-local prompt tokens at final
admission. It is nonterminal and does not authorize fallback. A positive
`capacity_delay_ms` means at least one allocation attempt was deferred. These
facts may coexist in any valid combination.

For example, a request can encounter a positive prefix during its first
attempt, wait for capacity while ordinary LRU activity reclaims that prefix,
and finally start as `D_MISS` with a positive capacity delay. This is an
accurate observation, not a routing failure.

## Coordinator interface

The coordinator snapshots the last committed binding when a turn starts:

```text
no D binding
    -> Prefill adapter
    -> existing NIXL transfer
    -> Decode generation

D binding present
    -> Decode adapter start_bound(turn)
    -> Decode append prefill and generation
```

The Decode adapter exposes one deep D-bound execution interface rather than an
admission result that the coordinator must interpret:

```python
execution = await decode.start_bound(turn)
observation = execution.observation
stream = execution.stream
```

`start_bound` owns the live Decode response. It consumes the internal execution
observation, preserves any normal chunks already received, and returns one
stream that yields every client-visible chunk exactly once. The coordinator
records the observation but cannot use it to switch adapters.

The D-bound request carries an explicit `decode_bound` marker and its terminal
cache action. It does not carry the Proxy's estimated cached-token count. The
coordinator already owns that estimate and compares it with the returned
observation when it builds the turn record.

The Prefill-side interface and NIXL transfer remain unchanged. Prefill-side
requests do not carry the D-binding marker.

## Decode scheduler behavior

The scheduler recognizes a D-bound request only from its explicit request
marker. The request follows this sequence:

1. Reconcile all relevant local KV cache groups and determine the currently
   reusable Decode-local prefix.
2. Attempt ordinary slot allocation for the unmatched prompt suffix and decode
   work.
3. If allocation cannot proceed, record the first deferral time and leave the
   request in normal Decode waiting.
4. On a later successful allocation, treat the then-current local prefix as the
   final observation, attach its blocks through normal allocation, and emit one
   internal execution observation.
5. Resume the same live request on the following scheduler step and perform
   model work normally.

Zero local hit takes the same allocation path. It must not stop or free the
request, emit a terminal finish reason, or invoke Prefill fallback. The
scheduler reports `D_MISS`, allocates the full append-prefill requirement, and
continues execution.

Capacity delay is independent of cache status. The scheduler no longer emits a
mutually exclusive `CAPACITY_DEFERRED` admission outcome. It emits the final
`D_HIT` or `D_MISS` cache status plus the measured delay.

The observation is internal metadata with no client-visible token. The native
adapter must consume it before exposing the response stream. Cache Action ACK
delivery remains in Decode's completion lifecycle.

## Binding commit lifecycle

The binding used by a current turn is immutable for that turn. Its `t_pred`
selects the cache action and candidate binding for the following turn.

```text
complete output + valid RETAIN_D ACK -> commit D binding
complete output + valid EVICT_D ACK  -> commit no D binding
failure, cancellation, malformed observation, or invalid/missing ACK
                                    -> preserve previous committed binding
```

The commit occurs while holding the existing per-session turn lock. Different
sessions can progress concurrently, while turns within one session remain
sequential.

If an engine action completed but its ACK was lost, the coordinator still does
not guess. Preserving the prior binding is safe: a subsequent D-bound request
can execute with zero cache reuse, and an unbound request can use Prefill and
NIXL regardless of retained Decode cache.

## Failure and invalid-run behavior

- A missing, duplicate, or malformed execution observation fails the current
  D-bound turn and does not commit a new binding.
- A client cancellation closes the live Decode stream, clears request-scoped
  capacity timing, and preserves the committed binding.
- A missing or inconsistent Cache Action ACK fails finalization and preserves
  the committed binding.
- KV event gaps, publisher overflow, malformed event frames, or process restarts
  invalidate an experimental run. They do not mutate live binding state or
  reroute an in-flight request.
- CUDA, NIXL, and output-consistency failures remain terminal test failures and
  cannot be summarized as a routing pass.

## Records and observability

Each turn record distinguishes control from observation:

```text
used_d_binding
execution_path: P_SIDE_AP | D_LOCAL_AP
local_cache_status: D_HIT | D_MISS | null
proxy_estimated_cached_tokens
actual_local_cached_tokens
prompt_tokens
locally_computed_tokens
hit_ratio
estimate_error
capacity_delay_ms
cache_action
cache_action_ack
```

`local_cache_status` is null for an unbound Prefill-side turn because that turn
does not perform D-bound local-prefix observation. Existing native KV event and
publisher/subscriber metrics remain independent from these request records.

## Required implementation changes

- Replace the coordinator's path-bearing re-entry plan with committed
  `next_turn_d_bound` state.
- Replace `AdmissionOutcome` with a cache-status type containing only `D_HIT`
  and `D_MISS`.
- Replace `AdmissionResult` with `DecodeExecutionObservation`; keep capacity
  delay as a separate numeric field.
- Replace the Decode adapter's public `admit_local` plus `stream_local` sequence
  with one `start_bound` execution interface.
- Replace `admission` request metadata and `admission_result` output metadata
  with `decode_bound` and `execution_observation` vocabulary.
- Remove terminal zero-hit handling and Proxy Prefill fallback.
- Remove engine consumption of `proxy_estimated_cached_tokens`.
- Preserve the existing Prefill/NIXL path, Cache Action completion hook,
  request-scoped eviction, KV event transport, and event mirror.
- Update documentation to remove the superseded zero-hit fallback contract and
  any fork-local issue references.

## Test strategy

Tests use module interfaces and observable scheduler outputs rather than private
adapter dictionaries, publisher queues, or block-pool storage fields.

### Coordinator and portable tests

- A first turn has no D binding and uses Prefill-side AP.
- A successful `RETAIN_D` ACK causes the next turn to use D-bound execution.
- Full, partial, and zero D-local hits all complete through the D adapter.
- `D_MISS` is recorded without a Prefill call or duplicated client output.
- A successful `EVICT_D` ACK clears the next-turn D binding.
- Failure, cancellation, malformed observation, and ACK failure do not commit a
  candidate binding.
- Capacity delay is recorded independently from both cache statuses.
- Concurrent sessions isolate bindings, failures, observations, and outputs.
- Completion and chat SSE streams hide internal observations and preserve every
  client-visible chunk exactly once.

### Scheduler and engine tests

- A D-bound zero-hit request allocates and schedules full append-prefill model
  work instead of producing a terminal stop.
- A positive partial hit schedules only the unmatched suffix.
- Capacity deferral followed by changed LRU residency stays on Decode and
  reports the final cache status with a positive delay.
- `RETAIN_D` leaves completed cache blocks ordinary and LRU-eligible.
- Prefill/NIXL requests without the D-binding marker retain existing behavior.
- Cancellation and completion clear request-scoped capacity timing.
- Cache Action eviction and native KV event behavior retain their existing
  focused regression coverage.

### Target-server acceptance

Run one CPU Proxy, one Prefill engine, and one Decode engine with the fixed
dual-GPU BF16 model configuration. Save exact revisions, requests, complete
responses and SSE, process logs, health snapshots, GPU memory snapshots, and
publisher/subscriber statistics for:

1. first-turn unbound Prefill/NIXL execution;
2. `RETAIN_D` followed by a D-bound full local hit;
3. D-bound partial local hit;
4. D-bound zero local hit performing full append prefill on Decode without
   Proxy fallback;
5. `EVICT_D` followed by unbound Prefill/NIXL execution;
6. D-bound capacity waiting and recovery without rerouting;
7. at least two concurrent sessions with isolated bindings and outputs.

Every accepted run requires output consistency, complete evidence, zero event
gaps, no malformed frames, and no CUDA or NIXL errors.

## Non-goals

- P binding or a general worker-affinity protocol.
- Hard cache pinning, leases, TTL retention, or session-owned physical blocks.
- Cache-hit-, mirror-, load-, or capacity-based routing.
- A synchronous cache query before dispatch.
- A new KV transfer mechanism or changes to the existing NIXL path.
- Multiple Prefill or Decode replicas, migration, or overlapping turns within
  one session.
