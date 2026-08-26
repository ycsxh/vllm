# TDAforward D-Binding Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the committed next-turn D-binding signal the sole authority for
Decode-local append prefill while reporting cache reuse and capacity delay as
independent observations.

**Architecture:** `ReentryCoordinator` stores only whether the next turn is
D-bound; no binding means the existing Prefill/NIXL path. A D-bound request is
one live Decode execution whose scheduler observation reports final local reuse
and capacity delay but cannot cause fallback. Binding changes commit only after
the complete output stream and a valid Cache Action ACK.

**Tech Stack:** Python 3.12, asyncio, FastAPI, httpx SSE, vLLM V1 Scheduler,
native KV events, NIXL, pytest, uv, pre-commit.

**Spec:**
`docs/superpowers/specs/2026-08-25-tda-forward-d-binding-execution-design.md`

## Global Constraints

- Use `uv` and `.venv/bin/python` for every Python and pytest command. Never use
  system Python or bare pip.
- An absent D binding always selects Prefill/NIXL. There is no P-binding state.
- A D-bound turn stays on Decode for full, partial, and zero local reuse and for
  capacity waiting.
- `RETAIN_D` adds no pin, lease, TTL, ownership record, or special eviction
  priority.
- `D_HIT` and `D_MISS` are observations; `capacity_delay_ms` is an independent
  observation. None of them can mutate routing.
- Do not modify the native KV event schema, NIXL transfer mechanism, or
  request-scoped `EVICT_D` behavior except for the test-surface cleanup listed
  below.
- Keep tests on module interfaces and observable results. Do not assert adapter
  dictionaries, publisher queues, block refcounts, or hash-map storage.
- Every code task uses red-green TDD and ends in a focused commit with
  `Co-authored-by: OpenAI Codex <codex@openai.com>` and the repository
  operator's `Signed-off-by` trailer.

## File responsibility map

- `examples/disaggregated/tda_forward/src/tda_forward/contracts.py`: portable
  binding, observation, execution, and record value types.
- `examples/disaggregated/tda_forward/src/tda_forward/coordinator.py`:
  per-session committed binding, adapter selection, ACK-gated commit, and
  records.
- `examples/disaggregated/tda_forward/src/tda_forward/fakes.py`: deterministic
  public fake adapters used by portable tests.
- `examples/disaggregated/tda_forward/src/tda_forward/native.py`: native SSE
  Decode execution adapter and internal observation/ACK parsing.
- `vllm/v1/core/sched/scheduler.py`: D-bound scheduler marker, local-reuse
  observation, capacity timing, and nonterminal zero-hit execution.
- `vllm/v1/core/sched/output.py`: scheduler-to-engine observation output field.
- `tests/entrypoints/openai/**`: preservation of internal control metadata
  through chat and completion streaming.
- `tests/distributed/test_events.py`: publisher/subscriber behavior through
  public calls and stats.
- `tests/v1/core/test_prefix_caching.py`: request-scoped eviction behavior
  through results, future lookup, and emitted events.
- `examples/disaggregated/tda_forward/README.md`: current binding contract and
  native launch description without fork-local metadata.

---

### Task 1: Separate committed D binding from Decode observations

**Files:**

- Modify: `examples/disaggregated/tda_forward/src/tda_forward/contracts.py`
- Modify: `examples/disaggregated/tda_forward/src/tda_forward/coordinator.py`
- Modify: `examples/disaggregated/tda_forward/src/tda_forward/fakes.py`
- Test: `examples/disaggregated/tda_forward/tests/test_coordinator.py`

**Interfaces:**

- Produces: `LocalCacheStatus.D_HIT | D_MISS`.
- Produces: `DecodeExecutionObservation(cache_status,
  actual_local_cached_tokens, prompt_tokens, locally_computed_tokens,
  capacity_delay_ms=0.0)`.
- Produces: `DecodeExecution(observation, stream)` where `stream` is an
  `AsyncIterator[dict[str, object]]`.
- Produces: `DecodeAdapter.start_bound(turn) -> DecodeExecution`.
- Produces: `ReentryCoordinator.is_d_bound(session_id: str) -> bool`.
- Removes: `ReentryPlan`, `PlannedPath`, `P_FALLBACK`,
  `AdmissionOutcome.CAPACITY_DEFERRED`, and routing based on admission results.

Add `AsyncIterator` from `collections.abc` to `contracts.py` for the
`DecodeExecution.stream` annotation.

- [ ] **Step 1: Replace coordinator tests with failing binding-first behavior**

Add focused tests with these observable assertions:

```python
@pytest.mark.asyncio
async def test_zero_hit_d_bound_turn_stays_on_decode() -> None:
    coordinator, prefill, decode = make_coordinator()
    await run_turn(coordinator, payload(t_pred=1.0))
    decode.queue_observation(
        "s",
        DecodeExecutionObservation(
            cache_status=LocalCacheStatus.D_MISS,
            actual_local_cached_tokens=0,
            prompt_tokens=8,
            locally_computed_tokens=8,
        ),
    )

    chunks = await run_turn(coordinator, payload(t_pred=1.0, stream=True))

    assert chunks == [{"choices": [{"text": "local"}]}]
    assert [call.kind for call in prefill.calls] == ["prefill"]
    assert coordinator.records[-1].execution_path is ExecutionPath.D_LOCAL_AP
    assert coordinator.records[-1].local_cache_status is LocalCacheStatus.D_MISS
```

```python
@pytest.mark.asyncio
async def test_ack_failure_preserves_committed_binding() -> None:
    coordinator, _, decode = make_coordinator()
    await run_turn(coordinator, payload(t_pred=1.0))
    assert coordinator.is_d_bound("s")
    decode.queue_ack(
        "s",
        CacheActionAck(status=CacheActionStatus.RETAINED),
    )

    with pytest.raises(RuntimeError, match="inconsistent with its action"):
        await run_turn(coordinator, payload(t_pred=50.0))

    assert coordinator.is_d_bound("s")
```

Update the boundary, first-turn, full/partial-hit, eviction, and concurrency
tests to use `is_d_bound`, `ExecutionPath`, and `local_cache_status`. Delete the
old zero-hit fallback expectation.

- [ ] **Step 2: Run the new tests and verify red**

Run:

```bash
cd examples/disaggregated/tda_forward
../../../.venv/bin/python -m pytest \
  tests/test_coordinator.py::test_zero_hit_d_bound_turn_stays_on_decode \
  tests/test_coordinator.py::test_ack_failure_preserves_committed_binding -v
```

Expected: collection fails because the new observation and execution interfaces
do not exist.

- [ ] **Step 3: Add the portable value types**

Replace the path/admission types with:

```python
class ExecutionPath(StrEnum):
    D_LOCAL_AP = "D_LOCAL_AP"
    P_SIDE_AP = "P_SIDE_AP"


class LocalCacheStatus(StrEnum):
    D_HIT = "D_HIT"
    D_MISS = "D_MISS"


@dataclass(frozen=True)
class DecodeExecutionObservation:
    cache_status: LocalCacheStatus
    actual_local_cached_tokens: int
    prompt_tokens: int
    locally_computed_tokens: int
    capacity_delay_ms: float = 0.0


@dataclass(frozen=True)
class DecodeExecution:
    observation: DecodeExecutionObservation
    stream: AsyncIterator[dict[str, object]]
```

Change `TurnRecord` to expose `used_d_binding`, `execution_path`, and
`local_cache_status`. Remove `planned_path`, `engine_outcome`, and
`fallback_reason`.

- [ ] **Step 4: Implement ACK-gated binding state**

Use this state and branch shape:

```python
@dataclass
class _SessionState:
    turn_sequence: int
    next_turn_d_bound: bool


def is_d_bound(self, session_id: str) -> bool:
    state = self._states.get(session_id)
    return False if state is None else state.next_turn_d_bound
```

Inside `_run_turn`, snapshot `used_d_binding` before dispatch. Call
`decode.start_bound(turn)` only when it is true; otherwise call the existing
Prefill/NIXL flow. After the stream and ACK validation, commit exactly:

```python
state.turn_sequence = turn_sequence
state.next_turn_d_bound = action is CacheAction.RETAIN_D
```

Do not update state in an exception or cancellation path.

- [ ] **Step 5: Update the fake Decode adapter**

Implement `queue_observation` and `start_bound`:

```python
async def start_bound(self, turn: PreparedTurn) -> DecodeExecution:
    self.calls.append(AdapterCall("start_bound", turn.session_id))
    queued = self._observations[turn.session_id]
    observation = queued.popleft() if queued else DecodeExecutionObservation(
        cache_status=LocalCacheStatus.D_MISS,
        actual_local_cached_tokens=0,
        prompt_tokens=len(turn.prompt_token_ids),
        locally_computed_tokens=len(turn.prompt_token_ids),
    )

    async def stream() -> AsyncIterator[dict[str, object]]:
        yield self._response_chunk(turn, "local")

    return DecodeExecution(observation=observation, stream=stream())
```

Remove the Proxy estimate from `AdapterCall`; the coordinator continues to read
the mirror estimate directly for its record.

- [ ] **Step 6: Run the complete portable coordinator suite**

Run:

```bash
cd examples/disaggregated/tda_forward
../../../.venv/bin/python -m pytest tests/test_coordinator.py -v
```

Expected: all coordinator tests pass; zero-hit D-bound execution makes no second
Prefill call, and failed finalization preserves the binding.

- [ ] **Step 7: Commit Task 1**

```bash
git add \
  examples/disaggregated/tda_forward/src/tda_forward/contracts.py \
  examples/disaggregated/tda_forward/src/tda_forward/coordinator.py \
  examples/disaggregated/tda_forward/src/tda_forward/fakes.py \
  examples/disaggregated/tda_forward/tests/test_coordinator.py
git commit -m "refactor: separate D binding from cache observation"
```

### Task 2: Make native D-bound execution one deep adapter operation

**Files:**

- Modify: `examples/disaggregated/tda_forward/src/tda_forward/native.py`
- Test: `examples/disaggregated/tda_forward/tests/test_native.py`

**Interfaces:**

- Consumes: `DecodeExecution` and `DecodeExecutionObservation` from Task 1.
- Produces: `VllmDecodeAdapter.start_bound(turn) -> DecodeExecution`.
- Wire input: `kv_transfer_params.tda_forward.decode_bound = true`.
- Wire output: `kv_transfer_params.tda_forward.execution_observation`.

- [ ] **Step 1: Write failing native adapter tests for hit and zero hit**

Use `httpx.MockTransport` and assert the public adapter result only:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "cached", "computed"),
    [("D_HIT", 4, 1), ("D_MISS", 0, 5)],
)
async def test_bound_decode_streams_after_cache_observation(
    status: str, cached: int, computed: int
) -> None:
    observation = {
        "cache_status": status,
        "actual_local_cached_tokens": cached,
        "prompt_tokens": 5,
        "locally_computed_tokens": computed,
        "capacity_delay_ms": 0.0,
    }
    # Mock one observation chunk, one text chunk, and one terminal ACK chunk.
    execution = await adapter.start_bound(_turn())
    chunks = [chunk async for chunk in execution.stream]
    ack = await adapter.finish(_turn(), CacheAction.RETAIN_D)

    assert execution.observation.cache_status.value == status
    assert [choice["text"] for chunk in chunks for choice in chunk["choices"]] \
        == ["ok", ""]
    assert ack.status is CacheActionStatus.RETAINED
    assert seen_payload["kv_transfer_params"]["tda_forward"] == {
        "decode_bound": True,
        "cache_action": "RETAIN_D",
    }
```

Replace the terminal-miss/fallback test with the `D_MISS` parameter above.
Remove assertions against `_sessions` and `_completed_acks`; missing ACK and
malformed stream tests must assert only `stream`/`finish` exceptions.

- [ ] **Step 2: Verify native tests fail on the old two-call interface**

Run:

```bash
cd examples/disaggregated/tda_forward
../../../.venv/bin/python -m pytest \
  tests/test_native.py::test_bound_decode_streams_after_cache_observation -v
```

Expected: FAIL because `start_bound` and `execution_observation` are absent.

- [ ] **Step 3: Implement `start_bound` and observation parsing**

Open one SSE response with:

```python
transfer = {
    "tda_forward": {
        "decode_bound": True,
        "cache_action": turn.cache_action.value,
    }
}
```

Consume lines until `_execution_observation` returns a value. Always retain the
same live session, including `D_MISS`, and return:

```python
return DecodeExecution(
    observation=observation,
    stream=self._stream_session(turn, session),
)
```

Rename `_admission_result` to `_execution_observation` and parse
`cache_status`, not `outcome`. Delete `admit_local`, `stream_local`, the
terminal-miss close, and all fallback-specific handling.

- [ ] **Step 4: Verify missing/malformed metadata fails without leaked success**

Keep tests that exhaust a stream without an ACK and that receive malformed JSON.
Assert:

```python
with pytest.raises(RuntimeError, match="without a Cache Action ACK"):
    await adapter.finish(turn, CacheAction.RETAIN_D)
```

For close failure, provide a custom public `httpx.AsyncByteStream` whose
`aclose()` raises `RuntimeError("stream close failed")`; do not reach into the
adapter's session dictionary.

- [ ] **Step 5: Run all native adapter tests**

Run:

```bash
cd examples/disaggregated/tda_forward
../../../.venv/bin/python -m pytest tests/test_native.py -v
```

Expected: all tests pass and both cache statuses complete through one Decode
connection.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  examples/disaggregated/tda_forward/src/tda_forward/native.py \
  examples/disaggregated/tda_forward/tests/test_native.py
git commit -m "refactor: make D-bound decode a single execution"
```

### Task 3: Make zero-hit and capacity-delayed D-bound requests executable

**Files:**

- Modify: `vllm/v1/core/sched/scheduler.py`
- Modify: `vllm/v1/core/sched/output.py`
- Test: `tests/v1/core/test_scheduler.py`

**Interfaces:**

- Consumes: request marker `tda_forward.decode_bound is True`.
- Produces: one internal `execution_observation` before model output.
- Produces: `cache_status` plus independent `capacity_delay_ms`.
- Preserves: normal request waiting, LRU eligibility, allocation, and model
  scheduling.

- [ ] **Step 1: Rewrite the scheduler zero-hit test to require model work**

Use only scheduler requests and outputs:

```python
def test_tda_bound_zero_hit_observes_miss_then_schedules_model_work(monkeypatch):
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=4,
        max_num_batched_tokens=64,
    )
    (request,) = create_requests(
        num_requests=1,
        num_tokens=9,
        max_tokens=1,
        block_size=4,
    )
    request.kv_transfer_params = {
        "tda_forward": {"decode_bound": True, "cache_action": "RETAIN_D"}
    }
    scheduler.add_request(request)

    observation_step = scheduler.schedule()
    outputs = scheduler.update_from_output(
        observation_step, _empty_model_runner_output()
    )
    observation = outputs[request.client_index].outputs[0].kv_transfer_params[
        "tda_forward"
    ]["execution_observation"]
    assert observation == {
        "cache_status": "D_MISS",
        "actual_local_cached_tokens": 0,
        "prompt_tokens": 9,
        "locally_computed_tokens": 9,
        "capacity_delay_ms": 0.0,
    }
    assert request.request_id in scheduler.requests

    model_step = scheduler.schedule()
    assert model_step.num_scheduled_tokens == {request.request_id: 9}
```

- [ ] **Step 2: Add the capacity/LRU reproducer**

Warm a shared prefix, force the first `allocate_slots` call to return `None`,
then clear ordinary prefix cache through `scheduler.reset_prefix_cache()` before
retrying. Assert the final result is a miss with delay and still schedules D
model work:

```python
deferred_step = scheduler.schedule()
assert deferred_step.total_num_scheduled_tokens == 0
assert scheduler.reset_prefix_cache()

observation_step = scheduler.schedule()
outputs = scheduler.update_from_output(
    observation_step, _empty_model_runner_output()
)
observation = outputs[request.client_index].outputs[0].kv_transfer_params[
    "tda_forward"
]["execution_observation"]
assert observation["cache_status"] == "D_MISS"
assert observation["actual_local_cached_tokens"] == 0
assert observation["capacity_delay_ms"] > 0
assert scheduler.schedule().num_scheduled_tokens == {request.request_id: 9}
```

- [ ] **Step 3: Run both tests and verify red**

Run:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py::test_tda_bound_zero_hit_observes_miss_then_schedules_model_work \
  tests/v1/core/test_scheduler.py::test_tda_bound_capacity_delay_reports_final_cache_state \
  -v
```

Expected: FAIL because zero hit is terminal and capacity is encoded as an
exclusive outcome.

- [ ] **Step 4: Replace admission helpers with binding/observation helpers**

Implement marker parsing:

```python
@staticmethod
def _tda_forward_decode_bound(request: Request) -> bool:
    params = request.kv_transfer_params
    if not isinstance(params, dict):
        return False
    value = params.get("tda_forward")
    return isinstance(value, dict) and value.get("decode_bound") is True
```

Add `self._tda_forward_observed_requests: set[str]` beside the capacity timing
map. An observation is pending only when the request is D-bound and its ID is
not in this set. Clear both structures in `_free_request`.

- [ ] **Step 5: Remove terminal miss and emit one orthogonal observation**

Delete the branch that sets `FINISHED_STOPPED`, calls `_free_request`, and emits
`tda_forward_d_miss`. After successful `allocate_slots`, emit:

```python
cache_status = (
    "D_HIT" if tda_forward_local_cached_tokens > 0 else "D_MISS"
)
observation = {
    "cache_status": cache_status,
    "actual_local_cached_tokens": tda_forward_local_cached_tokens,
    "prompt_tokens": request.num_prompt_tokens,
    "locally_computed_tokens": max(
        request.num_prompt_tokens - tda_forward_local_cached_tokens, 0
    ),
    "capacity_delay_ms": capacity_delay_ms,
}
self._tda_forward_observed_requests.add(request_id)
```

Attach it under `execution_observation`, leave the request alive, and break so
the observation is delivered before model output. Keep the Decode-bound marker
excluding external/NIXL cache from the local observation path on subsequent
steps.

Rename `tda_forward_admission_outputs` to
`tda_forward_observation_outputs` in `SchedulerOutput` and
`update_from_output`.

- [ ] **Step 6: Run the four scheduler behavior tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py::test_tda_bound_hit_observes_reuse_before_model_work \
  tests/v1/core/test_scheduler.py::test_tda_bound_zero_hit_observes_miss_then_schedules_model_work \
  tests/v1/core/test_scheduler.py::test_tda_bound_reconciles_every_cache_group \
  tests/v1/core/test_scheduler.py::test_tda_bound_capacity_delay_reports_final_cache_state \
  -v
```

Expected: 4 passed. The zero-hit request remains in `scheduler.requests`; the
capacity test reports `D_MISS` plus positive delay rather than fallback.

- [ ] **Step 7: Commit Task 3**

```bash
git add \
  vllm/v1/core/sched/scheduler.py \
  vllm/v1/core/sched/output.py \
  tests/v1/core/test_scheduler.py
git commit -m "fix: keep bound append prefill on decode"
```

### Task 4: Preserve observations through OpenAI streaming and public tests

**Files:**

- Modify: `tests/entrypoints/openai/chat_completion/test_serving_chat.py`
- Modify: `tests/entrypoints/openai/completion/test_completion_error.py`
- Modify: `examples/disaggregated/tda_forward/tests/test_http.py`
- Modify only if the tests expose a real propagation bug:
  `vllm/entrypoints/openai/chat_completion/serving.py`
- Modify only if the tests expose a real propagation bug:
  `vllm/entrypoints/openai/completion/serving.py`

**Interfaces:**

- Consumes: internal `execution_observation` metadata from Task 3.
- Produces: the same metadata in native SSE for the adapter, without adding it
  to normal aggregated OpenAI response fields.
- Tests: HTTP behavior only through `create_app` and ASGI requests.

- [ ] **Step 1: Rename the native metadata propagation fixtures**

Use this fixture in both chat and completion streaming tests:

```python
transfer = {
    "tda_forward": {
        "execution_observation": {
            "cache_status": "D_HIT",
            "actual_local_cached_tokens": 16,
            "prompt_tokens": 17,
            "locally_computed_tokens": 1,
            "capacity_delay_ms": 0.0,
        }
    }
}
```

Assert the decoded SSE chunk carries the exact transfer object once.

- [ ] **Step 2: Replace private HTTP aggregation testing with an ASGI request**

Stop importing `_aggregate_stream`. Add a test Decode fake whose public
`stream_from_prefill` yields the three chat deltas and usage chunk, inject it
through `ReentryCoordinator`, then POST `stream: false` to
`/v1/chat/completions`:

```python
response = await client.post(
    "/v1/chat/completions",
    json={
        "model": "fake",
        "messages": [{"role": "user", "content": "hello"}],
        "session_id": "s",
        "t_pred": 50.0,
        "stream": False,
    },
)
assert response.json()["choices"][0]["message"] == {
    "role": "assistant",
    "content": "hello world",
}
assert response.json()["usage"] == {"total_tokens": 7}
```

- [ ] **Step 3: Run the propagation and HTTP tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/entrypoints/openai/chat_completion/test_serving_chat.py::test_streaming_response_carries_kv_transfer_control_metadata \
  tests/entrypoints/openai/completion/test_completion_error.py::test_completion_stream_carries_kv_transfer_control_metadata \
  examples/disaggregated/tda_forward/tests/test_http.py -v
```

Expected: all pass. Change serving code only if a new test fails because the
metadata is dropped; keep any fix to the existing generic
`kv_transfer_params` propagation.

- [ ] **Step 4: Commit Task 4**

```bash
git add \
  tests/entrypoints/openai/chat_completion/test_serving_chat.py \
  tests/entrypoints/openai/completion/test_completion_error.py \
  examples/disaggregated/tda_forward/tests/test_http.py \
  vllm/entrypoints/openai/chat_completion/serving.py \
  vllm/entrypoints/openai/completion/serving.py
git commit -m "test: cover D-bound metadata through public streams"
```

Before staging, omit unchanged serving files from `git add`.

### Task 5: Remove private-state assertions from existing branch regressions

**Files:**

- Modify: `tests/distributed/test_events.py`
- Modify: `tests/v1/core/test_prefix_caching.py`
- Modify: `examples/disaggregated/tda_forward/tests/test_native.py`

**Interfaces:**

- Consumes: public `publish`, `receive_one`, `stats`, `shutdown`,
  `evict_request_blocks`, `get_computed_blocks`, `take_events`, adapter stream,
  and adapter `finish` behavior.
- Produces: the same nonblocking, sequence-gap, request-scoped eviction, and
  cleanup coverage without private storage access.

- [ ] **Step 1: Control the publisher through its thread dependency**

Replace monkeypatches of `_publisher_thread` and `_event_queue` with a test fake
for the imported `threading.Thread`. Capture the real thread class before
patching; the fake stores the constructor target and exposes `release()`:

```python
real_thread = threading.Thread

class ControlledThread:
    instances: list["ControlledThread"] = []

    def __init__(self, *, target, daemon: bool, name: str) -> None:
        self.target = target
        self.daemon = daemon
        self.name = name
        self.delegate: threading.Thread | None = None
        self.instances.append(self)

    def start(self) -> None:
        return

    def release(self) -> None:
        self.delegate = real_thread(
            target=self.target, daemon=self.daemon, name=self.name
        )
        self.delegate.start()

    def is_alive(self) -> bool:
        return self.delegate is not None and self.delegate.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self.delegate is not None:
            self.delegate.join(timeout)
```

With the publisher queue size set to one, publish two batches while the fake is
unreleased and assert `stats.dropped_batches == 1`. Release the fake, receive
sequence 0, publish again, and receive sequence 2. The test touches no publisher
attribute whose name starts with `_`.

- [ ] **Step 2: Split native round-trip and metrics behaviors**

Keep one test asserting `(sequence, decoded) == (0, batch)`. Add separate tests
for publisher stats and subscriber stats so each test has one named behavior.

- [ ] **Step 3: Express request-scoped eviction through future lookup**

After `evict_request_blocks`, assert the aggregate result and remove events.
Create a new request with the same prompt and assert:

```python
computed, num_computed_tokens, _ = manager.get_computed_blocks(retry)
assert num_computed_tokens == 0
assert not computed.blocks[0]
```

For the shared-active case, assert `status == "DEFERRED"`, zero immediately
reusable blocks, two deferred blocks, and future lookup miss. Free both requests
and allocate an unrelated request to prove normal progress. Remove direct
assertions on `ref_cnt`, `block_hash`, and internal cache maps.

- [ ] **Step 4: Keep native failure tests on the public adapter surface**

Remove the remaining `_sessions` and `_completed_acks` assertions. A malformed
stream must raise while iterating `execution.stream`; a missing ACK must raise
from `finish`. This preserves the failure contract without checking storage.

- [ ] **Step 5: Run all remediated tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/distributed/test_events.py \
  tests/v1/core/test_prefix_caching.py \
  examples/disaggregated/tda_forward/tests/test_native.py -v
```

Expected: all pass with no new xfails or timing retries.

- [ ] **Step 6: Commit Task 5**

```bash
git add \
  tests/distributed/test_events.py \
  tests/v1/core/test_prefix_caching.py \
  examples/disaggregated/tda_forward/tests/test_native.py
git commit -m "test: verify TDAforward through module interfaces"
```

### Task 6: Align documentation and complete local verification

**Files:**

- Modify: `examples/disaggregated/tda_forward/README.md`
- Modify: any directly affected TDAforward `__init__.py` exports.

**Interfaces:**

- Documents: unbound default Prefill/NIXL, ACK-committed D binding, nonterminal
  `D_MISS`, independent capacity delay, and no pin/TTL.

- [ ] **Step 1: Rewrite the README contract**

Remove fork-local issue references. State explicitly:

```text
No D binding -> Prefill-side AP and NIXL transfer.
D binding -> Decode-local append prefill, including zero local reuse.
D_HIT/D_MISS report final local reuse and never select the worker.
capacity_delay_ms reports Decode waiting and never causes rerouting.
The next-turn binding changes only after a complete stream and valid Cache
Action ACK. RETAIN_D remains ordinary LRU-eligible soft retention.
```

Delete every statement that zero hit is terminal or causes Proxy fallback.

- [ ] **Step 2: Run the portable suite**

Run:

```bash
cd examples/disaggregated/tda_forward
../../../.venv/bin/python -m pytest tests -v
```

Expected: all portable tests pass.

- [ ] **Step 3: Run the focused scheduler, event, and NIXL suites**

Run:

```bash
.venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py \
  tests/v1/core/test_prefix_caching.py \
  tests/distributed/test_events.py \
  tests/v1/kv_connector/unit/test_nixl_connector_hma.py \
  tests/v1/kv_connector/unit/test_nixl_heartbeat.py \
  tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py \
  tests/v1/kv_connector/unit/test_bidirectional_kv_transfer.py -v
```

Expected: all selected tests pass. Network-denied fixture failures must be
rerun with the same command in the repository's allowed environment; do not
weaken or skip them.

- [ ] **Step 4: Lint the documentation change and inspect the exact patch**

Run:

```bash
.venv/bin/pre-commit run markdownlint-cli2 --files \
  examples/disaggregated/tda_forward/README.md
git diff --check origin/main...HEAD
git diff --check -- examples/disaggregated/tda_forward/README.md
git status --short --branch
```

Expected: markdownlint and both diff checks pass; the README is the only
task-owned unstaged path.

- [ ] **Step 5: Commit Task 6**

```bash
git add examples/disaggregated/tda_forward/README.md
git commit -m "docs: align TDAforward with D-binding execution"
```

- [ ] **Step 6: Run branch-wide hooks after the documentation commit**

Run:

```bash
.venv/bin/pre-commit run --from-ref origin/main --to-ref HEAD
git diff --check origin/main...HEAD
git status --short --branch
```

Expected: all hooks and the branch diff check pass; no task-owned changes remain
uncommitted.
