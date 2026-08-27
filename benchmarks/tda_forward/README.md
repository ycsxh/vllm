# TDAforward Issue #18 acceptance harness

This package prepares and audits the target-server evidence for the fixed 1P1D
TDAforward slice. It does not change the binding-first contract: a committed D
binding always executes append prefill on Decode, including `D_MISS`; only an
unbound turn uses Prefill/NIXL.

Portable tests validate planning, parsing, aggregation, invalidation, and
reporting only. They cannot establish Issue #18 PASS.

## One entry point

From a clean checkout of the exact personal-fork commit:

```bash
.venv/bin/python -m benchmarks.tda_forward.runner plan
.venv/bin/python -m benchmarks.tda_forward.runner run-all \
  --evidence-dir /absolute/path/outside/the/repository/issue-18-evidence
```

To resume deliberately after inspecting a failed run, use `run --run-id` with a
new evidence directory. Existing run directories are never overwritten. Rebuild
a report from preserved runs with:

```bash
.venv/bin/python -m benchmarks.tda_forward.runner report \
  --evidence-dir /absolute/path/to/issue-18-evidence
```

## Target-server checklist

1. Check out the exact commit reported in the handoff and confirm `git status`
   is clean. Install the repository environment with `uv`; do not use system
   Python or bare pip. The TDAforward proxy uses the same repository `.venv` and
   its source directory through `PYTHONPATH`.
2. Confirm exactly two RTX 3090 GPUs are visible, NIXL is importable in the
   repository environment, and `numactl`, `nvidia-smi`, and `nvcc` are present.
3. Set all required variables. Revisions must be full immutable 40-character
   lowercase SHAs:

   ```bash
   export TDA_MODEL_REVISION=<Qwen-Qwen3.5-4B-commit>
   export TDA_TOKENIZER_REVISION=<matching-tokenizer-commit>
   export TDA_ATTENTION_BACKEND=FLASH_ATTN
   export TDA_PREFILL_CPUS=<GPU-0-local-cpu-list>
   export TDA_PREFILL_NUMA_NODE=<GPU-0-numa-node>
   export TDA_DECODE_CPUS=<GPU-1-local-cpu-list>
   export TDA_DECODE_NUMA_NODE=<GPU-1-numa-node>
   ```

4. Put the evidence directory outside the checkout. Run the plan command, review
   every process command and fixed value, then run `run-all` without editing the
   committed config between repetitions.
5. If a controlled run is invalid, preserve it, diagnose it, and rerun the full
   affected enabled/disabled seed pair in a new evidence directory. Do not copy
   a replacement summary over the invalid run.

The committed matrix contains a sequential direct-Decode correctness oracle and
disabled/enabled steady-load comparisons for seeds 17, 29, and 43, three enabled
healthy-consumer high-churn repetitions, and one deliberately stalled-consumer
saturation drive. Normal workload turns are serialized inside each session while
independent session coroutines overlap.

## Evidence layout

The top directory contains the committed config copy, environment and hardware
commands, `acceptance-summary.json`, `runs.csv`, and `report.md`. Each run keeps:

- `manifest.json`: commit, model/tokenizer revisions, comparison fingerprint,
  complete launch commands, ports, GPU assignments, and selected environment;
- `workload.json`: every deterministic session and turn;
- `logs/`: unmodified Prefill, Decode, and Proxy logs;
- `raw/responses.jsonl`: request IDs, turn definitions, timings, and raw SSE
  chunks;
- `raw/{prefill,decode}-metrics-{before,after}.txt`: Prometheus exposition;
- `raw/cpu.jsonl`: one-second process-tree CPU time samples;
- `raw/proxy-state.json`: per-D-attempt reconciliation records, mirror state,
  event lag, subscriber counts, and sequence gaps;
- `raw/{prefill,decode}-instrumentation.json`: raw Scheduler-step samples and
  event construction/enqueue samples plus publisher counters;
- `run-summary.json`: derived metrics, observed scenario coverage, integrity
  counters, and explicit invalid reasons.

## Decision rules

Controlled and healthy-consumer runs require zero publisher overflow, zero
sequence gaps, no malformed event, no Proxy/Decode restart, and no failed
request. The three-seed enabled median throughput must be at least 95% of the
disabled median. Enabled p95 TTFT, TPOT, and Scheduler-step wall time may each
increase by at most 5%.

Healthy runs additionally require observed full, partial, and zero reuse;
`EVICT_D`; capacity delay followed by recovery; and shared-prefix reuse. Every
D-local record includes Proxy estimate, engine actual cached/prompt/computed
tokens, hit ratio, signed error, and capacity delay. Response SHA-256 digests
must match the fixed-seed sequential oracle, which bypasses the Proxy and does
not perform concurrent eviction, including for the EVICT_D drive.

The stalled drive fixes the publisher queue to one and delays its
acceptance-only consumer while leaving the network subscriber healthy. It passes
only if requests continue
completing, publisher queue overflow is visible, and a later source gap is
visible. It is always marked
`liveness_only` and excluded from routing-quality and normal-latency conclusions.
Publisher enqueue p99 must remain at or below the committed 50 ms liveness bound,
far below the intentional two-second consumer delay.
Missing artifacts produce `INCOMPLETE`; integrity failures produce `INVALID`;
threshold misses produce `FAIL`. Only complete real target-server evidence can
produce `PASS`.
