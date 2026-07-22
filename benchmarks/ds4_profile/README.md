# DS4-informed Qwen3.5 1P1D profile

This directory is being refactored into a controlled serving-metrics profile
for a fixed dual-RTX-3090 deployment.

[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md) is the single source of truth
for requirements, metric semantics, ticket scope, experiment construction,
testing, acceptance, and output artifacts. If another note or historical file
conflicts with it, the authoritative specification wins.

## Target experiment

- `Qwen/Qwen3.5-4B`, BF16, language-model-only;
- one TP=1 prefill instance and one TP=1 decode instance;
- real P-to-D state transfer through NIXL and a 1P1D proxy;
- DS4 trajectories used only as the request dataset;
- controlled P-side prefix-cache hit ratios of 0%, 25%, 50%, 75%, 85%, and
  90%;
- P-side `max_num_batched_tokens` as the chunked-prefill experiment axis;
- explicit selected points rather than an implicit Cartesian product;
- TTFT, request-level TPOT, and output-token throughput as primary results.

This is a serving-oriented metrics experiment. It is not a production traffic
model and does not study RPS, Poisson arrivals, SLO goodput, routing, or
autoscaling.

## Implementation tickets

| Ticket | Scope | Exit result |
| --- | --- | --- |
| 1 | Minimal DS4-to-CustomDataset adapter | Deterministic profile-ready JSONL |
| 2 | Qwen3.5 NIXL 1P1D feasibility and fixed launcher | Target-machine cold and prefix-hit smoke |
| 3 | Controlled serving-metric MVP | First visible point and six-point minimum matrix |
| 4 | Selected measurements and report | Aggregated pilot, concise report, replacement acceptance |

Execution is risk-first: Ticket 2 smoke precedes the remaining implementation.
The detailed gates are in
[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md#11-four-implementation-tickets).

## Current state

- The replacement design is approved.
- The pinned DS4 snapshot/manifest work remains reusable.
- The new Qwen3.5 1P1D launcher, minimal adapter, and controlled runner are not
  implemented yet.
- Existing Qwen2.5, normalization, workload, container, and profile-spine code
  is legacy implementation retained temporarily for traceability.
- Legacy code must not be extended or treated as the new experiment path.
- It is retired only after the replacement passes the hardware and profile
  acceptance gates.

Follow [`WORKFLOW.md`](WORKFLOW.md) for the implementation and acceptance order.
Do not copy commands from legacy handoffs or historical design documents into
new work.

## Reuse boundary

Reuse the official and existing contracts named in the authoritative
specification, especially:

- `vllm bench serve` and its detailed JSON output;
- CustomDataset JSONL;
- the OpenAI-compatible completion endpoint;
- `NixlConnector`, `/metrics`, and `/reset_prefix_cache`;
- official NIXL integration-test launch and metrics patterns;
- the existing pinned snapshot, SHA validation, fixtures, NUMA checks, and
  exact-commit evidence discipline.

Do not reuse the old `GPUModelRunner` measurement boundary, teacher-forced
replay, DS4-to-Qwen token mapping, custom `SchedulerOutput`, LRU replay, or
legacy Parquet result contract.

## Non-goals

The replacement does not implement LRU/eviction research, natural DS4 hit-rate
analysis, full trajectory replay, tool execution, kernel profiling, transfer
microbenchmarks, quantization, FP8 KV, speculative decoding, or multi-P/multi-D
scheduling.

## Documentation status

These files are current:

- [`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md): normative design and
  acceptance contract;
- [`WORKFLOW.md`](WORKFLOW.md): execution and handoff sequence;
- this README: project entry and current status.

`HANDOFF.md`, `TICKET_04_HANDOFF.md`, `container/README.md`, and the existing
Ticket 01-04 implementation document only the pre-refactor path. They are
historical evidence, not instructions for the replacement.
