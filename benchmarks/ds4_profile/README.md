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

| Ticket | Scope | Current gate |
| --- | --- | --- |
| 1 | Minimal DS4-to-CustomDataset adapter | Gate B `remote_verified` |
| 2 | Qwen3.5 NIXL 1P1D feasibility and fixed launcher | Gate A `remote_verified` |
| 3 | Controlled serving-metric MVP | Gate C `remote_verified` |
| 4 | Selected measurements and report | Target run retained; Gate D blocked on class A |

Execution is risk-first: Ticket 2 smoke precedes the remaining implementation.
The detailed gates are in
[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md#11-four-implementation-tickets).

## Current state

- The replacement design is approved.
- The pinned DS4 snapshot/manifest work remains reusable.
- The prompt-only adapter and fixed Qwen3.5 1P1D launcher are implemented with
  network-free CPU contract tests.
- Ticket 1 passed the exact-revision Qwen3.5 tokenizer Gate B.
- Ticket 2 passed the dual-RTX-3090 Gate A with positive P cache-hit,
  D external-transfer, and NIXL-success evidence.
- Both gates validate final runtime delivery
  `4415bbe8f04c11c5beab7057effe208659f1b91f`.
- The controlled point runner, explicit six-point plan, CPU contract tests, and
  point-specific P deployment restart are implemented.
- Ticket 3 separates the configured 128-token isolation block from the
  accepted runtime's 640-token effective HMA cache page.
- Ticket 3 Gate C passed on the target dual-RTX-3090 machine against immutable
  delivery `163935c12db0545c12eba694bfd6316be1f4094a`.
- Ticket 4 now has a checked-in 30-point selected pilot, explicit report
  comparisons, OOM/unsupported retention, optional one-point eager diagnostics,
  and an auditable `summary.csv`/Markdown/SVG report generator.
- The frozen Ticket 4 target run retained 20 valid and 10 failed points. Class
  B has a locally reviewed token-accurate ITL fix; class A remains reproduced
  but undiagnosed. Gate D human acceptance remains blocked.
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
- [`TICKET_01_SERVER_HANDOFF.md`](TICKET_01_SERVER_HANDOFF.md): pinned-tokenizer
  Gate B completion;
- [`TICKET_02_SERVER_HANDOFF.md`](TICKET_02_SERVER_HANDOFF.md): dual-3090 Gate A
  completion;
- [`TICKET_03_SERVER_HANDOFF.md`](TICKET_03_SERVER_HANDOFF.md): controlled MVP
  Gate C acceptance and historical target-server procedure;
- [`TICKET_04_SERVER_HANDOFF.md`](TICKET_04_SERVER_HANDOFF.md): replacement
  Ticket 4 target-server execution, report, and Gate D procedure;
- [`TICKET_04_MEASUREMENT_FIX_HANDOFF.md`](TICKET_04_MEASUREMENT_FIX_HANDOFF.md):
  current class-A/class-B diagnosis, fix, and next-session state;
- [`HANDOFF.md`](HANDOFF.md): current replacement-project handoff state;
- this README: project entry and current status.

`TICKET_04_HANDOFF.md`, `container/README.md`, and the existing Ticket 01-04
implementation document only the pre-refactor path. They are historical
evidence, not instructions for the replacement.
