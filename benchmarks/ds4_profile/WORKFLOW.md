# Qwen3.5 1P1D profile workflow

This workflow turns the approved profile design into a small, auditable result.
It applies only to `benchmarks/ds4_profile/` and does not replace repository-wide
vLLM development policy.

[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md) is normative. This file defines
the order of work, environment boundaries, handoff evidence, and promotion
gates. It deliberately contains no commands for modules that have not been
implemented yet.

## Working rules

- Optimize for a visible, trustworthy profile result rather than a reusable
  research platform.
- Use official vLLM serving, NIXL, metrics, cache-reset, and benchmark
  interfaces wherever they satisfy the experiment contract.
- Run explicit experiment points. Do not generate a hidden Cartesian product.
- Keep local contract testing separate from target-machine hardware evidence.
- Never convert a skipped, unsupported, noisy, OOM, or failed point into a
  passing result.
- Preserve unrelated working-tree changes.
- Keep model caches, raw datasets, and full run directories outside Git.

## Environment boundary

| Environment | Responsibilities | Must not claim |
| --- | --- | --- |
| Developer workstation | Adapter/protocol implementation, CPU tests, fixtures, result parsing, documentation | Qwen3.5 1P1D hardware acceptance |
| Dual-3090 server | NIXL/NUMA smoke, controlled measurements, raw evidence, failure diagnosis | Validation of code other than the checked-out revision |
| Personal fork | Reviewable commits, branch history, concise evidence summaries | Mutation of `vllm-project/vllm` |

All Git state changes target the personal fork `ycsxh/vllm`. Upstream is a
read-only reference. Before a server handoff, record the exact commit and
working-tree state. A result from an older commit does not validate a newer
one.

## Phase 0 — Freeze inputs

Before implementing or running a ticket:

1. confirm the authoritative specification has not changed;
2. record the vLLM commit and dirty state;
3. pin the full Qwen3.5 model/tokenizer revision;
4. retain the existing immutable DS4 manifest and file hashes;
5. record the two-GPU topology and intended NUMA assignment;
6. choose external directories for model/data caches and run artifacts.

Changing the model, precision, metric definitions, cache protocol, experiment
axes, or ticket boundaries requires updating the authoritative specification
before implementation continues.

## Phase 1 — Ticket 2 hardware feasibility

Ticket 2 runs first because Qwen3.5 hybrid state transfer on the target machine
is the highest-risk assumption.

Build only a fixed-topology launcher based on official NIXL patterns:

- P on GPU 0 and its local CPU NUMA node;
- D on GPU 1 and its local CPU NUMA node;
- TP=1 for both roles;
- identical pinned model, BF16, backend, cache dtype, block size, and cache
  mode;
- prefix caching and chunked prefill enabled;
- NIXL configured to fail closed;
- bounded readiness, request, shutdown, and cleanup timeouts;
- complete P, D, and proxy logs.

Run one cold deterministic request and one repeated-prefix request. The smoke
proves functional transfer; it is not a transfer-performance experiment.

The local launcher entry point is
`.venv/bin/python -m benchmarks.ds4_profile.run_pd`. Its dry-run freezes and
prints the exact process plan without starting children. Follow
[`TICKET_02_SERVER_HANDOFF.md`](TICKET_02_SERVER_HANDOFF.md) for the target
machine run; a local test or dry-run cannot satisfy Gate A.

### Gate A — continue or stop

Continue only when:

- Qwen3.5-4B BF16 completes the real P-to-D request;
- cold and repeated requests produce identical greedy output;
- D external-transfer tokens and successful transfer metrics increase;
- the repeated request shows the expected prefix-cache evidence;
- no hang, OOM, compatibility mismatch, failed transfer, or silent fallback
  occurs.

If 4B fails, use Qwen3.5-0.8B only to distinguish environment failure from a
4B/HMA failure. It cannot satisfy Gate A. Stop and review the model/runtime
decision if the 4B smoke remains invalid.

## Phase 2 — Ticket 1 minimal dataset adapter

After or alongside stabilization of Gate A, implement only the dataset contract
required by the profile:

1. verify the pinned manifest and raw file hashes;
2. select assistant-turn prompt cut points deterministically;
3. render prompts with the pinned Qwen3.5 chat template;
4. calculate input token lengths;
5. write prompt-only CustomDataset JSONL plus a small provenance/row sidecar.

The adapter entry point is
`.venv/bin/python -m benchmarks.ds4_profile.prepare_dataset`. Follow
[`TICKET_01_SERVER_HANDOFF.md`](TICKET_01_SERVER_HANDOFF.md) to bind its output
to the cached, immutable Qwen3.5 tokenizer revision.

Do not generate normalized Parquet, trajectory analytics, natural hit-rate
estimates, teacher-forced tokens, or DS4-to-Qwen vocabulary mappings.

### Gate B — adapter acceptance

- repeated runs with the same inputs are byte-identical;
- recorded input lengths match prompt token IDs;
- every row has source identity;
- source, revision, hash, tokenizer, and rendering failures fail closed;
- focused CPU tests use the existing network-free fixtures.

## Phase 3 — Ticket 3 controlled MVP

Implement one narrow orchestrator around official interfaces. It accepts an
explicit point file and prepared dataset, starts or validates the fixed 1P1D
deployment, controls cache state, invokes `vllm bench serve`, and saves raw
official results plus metrics deltas.

The cache protocol for every measured repetition is:

```text
wait idle
→ reset P and D
→ warm each nonzero P prefix
→ wait for transfers
→ reset D only
→ snapshot P/D metrics
→ run the official benchmark
→ snapshot metrics again
→ validate and persist the point
```

Engine/CUDA Graph warmup happens before measured repetitions. Benchmark
readiness and implicit warmup requests are disabled so they cannot mutate the
prepared cache state. P and D run with `VLLM_SERVER_DEV_MODE=1` only in this
isolated experiment environment to expose the reset endpoint.

The first visible point is:

```text
hit=75%
P max_num_batched_tokens=4096
max_concurrency=1
output_tokens=1
```

After it succeeds, run the six-point minimum matrix defined in
[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md#92-minimum-credible-matrix).

### Gate C — MVP acceptance

- the first point has complete benchmark, metrics, log, configuration, and
  provenance artifacts;
- every minimum-matrix point has three measured runs or an explicit retained
  failure;
- actual P hit agrees with the aligned plan within one block;
- 0% points show no unintended local reuse;
- D external-transfer evidence is present;
- one-token TTFT points omit TPOT instead of reporting zero;
- no result depends on a private GPU runner or custom latency calculation.

If the six points do not show interpretable differences, stop before expanding
the experiment and diagnose construction or measurement semantics.

## Phase 4 — Ticket 4 selected pilot and report

Only after Gate C:

- run the controlled hit-ratio main effect;
- run the chunk-budget main effect;
- run concurrency and input/output-length main effects;
- add only the selected two-factor interactions in the authoritative plan;
- aggregate three runs, report p50/p90/p95, mean/CV, and label noisy points;
- generate `summary.csv`, a concise `report.md`, and only necessary plots.

Profiler-enabled runs, if a representative point needs diagnosis, are stored
separately and excluded from latency statistics.

### Gate D — replacement acceptance

The human reviewer verifies that every claim traces to official benchmark JSON,
P/D metric deltas, frozen configuration, and logs. The report must state the
hardware, topology, model revision, BF16 precision, cache mode, limitations,
unsupported points, and noisy results.

After Gate D passes:

1. promote the new commands into this README and workflow;
2. retire the legacy `gpu_profile.py` and `profile_spine.py` path;
3. remove Qwen2.5 mapping, teacher forcing, and old result-contract tests;
4. move retained historical evidence to an explicitly discarded/archive area;
5. keep only one documented future workflow.

## Ticket verification and handoff

Each ticket handoff records:

- branch, exact commit, and dirty state;
- local commands run and exact results;
- unresolved hardware-only checks;
- target-machine command/configuration when implemented;
- model and dataset revisions;
- result path, checksums, and acceptance verdict;
- failures, reruns, and noise decisions without omission.

Use `.venv/bin/python -m pytest` for focused Python tests and the repository's
`pre-commit` hooks for changed files. Do not use system Python or bare `pip`.
Exact focused test commands belong beside the implementation that introduces
them; this workflow must not advertise nonexistent modules or stale commands.

## Failure handling

- Cache reset failure: wait for idle and retry a bounded number of times, then
  retain an invalid repetition.
- NIXL failure or silent fallback: invalidate the point.
- OOM: record the point as unsupported; do not silently lower its parameters.
- Hit mismatch beyond one block: invalidate and diagnose before proceeding.
- CV above 5%: label noisy and selectively rerun only that point.
- Partial execution: retain completed artifacts and the exact failed phase.

No latency value is an acceptance threshold. Functional evidence and result
integrity determine whether a stage passes.
