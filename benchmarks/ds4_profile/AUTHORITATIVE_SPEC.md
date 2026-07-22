# DS4-informed Qwen3.5 1P1D Serving-Metrics Profile

- Status: **authoritative for all new work**
- Effective date: 2026-07-22
- Repository baseline at approval: `65de0de0ab`

This document is the single source of truth for the replacement of the legacy
12-ticket DS4 profile design. If another planning note, scratch specification,
ticket, handoff, or README conflicts with this document, this document wins.

The existing `GPUModelRunner`-based implementation remains historical evidence
until the replacement path passes hardware acceptance. It must not be extended
for the new profile.

## 1. Objective

Produce useful performance results quickly for a fixed dual-RTX-3090 system:

- model: `Qwen/Qwen3.5-4B`;
- weights: BF16, no weight quantization;
- topology: one prefill instance on GPU 0 and one decode instance on GPU 1;
- each role uses TP=1;
- cache transfer: real NIXL P-to-D transfer through a 1P1D proxy;
- workload source: pinned DS4 SWE-bench trajectories;
- measurement style: controlled serving-oriented metrics, not production
  arrival-process modeling;
- primary variables: P-side prefix-cache hit ratio,
  `max_num_batched_tokens`, input/output length, and offered concurrency;
- primary outputs: TTFT, request-level TPOT, and output-token throughput.

The project is successful when it produces an auditable small experiment that
shows how controlled prefix hits, chunked-prefill budget, and concurrency affect
the fixed 1P1D deployment. Building a reusable research platform is not a goal.

## 2. Questions the profile answers

1. How does controlled P-side prefix-cache hit ratio affect full 1P1D TTFT?
2. How does the P scheduler token budget affect full 1P1D TTFT?
3. How do offered concurrency and output length affect request-level TPOT and
   output-token throughput?
4. Are the requested cache-hit conditions actually observed by P, and is D
   actually receiving state through NIXL?
5. Which tested points are unsupported, noisy, or out of memory on the target
   dual-3090 machine?

The profile does not attempt to isolate P-to-D transfer latency from TTFT. NIXL
metrics are collected only to prove that the intended transfer path ran and to
explain failures.

## 3. Fixed terminology and metric semantics

### 3.1 Serving-oriented, not production traffic

The official HTTP serving path is used because it already measures streaming
TTFT, TPOT, ITL, and throughput correctly. The experiment does not model
Poisson arrivals, production RPS, queueing policy, goodput, or an SLO.

`max_concurrency` is the client control. It is not called batch size. The actual
continuous batch assembled by the scheduler is an observation, not a directly
controlled variable.

### 3.2 TTFT

TTFT is measured at the benchmark client and includes:

```text
proxy + P cache lookup/compute + P-to-D transfer + D first token
```

It must be labeled `1P1D TTFT`. It must not be labeled P-only latency.

### 3.3 TPOT

TPOT uses the official request-level definition:

```text
(request latency - TTFT) / (output tokens - 1)
```

TPOT is undefined for one-token outputs. TTFT experiments therefore use
`output_tokens=1`, while decode experiments use multi-token outputs.

### 3.4 Chunked prefill

`max_num_batched_tokens` is the total scheduler token budget per iteration. It
is not a guaranteed per-request chunk size. With concurrency 1 it is a useful
approximation of the single-request prefill chunk; with concurrency greater
than 1 requests share the budget.

Chunk-budget sweeps change the P configuration. D configuration remains fixed.
A distinct server configuration is restarted before measurement rather than
hot-mutated.

### 3.5 Prefix-cache hit ratio

The controlled nominal ratios are:

```text
0%, 25%, 50%, 75%, 85%, 90%
```

The planned prefix length is aligned down to the configured cache block size:

```text
planned_cached_tokens = floor(input_tokens * ratio / block_size) * block_size
nominal_aligned_ratio = planned_cached_tokens / input_tokens
```

The authoritative observed value is:

```text
actual_P_hit_ratio =
  delta(P prompt_tokens_by_source{source="local_cache_hit"})
  / sum(delta(P prompt_tokens_by_source{source=each source}))
```

The report always carries requested, block-aligned, and observed ratios. If two
requested ratios collapse to the same aligned prefix length, the duplicate
point is rejected before GPU execution.

## 4. Fixed runtime configuration

Every valid run records the effective values below. A missing model revision or
dirty-state record invalidates reproducibility, but does not turn a completed
measurement into a fabricated pass.

| Setting | Required value or rule |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| Model revision | Full immutable Hugging Face commit, required in run config |
| Mode | `--language-model-only` |
| Weights | BF16 |
| Weight quantization | Disabled |
| P/D TP | 1/1 |
| Prefix caching | Enabled |
| Mamba/GDN cache | `align` |
| Chunked prefill | Enabled |
| Speculative decoding/MTP | Disabled |
| P/D model, dtype, backend, cache dtype | Identical |
| P/D block size | Identical; fixed for a comparison |
| Compile/CUDA Graph | Default optimized mode for main results |
| Eager mode | At most one explicitly labeled diagnostic point |
| GPU assignment | P=GPU0, D=GPU1 unless a run explicitly overrides it |
| CPU/NUMA | Each role bound to the CPUs local to its assigned GPU |
| NIXL failure policy | Fail closed; no silent local recompute fallback |
| Development endpoints | `VLLM_SERVER_DEV_MODE=1` on P and D for cache reset |

Qwen3.5 is a hybrid Attention/Gated-DeltaNet model. Basic NIXL PD support and
hybrid cache layout tests in the pinned vLLM revision are prior evidence, not a
substitute for the target-machine hardware smoke.

## 5. DS4 dataset contract

DS4 is an input dataset, not an object of deep workload research.

The adapter consumes the existing pinned manifest and immutable trajectory
files. One output row represents one selected assistant turn. Rendering and
length calculation follow this exact contract:

- the prompt contains all messages before that assistant turn;
- `prompt_text` is produced with the pinned Qwen3.5 tokenizer's
  `apply_chat_template(messages_before, add_generation_prompt=True,
  tokenize=False)`;
- `prompt_ids` are the token IDs of `prompt_text` using that same tokenizer;
- `full_ids` are produced by applying the same chat template to
  `messages_before + [source_assistant_message]` with
  `add_generation_prompt=False` and tokenizing the result;
- `output_tokens` is `len(full_ids) - lcp(prompt_ids, full_ids)`, where `lcp`
  is the longest common token prefix; the adapter rejects non-positive values;
- selection order and any sampling are deterministic;
- raw DS4 files are never modified.

This rule lets the model's chat template serialize reasoning content and tool
calls without inventing a second message format. The adapter pins the tokenizer
revision and records it in provenance.

The profile-ready JSONL uses the official CustomDataset minimum contract:

```json
{"prompt":"<rendered completion prompt>","output_tokens":128}
```

A sidecar row map may carry `request_id`, source path, task, mode, turn index,
input token count, output token count, and prompt token IDs. These fields are
for provenance and controlled-prefix preparation; they do not create a new
benchmark dataset framework.

The adapter does not produce:

- natural DS4 cache-hit estimates;
- trajectory reuse-distance analysis;
- LRU or capacity analysis;
- tool-ready timing;
- teacher-forced tokens;
- DS4-to-Qwen vocabulary mappings;
- Parquet artifacts;
- semantic or SWE-bench correctness claims.

## 6. Reuse policy

Reuse means reusing validated interfaces, contracts, experiment protocols, and
design knowledge. It does not require copying test code unchanged.

### 6.1 Directly use stable vLLM interfaces

- `vllm bench serve` for request scheduling and metric calculation;
- CustomDataset JSONL for per-request prompts and output lengths;
- OpenAI-compatible completion requests;
- `NixlConnector` for P-to-D state transfer;
- `/metrics` for P, D, scheduler, and NIXL counters;
- `/reset_prefix_cache` for controlled cache state;
- official benchmark JSON produced by `--save-result --save-detailed`.

### 6.2 Adapt established implementation patterns

- NIXL producer/consumer settings, side-channel ports, health checks, and
  proxy request flow from the NIXL integration tests and usage guide;
- prefix/suffix length construction from RandomDataset, Prefix Repetition, and
  `benchmarks/auto_tune/auto_tune.sh`;
- metrics before/after deltas from the hybrid/Mamba PD cache tests;
- explicit point files, repeated runs, resume-by-existing-result, and plotting
  conventions from `vllm bench sweep`;
- dataset revision, SHA validation, fixtures, NUMA checks, and server handoff
  discipline from the existing DS4 project.

### 6.3 Use as reference only

- `benchmark_prefix_caching.py` is offline `LLM.generate`, not the 1P1D path;
- `auto_tune.sh` targets tuning/goodput and does not implement the required
  warm-P/reset-D protocol;
- CI NIXL scripts contain matrices and cleanup behavior not needed by the fixed
  experiment;
- the toy proxy is acceptable for the first smoke but test-file stability is
  not treated as a long-term public contract;
- hybrid prefix tests validate repeated-request behavior, not the controlled
  P-side hit ratio defined here;
- profiler traces are diagnostic and never provide the main latency samples.

### 6.4 Do not reuse the legacy measurement path

The replacement must not call or extend:

- `gpu_profile.py`;
- `profile_spine.py`;
- Qwen2.5 execution-token mapping;
- teacher-forced replay;
- custom `SchedulerOutput` construction;
- KVCacheManager/LRU replay;
- the legacy Parquet artifact contract.

## 7. Replacement modules and interfaces

Only two genuinely new seams are required:

1. **DS4 adapter seam**: pinned DS4 snapshot to CustomDataset JSONL plus a
   provenance sidecar.
2. **Controlled-cache experiment seam**: explicit point plus prepared dataset
   to official benchmark result plus metrics deltas.

The fixed P/D launcher is implementation supporting the second seam. It is not
a connector abstraction or a general deployment framework.

The intended command shape is deliberately small:

```bash
.venv/bin/python -m benchmarks.ds4_profile.prepare_dataset \
  --manifest <pinned-manifest> \
  --model Qwen/Qwen3.5-4B \
  --output-dir <prepared-dir>

benchmarks/ds4_profile/run_pd.sh \
  --config <server-config> \
  --results-dir <run-dir>

.venv/bin/python -m benchmarks.ds4_profile.run_points \
  --prepared-dir <prepared-dir> \
  --plan <explicit-points.json> \
  --results-dir <run-dir>
```

Names may change during implementation only if the interface remains equally
small and this document is updated first.

## 8. Controlled-cache measurement protocol

The protocol is part of the experiment contract.

### 8.1 Engine warmup

1. Start P, D, and the proxy.
2. Wait for all health endpoints.
3. Send non-measured canary requests until compilation and CUDA Graph capture
   needed by the selected configuration have completed.
4. Wait until no requests or NIXL transfers are in flight.
5. Reset P and D caches and require successful responses.

Engine warmup is not prefix preparation and is not a measured repetition.

### 8.2 Per-request cache isolation

DS4 prompts may share a system prefix. Each measured request therefore receives
a deterministic, tokenizer-stable, request-unique first cache block. The block
is included in the configured total input length and in the warm prefix. This
prevents requests in the same benchmark batch from creating accidental hits for
one another without introducing a custom server endpoint.

The adapter or point runner verifies that decoding and re-encoding the isolation
block preserves its token IDs. Failure to preserve the IDs rejects the point.

### 8.3 Prefix preparation and measurement

For each repetition:

1. Wait for P, D, and NIXL to become idle.
2. Reset P and D local prefix caches. Retry only bounded transient failures.
3. For each measured request with a nonzero planned prefix, send a prefix-only
   warm request through the full proxy using the exact prefix token IDs.
4. Wait for all warm transfers to complete.
5. Reset only D's local prefix cache and require success. P retains the warmed
   prefixes; D starts empty so the measured request exercises P-to-D transfer.
6. Snapshot P and D `/metrics`.
7. Invoke `vllm bench serve` for the measured requests.
8. Snapshot P and D `/metrics` again.
9. Calculate metric deltas and validate observed P hits and D external transfer.
10. Save the official benchmark result, metrics, effective configuration, and
    logs before advancing to the next repetition.

For the 0% condition, step 3 is skipped. The unique first block and clean P
cache prevent accidental shared-prefix hits.

The benchmark invocation disables its implicit readiness and warmup requests,
because either would alter the prepared cache state:

```text
--ready-check-timeout-sec 0
--num-warmups 0
```

Prepared prompts already contain the Qwen chat template, so CustomDataset uses
`--skip-chat-template`. Runs also use `--disable-shuffle`, `--no-oversample`,
`--custom-output-len -1`, `--ignore-eos`, `--save-result`, and
`--save-detailed`. The fixed arrival mode is `--request-rate inf`; the runner
sets `--num-prompts` to the exact prepared-row count.

## 9. Experiment plan

Experiment points are written explicitly. The runner must not create an
implicit Cartesian product.

### 9.1 First visible result

Before implementing the larger matrix, run:

```text
Qwen3.5-4B BF16
real NIXL 1P1D
one fixed DS4-derived input length
hit=75%
max_num_batched_tokens=4096 on P
max_concurrency=1
output_tokens=1
```

This result is a technical profile point, not a statistically complete claim.

### 9.2 Minimum credible matrix

TTFT points:

```text
hit: 0%, 75%
P max_num_batched_tokens: 2048, 4096
max_concurrency: 1
output_tokens: 1
```

Decode points:

```text
hit: 75%
P max_num_batched_tokens: 4096
max_concurrency: 1, 4
output_tokens: 128
```

These six points are the first acceptance milestone.

### 9.3 Selected full pilot

After the minimum matrix passes:

- hit main effect: 0%, 25%, 50%, 75%, 85%, 90% at one fixed chunk budget,
  length, and concurrency;
- chunk main effect: 1024, 2048, 4096, 8192 at hit 75% and concurrency 1;
- selected chunk-by-hit interaction: hit 0%, 75%, and 90% only;
- concurrency main effect: 1, 2, 4, and 8 at hit 75% and chunk budget 4096;
- selected concurrency-by-hit interaction: hit 0%, 75%, and 90% only;
- input/output lengths: a small deterministic selection from the prepared DS4
  rows, not a deep DS4 distribution analysis.

Unsupported or out-of-memory points remain recorded as such. Parameters are not
silently reduced to obtain a passing result.

## 10. Metrics and statistics

### 10.1 Primary reported metrics

- 1P1D TTFT: p50 and p90;
- request-level TPOT: p50 and p90 for outputs longer than one token;
- output-token throughput;
- actual P local cache-hit ratio.

### 10.2 Diagnostic metrics

- ITL p99;
- P prompt tokens by source: local compute and local cache hit;
- D prompt tokens from external KV transfer;
- NIXL failed transfers and expired requests;
- configured `max_num_batched_tokens` and `max_concurrency`;
- completed and failed request counts.

Per-step scheduler traces, inferred P-only latency, and transfer microbenchmark
statistics are not required for the main result.

### 10.3 Repetition policy

- engine/CUDA Graph warmup occurs before measured repetitions;
- prefix preparation occurs before every measured repetition;
- each point uses three independent measured runs initially;
- each run uses enough prepared requests for request-level percentiles, with a
  default target of 20 and an allowed range of 20-50;
- report request-level percentiles within runs and mean/CV across run summaries;
- CV above 5% is marked noisy; only noisy or failed points are selectively rerun;
- profiler-enabled runs are stored separately and excluded from the statistics.

## 11. Four implementation tickets

### Ticket 1 — Minimal DS4 dataset adapter

Build:

- consume the existing pinned manifest and verify file hashes;
- select assistant turns deterministically;
- render Qwen3.5 completion prompts;
- compute input/output token counts;
- emit CustomDataset JSONL and a small provenance sidecar;
- reuse existing network-free DS4 fixtures.

Do not build trajectory analytics, Parquet schemas, workload planners, or token
mapping.

Acceptance:

- the same input/config produces byte-identical outputs;
- fixture prompt token IDs match recorded input lengths;
- every row has a positive output length and traceable source identity;
- invalid revision, hash, source format, or tokenizer configuration fails
  closed.

### Ticket 2 — Qwen3.5 1P1D feasibility and fixed launcher

Build:

- a thin fixed-topology P/D/proxy launcher based on official NIXL patterns;
- GPU and CPU/NUMA binding;
- readiness, log capture, bounded timeout, and child-process cleanup;
- one cold and one repeated-prefix smoke;
- provenance for model/vLLM/NIXL/CUDA/topology configuration.

Do not build a general topology manager, connector interface, dynamic routing,
or multi-P/multi-D support.

Acceptance:

- Qwen3.5-4B BF16 completes a real P-to-D request on the target machine;
- cold and repeated deterministic requests return identical greedy output;
- D external-transfer tokens and successful NIXL transfer metrics increase;
- no hang, OOM, compatibility-hash error, failed transfer, or silent fallback;
- if 4B fails, a temporary Qwen3.5-0.8B diagnostic distinguishes environment
  failure from 4B/HMA failure but does not replace the 4B acceptance result.

### Ticket 3 — Controlled serving-metric MVP

Build:

- explicit point JSON;
- isolation-block and block-aligned warm-prefix preparation;
- reset-P/reset-D, warm-P, reset-D, measure protocol;
- metrics before/after collection and validation;
- official `vllm bench serve` invocation;
- the six-point minimum credible matrix;
- raw result preservation.

Do not build a generic sweep engine, database, custom latency calculator,
profiler pipeline, or per-step scheduler instrumentation.

Acceptance:

- the first 75%/4096/concurrency-1 point produces complete artifacts;
- all six minimum points finish or retain an explicit failure record;
- measured P hit ratios agree with aligned planned ratios within one cache block;
- 0% points do not show unintended local prefix reuse;
- TPOT is omitted rather than reported as zero for one-token TTFT points;
- every point is reproducible from its frozen inputs and command.

### Ticket 4 — Selected measurements, report, and replacement acceptance

Build:

- the selected pilot matrix from section 9.3;
- three-run aggregation, CV, and noisy-point labeling;
- `summary.csv`, a concise Markdown report, and only the plots needed to answer
  the profile questions;
- a comparison of default optimized mode with one eager diagnostic point only
  if needed to explain a result;
- replacement acceptance and the old-path retirement checklist.

Do not build an automatic experiment optimizer, arbitrary Cartesian planner,
dashboard, production traffic model, or publication pipeline.

Acceptance:

- report claims are backed by raw official benchmark JSON and metrics deltas;
- plots separate variables and label unsupported/noisy points;
- DS4 is described only as the input dataset;
- the report states hardware, model, revision, precision, topology, cache mode,
  and experiment limitations;
- after human review, the new path becomes the only documented future workflow.

## 12. Testing strategy

### 12.1 Local CPU tests

Test observable contracts only:

- manifest/hash/source validation;
- deterministic DS4 row selection;
- Qwen prompt rendering and token lengths on pinned fixtures;
- isolation-block encode/decode stability;
- block alignment and duplicate-point rejection;
- explicit point parsing without hidden Cartesian expansion;
- controlled protocol ordering using fake HTTP endpoints;
- reset failure, metrics absence, benchmark failure, and partial-artifact
  retention;
- official result parsing and summary calculations.

Do not mock `GPUModelRunner`, construct `SchedulerOutput`, or test private vLLM
cache-manager behavior.

### 12.2 Hardware smoke

The hardware smoke is a functional gate, not a performance threshold:

- exactly two exclusive RTX 3090 GPUs are visible;
- recorded topology and NUMA placement match the assigned roles;
- P, D, and proxy become healthy;
- Qwen3.5-4B BF16 cold request succeeds;
- prefix-cache request succeeds;
- P local-hit and D external-transfer evidence is nonzero where expected;
- output is deterministic for the smoke seed/settings;
- no failed NIXL transfers or compatibility mismatch occurs.

### 12.3 Profile acceptance

The six-point minimum matrix must produce:

- three measured runs per point or a retained explicit failure;
- request counts sufficient for the requested percentiles;
- official detailed benchmark results;
- P/D metrics before and after;
- actual hit calculation;
- effective server/client configuration;
- logs and provenance;
- a summary that does not invent missing values.

Timing tests never assert machine-specific absolute latency in CPU or CI tests.

## 13. Run artifacts

Use a simple filesystem contract:

```text
<run-dir>/
  run-manifest.json
  prepared/
    dataset.jsonl
    rows.jsonl
    provenance.json
  plan.json
  server/
    p.log
    d.log
    proxy.log
    topology.txt
  points/
    <point-id>/
      point.json
      run-01/
        bench-result.json
        p-metrics-before.txt
        p-metrics-after.txt
        d-metrics-before.txt
        d-metrics-after.txt
        derived.json
      run-02/
      run-03/
  summary.csv
  report.md
  plots/
```

Large model/data caches and run artifacts stay outside Git. The repository keeps
only fixtures, scripts, configuration examples, tests, and concise evidence
summaries.

## 14. Failure and stopping rules

- Reset failure: retry a bounded number of times after waiting for idle; then
  fail the repetition and retain logs.
- NIXL failure or silent fallback: invalidate the point.
- Model OOM: record unsupported; do not lower lengths or memory settings under
  the same point ID.
- Actual hit mismatch greater than one block: invalidate and diagnose before
  continuing the matrix.
- Noisy point: mark it and selectively rerun; do not rerun every clean point.
- Qwen3.5-4B PD smoke cannot pass after environment and 0.8B diagnostics: stop
  and review the model choice rather than implementing the remaining tickets.
- The minimum six-point matrix does not show interpretable differences: stop
  before building the selected full pilot and review experiment construction.

## 15. Explicit non-goals

- LRU, eviction policy, capacity curves, reuse distance, or cache-manager replay;
- natural DS4 hit-rate analysis or full trajectory replay;
- tool execution, tool correctness, SWE-bench score, or model-quality eval;
- teacher-forced decode or sampled-token injection;
- operator/kernel microbenchmarks;
- always-on Torch/Nsight profiling;
- independent P-to-D transfer latency/bandwidth sweep;
- P-to-P or D-to-P migration;
- quantized weights, FP8 KV, MTP, speculative decoding, or LoRA;
- production RPS, Poisson traffic, goodput, SLO, or queue-model conclusions;
- multi-P/multi-D scheduling, routing, or autoscaling;
- arbitrary experiment planners, dashboards, or custom storage systems;
- an upstream vLLM PR.

## 16. Implementation order and expected time

Implementation is risk-first:

1. Ticket 2 hardware feasibility slice using official patterns;
2. Ticket 1 minimal adapter while the fixed runtime is stabilized;
3. Ticket 3 first point, then the six-point minimum matrix;
4. Ticket 4 only after the minimum matrix is interpretable.

Assuming cached model access, two exclusive GPUs, and a working NIXL/UCX base:

| Milestone | Expected time |
| --- | ---: |
| Qwen3.5-4B real 1P1D smoke | 0.5-1.5 working days |
| First controlled DS4 point | 2-3 working days total |
| Six-point minimum credible matrix | 3-4 working days total |
| Selected pilot and concise report | 5-8 working days total |

NIXL/UCX, cross-NUMA, or Qwen3.5 HMA issues may add 2-5 working days. The
schedule is an estimate, not an acceptance threshold.

## 17. Legacy retirement policy

Before replacement hardware acceptance:

- preserve existing implementation and hardware evidence;
- do not extend the legacy runner/spine;
- mark old planning specifications as superseded;
- keep large artifacts outside Git.

After Ticket 4 and human review:

- delete or move the legacy `gpu_profile.py` and `profile_spine.py` path;
- remove Qwen2.5 execution mapping and teacher-forcing configuration/tests;
- replace legacy README/workflow instructions with the accepted 1P1D path;
- retain historical evidence only in a clearly labeled discarded/archive area;
- maintain one documented future workflow.

## 18. Authority maintenance

Any implementation decision that changes the objective, fixed model/precision,
metric semantics, cache protocol, experiment axes, four-ticket scope, or
acceptance gates must update this document before code changes proceed.

Implementation notes may add commands and troubleshooting details, but they may
not silently redefine the experiment.

## 19. Repository references used by this design

- `docs/benchmarking/cli.md`
- `docs/benchmarking/sweeps.md`
- `docs/configuration/optimization.md`
- `docs/contributing/profiling.md`
- `docs/features/nixl_connector_compatibility.md`
- `docs/features/nixl_connector_usage.md`
- `vllm/benchmarks/serve.py`
- `vllm/benchmarks/datasets/datasets.py`
- `vllm/v1/metrics/stats.py`
- `vllm/v1/metrics/loggers.py`
- `tests/v1/kv_connector/nixl_integration/`
- `benchmarks/auto_tune/auto_tune.sh`
- `benchmarks/ds4_profile/prepare_snapshot.py`
- `benchmarks/ds4_profile/WORKFLOW.md`
