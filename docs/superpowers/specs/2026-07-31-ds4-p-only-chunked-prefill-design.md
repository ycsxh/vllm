# DS4 P-only Chunked-Prefill Profile Design

## Status

Approved for implementation on 2026-07-31.

This design adds a P-engine-local chunked-prefill experiment beside the
existing unchunked fixed-batch P/D profile. The existing fixed-batch contracts
remain unchanged.

## Purpose

The profile measures how the batch-wide `max_num_batched_tokens` budget changes
P-engine prefill-completion latency for exact batches of 13,723-token requests
under controlled prefix-cache reuse.

The primary metric is the sum of built-in `elapsed_ms` values for every valid
context iteration belonging to one target batch. It is a P-side TTFT proxy, not
client-observed TTFT. It excludes cache preparation, P-to-D transfer, the D
worker, proxy and network work, client timing, and the one-token completion
phase.

## Frozen experiment

The checked-in main plan contains exactly these 36 explicit points:

- input tokens: 13,723;
- requested cache hit: 0%, 75%, and 90%;
- exact submitted batch size: 1, 2, and 4;
- batch-wide `max_num_batched_tokens`: 512, 1,024, 2,048, and 4,096;
- output tokens: 1 with EOS ignored;
- repetitions: 3;
- one unmeasured warmup target and three measured target batches per
  repetition.

Complete 640-token HMA pages determine the planned cached tokens:

| Requested hit | Planned cached tokens | Aligned hit |
| ---: | ---: | ---: |
| 0% | 0 | 0% |
| 75% | 10,240 | 74.6193% |
| 90% | 12,160 | 88.6104% |

The model and tokenizer remain frozen to Qwen/Qwen3.5-4B BF16 at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The runtime remains frozen to
TP=1, BF16 KV, 128-token blocks, 640-token effective HMA pages, HND KV layout,
Mamba/GDN `align`, prefix-cache retention interval zero, prefix caching
enabled, optimized execution, synchronous scheduling, the accepted attention
backend, and the accepted P GPU/CPU/NUMA placement.

Chunked prefill is enabled, `max_num_seqs` is always 4, and
`max_num_batched_tokens` is the selected batch-wide budget. The budget is never
described as a per-request chunk size.

## Additive module boundary

The new workflow has three modules:

1. `chunked_prefill.py` owns explicit-plan validation, deterministic requests,
   engine grouping, cache preparation, target validation, raw artifacts,
   statistics, failure retention, and the run CLI.
2. `chunked_prefill_runtime.py` owns the persistent subprocess client used for
   one token-budget engine group. The worker continues to use the public
   `OfflineLLMRuntime` adapter.
3. `chunked_prefill_report.py` re-audits raw observations and frozen provenance,
   then writes CSV, Markdown, and SVG outputs.

The implementation may reuse stable data records and pure statistical helpers
from the fixed-batch workflow. It does not add a chunked mode to
`fixed_batch.py`, change its plan schema, relax its single-iteration P rule, or
change its D behavior.

No code calls a private scheduler, model runner, cache manager, or engine-core
method. The only measurement boundary is public `LLM.generate`, public
`LLM.reset_prefix_cache`, returned `num_cached_tokens`, and built-in
iteration-detail logging.

## Plan and engine grouping

The plan lists every point directly. Loading a plan never expands a Cartesian
product. Point IDs are unique and each point carries its selected batch-wide
token budget.

The main-plan contract verifies the exact 36-point set. The smoke plan contains
only:

- B=1, requested hit=75%, budget=512; and
- B=4, requested hit=0%, budget=512.

Points are grouped by token budget while preserving their explicit order.
Production main execution launches exactly four engines, one for each budget,
and runs all nine matching points before closing that engine. The resolved plan
records both the point order and the four engine groups.

An engine configuration is shared by a group and retained once under the
group's artifact directory. Each point retains an engine reference and the
exact effective configuration used for its group.

## Deterministic requests and cache preparation

Every point constructs exactly B complete prompts. Prompt token IDs are stable
for the execution seed. Each request has a request-unique first 128-token block,
preventing requests in one target batch from populating cache entries for one
another.

For positive-hit requests, the warm prompt is the planned prefix plus one
token. Each sample, including the unmeasured warmup target, follows this
sequence:

1. wait for the synchronous engine to be idle;
2. reset the public prefix cache and require success;
3. generate each request's warm prompt separately when the hit is positive;
4. retain every cache-warm observation as setup evidence;
5. wait for idle;
6. submit exactly B complete prompts in one synchronous `LLM.generate` call;
7. request exactly one output token with EOS ignored;
8. validate and retain the target observation before advancing.

The 0% point skips warm generation but keeps prefix caching enabled and still
resets before every sample.

The unmeasured target uses the same preconditions and validation as measured
targets. It is retained under the run directory but never contributes to
statistics.

## Target validation and primary sample

A target is valid only when all of these observable conditions hold:

- exactly B requests are returned;
- every returned prompt length is exactly 13,723;
- every request returns exactly one output token;
- every request reports the exact planned cached-token count;
- at least one iteration-detail record exists;
- every retained target iteration is context-only, with positive context work,
  zero generation requests, and zero generation tokens;
- no iteration's `context_tokens` exceeds the configured batch-wide token
  budget;
- context request counts are positive and do not exceed B;
- total context tokens equal
  `B * (13,723 - planned_cached_tokens)`;
- every iteration latency is positive and finite.

The token-total equality rejects missing, duplicated, incomplete, and
preempted/recomputed context work. Unexpected generation work rejects the
sample.

For a valid target:

```text
prefill_completion_latency_ms =
  sum(context_iteration.elapsed_ms)

computed_token_throughput_per_s =
  total_context_tokens / (prefill_completion_latency_ms / 1000)
```

Latency is never divided by B. The runtime `LLM.generate` wall time remains a
diagnostic field only. The raw artifact also retains the context-iteration
count and every iteration's complete composition.

## Repetitions and statistics

Each primary point has three repetitions. Each repetition contains one
validated unmeasured warmup target and three validated measured target
batches. This gives three primary samples per run and nine per point.

For each run, the workflow reports p50 and p90 for:

- P-side prefill-completion latency;
- context-iteration count;
- computed-token throughput.

The point summary reports:

- the three run-p50 values;
- mean and sample CV of the run-p50 values;
- a noisy label when latency run-p50 CV exceeds 5%;
- global nine-sample p50 and p90 as diagnostic evidence.

Stored summaries are convenient outputs, not audit authorities.

## Artifact contract

One run uses this layout:

```text
<results>/
  resolved-plan.json
  execution-order.json
  status.json
  engines/
    budget-0512/
      engine-config.json
      provenance.json
      runtime-invocation.json
      runtime-stderr.log
      runtime-stdout.log
      iteration-details.log
  points/
    <point-id>/
      point.json
      requests.json
      engine-reference.json
      point-summary.json
      status.json
      run-01/
        warmup-01.json
        measured-01.json
        measured-02.json
        measured-03.json
        run-summary.json
      run-02/
      run-03/
```

Each sample artifact contains:

- cache-warm observations;
- target wall time;
- returned request IDs, prompt lengths, cached-token counts, and output IDs;
- every target iteration;
- the derived primary sample for valid targets.

Failure artifacts are written before a point failure is surfaced. A new
hardware attempt always uses a new results directory.

When a sample fails after setup starts, its active run also retains
`partial-sample.json`: all completed cache-warm observations, the precise
failure stage, and any invalid warm or target observation. Report audit binds
that partial artifact to the point's phase, batch, completed repetitions, and
exact preceding run/sample topology.

## Failure policy

Cache reset failure, cache mismatch, invalid iteration composition, unexpected
generation work, nonpositive timing, incomplete work, preemption/recompute,
or an unrecognized runtime error fails the point.

A recognized CUDA OOM or verified vLLM cache/capacity limitation is retained as
`unsupported`. Unsupported, failed, and noisy points remain visible in the
status manifest and reports. No retry changes input length, batch size, hit
condition, or token budget under the same point ID.

If a runtime failure makes a shared engine unusable, remaining points in that
engine group receive explicit retained failures. A later retry starts a new
artifact directory; it never overwrites the original evidence.

## Report re-audit

The report builder accepts the checked-in plan, results directory, report
directory, and frozen hardware description. Before producing output it:

- verifies the resolved plan and execution order;
- verifies exactly one engine configuration per selected token budget;
- verifies model, revisions, clean-tree state, public API boundary, physical
  GPU, CPU/NUMA placement, runtime environment, and engine invocation;
- reparses every measured raw observation;
- reruns all token, cache, iteration, budget, and output validation;
- recomputes every run and point statistic;
- rejects inconsistent stored summaries, statuses, engine references, or
  provenance.

The report contains:

- `summary.csv`, one row per explicit point;
- `report.md`, including frozen scope, exclusions, audited status table, and
  metric interpretation;
- `p-prefill-completion-latency.svg`;
- `p-context-iterations.svg`;
- `p-computed-token-throughput.svg`.

Each plot uses batch-wide token budget on the x-axis, with distinct series for
every batch-size and requested-hit combination. Noisy points are marked and
unsupported/failed points are annotated.

The report explicitly distinguishes the P-engine-local proxy from historical
client-observed 1P1D TTFT.

## Behavioral test seams

Tests use public runner/report boundaries and an injected fake runtime.

### Plan seam

The checked-in main plan must contain exactly the required 36 unique points and
four engine groups. Tests independently assert the permitted input length,
hits, batch sizes, budgets, output length, repetition policy, cached-token
counts, and aligned ratios.

### Observation seam

A worked multi-iteration example verifies that the primary latency is the sum
of context-iteration elapsed times, not wall time, the last iteration, or a
per-request value. Focused negative cases reject wrong request count, prompt or
output length, cached tokens, total context tokens, budget overflow,
generation work, empty work, nonpositive latency, and preempted/recomputed
work.

### Runner seam

An injected runtime verifies four engines for the main grouping, exact
reset/warm/target ordering, exact B-sized submissions, warm evidence retention,
warmup exclusion, nine measured samples per point, deterministic requests, and
partial evidence for failed or unsupported points.

### Runtime seam

The existing public `OfflineLLMRuntime` contract remains covered for
`LLM.generate`, `reset_prefix_cache`, `num_cached_tokens`, supported LLM
arguments, and iteration log parsing. The new grouped subprocess client is
tested for persistent engine reuse and frozen invocation/environment
provenance.

### Report seam

Tests modify stored summaries to prove that raw observations win, then modify
raw accounting or provenance to prove that the report fails closed. Noisy and
unsupported points must appear in CSV, Markdown, and SVG output.

CPU tests assert behavior and formulas only. Hardware timing thresholds are not
placed in pytest.

## Verification and acceptance

Local gates are:

1. focused pytest;
2. Ruff check;
3. Ruff format check;
4. CLI dry-run of smoke and main plans;
5. report generation and raw-artifact audit against deterministic fake
   evidence.

Target gates are:

1. record clean exact commit and frozen environment;
2. run the required two-point smoke on the dual RTX 3090 host;
3. independently audit both smoke observations;
4. only after smoke passes, run all 36 main points;
5. generate and audit CSV, Markdown, raw JSON, and SVG reports;
6. independently check every point's token accounting, cache hit, iterations,
   budget, provenance, and statistics;
7. retain every failure, unsupported result, noisy label, and retry;
8. archive the complete evidence with a SHA-256 inventory;
9. write a target acceptance record with commands, outputs, paths, checksums,
   and requirement-by-requirement PASS/FAIL.

Acceptance requires all implementation, local, smoke, matrix, report, audit,
and evidence requirements to pass. Historical client-observed 1P1D TTFT remains
a separate measurement and is not used as the profile baseline.
