# DS4 fixed-batch profile: target-server handoff

This handoff starts after the local implementation commit is available on the
target server. It covers real-GPU smoke, matrix execution, report generation,
and human acceptance. The local delivery does not claim GPU validation.

[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md#20-additive-fixed-batch-node-profile)
is normative. This file supplies commands and gates without redefining the
experiment.

## 1. Delivery contents

The fixed-batch workflow consists of:

- `fixed_batch.py`: explicit plan validation, deterministic 12,800-token
  requests, P/D orchestration, iteration validation, summaries, and the CLI;
- `fixed_batch_runtime.py`: one isolated public offline `LLM` process per point,
  built-in iteration-detail log capture, cache reset, and optional profiler
  control;
- `fixed_batch_report.py`: raw-artifact re-audit plus CSV, Markdown, and SVG
  report generation;
- `config/fixed-batch-smoke.json`: one P and one D construction smoke;
- `config/fixed-batch-main.json`: the common P B<=8 matrix, formal P B=16 at
  90% hit, and D B<=16 matrix;
- `config/fixed-batch-frontier.json`: conditional P B=16 smoke/formal pairs at
  75% and 0% hit; and
- `config/fixed-batch-diagnostics.json`: separate B=1/B=8/B=16 Torch-profiler
  diagnostics.

The accepted HTTP/NIXL 1P1D serving path is unchanged. Its client-observed TTFT
and TPOT remain separate from the node-level proxies produced here.

## 2. Establish the exact target state

Use the same local model/tokenizer snapshots, attention backend, CUDA runtime,
hybrid-model environment, CPU placement, and NUMA placement as the accepted
Ticket 4 run.

```bash
set -euo pipefail

REPO_ROOT=/path/to/vllm
EXPECTED_COMMIT=<delivery-commit>
MODEL_REVISION=<full-40-character-hugging-face-commit>
TOKENIZER_REVISION=<full-40-character-hugging-face-commit>
P_CPUS=<prefill-local-cpu-list>
P_NUMA=<prefill-numa-node>
D_CPUS=<decode-local-cpu-list>
D_NUMA=<decode-numa-node>
ARTIFACT_ROOT=/path/outside/git/ds4-fixed-batch

cd "$REPO_ROOT"
git fetch origin
git checkout "$EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test -x .venv/bin/python
command -v nvidia-smi
mkdir -p "$ARTIFACT_ROOT"
```

Do not execute from a dirty checkout. Do not write run artifacts inside Git.
Confirm that both model revisions resolve from the immutable local cache before
launching a GPU point.

## 3. CPU contract and static gate

Run the focused contract suite and formatting checks first:

```bash
.venv/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_fixed_batch.py -q

.venv/bin/python -m ruff check \
  benchmarks/ds4_profile/fixed_batch.py \
  benchmarks/ds4_profile/fixed_batch_runtime.py \
  benchmarks/ds4_profile/fixed_batch_report.py \
  tests/benchmarks/ds4_profile/test_fixed_batch.py

.venv/bin/python -m ruff format --check \
  benchmarks/ds4_profile/fixed_batch.py \
  benchmarks/ds4_profile/fixed_batch_runtime.py \
  benchmarks/ds4_profile/fixed_batch_report.py \
  tests/benchmarks/ds4_profile/test_fixed_batch.py
```

Then resolve the smoke without importing vLLM or touching a GPU:

```bash
.venv/bin/python -m benchmarks.ds4_profile.fixed_batch \
  --plan benchmarks/ds4_profile/config/fixed-batch-smoke.json \
  --results-dir "$ARTIFACT_ROOT/unused-smoke-results" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --prefill-cpus "$P_CPUS" \
  --prefill-numa-node "$P_NUMA" \
  --decode-cpus "$D_CPUS" \
  --decode-numa-node "$D_NUMA" \
  --dry-run > "$ARTIFACT_ROOT/smoke-resolved-plan.json"

jq -e '
  .execution.vllm_dirty == false and
  (.points | length) == 2 and
  all(.engine_configs[];
    .model == "Qwen/Qwen3.5-4B" and
    .dtype == "bfloat16" and
    .kv_cache_dtype == "bfloat16" and
    .tensor_parallel_size == 1 and
    .enable_prefix_caching == true and
    .enable_chunked_prefill == false and
    .block_size == 128 and
    .cache_alignment_tokens == 640 and
    .max_model_len >= 12801)
' "$ARTIFACT_ROOT/smoke-resolved-plan.json"

test ! -e "$ARTIFACT_ROOT/unused-smoke-results"
```

If the resolved plan differs, stop. Do not edit a point or lower a resource
limit under the same point ID.

## 4. Real P and D smoke

Run the checked-in two-point smoke:

```bash
SMOKE_RESULTS="$ARTIFACT_ROOT/smoke-$EXPECTED_COMMIT"
test ! -e "$SMOKE_RESULTS"

.venv/bin/python -m benchmarks.ds4_profile.fixed_batch \
  --plan benchmarks/ds4_profile/config/fixed-batch-smoke.json \
  --results-dir "$SMOKE_RESULTS" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --prefill-cpus "$P_CPUS" \
  --prefill-numa-node "$P_NUMA" \
  --decode-cpus "$D_CPUS" \
  --decode-numa-node "$D_NUMA" \
  2>&1 | tee "$ARTIFACT_ROOT/smoke-$EXPECTED_COMMIT.launcher.log"
```

The smoke passes only when:

- top-level status is `valid`;
- P reports exactly 9,600 cached tokens, 3,200 context tokens, and one context
  request in its target iteration;
- D reports one 12,800-token setup iteration followed by 127 pure decode
  iterations;
- every D decode iteration has one generation request and one generated token;
- both requests produce the exact configured output length;
- every engine config records chunked prefill disabled, prefix caching enabled,
  the expected physical GPU, CPU list, and NUMA node; and
- runtime logs contain no OOM, preemption, cache mismatch, or unrecognized
  failure.

Useful audit commands:

```bash
jq . "$SMOKE_RESULTS/status.json"
jq '.observation.requests, .observation.iterations' \
  "$SMOKE_RESULTS/points/p-b1-hit75-smoke/run-01/measured-01.json"
jq '{
  setup: .setup_iteration,
  first_decode: .first_decode_iteration,
  steady_count: (.steady_decode_iterations | length)
}' "$SMOKE_RESULTS/points/d-b1-smoke/run-01/measured-01.json"
```

Do not start the main matrix until both points pass.

## 5. Main matrix

The main plan has 18 explicit points:

- P: B=1/2/4/8 at 0%/75%/90% hit;
- P: formal B=16 at 90% hit; and
- D: B=1/2/4/8/16 with no hit-ratio axis.

```bash
MAIN_RESULTS="$ARTIFACT_ROOT/main-$EXPECTED_COMMIT"
test ! -e "$MAIN_RESULTS"

.venv/bin/python -m benchmarks.ds4_profile.fixed_batch \
  --plan benchmarks/ds4_profile/config/fixed-batch-main.json \
  --results-dir "$MAIN_RESULTS" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --prefill-cpus "$P_CPUS" \
  --prefill-numa-node "$P_NUMA" \
  --decode-cpus "$D_CPUS" \
  --decode-numa-node "$D_NUMA" \
  2>&1 | tee "$ARTIFACT_ROOT/main-$EXPECTED_COMMIT.launcher.log"
```

The runner continues after a recognized capacity/OOM result and retains it as
`unsupported`. An unrecognized failure makes the CLI return nonzero after
retaining partial evidence. Never rerun in place; use a new results directory.

Audit the manifest and every point:

```bash
jq . "$MAIN_RESULTS/status.json"
find "$MAIN_RESULTS/points" -name status.json -print0 \
  | sort -z \
  | xargs -0 -n1 jq '{status, error, reason, completed_repetitions}'
```

For every valid formal point, confirm three `run-summary.json` files, five
warmups and ten measured batches per run, exact cached-token counts for P, exact
iteration composition, p50/p90/p95 summaries, and CV/noisy labels.

## 6. Conditional P B=16 frontier

Run the frontier plan only after the main matrix is retained. It executes a
one-run feasibility smoke before each matching formal point. A failed or
unsupported smoke prevents its formal dependent point from running.

```bash
FRONTIER_RESULTS="$ARTIFACT_ROOT/frontier-$EXPECTED_COMMIT"
test ! -e "$FRONTIER_RESULTS"

.venv/bin/python -m benchmarks.ds4_profile.fixed_batch \
  --plan benchmarks/ds4_profile/config/fixed-batch-frontier.json \
  --results-dir "$FRONTIER_RESULTS" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --prefill-cpus "$P_CPUS" \
  --prefill-numa-node "$P_NUMA" \
  --decode-cpus "$D_CPUS" \
  --decode-numa-node "$D_NUMA"
```

Retain unsupported points. Do not enable chunking, lower input length, lower B,
or change memory limits while preserving the point ID.

## 7. Report re-audit

Generate the core report from the main raw artifacts:

```bash
MAIN_REPORT="$ARTIFACT_ROOT/main-report-$EXPECTED_COMMIT"
test ! -e "$MAIN_REPORT"

.venv/bin/python -m benchmarks.ds4_profile.fixed_batch_report \
  --plan benchmarks/ds4_profile/config/fixed-batch-main.json \
  --results-dir "$MAIN_RESULTS" \
  --report-dir "$MAIN_REPORT" \
  --gpu-count 2 \
  --gpu-model "NVIDIA GeForce RTX 3090" \
  --topology "1P1D TP=1"
```

The report builder recomputes every metric from each measured raw observation;
it does not trust stored point summaries. It also cross-checks the resolved
plan, engine configurations, runtime invocations, revisions, GPU model, and
runtime provenance. The CSV and Markdown retain the first pure-decode step
separately from the 126-step steady distribution. Review `summary.csv`,
`report.md`, and all four SVG plots. Keep frontier results as a clearly labeled
supplemental table/curve until they are deliberately combined into a later
approved report.

## 8. Optional diagnostics

Diagnostics are never primary statistics. After reviewing the main statuses,
run only diagnostic points whose matching main B is supported. The checked-in
plan contains representative P and D B=1/B=8/B=16 points. If either B=16 main
point is unsupported, create an explicitly reviewed filtered diagnostic plan
outside Git rather than profiling that unsupported condition.

The D profiler is configured with one delayed iteration so the setup/prefill
iteration is excluded; it then captures at most 127 pure decode iterations.
The P profiler captures one target prefill iteration. Trace-enabled latency
must not be copied into the main report.

## 9. Existing 1P1D serving validation

After the curves are reviewed, choose a small number of representative
supported conditions and validate the corresponding high-level behavior
through the already accepted real 1P1D serving path. Keep its official
client-observed TTFT and request-level TPOT names. Do not relabel a node proxy
as a serving metric or alter the accepted serving runner to force an offline
batch.

## 10. Acceptance record

Before accepting the hardware result, record:

- exact delivery commit and clean-tree evidence;
- full model and tokenizer revisions;
- GPU topology, CPU/NUMA placement, CUDA/driver/PyTorch/vLLM versions;
- smoke and matrix result directories plus SHA-256 inventory;
- valid, unsupported, failed, and noisy point counts;
- confirmation that every P target is one context iteration;
- confirmation that every D target retains 127 pure decode iterations and 126
  steady samples;
- confirmation that profiler traces are excluded from primary statistics;
- report re-audit command and outputs; and
- selected 1P1D serving validation results, if run.

Only the target-server record can change the implementation status from
locally verified to hardware validated and accepted.
