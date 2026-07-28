# Ticket 3 controlled-metric server handoff

Status: `local_ready`, Gate C `remote_pending`.

Ticket 1 Gate B and Ticket 2 Gate A are already `remote_verified`. Ticket 3
must continue on the same target dual-RTX-3090 machine. This handoff covers the
first controlled point and the six-point minimum matrix; it does not authorize
Ticket 4.

## Local delivery

The delivery adds:

- `run_points.py`, the explicit controlled-cache orchestrator;
- `config/controlled-mvp-points.json`, the six ordered points;
- immutable benchmark-tokenizer snapshot resolution from the pinned revision;
- point-specific P restarts for `max_num_batched_tokens=2048` or `4096`, while
  D remains fixed at `4096`;
- deterministic request-unique 128-token isolation blocks;
- 640-token warm-prefix alignment from the accepted Qwen3.5 HMA runtime;
- bounded reset and idle checks, official `vllm bench serve`, Prometheus delta
  validation, and partial failure artifacts;
- request-level TTFT and TPOT summaries derived separately from the unmodified
  official detailed JSON;
- point-level means, sample CVs, and CV-over-5% noisy markers across the three
  run summaries;
- network-free CPU tests for the public point, protocol, metric, and CLI
  contracts.

The first point in the checked-in file is the required
75%/4096/concurrency-1/one-token point. The remaining five complete the minimum
matrix.

## Before target execution

Start from a new target-server worktree at the exact Ticket 3 delivery commit.
Do not reuse the accepted Ticket 2 results directory or a dirty checkout.

```bash
set -euo pipefail
export EXPECTED_COMMIT='<40-character Ticket 3 delivery commit>'
export MODEL_REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
export TOKENIZER_REVISION="$MODEL_REVISION"
export REPO_ROOT='/home/lyc/vllm/.worktrees/ticket-03'
export PREPARED_DIR='/home/lyc/ds4-storage/runs/ds4-ticket-01-4415bbe8f-a'
export RESULTS_DIR='/home/lyc/ds4-storage/runs/ds4-ticket-03-<commit>-attempt-01'
export PLAN='benchmarks/ds4_profile/config/controlled-mvp-points.json'
export HF_HOME='/home/lyc/ds4-storage/model-cache'
export HF_HUB_CACHE="$HF_HOME/huggingface"
export CUDA_HOME='<accepted CUDA 13 root from Ticket 2 attempt-08>'
export CUDA_COMPAT_DIR='/home/lyc/ds4-storage/runtime/ticket-03-cuda-link-compat'
export P_CPUS='<GPU 0 local CPU list from Ticket 2 attempt-08>'
export P_NUMA='<GPU 0 NUMA node from Ticket 2 attempt-08>'
export D_CPUS='<GPU 1 local CPU list from Ticket 2 attempt-08>'
export D_NUMA='<GPU 1 NUMA node from Ticket 2 attempt-08>'
export PATH="$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib"
export LIBRARY_PATH="$CUDA_COMPAT_DIR:$CUDA_HOME/lib"
export FLASHINFER_JIT_VERBOSE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'
test "${#EXPECTED_COMMIT}" -eq 40
test "${#MODEL_REVISION}" -eq 40
test "$(pwd -P)" = "$REPO_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test ! -e "$RESULTS_DIR"
test -f "$PREPARED_DIR/dataset.jsonl"
test -f "$PREPARED_DIR/rows.jsonl"
test -f "$PREPARED_DIR/provenance.json"
git remote get-url origin |
  grep -Ex '(git@github.com:|https://github.com/)ycsxh/vllm(\.git)?'
```

Reuse the exact accepted Ticket 2 CUDA, cache, NUMA, and CPU values. Do not
replace them with guessed paths. The CUDA compatibility directory stays
outside `RESULTS_DIR`, because live execution requires a new results path:

```bash
set -euo pipefail
mkdir -p "$CUDA_COMPAT_DIR"
test -e "$CUDA_HOME/lib/libcudart.so.13"
if test -L "$CUDA_COMPAT_DIR/libcudart.so"; then
  ticket3_link_target="$(readlink "$CUDA_COMPAT_DIR/libcudart.so")"
  test "$ticket3_link_target" = "$CUDA_HOME/lib/libcudart.so.13"
else
  test ! -e "$CUDA_COMPAT_DIR/libcudart.so"
  ln -s "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_COMPAT_DIR/libcudart.so"
fi
```

Repeat the Ticket 2 read-only GPU, topology, listener, package, and rollback
preflight. Exactly two idle RTX 3090 GPUs and no listeners on ports 8000, 8100,
8200, 5600, or 5601 are mandatory.

## Local tests on the target checkout

```bash
set -euo pipefail
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py -q
.venv/bin/ruff check \
  benchmarks/ds4_profile/prepare_dataset.py \
  benchmarks/ds4_profile/run_pd.py \
  benchmarks/ds4_profile/pd_proxy.py \
  benchmarks/ds4_profile/run_points.py \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py
.venv/bin/ruff format --check \
  benchmarks/ds4_profile/prepare_dataset.py \
  benchmarks/ds4_profile/run_pd.py \
  benchmarks/ds4_profile/pd_proxy.py \
  benchmarks/ds4_profile/run_points.py \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py
```

These tests do not claim Gate C. They validate only the CPU-visible contracts.

## Required dry run

The dry run loads the pinned tokenizer offline, verifies every prepared prompt,
constructs the isolation and warm-prefix token IDs, rejects aligned duplicate
conditions, and serializes all six server plans. It launches no process and
does not create `RESULTS_DIR`.

```bash
set -euo pipefail
.venv/bin/python -m benchmarks.ds4_profile.run_points \
  --prepared-dir "$PREPARED_DIR" \
  --plan "$PLAN" \
  --results-dir "$RESULTS_DIR" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --prefill-cpus "$P_CPUS" \
  --prefill-numa-node "$P_NUMA" \
  --decode-cpus "$D_CPUS" \
  --decode-numa-node "$D_NUMA" \
  --readiness-timeout 900 \
  --request-timeout 300 \
  --shutdown-timeout 30 \
  --dry-run > /tmp/ds4-ticket-03-plan.json
jq -e '.schema_version == 1 and (.points | length) == 6' \
  /tmp/ds4-ticket-03-plan.json
jq -e '.points[0].id ==
  "ttft-hit-75-chunk-4096-concurrency-1"' \
  /tmp/ds4-ticket-03-plan.json
jq -e 'all(.points[] | select(.id | contains("hit-75"));
  .planned_cached_tokens > 0)' /tmp/ds4-ticket-03-plan.json
test ! -e "$RESULTS_DIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

Human review is required before live execution. Inspect all six P/D/proxy
commands, child environments, revisions, CPU/NUMA bindings, token budgets,
request counts, and planned cached-token totals. In particular, confirm the
75% points align to at least one 640-token effective page.

## Gate C live execution

The runner owns one clean P/D/proxy lifecycle per point. It sends non-measured
canaries, resets both caches, runs three measured repetitions, and performs
bounded process-group cleanup before moving to the next point.

```bash
set -euo pipefail
.venv/bin/python -m benchmarks.ds4_profile.run_points \
  --prepared-dir "$PREPARED_DIR" \
  --plan "$PLAN" \
  --results-dir "$RESULTS_DIR" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --prefill-cpus "$P_CPUS" \
  --prefill-numa-node "$P_NUMA" \
  --decode-cpus "$D_CPUS" \
  --decode-numa-node "$D_NUMA" \
  --readiness-timeout 900 \
  --request-timeout 300 \
  --shutdown-timeout 30 \
  2>&1 | tee "$RESULTS_DIR.launcher.log"
```

`set -o pipefail` preserves a nonzero runner result. Do not create
`RESULTS_DIR` first. The adjacent launcher transcript is intentionally outside
the runner-owned directory.

The required first point is a hard gate: if it fails, the runner retains its
artifacts and does not start the remaining five. After the first point passes,
the runner continues past later retained point failures, returns nonzero if any
point failed, and stops immediately with status 130 after an operator interrupt.
Never delete or overwrite a failed point directory.

## Gate C review

Start with the machine-readable manifest:

```bash
set -euo pipefail
jq . "$RESULTS_DIR/run-manifest.json"
jq -e '.status == "valid" and (.points | length) == 6 and
  all(.points[]; .status == "valid" and .completed_repetitions == 3)' \
  "$RESULTS_DIR/run-manifest.json"
test "$(find "$RESULTS_DIR/points" -name bench-result.json | wc -l)" -eq 18
test "$(find "$RESULTS_DIR/points" -name derived.json | wc -l)" -eq 18
test "$(find "$RESULTS_DIR/points" -name point-summary.json | wc -l)" -eq 6
```

For every repetition, verify:

- `bench-result.json` contains aligned `input_lens`, `output_lens`, `ttfts`,
  `itls`, `start_times`, `generated_texts`, and `errors`;
- `derived.json` is `valid`, D external-transfer tokens are positive, all NIXL
  failure deltas are zero, and the observed P hit passed the one-effective-page
  tolerance;
- requested, 640-token-aligned planned, and observed P hit ratios are all
  present, with the observed denominator covering every P prompt-token source;
- one-token points contain no derived TPOT keys;
- `point-summary.json` reports mean/CV across three runs and marks every metric
  above 5% CV as noisy;
- 0% points have zero P local-cache-hit tokens;
- P, D, and proxy logs contain no OOM, compatibility mismatch, failed transfer,
  silent recompute, or unbounded hang;
- cleanup leaves no GPU compute process or fixed-port listener.

The first point must be reviewed before treating the other five as performance
evidence. Complete artifacts alone are not Gate C: the human reviewer must
confirm that the metric sources and aligned ratios mean what the specification
claims.

## Server-side fixes

If target hardware exposes a source bug, fix it on a new personal-fork branch,
add the smallest regression test, and rerun the focused CPU suite. Use a new
results directory for every changed delivery commit. Rerun only the affected
point after diagnosis, then rerun the complete six-point matrix only when its
construction remains comparable. Preserve all earlier failure directories.

Do not start Ticket 4 until all six points are valid or explicitly retained as
unsupported and a human judges the minimum matrix interpretable.
