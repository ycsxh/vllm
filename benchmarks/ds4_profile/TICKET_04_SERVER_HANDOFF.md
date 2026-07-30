# Ticket 4 selected-pilot server handoff

Status: `local_ready`, Gate D `remote_pending`.

Tickets 1–3 are `remote_verified`. Ticket 4 local development is complete, but
no local command in this handoff claims dual-GPU execution, performance
validity, report acceptance, or legacy retirement.

## Local delivery

The delivery adds:

- `config/selected-pilot-points.json`, a version 2 plan with 30 explicit
  optimized-mode points and eight report comparisons;
- the selected hit/chunk and concurrency/hit interactions, with overlapping
  main-effect points stored only once;
- deterministic input cut points 0, 17, and 35 from
  `data/no_think/astropy__astropy-13236.traj.json`;
- explicit output lengths 1, 32, and 128;
- ITL p99 aggregation in addition to TTFT/TPOT p50, p90, and p95;
- fail-closed retention of recognized CUDA OOMs as `unsupported`;
- support for at most one separately labeled `eager_diagnostic` point; and
- `report_results.py`, which re-derives every valid result from official
  detailed benchmark JSON and P/D metric snapshots before writing
  `summary.csv`, `report.md`, and SVG plots.

The checked-in main plan contains no eager point. It does not create a hidden
Cartesian product, automatic optimizer, dashboard, production traffic model,
or publication pipeline.

## Local verification

The replacement-path CPU suite passed:

```text
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py \
  tests/benchmarks/ds4_profile/test_report_results.py -q

88 passed
```

The repository's ruff check, ruff format, typos, Markdown lint, and manual
mypy-3.12 hooks passed for the changed files. The aggregate pre-commit command
could not finish installing the unrelated actionlint environment because the
Go module proxy timed out repeatedly; all hooks applicable to these changed
Python, JSON, and Markdown files were run directly.

The complete legacy-inclusive `tests/benchmarks/ds4_profile` directory was also
attempted. Collection stopped at historical `test_profile_spine.py` because
this local environment has no `torch` installation. Do not present that as a
test failure or a pass; rerun the complete directory in the target environment.

## Target checkout and frozen inputs

Use a new target-server worktree at the exact delivered commit. Reuse the
accepted Ticket 1 prepared dataset and the exact Ticket 2/3 CUDA, model cache,
CPU, NUMA, and environment values. Never reuse or overwrite a prior results
directory.

```bash
set -euo pipefail
export EXPECTED_COMMIT='<40-character Ticket 4 delivery commit>'
export MODEL_REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
export TOKENIZER_REVISION="$MODEL_REVISION"
export REPO_ROOT='/home/lyc/vllm/.worktrees/ticket-04'
export PREPARED_DIR='/home/lyc/ds4-storage/runs/ds4-ticket-01-4415bbe8f-a'
export RESULTS_DIR='/home/lyc/ds4-storage/runs/ds4-ticket-04-<commit>-attempt-01'
export PLAN='benchmarks/ds4_profile/config/selected-pilot-points.json'
export REPORT_METADATA='benchmarks/ds4_profile/config/selected-pilot-report-metadata.json'
export HF_HOME='/home/lyc/ds4-storage/model-cache'
export HF_HUB_CACHE="$HF_HOME/huggingface"
export CUDA_HOME='<accepted CUDA 13 root from Ticket 2/3>'
export CUDA_COMPAT_DIR='/home/lyc/ds4-storage/runtime/ticket-04-cuda-link-compat'
export P_CPUS='<accepted GPU 0 local CPU list>'
export P_NUMA='<accepted GPU 0 NUMA node>'
export D_CPUS='<accepted GPU 1 local CPU list>'
export D_NUMA='<accepted GPU 1 NUMA node>'
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

Recreate or verify the accepted unversioned CUDA compatibility link outside
`RESULTS_DIR` exactly as in the Ticket 3 handoff. Repeat its read-only GPU,
topology, listener, package, and rollback preflight. Exactly two idle RTX 3090
GPUs and no listener on ports 8000, 8100, 8200, 5600, or 5601 are mandatory.
Capture `nvidia-smi` GPU identity/topology and `numactl --hardware`; those files
are the evidence for the report metadata's hardware statement.

## Target-side CPU checks

```bash
set -euo pipefail
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py \
  tests/benchmarks/ds4_profile/test_report_results.py -q
.venv/bin/ruff check \
  benchmarks/ds4_profile/prepare_dataset.py \
  benchmarks/ds4_profile/run_pd.py \
  benchmarks/ds4_profile/pd_proxy.py \
  benchmarks/ds4_profile/run_points.py \
  benchmarks/ds4_profile/report_results.py \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py \
  tests/benchmarks/ds4_profile/test_report_results.py
.venv/bin/ruff format --check \
  benchmarks/ds4_profile/prepare_dataset.py \
  benchmarks/ds4_profile/run_pd.py \
  benchmarks/ds4_profile/pd_proxy.py \
  benchmarks/ds4_profile/run_points.py \
  benchmarks/ds4_profile/report_results.py \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py \
  tests/benchmarks/ds4_profile/test_run_points.py \
  tests/benchmarks/ds4_profile/test_report_results.py
```

These commands validate CPU-visible contracts only.

## Required real-tokenizer dry run

The dry run is a hard gate. It loads the immutable cached tokenizer offline,
resolves all 30 source rows, verifies the 128/640-token cache geometry, rejects
aligned duplicate conditions, and serializes every point-specific server and
benchmark command without launching a process.

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
  --dry-run > /tmp/ds4-ticket-04-plan.json
jq -e '.schema_version == 1 and (.points | length) == 30' \
  /tmp/ds4-ticket-04-plan.json
jq -e 'all(.points[];
  .server_plan.compatibility.enforce_eager == false)' \
  /tmp/ds4-ticket-04-plan.json
jq -e 'all(.points[] | select(.id | contains("hit-0") | not);
  .planned_cached_tokens > 0)' /tmp/ds4-ticket-04-plan.json
jq -e '.report.comparisons | length == 8' "$PLAN"
test ! -e "$RESULTS_DIR"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

Human-review all 30 resolved point summaries before live execution. Confirm:

- assistant cut points 0, 17, and 35 exist and have strictly increasing actual
  Qwen3.5 input lengths;
- all six requested hit ratios at the long cut point resolve to distinct
  640-token-aligned prefixes;
- every nonzero hit point spans at least one effective page;
- the 1024/2048/4096/8192 chunk grid, 1/2/4/8 concurrency grid, and 1/32/128
  output lengths match the checked-in plan;
- no main point is eager; and
- all P/D commands preserve the accepted BF16, HND, Mamba `align`, NIXL,
  prefix-cache, chunked-prefill, CPU, NUMA, and revision settings.

If the dry run rejects a collapsed hit condition, stop. Do not weaken the
duplicate check or silently change a source row; update the authoritative plan
and repeat local review first.

## Live selected pilot

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

The first point remains a hard construction/environment gate. Later recognized
CUDA OOMs are retained as `unsupported` and the matrix continues. Any other
point failure makes the runner return nonzero while preserving all completed
and partial artifacts. Do not delete, edit, or rerun in place.

The final manifest must be `valid` or `complete_with_unsupported`. Every valid
point must contain three official detailed results, three independently derived
results, and one point summary. A later unsupported point must have
`point-failure.json` and `status.json` with the exact OOM reason.

## Audited report

First verify the checked-in hardware metadata against the captured preflight.
If the target hardware differs, do not edit the result directory silently;
prepare a reviewed metadata file beside the evidence and pass that exact path.

```bash
set -euo pipefail
.venv/bin/python -m benchmarks.ds4_profile.report_results \
  --results-dir "$RESULTS_DIR" \
  --metadata "$REPORT_METADATA"
test -f "$RESULTS_DIR/summary.csv"
test -f "$RESULTS_DIR/report.md"
test "$(find "$RESULTS_DIR/plots" -maxdepth 1 -name '*.svg' | wc -l)" -eq 8
test "$(wc -l < "$RESULTS_DIR/summary.csv")" -eq 31
grep -F 'DS4 supplies input prompts only' "$RESULTS_DIR/report.md"
grep -F 'Gate D human acceptance' "$RESULTS_DIR/report.md"
grep -F 'Legacy retirement checklist' "$RESULTS_DIR/report.md"
```

The report command fails if a preserved `derived.json` or point summary differs
from recomputation against raw official JSON and metrics. This structural audit
does not replace log review, plot review, hardware verification, or human
acceptance.

## Optional eager diagnosis

Do not run eager mode by default. If one optimized result requires diagnosis:

1. choose exactly one optimized point and state the anomaly being tested;
2. create a separate one-point version 2 plan with a new ID and
   `"execution_mode": "eager_diagnostic"`;
3. keep every other point field identical;
4. run it in a new result directory with the same frozen target inputs;
5. compare its audited row only with the matching optimized row; and
6. exclude the eager sample from all main-result means, CVs, and plots.

`load_experiment_plan` rejects more than one eager point. Preserve the
diagnostic plan, raw result, reason, and comparison even if eager mode does not
explain the anomaly.

## Gate D review

The human reviewer must verify:

- all factual report rows trace to official detailed JSON and P/D deltas;
- TTFT is labeled full 1P1D TTFT and TPOT is absent for one-token points;
- p50/p90/p95, mean, sample CV, ITL p99, throughput, and observed hit values
  agree with `summary.csv`;
- every plot isolates its named variable and visibly labels noisy and
  unsupported points;
- D external-transfer evidence is positive for every valid repetition and all
  NIXL failure/expiry deltas are zero;
- hardware, model, immutable revisions, BF16 precision, topology, HND/Mamba
  cache mode, and limitations are accurate;
- DS4 is described only as the input dataset;
- OOM and noisy decisions are retained without parameter substitution; and
- cleanup leaves no GPU process, related survivor, or fixed-port listener.

Seal the plan, manifest, official JSON, metric snapshots, derived files,
summaries, metadata, report, plots, logs, preflight, cleanup evidence, and
checksums. Record the exact commit and acceptance verdict in this handoff or a
follow-up closeout commit.

## Post-acceptance legacy retirement

Do not perform this section until Gate D has explicit human acceptance. Then:

1. remove or archive `gpu_profile.py` and `profile_spine.py`;
2. remove the Qwen2.5 execution mapping, teacher forcing, legacy profile-spine
   configuration, and their old result-contract tests;
3. move retained historical evidence to a clearly labeled discarded/archive
   area;
4. replace legacy README/workflow commands with the accepted Qwen3.5 1P1D
   commands; and
5. verify that one documented future workflow remains.

Run the affected CPU suites and repository checks again after retirement.
Legacy removal is a new reviewed change; Gate D validates the measurement
delivery, not unreviewed deletions made afterward.
