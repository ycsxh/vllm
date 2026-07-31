# DS4 P-only chunked-prefill target-server handoff

This procedure runs the additive P-only chunked-prefill profile after its
implementation commit is available on the dual-RTX-3090 target. It does not
change or reinterpret the existing unchunked fixed-batch P/D workflow or
client-observed 1P1D results.

[`AUTHORITATIVE_SPEC.md`](AUTHORITATIVE_SPEC.md#21-additive-p-only-chunked-prefill-profile)
is normative.

## 1. Frozen target

Use:

- model/tokenizer: `Qwen/Qwen3.5-4B`;
- model/tokenizer revision:
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- BF16 weights and KV cache, TP=1;
- FLASH_ATTN, HND KV layout, Mamba/GDN `align`;
- configured block size 128 and effective HMA page size 640;
- prefix-cache retention interval zero;
- P on physical GPU 0, CPUs `0,2,4,6,8,10`, NUMA node 0;
- the accepted target venv, precompiled extensions, model cache, CUDA
  compatibility directory, and driver; and
- physical GPU 1 idle and retained as part of the frozen dual-3090 system.

The main profile launches four P engines, one for each batch-wide token budget.
No D worker, NIXL process, or proxy participates.

## 2. Establish an exact clean checkout

```bash
set -euo pipefail

REPO_ROOT=/path/to/clean/vllm-worktree
EXPECTED_COMMIT=<full-implementation-commit>
MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
TOKENIZER_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
ARTIFACT_ROOT=/home/lyc/ds4-storage/runs
ATTEMPT=01

cd "$REPO_ROOT"
test "$(git remote get-url origin)" = "https://github.com/ycsxh/vllm.git"
test "$(git remote get-url upstream)" = \
  "https://github.com/vllm-project/vllm.git"
git fetch origin
git checkout "$EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git merge-base --is-ancestor \
  1dab44455972017a97366cd6bd645ae014a9db45 HEAD
test -z "$(git status --porcelain)"
test -x .venv/bin/python

export HF_HOME=/home/lyc/ds4-storage/cache
export HF_HUB_CACHE=/home/lyc/ds4-storage/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/home/lyc/ds4-storage/runtime/ticket-03-venv-47438683/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=/home/lyc/ds4-storage/runtime/ticket-04-cuda-link-compat-0762f2afe-a01:$CUDA_HOME/lib
```

Never execute production evidence from a dirty checkout. Never reuse a
results, report, audit, or archive path.

## 3. Hardware and process preflight

```bash
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,driver_version,uuid \
  --format=csv
nvidia-smi topo -m
numactl --hardware
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

The gate requires two NVIDIA GeForce RTX 3090 GPUs, 24,576 MiB each, no
unexpected compute process, and P's documented CPU/NUMA placement.

## 4. CPU contract, lint, and dry-run gates

```bash
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_chunked_prefill.py \
  tests/benchmarks/ds4_profile/test_fixed_batch.py -q

.venv/bin/python -m ruff check \
  benchmarks/ds4_profile/chunked_prefill.py \
  benchmarks/ds4_profile/chunked_prefill_runtime.py \
  benchmarks/ds4_profile/chunked_prefill_report.py \
  benchmarks/ds4_profile/fixed_batch_runtime.py \
  tests/benchmarks/ds4_profile/test_chunked_prefill.py

.venv/bin/python -m ruff format --check \
  benchmarks/ds4_profile/chunked_prefill.py \
  benchmarks/ds4_profile/chunked_prefill_runtime.py \
  benchmarks/ds4_profile/chunked_prefill_report.py \
  benchmarks/ds4_profile/fixed_batch_runtime.py \
  tests/benchmarks/ds4_profile/test_chunked_prefill.py
```

Resolve both plans without importing vLLM or touching a GPU:

```bash
PREFLIGHT="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-preflight-a${ATTEMPT}"
test ! -e "$PREFLIGHT"
mkdir -p "$PREFLIGHT"

.venv/bin/python -m benchmarks.ds4_profile.chunked_prefill \
  --plan benchmarks/ds4_profile/config/chunked-prefill-smoke.json \
  --results-dir "$PREFLIGHT/unused-smoke-results" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --p-gpu 0 \
  --prefill-cpus 0,2,4,6,8,10 \
  --prefill-numa-node 0 \
  --dry-run > "$PREFLIGHT/smoke-resolved-plan.json"

.venv/bin/python -m benchmarks.ds4_profile.chunked_prefill \
  --plan benchmarks/ds4_profile/config/chunked-prefill-main.json \
  --results-dir "$PREFLIGHT/unused-main-results" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --p-gpu 0 \
  --prefill-cpus 0,2,4,6,8,10 \
  --prefill-numa-node 0 \
  --dry-run > "$PREFLIGHT/main-resolved-plan.json"

test ! -e "$PREFLIGHT/unused-smoke-results"
test ! -e "$PREFLIGHT/unused-main-results"

jq -e '
  (.points | length) == 2 and
  (.engine_configs | length) == 1 and
  all(.engine_configs[];
    .enable_chunked_prefill == true and
    .enable_prefix_caching == true and
    .max_num_seqs == 4 and
    .max_num_batched_tokens == 512)
' "$PREFLIGHT/smoke-resolved-plan.json"

jq -e '
  (.points | length) == 36 and
  (.engine_configs | length) == 4 and
  ([.engine_configs[].max_num_batched_tokens] ==
    [512, 1024, 2048, 4096]) and
  all(.engine_configs[];
    .model == "Qwen/Qwen3.5-4B" and
    .dtype == "bfloat16" and
    .kv_cache_dtype == "bfloat16" and
    .tensor_parallel_size == 1 and
    .enable_chunked_prefill == true and
    .enable_prefix_caching == true and
    .max_num_seqs == 4 and
    .max_model_len == 13724 and
    .block_size == 128 and
    .cache_alignment_tokens == 640 and
    .cuda_visible_devices == "0" and
    .cpu_affinity == "0,2,4,6,8,10" and
    .numa_node == 0)
' "$PREFLIGHT/main-resolved-plan.json"
```

Stop if either dry-run differs. Do not edit a checked-in point to pass the
gate.

## 5. Required two-point smoke

Use a new directory for every attempt:

```bash
SMOKE_RESULTS="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-smoke-a${ATTEMPT}"
SMOKE_REPORT="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-smoke-report-a${ATTEMPT}"
SMOKE_LOG="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-smoke-a${ATTEMPT}.launcher.log"
test ! -e "$SMOKE_RESULTS"
test ! -e "$SMOKE_REPORT"
test ! -e "$SMOKE_LOG"

.venv/bin/python -m benchmarks.ds4_profile.chunked_prefill \
  --plan benchmarks/ds4_profile/config/chunked-prefill-smoke.json \
  --results-dir "$SMOKE_RESULTS" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --p-gpu 0 \
  --prefill-cpus 0,2,4,6,8,10 \
  --prefill-numa-node 0 \
  2>&1 | tee "$SMOKE_LOG"

.venv/bin/python -m benchmarks.ds4_profile.chunked_prefill_report \
  --plan benchmarks/ds4_profile/config/chunked-prefill-smoke.json \
  --results-dir "$SMOKE_RESULTS" \
  --report-dir "$SMOKE_REPORT" \
  --gpu-count 2 \
  --gpu-model "NVIDIA GeForce RTX 3090" \
  --topology "P=GPU0/NUMA0 on dual RTX 3090"
```

The smoke passes only when:

- top-level status is `valid`;
- both point statuses are `valid`;
- B=1/requested-75% returns exactly 10,240 cached tokens and 3,483
  total context tokens;
- B=4/requested-0% returns four zero-cache requests and 54,892 total context
  tokens;
- every target context iteration is no larger than 512 tokens;
- every target contains context work only;
- primary latency equals the sum of all retained context-iteration times;
- cache-warm evidence exists only for the positive-hit point and is excluded
  from the measured sample; and
- report re-audit succeeds with no provenance mismatch.

Useful review commands:

```bash
jq . "$SMOKE_RESULTS/status.json"
find "$SMOKE_RESULTS/points" -name status.json -print0 |
  sort -z |
  xargs -0 -n1 jq '{status, error, phase, completed_repetitions}'
sed -n '1,5p' "$SMOKE_REPORT/summary.csv"
```

Do not start the main matrix if either smoke point fails or is unsupported.
Diagnose against the retained attempt and retry only in a new directory.

## 6. Complete 36-point matrix

```bash
MAIN_RESULTS="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-main-a${ATTEMPT}"
MAIN_REPORT="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-main-report-a${ATTEMPT}"
MAIN_LOG="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-main-a${ATTEMPT}.launcher.log"
test ! -e "$MAIN_RESULTS"
test ! -e "$MAIN_REPORT"
test ! -e "$MAIN_LOG"

.venv/bin/python -m benchmarks.ds4_profile.chunked_prefill \
  --plan benchmarks/ds4_profile/config/chunked-prefill-main.json \
  --results-dir "$MAIN_RESULTS" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --p-gpu 0 \
  --prefill-cpus 0,2,4,6,8,10 \
  --prefill-numa-node 0 \
  2>&1 | tee "$MAIN_LOG"

.venv/bin/python -m benchmarks.ds4_profile.chunked_prefill_report \
  --plan benchmarks/ds4_profile/config/chunked-prefill-main.json \
  --results-dir "$MAIN_RESULTS" \
  --report-dir "$MAIN_REPORT" \
  --gpu-count 2 \
  --gpu-model "NVIDIA GeForce RTX 3090" \
  --topology "P=GPU0/NUMA0 on dual RTX 3090"
```

The runner performs exactly four engine launches and retains all 36 statuses.
The report command independently checks all raw measured observations,
deterministic prompts, cache preparation, token totals, iteration budgets,
stored statistics, execution order, runtime environment, and provenance.

Unsupported, noisy, and failed points remain in the CSV, Markdown, and SVG
outputs. Never rerun in place or modify an axis under the same point ID.

## 7. Independent status and artifact audit

```bash
jq . "$MAIN_RESULTS/status.json"
test "$(find "$MAIN_RESULTS/points" -mindepth 1 -maxdepth 1 -type d | wc -l)" = 36
test "$(find "$MAIN_RESULTS/engines" -mindepth 1 -maxdepth 1 -type d | wc -l)" = 4
test "$(find "$MAIN_RESULTS/points" -name 'measured-*.json' | wc -l)" = 324
test "$(find "$MAIN_RESULTS/points" -name 'run-summary.json' | wc -l)" = 108
test "$(wc -l < "$MAIN_REPORT/summary.csv")" = 37
test "$(find "$MAIN_REPORT" -name '*.svg' | wc -l)" = 3

find "$MAIN_RESULTS/points" -name status.json -print0 |
  sort -z |
  xargs -0 -n1 jq '{status, error, phase, completed_repetitions}'

rg -n -i \
  'out of memory|preempt|cache|error|failed|unsupported' \
  "$MAIN_RESULTS" "$MAIN_LOG" > \
  "$MAIN_REPORT/runtime-failure-scan.txt" || true
```

Review all 36 CSV rows and the three SVGs. Confirm that every valid row has
nine global samples, exact actual cached tokens, a run-p50 mean/CV, global p50
and p90, and the expected context-iteration trend. Distinguish the P-engine
proxy from historical client-observed 1P1D TTFT in every conclusion.

## 8. Checksums, archive, and acceptance record

Create new output paths:

```bash
AUDIT_DIR="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-audit-a${ATTEMPT}"
ARCHIVE="$ARTIFACT_ROOT/ds4-chunked-prefill-${EXPECTED_COMMIT:0:10}-evidence-a${ATTEMPT}.tar.zst"
test ! -e "$AUDIT_DIR"
test ! -e "$ARCHIVE"
mkdir -p "$AUDIT_DIR"

(
  cd "$ARTIFACT_ROOT"
  find \
    "$(basename "$PREFLIGHT")" \
    "$(basename "$SMOKE_RESULTS")" \
    "$(basename "$SMOKE_REPORT")" \
    "$(basename "$MAIN_RESULTS")" \
    "$(basename "$MAIN_REPORT")" \
    -type f -print0 |
    sort -z |
    xargs -0 sha256sum
) > "$AUDIT_DIR/evidence-checksums.sha256"

tar --zstd -cf "$ARCHIVE" -C "$ARTIFACT_ROOT" \
  "$(basename "$PREFLIGHT")" \
  "$(basename "$SMOKE_RESULTS")" \
  "$(basename "$SMOKE_REPORT")" \
  "$(basename "$MAIN_RESULTS")" \
  "$(basename "$MAIN_REPORT")" \
  "$(basename "$AUDIT_DIR")"

sha256sum "$AUDIT_DIR/evidence-checksums.sha256" "$ARCHIVE" \
  "$SMOKE_LOG" "$MAIN_LOG" > "$AUDIT_DIR/delivery-checksums.sha256"
```

Write `acceptance.md` in the audit directory with:

- exact clean source commit and baseline-ancestor result;
- origin/upstream URLs and confirmation of no upstream mutation;
- model/tokenizer revision;
- GPU, driver, CUDA, PyTorch, vLLM, CPU, NUMA, and runtime environment;
- every test, lint, dry-run, smoke, matrix, report, and audit command plus
  output;
- absolute evidence/report/archive paths and SHA-256 values;
- a 36-point PASS/unsupported/failed/noisy table;
- token/cache/iteration/budget/provenance/statistics audit verdicts;
- chunk-budget, requested-hit, and exact-B main effects and interactions;
- an explicit distinction from historical client-observed 1P1D TTFT;
- retained failure/retry history and remaining risks; and
- requirement-by-requirement PASS/FAIL with a final acceptance verdict.

Only a complete target record can mark this profile `ACCEPTED`.
