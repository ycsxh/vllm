# Ticket 2 dual-RTX-3090 server handoff

Status: `remote_pending`. No GPU, NIXL, model download, or server process was
run on the developer workstation. This handoff cannot become
`remote_verified` until its evidence is bound to the clean delivery commit.

## Frozen inputs

Before any live action, replace the two unset values below with full immutable
40-character commits. `EXPECTED_COMMIT` is the commit containing this launcher;
`MODEL_REVISION` is the approved Hugging Face commit for both model and
tokenizer. Do not use a tag, branch, `main`, or `latest`.

```bash
set -euo pipefail
export EXPECTED_COMMIT='<DELIVERY_COMMIT_40_HEX>'
export MODEL_REVISION='<QWEN3_5_4B_COMMIT_40_HEX>'
export TOKENIZER_REVISION="$MODEL_REVISION"
export REPO_ROOT='/srv/vllm'
export RUN_DIR='/srv/vllm-runs/ds4-ticket-02'
export HF_HOME='/srv/model-cache/huggingface'
export P_CPUS='<GPU0_LOCAL_CPU_LIST>'
export P_NUMA='<GPU0_NUMA_NODE>'
export D_CPUS='<GPU1_LOCAL_CPU_LIST>'
export D_NUMA='<GPU1_NUMA_NODE>'
test "${#EXPECTED_COMMIT}" -eq 40
test "${#MODEL_REVISION}" -eq 40
test "${#TOKENIZER_REVISION}" -eq 40
test "$P_NUMA" != "$D_NUMA"
```

The fixed runtime is Qwen/Qwen3.5-4B, BF16, TP=1, P=GPU0, D=GPU1,
NIXL pull transfer, HND BF16 cache, Mamba/GDN `align`, block size 128,
prefix caching, chunked prefill, and fail-closed KV loading. The fixed ports are
P 8100, D 8200, proxy 8000, and NIXL side channels 5600/5601.

## Target identity, topology, and rollback preflight

Run from the server checkout. These are read-only checks. Save the output with
the run artifacts; failure leaves the status `remote_failed` and stops the
handoff.

```bash
set -euo pipefail
test "$(pwd -P)" = "$REPO_ROOT"
git remote -v
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
git remote get-url origin | \
  grep -Ex '(git@github.com:|https://github.com/)ycsxh/vllm(\.git)?'
mkdir -p "$RUN_DIR/server"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version \
  --format=csv,noheader | tee "$RUN_DIR/server/gpus.txt"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 2
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | \
  grep -c '^NVIDIA GeForce RTX 3090$')" -eq 2
nvidia-smi topo -m | tee "$RUN_DIR/server/topology.txt"
numactl --hardware | tee "$RUN_DIR/server/numa.txt"
lscpu -p=CPU,NODE | tee "$RUN_DIR/server/cpu-node-map.txt"
for port in 8000 8100 8200 5600 5601; do
  test -z "$(lsof -tiTCP:"$port" -sTCP:LISTEN)"
done
.venv/bin/python -c \
  'import importlib.metadata as m; print("vllm", m.version("vllm")); print("nixl", m.version("nixl")); print("torch", m.version("torch"))' \
  | tee "$RUN_DIR/server/python-packages.txt"
```

Human verification is required before launch: confirm `gpus.txt` lists exactly
the two exclusive RTX 3090s, and confirm `P_CPUS`/`P_NUMA` and
`D_CPUS`/`D_NUMA` match the PCI/NUMA locality shown above. The rollback target
is only the three process groups started by this launcher. `Ctrl-C` and SIGTERM
invoke bounded group cleanup. Before launch, record this recovery procedure:

1. Inspect listeners with `lsof -nP -iTCP:8000 -iTCP:8100 -iTCP:8200
   -sTCP:LISTEN`.
2. For each reported PID, run `ps -o pid,pgid,lstart,args -p <PID>` and verify
   its command is this checkout's P, D, or proxy invocation.
3. Only after that identity check, terminate its process group with
   `kill -TERM -- -<PGID>`, wait up to 30 seconds, then use
   `kill -KILL -- -<PGID>` only for a group that remains.
4. Re-run the listener check and preserve all logs. Never use a broad `pkill`.

## Plan and live run

First generate and review the deterministic plan. Dry-run creates no run
directory and launches nothing.

```bash
set -euo pipefail
.venv/bin/python -m benchmarks.ds4_profile.run_pd \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --prefill-cpus "$P_CPUS" \
  --prefill-numa-node "$P_NUMA" \
  --decode-cpus "$D_CPUS" \
  --decode-numa-node "$D_NUMA" \
  --run-dir "$RUN_DIR" \
  --readiness-timeout 900 \
  --request-timeout 300 \
  --shutdown-timeout 30 \
  --dry-run > /tmp/ds4-ticket-02-plan.json
.venv/bin/python -m json.tool /tmp/ds4-ticket-02-plan.json
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

Review every command, environment value, port, revision, CPU, and NUMA binding.
The live action requires separate operator approval. With the model already in
the external cache, prevent implicit network access and run:

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python -m benchmarks.ds4_profile.run_pd \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --attention-backend FLASH_ATTN \
  --prefill-cpus "$P_CPUS" \
  --prefill-numa-node "$P_NUMA" \
  --decode-cpus "$D_CPUS" \
  --decode-numa-node "$D_NUMA" \
  --run-dir "$RUN_DIR" \
  --readiness-timeout 900 \
  --request-timeout 300 \
  --shutdown-timeout 30 \
  2>&1 | tee "$RUN_DIR/launcher.log"
```

`set -o pipefail` preserves the launcher's nonzero exit status through `tee`.
The launcher writes the exact plan and invocation, provenance, P/D/proxy logs,
three P and D metric snapshots, and cold/repeated responses. It refuses live
execution from a dirty checkout and always attempts bounded process-group
cleanup.

## Gate A evidence and verdict

All checks below are `remote_pending`. A completed request alone is not Gate A.

```bash
set -euo pipefail
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
jq -e '.outputs_identical == true' "$RUN_DIR/smoke-result.json"
jq -e '.cold_response.choices[0].text == \
  .repeated_response.choices[0].text' "$RUN_DIR/smoke-result.json"
jq -e '.compatibility.nixl_load_failure_policy == "fail"' \
  "$RUN_DIR/launch-plan.json"
! grep -Eai 'out of memory|compatib.*mismatch|failed transfer|recomput' \
  "$RUN_DIR/server/p.log" "$RUN_DIR/server/d.log" \
  "$RUN_DIR/server/proxy.log"
grep -F 'Num successful transfers=' "$RUN_DIR/server/p.log" \
  "$RUN_DIR/server/d.log" || true
```

The periodic successful-transfer log line may not be emitted before this short
smoke shuts down, so its absence alone is not a failure. Preserve and inspect
all raw snapshots; do not substitute log inference for a missing counter. The
checked-out revision documents
`vllm:prompt_tokens_by_source_total` and the `vllm:nixl_*` counters. Locate the
actual exported series before calculating deltas:

```bash
set -euo pipefail
grep -HE 'prompt_tokens_by_source_total|vllm:nixl_' \
  "$RUN_DIR"/server/{p,d}-metrics-{before,cold,repeated}.txt
```

The human evidence record must show these semantic comparisons and retain the
exact source lines used:

- P `source="local_cache_hit"`: cold to repeated delta is positive;
- D `source="external_kv_transfer"`: before to cold delta is positive;
- a documented NIXL successful-transfer series or the periodic
  `Num successful transfers` log increases for the cold transfer;
- failed-transfer and failed-notification deltas remain zero;
- expired-request deltas remain zero.

If any required series is absent, ambiguous, reset, or emitted on an unexpected
role, keep the verdict `remote_pending` and diagnose it from the raw snapshots
and logs. Do not guess a zero or move a metric between P and D.

Gate A becomes `remote_verified` only when every command succeeds, the exact
commit and immutable revisions match the saved plan, both greedy outputs are
identical, P repeated-prefix hits and D external-transfer/success counters rise,
all failure counters remain zero, logs show no hang/OOM/compatibility failure or
fallback, and the human reviewer confirms topology/NUMA placement. Otherwise
record `remote_failed`, retain the complete run directory, and stop. A temporary
Qwen3.5-0.8B diagnostic may distinguish environment failure, but can never
satisfy Gate A or replace the fixed 4B configuration.
