# Ticket 1 pinned-tokenizer handoff

Status: `local_contract_verified`, `real_tokenizer_pending`. The network-free
tests validate the adapter contract with a deterministic tokenizer double. A
run against the exact cached Qwen3.5 tokenizer revision is still required
before Gate B is recorded as complete.

## Local verification

The delivery checkout passed the network-free public-seam tests:

```text
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py -q
14 passed

.venv/bin/ruff check benchmarks/ds4_profile/prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py
All checks passed!

.venv/bin/ruff format --check benchmarks/ds4_profile/prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py
2 files already formatted
```

These results validate the adapter contract with network-free tokenizer
doubles. They do not replace the immutable real-tokenizer run below.

## Frozen inputs

Run from a clean checkout of the delivery commit. Replace every placeholder
with an immutable path or 40-character commit; do not use `main`, `latest`, or
an unpinned tokenizer cache entry.

```bash
set -euo pipefail
export EXPECTED_COMMIT='<DELIVERY_COMMIT_40_HEX>'
export MODEL_REVISION='<QWEN3_5_4B_COMMIT_40_HEX>'
export REPO_ROOT='/srv/vllm'
export MANIFEST='<PINNED_DS4_SNAPSHOT>/manifest.json'
export OUTPUT_A='/srv/vllm-runs/ds4-ticket-01-a'
export OUTPUT_B='/srv/vllm-runs/ds4-ticket-01-b'
export HF_HOME='/srv/model-cache/huggingface'
test "${#EXPECTED_COMMIT}" -eq 40
test "${#MODEL_REVISION}" -eq 40
test -f "$MANIFEST"
test ! -e "$OUTPUT_A"
test ! -e "$OUTPUT_B"
test "$(pwd -P)" = "$REPO_ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
```

The manifest remains the source of truth for the DS4 dataset revision, file
paths, and hashes. The adapter verifies every file before loading the tokenizer
and does not modify the snapshot.

## Prepare twice without network access

The exact model/tokenizer revision must already exist under the external Hugging
Face cache. Both runs use the same frozen inputs and separate new directories.

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
for output in "$OUTPUT_A" "$OUTPUT_B"; do
  .venv/bin/python -m benchmarks.ds4_profile.prepare_dataset \
    --manifest "$MANIFEST" \
    --model Qwen/Qwen3.5-4B \
    --tokenizer-revision "$MODEL_REVISION" \
    --output-dir "$output"
done
cmp "$OUTPUT_A/dataset.jsonl" "$OUTPUT_B/dataset.jsonl"
cmp "$OUTPUT_A/rows.jsonl" "$OUTPUT_B/rows.jsonl"
cmp "$OUTPUT_A/provenance.json" "$OUTPUT_B/provenance.json"
```

Any missing cache entry, revision mismatch, source-format mismatch, hash
mismatch, or rendering failure keeps the status `real_tokenizer_pending` or
marks it `remote_failed`; do not retry with a floating revision.

## Gate B evidence

Validate the prompt-only dataset and its sidecars without deriving generation
length from historical DS4 assistant responses.

```bash
set -euo pipefail
jq -e -s 'all(.[]; has("prompt") and (keys == ["prompt"]))' \
  "$OUTPUT_A/dataset.jsonl" >/dev/null
jq -e -s 'all(.[]; has("request_id") and has("source_path") and \
  has("source_sha256") and has("input_tokens") and has("prompt_ids") and \
  (.input_tokens == (.prompt_ids | length)) and \
  (has("output_tokens") | not))' "$OUTPUT_A/rows.jsonl" >/dev/null
jq -e --arg revision "$MODEL_REVISION" \
  '.tokenizer.model == "Qwen/Qwen3.5-4B" and \
   .tokenizer.revision == $revision and \
   .selection.selected_assistant_turns == "all" and \
   .row_count > 0' "$OUTPUT_A/provenance.json" >/dev/null
test "$(wc -l < "$OUTPUT_A/dataset.jsonl")" -eq \
  "$(wc -l < "$OUTPUT_A/rows.jsonl")"
sha256sum "$MANIFEST" "$OUTPUT_A"/*.json "$OUTPUT_A"/*.jsonl
```

Gate B becomes complete only when the checkout is clean at `EXPECTED_COMMIT`,
both executions are byte-identical, provenance names the immutable tokenizer
revision, every source hash passes, and every sidecar input length equals its
recorded prompt-token count. Preserve `OUTPUT_A`, the command transcript, and
the final checksums as the Ticket 1 evidence package.
