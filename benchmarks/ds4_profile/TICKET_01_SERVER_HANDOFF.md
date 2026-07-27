# Ticket 1 pinned-tokenizer handoff

Status: `remote_verified`. Gate B passed against the cached immutable Qwen3.5
tokenizer revision on delivery commit
`4415bbe8f04c11c5beab7057effe208659f1b91f`. The original failed attempt from
baseline `c27a4fdf8969e2927973197257510c1666f47b64` and the earlier accepted
`faa5b9ef8` evidence remain preserved.

## Local verification

The final delivery checkout passed the combined focused suite:

```text
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_run_pd.py \
  tests/benchmarks/ds4_profile/test_pd_proxy.py -q
49 passed

.venv/bin/ruff check benchmarks/ds4_profile/prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py
All checks passed!

.venv/bin/ruff format --check benchmarks/ds4_profile/prepare_dataset.py \
  tests/benchmarks/ds4_profile/test_prepare_dataset.py
2 files already formatted
```

These results validate the adapter contract with network-free tokenizer
doubles. They do not replace the immutable real-tokenizer run below.

## Completed Gate B evidence

The verified run used:

```text
delivery commit:
  4415bbe8f04c11c5beab7057effe208659f1b91f
model and tokenizer:
  Qwen/Qwen3.5-4B
model/tokenizer revision:
  851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
dataset revision:
  4da61f3d06b48b6817a62b99e9c47035c8e59787
manifest:
  /home/lyc/ds4-storage/snapshot/4da61f3d06b48b6817a62b99e9c47035c8e59787/manifest.json
output A:
  /home/lyc/ds4-storage/runs/ds4-ticket-01-4415bbe8f-a
output B:
  /home/lyc/ds4-storage/runs/ds4-ticket-01-4415bbe8f-b
transcript and checksums:
  /home/lyc/ds4-storage/runs/ds4-ticket-01-4415bbe8f-evidence
```

Both preparations ran with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. The three artifact pairs were byte-identical, all
handoff `jq` assertions passed, all 667 dataset rows had matching
`input_tokens` and `prompt_ids` lengths, and no row carried `output_tokens`.
The final outputs were also byte-identical to the earlier accepted
`faa5b9ef8` outputs. The evidence manifest `evidence-checksums.txt` has
SHA-256:

```text
ede34a26e8e289569fa70db61534f6d8509ade1be608df2dc437d249b54ad9aa
```

The retained checksums are:

```text
c228d3f670b20185a3bde4be08a67b73877179795f1b899ae4f6a23d64371c6a  manifest.json
18be24424d424fddc7212b15cb6e855f12775f90e0babdfb2f5afaf5746a5df3  provenance.json
60b716d8f58e8d31f1fba8652a983b16e592c2fdee5a7a8df18b3571a9db01d0  dataset.jsonl
2fda86403091483296cf87981e675eeac0ee485fa9839fed81bc31dafb2a832d  rows.jsonl
```

The first real-tokenizer attempt at baseline `c27a4fdf...` failed closed
because the installed Transformers version did not expose `_commit_hash` on
the loaded tokenizer. Its transcript remains at
`/home/lyc/ds4-storage/runs/ds4-ticket-01-c27a4fdf-evidence`. The accepted
implementation separately resolves the cached `tokenizer_config.json` commit
and rejects a missing or mismatched revision. A second diagnostic exposed DS4
tool-call arguments as JSON strings; the adapter now parses only JSON objects
for Qwen's chat template and rejects malformed or non-object arguments.

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
