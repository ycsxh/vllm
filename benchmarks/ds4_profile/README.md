# DS4 trajectory normalization

The local-development to school-server acceptance process for this DS4
dual-3090 project is defined in [`WORKFLOW.md`](WORKFLOW.md). It also records
the personal-fork-only Git boundary and the inputs and outputs of each stage.

This directory contains the offline ingestion step for the DS4 Agent profile.
It consumes a local snapshot of the third-party
`Yi30/deepseek-v4-swebench-trajectories` Hugging Face repository and never
modifies raw trajectory files. The repository contains trajectories for
SWE-bench Verified tasks, but is not an official SWE-bench publication. It is
registered as a Hugging Face model repository even though its contents are a
trajectory dataset.

## Snapshot contract

Download the complete pinned snapshot first:

```bash
hf download Yi30/deepseek-v4-swebench-trajectories \
  --revision 4da61f3d06b48b6817a62b99e9c47035c8e59787 \
  --include 'data/**/*.traj.json' \
  --local-dir /path/to/snapshot \
  --max-workers 1
```

Do not pass `--repo-type dataset`: this repository is registered as type
`model`. Generate `manifest.json` beside the downloaded `data/` tree:

```bash
.venv/bin/python -m benchmarks.ds4_profile.prepare_snapshot \
  --snapshot-dir /path/to/snapshot \
  --manifest /path/to/snapshot/manifest.json
```

The manifest records the immutable Hugging Face commit, pilot coverage, and a
SHA-256 for every included raw file:

```json
{
  "dataset": {
    "repo_id": "Yi30/deepseek-v4-swebench-trajectories",
    "repo_type": "model",
    "revision": "4da61f3d06b48b6817a62b99e9c47035c8e59787"
  },
  "pilot_coverage": {
    "domains": ["astropy"],
    "reasoning_modes": ["no_think", "think_high"],
    "trajectory_count": 20,
    "unique_task_count": 10
  },
  "files": [
    {
      "path": "data/no_think/astropy__astropy-12907.traj.json",
      "sha256": "<sha256-of-the-raw-file>"
    }
  ]
}
```

The revision must be a full 40-character commit. Files are accepted only from
the `no_think` and `think_high` directories, and must identify both a DS4 model
and mini-swe-agent format. Any hash or source-identity mismatch stops the run.

## Normalize

Install the benchmark dependency through the repository environment and run:

```bash
uv pip install -r benchmarks/ds4_profile/requirements.txt
.venv/bin/python -m benchmarks.ds4_profile.normalize \
  --manifest /path/to/snapshot/manifest.json \
  --output-dir /path/to/normalized \
  --require-complete-pilot
```

The complete-pilot gate requires exactly 20 trajectories covering 10 paired
tasks across `no_think` and `think_high`. The command writes:

- `turns.parquet`: one row per assistant turn, including trajectory identity,
  reasoning mode, turn position, source usage, structured tool calls/results,
  parallel-call counts, source tool duration, and strict tool readiness.
- `provenance.json`: source revision, every raw path/hash, parser/schema
  versions, pilot coverage, and artifact row counts.

The Parquet schema metadata repeats the revision, parser/schema versions, and
pilot coverage so downstream consumers can validate an artifact without its
sidecar. The pinned, network-free contract fixture lives under
`tests/benchmarks/ds4_profile/fixtures/pinned/`.

## Build deterministic workloads

The workload build requires `uv`. On a new server, check for it explicitly:

```bash
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi
```

Install `uv` through the server's package manager where possible. Do not hide a
moving `curl | sh` install inside the workload command. Then create the local
environment and install the pinned CPU-side artifact dependencies:

```bash
uv venv --python 3.12
UV_CACHE_DIR=/tmp/uv-cache uv pip install \
  --python .venv/bin/python \
  -r benchmarks/ds4_profile/requirements.txt
```

Only tokenizer files are required; model weights are not downloaded. Pin both
repositories to immutable commits:

```bash
.venv/bin/hf download deepseek-ai/DeepSeek-V4-Flash \
  tokenizer.json tokenizer_config.json \
  --revision 60d8d70770c6776ff598c94bb586a859a38244f1 \
  --local-dir /path/to/tokenizers/deepseek-v4-flash/60d8d70770c6776ff598c94bb586a859a38244f1

.venv/bin/hf download Qwen/Qwen2.5-Coder-7B-Instruct \
  tokenizer.json tokenizer_config.json vocab.json merges.txt \
  --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 \
  --local-dir /path/to/tokenizers/qwen2.5-coder-7b-instruct/c03e6d358207e414f1eca0bb1891e29f1db0e242
```

Build Ticket 02 from the immutable Ticket 01 manifest, raw snapshot, and
normalized Parquet:

```bash
.venv/bin/python -m benchmarks.ds4_profile.workloads \
  --manifest /path/to/snapshot/manifest.json \
  --normalized-turns /path/to/ticket-01/turns.parquet \
  --ds4-tokenizer /path/to/tokenizers/deepseek-v4-flash \
  --qwen-tokenizer /path/to/tokenizers/qwen2.5-coder-7b-instruct \
  --output-dir /path/to/ticket-02 \
  --block-size 16 \
  --seed 20260715
```

The command writes `rendered_turns.parquet`, `workload_plan.json`, and
`provenance.json`. Full chat histories remain only in the immutable raw
snapshot. Execution token IDs are materialized only for the unique turns used
by exact replays or mixed batches.

## Run the Ticket 04 profile spine

Ticket 04 adds a CPU-testable artifact harness and a hardware-gated path at the
`GPUWorker`/`GPUModelRunner` boundary. It emits versioned raw sample and
aggregate Parquet, frozen configuration, provenance, and a minimal Markdown
result. Raw rows distinguish prompt, context, cached, scheduled, and new-token
counts; a hardware-valid run also records the observed CUDA Graph runtime mode
for every warmup and steady sample. The school-server command and acceptance
procedure are documented in
[`container/README.md`](container/README.md#ticket-04-profile-spine).
The exact local-to-server continuation is recorded in
[`TICKET_04_HANDOFF.md`](TICKET_04_HANDOFF.md).

On a developer workstation, use only result validation, the focused CPU
contract tests, and `profile-spine --print-plan`. Do not load the model or opt
into `DS4_PROFILE_SPINE_GPU_SMOKE`; the real profile and GPU-gated pytest are
school-server acceptance steps.

Validate an existing result without loading a model:

```bash
.venv/bin/python -m benchmarks.ds4_profile.profile_spine validate \
  --result-dir /path/to/profile-spine-result
```

## Replay Ticket 07 KV cache metadata

Ticket 07 replays one complete DS4 prompt sequence through the real CPU-side
`KVCacheManager` metadata path. It hashes prompt token IDs only. It does not
read decode tokens, allocate KV tensors, use a GPU, establish HBM residency,
or measure Prefill/Decode latency. Local development runs focused CPU
contracts; the full planner and container artifact acceptance run on the
school server.

The replay uses the real `Request` → `KVCacheManager.get_computed_blocks` →
`allocate_slots` → `take_events` → `free` path. Native `BlockRemoved` events
are the eviction authority; compulsory, capacity, prefix-mismatch, and
manager-forced-recompute outcomes remain distinct. A successful run is only
**metadata-only validated** and must say **GPU/HBM validated: no**.

See the [Ticket 07 container runbook](container/README.md#ticket-07-cpu-metadata-replay)
for the school-server commands and [the current handoff](TICKET_07_HANDOFF.md)
for the exact accepted and outstanding evidence.

The current full-data planner failed closed: all 20 complete trajectories were
admitted at their minimum usable capacity, but none emitted a native
`BlockRemoved` event. The selection therefore remains `unselected`; Ticket 07
is not metadata-only validated and is not mergeable without an approved design
revision. See the handoff for the retained planning record and checksums.
