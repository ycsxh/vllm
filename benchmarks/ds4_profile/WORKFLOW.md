# DS4 dual-3090 profile workflow

This workflow applies only to the staged DS4 P/D profiling project under
`benchmarks/ds4_profile/`. It is not a repository-wide vLLM contribution rule.
Its purpose is to keep local development, school-server execution, evidence,
and personal-fork Git history tied to the same source revision.

The words **must** and **must not** below are reserved for result-integrity and
remote-safety gates. Branch names, archive layouts, and example commands are
recommendations that may be adapted to the current ticket.

## Stage overview

| Stage | Primary environment | Inputs | Outputs | Exit gate |
| --- | --- | --- | --- | --- |
| Local development | Developer workstation Codex | Ticket, pinned prior artifacts, personal-fork `main` | Tests, fixture artifacts, feature-branch commit, handoff | Commit is pushed to the personal fork and identified by SHA |
| Hardware acceptance | School-server Codex CLI | Exact feature-branch SHA, persistent caches, two RTX 3090s | GPU artifacts, checksums, acceptance summary | Hardware result passes on the same SHA and clean/dirty state is recorded |
| Integration | Personal fork | Reviewed code and hardware evidence | PR and merge to personal-fork `main` | Human review is complete |

Local fixture success is not hardware validation. A preflight or GPU failure
must remain `skipped` or `invalid`; it must not be rewritten as a passing run.

## Remote boundary

The working remote is the personal fork:

```text
origin    https://github.com/ycsxh/vllm.git
upstream  https://github.com/vllm-project/vllm.git
```

This project may fetch upstream for comparison, but it must not push branches,
open issues or PRs, or otherwise mutate `vllm-project/vllm`. Feature branches,
PRs, and merges described here target `origin` only.

Check the boundary before work that uses a remote:

```bash
git remote -v
git status --short --branch
```

## Stage 1: local development

Start from the personal fork's reviewed baseline and use one feature branch per
ticket. Preserve unrelated working-tree files.

Inputs:

- the ticket and overall DS4 profile specification;
- immutable Ticket 01/02 artifacts and revisions;
- the Ticket 03 container contract and persistent-mount layout; and
- the personal fork's current `main` commit.

Responsibilities:

- design the public CLI and artifact seam before implementation;
- implement and run CPU/static contract tests locally;
- exercise a deterministic fixture through raw samples, aggregates,
  provenance, and Markdown rendering;
- keep hardware-dependent tests gated and make skipped status explicit; and
- document the exact server command and expected result files.

The local stage must stay lightweight. It must not load the Qwen model, set
`DS4_PROFILE_SPINE_GPU_SMOKE=1`, run `profile-spine` without `--print-plan`, or
attempt to reproduce the dual-GPU acceptance path on CPU. Those commands are
reserved for Stage 2. During ordinary iteration, run the focused CPU contract
test or static plan inspection; rerun the broader DS4 suite only when a change
affects its shared contracts.

Recommended checks for Ticket 04 are:

```bash
.venv/bin/python -m pytest \
  --confcutdir=tests/benchmarks/ds4_profile \
  tests/benchmarks/ds4_profile/test_profile_spine.py -v

.venv/bin/python -m benchmarks.ds4_profile.container.runtime \
  profile-spine --print-plan
```

The lightweight local environment may not contain torch. In that case the CPU
contract suite remains valid with `--confcutdir`; GPU execution remains a
school-server responsibility.

Ticket 07 has a separate local/server gate. Locally, exercise the focused real
`Request`/`KVCacheManager` CPU metadata contracts and the smallest relevant
upstream KV cache regressions with both hash environment variables set before
Python starts. Do not treat a fixture-only harness as real-manager evidence.
The complete input planner, selection pin, replay, independent artifact
validation, and checksum capture run on the school server. This remains a
CPU-only metadata acceptance path and never constitutes GPU/HBM validation.
The exact commands and result layout live only in the
[Ticket 07 container runbook](container/README.md#ticket-07-cpu-metadata-replay).

Before handoff, commit the implementation on the feature branch and push that
branch to `origin`. Record:

```bash
git rev-parse HEAD
git status --short
git log -1 --oneline
```

The handoff must name the branch and exact commit, list local verification and
its results, identify every unresolved hardware-only check, and link this
workflow and the container runbook. Large model caches and raw result Parquet
remain outside Git.

## Stage 2: school-server acceptance

The server must execute the exact pushed commit named by the handoff. Fetch the
personal fork, check out that revision, and verify source state before building:

```bash
git fetch origin <feature-branch>
git switch --detach <handoff-commit>
test "$(git rev-parse HEAD)" = "<handoff-commit>"
git status --short
```

A clean checkout is recommended. If a diagnostic patch is necessary, the run
may proceed only when the dirty state and diff are retained in provenance; it
cannot validate the original clean commit.

Inputs:

- the handoff commit and feature-branch name;
- `container-contract.json` and `profile-spine.json` from that commit;
- pinned raw, Ticket 01, Ticket 02, and tokenizer mounts;
- the pinned Qwen model cache; and
- two exclusive RTX 3090 GPUs with the documented NUMA assignment.

Run the Ticket 03 prechecks, then the Ticket 04 spine through the same image:

```bash
"${DS4_RUN[@]}" preflight
"${DS4_RUN[@]}" cpu-dry-run
"${DS4_RUN[@]}" profile-spine
```

`profile-spine` prints its run-specific result directory. A complete result
contains:

- `run-config.json`;
- `raw_samples.parquet`;
- `aggregates.parquet`;
- `provenance.json`; and
- `result.md`.

`run-config.json` freezes the effective engine, sampling, compile, CUDA Graph,
cache, model-length, and measurement parameters. `raw_samples.parquet` records
the actual CUDA Graph runtime mode and prompt/context/cached/scheduled/new-token
counts; partial failures retain completed rows and the failed point/phase.

Validate the artifact contract independently:

```bash
"${DS4_RUN[@]}" exec \
  --output /mnt/ds4/results/ticket-04-validation.json \
  -- /opt/ds4-profile/bin/python -m benchmarks.ds4_profile.profile_spine \
  validate --result-dir /mnt/ds4/results/ticket-04/<run-id>
```

Archive or checksum the result directory outside the repository. The server
handoff back to local development should include the source SHA and dirty
state, image ID, exact invocation, result path, validation result, checksums,
and any rerun/noise decision.

## Stage 3: personal-fork integration

Only hardware-validated work proceeds to a PR against the personal fork's
`main`. Evidence summaries or runbook corrections may be committed to the same
feature branch; large results stay in persistent storage and are referenced by
path and checksum.

The submitting human reviews every changed line, understands the result
contract, and confirms the test and hardware evidence before merge. The school
server then returns to the merged personal-fork `main` for the next ticket.

If hardware validation changes code, push the updated feature branch and rerun
against its new exact commit. Evidence from an older commit does not validate a
newer one.

For Ticket 07, replace “hardware validation” in this integration gate with
“metadata-only validation”: merge eligibility requires an accepted clean
source SHA, immutable Ticket 07 image ID, pinned planning record, complete
ordered replay, independent validator success, and five artifact checksums.
The handoff must still state `GPU/HBM validated: no`.

The current Ticket 07 planner produced no eligible candidate: 20 complete
sessions passed admission, and all had zero native evictions. The selection
remains `unselected`, so the container replay and integration gate must not run
until a revised design is approved. This is a metadata-only planning failure,
not GPU/HBM evidence.
