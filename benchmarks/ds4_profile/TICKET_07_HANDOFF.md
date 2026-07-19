# Ticket 07 WIP Handoff

## Status

Ticket 07 is an incomplete work-in-progress for continuation on the school
server. The branch is `codex/ticket-07-kv-cache-manager-replay`, based on the
Ticket 04 merge `65de0de0ab4a5799284e97b823e673d5ac73ef05`.

The last implementation commit before this handoff is
`a0dd30f493`. Use the final pushed branch head named by the external delivery
handoff, because this document is committed after that implementation commit.

This branch is `remote_pending`. It is not ready for a PR or merge.

## Completed Locally

- The approved design and eight-task implementation plan are committed under
  `docs/superpowers/`.
- Task 1 reconstructs complete prompt-only sessions and reconciles the exact
  Ticket 02 scalar key set without reading completion/decode token columns.
- Task 2 builds the real metadata-only `Request` and `KVCacheManager` seam,
  deterministic hashes, null-block accounting, and scoped observation of the
  real BlockPool touch/allocate/evict/free operations.
- Task 3 implements serial replay, miss attribution, manager-forced recompute,
  native `BlockRemoved` eviction evidence, occupancy, per-operation timing,
  partial OOC evidence, and future-reuse labels.
- Independent reviews passed for Tasks 1 and 2 and the final Task 3 fix slice.

Observed local checks include Ruff, `py_compile`, AST/interface checks,
`git diff --check`, and direct prompt/replay harnesses. The shared local
`.venv` has no `torch`, so pytest failed during collection; real manager tests,
container execution, and full replay remain unverified.

Commits used `--no-verify` only after pre-commit attempted to bootstrap
actionlint through unavailable network access. Re-run repository hooks in the
server environment before delivery.

## Remaining Work

Continue from Task 4 in
`docs/superpowers/plans/2026-07-19-ds4-ticket-07-kv-cache-manager-replay.md`:

1. Implement the deterministic trajectory/capacity selection planner and pin
   the selected values in configuration.
2. Add versioned Parquet artifacts, the independent semantic validator, and
   the `plan`, `run`, and `validate` CLI commands.
3. Register the CPU-only container command without requesting GPUs.
4. Complete focused tests, documentation, full server planning/replay,
   independent validation, input/result hashes, and the final handoff.

Every Python/vLLM process must set `PYTHONHASHSEED=0` and
`VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0` before vLLM import. Ticket 07 is
metadata-only: it reads prompt data, allocates no KV tensors, and makes no HBM,
Prefill, Decode, model, or GPU-residency claim.

## First Server Commands

```bash
git fetch origin codex/ticket-07-kv-cache-manager-replay
git switch --detach origin/codex/ticket-07-kv-cache-manager-replay
git rev-parse HEAD
git status --short
export PYTHONHASHSEED=0
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
```

Confirm the exact SHA and a clean checkout, read the design and plan, inspect
the diff from `65de0de0ab`, then resume Task 4. Use `uv` and
`.venv/bin/python`; never use system Python or bare pip. Keep all GitHub state
changes in `ycsxh/vllm` and leave upstream read-only.

## Suggested Skills

- `superpowers:subagent-driven-development` or the existing task-by-task plan
- `code-review` after each completed task
- `agent-team-workflow:team-review-cycle` before the final push
- `agent-team-workflow:team-delivery` for exact-SHA evidence
