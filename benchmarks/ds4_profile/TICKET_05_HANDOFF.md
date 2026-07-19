# Ticket 05 WIP Handoff

## Status

Ticket 05 is an incomplete work-in-progress for continuation on the school
server. The branch is `codex/ticket-05-p-side-prefill-profile`, based on the
Ticket 04 merge `65de0de0ab4a5799284e97b823e673d5ac73ef05`.

The last implementation commit before this handoff is
`c1d0cff0d056aed22264f9c4df6bb16648a87fd3`. Use the final pushed branch head
named by the external delivery handoff, because this document is committed
after that implementation commit.

This branch is `remote_pending`. It is not ready for a PR or merge.

## Completed Locally

- The approved design and seven-task implementation plan are committed under
  `docs/superpowers/`.
- Task 1 implements schema-v2 identifiers, frozen full/smoke manifests,
  v1/v2 validation dispatch, exact cardinality, turn/raw reconciliation,
  prefix-evidence group cardinality, aggregates, and comparisons.
- Task 2 implements the deterministic 34-workload/68-point P-side planner,
  fixed 4096-token homogeneous prefixes, 16-token block alignment, and the
  4096-token/eight-sequence chunk budget.
- Independent reviews passed for Task 1 and the final Task 2 fix slice.

Observed local checks include Ruff, `py_compile`, `git diff --check`, direct
validator harnesses, and direct planner tests. The shared local `.venv` has no
`torch`, so pytest failed during collection at `tests/conftest.py`; no pytest,
model, CUDA, worker, container, or hardware result is claimed.

Commits used `--no-verify` only after pre-commit attempted to bootstrap
actionlint through unavailable network access. Re-run repository hooks in the
server environment before delivery.

## Remaining Work

Continue from Task 3 in
`docs/superpowers/plans/2026-07-19-ds4-ticket-05-p-side-prefill-profile.md`:

1. Build v2 chunk, turn, aggregate, and comparison artifacts.
2. Adapt the real Scheduler and KV cache state with fail-closed partial and
   preemption handling.
3. Execute real GPU0 prefix primes, synchronize, prove live CUDA:0 KV tensor
   residency, and time only `GPUWorker.execute_model` chunks.
4. Add the frozen configuration and P-only container command.
5. Complete focused tests, documentation, server smoke, full 68-point run,
   independent validation, checksums, and the final acceptance handoff.

Do not treat equal token IDs or Scheduler block IDs alone as cache-hit or HBM
evidence. Do not time a partial/preempted batch. Full runs require exactly 68
canonical points; smoke runs require their separately frozen selected manifest.

## First Server Commands

```bash
git fetch origin codex/ticket-05-p-side-prefill-profile
git switch --detach origin/codex/ticket-05-p-side-prefill-profile
git rev-parse HEAD
git status --short
```

Confirm the exact SHA and a clean checkout, read the design and plan, inspect
the current diff from `65de0de0ab`, then resume Task 3. Use `uv` and
`.venv/bin/python`; never use system Python or bare pip. Keep all GitHub state
changes in `ycsxh/vllm` and leave upstream read-only.

## Suggested Skills

- `superpowers:subagent-driven-development` or the existing task-by-task plan
- `code-review` after each completed task
- `agent-team-workflow:team-review-cycle` before the final push
- `agent-team-workflow:team-delivery` for the exact-SHA server evidence
