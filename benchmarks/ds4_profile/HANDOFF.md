# Ticket 03 Handoff

## Purpose

Use this handoff when reviewing the Ticket 03 acceptance archive or starting
follow-up DS4 profiling work. Ticket 03 itself is complete; no required work is
left open.

## Current State

- Personal-fork PR: <https://github.com/ycsxh/vllm/pull/1> (`MERGED`)
- Validated feature head: `c855e2bd31ff31f0d6b6123f550b000f24b7386e`
- Personal-fork merge commit: `64f86cd0f51d69289753bbad8da0bfda6929d2f1`
- The remote feature branch was deleted after merge.
- The submitting human reviewed the Ticket 03 change before merge.

The implementation, test commands, acceptance summary, AI-assistance
disclosure, and duplicate-work result are already captured in PR #1. Do not
duplicate them here.

## Authoritative References

- Runtime contract and operations:
  [`container/README.md`](container/README.md)
- DS4 workflow overview: [`README.md`](README.md)
- Cross-machine development and acceptance stages: [`WORKFLOW.md`](WORKFLOW.md)
- Container implementation: [`container/`](container/)
- Focused tests: [`../../tests/benchmarks/ds4_profile/`](../../tests/benchmarks/ds4_profile/)
- Target-server evidence: `$HOME/ds4-storage/results/`
- Downloadable review archive:
  `$HOME/ds4-storage/ticket-03-review-20260717.tar.gz`

Review the archive in this order:

1. `results/gpu-smoke.json`
2. `results/cpu-dry-run/provenance.json`
3. `results/cache-model.json`
4. `results/image-inspect.json`
5. `results/pull-request.md`

The remaining JSON, Parquet, and build-log files provide supporting detail.

## Decisions and Constraints

- Python setup, tests, linting, and runtime validation were isolated in
  containers; the host/base Python environment was not modified.
- The existing compiled base image was reused. Only the thin Ticket 03 overlay
  was rebuilt after runtime fixes.
- Model and runtime caches remain outside the repository under
  `$HOME/ds4-storage`.
- Do not write to `vllm-project/vllm` or to Hugging Face data sources. PR #1
  and its merge exist only in `ycsxh/vllm`.
- The PR's red `pre-run-check` was an inherited contributor-governance gate:
  it required a maintainer label or four prior merged PRs. It was not a code or
  test failure. Satisfying it in the fork would have queued an unavailable
  upstream self-hosted runner, so the verified container and dual-GPU evidence
  was used instead.

## Next Session

For evidence review:

```bash
sha256sum "$HOME/ds4-storage/ticket-03-review-20260717.tar.gz"
tar -tzf "$HOME/ds4-storage/ticket-03-review-20260717.tar.gz"
```

Extract the archive outside the repository. If follow-up implementation is
requested, start a new branch from the personal fork's current `main`, preserve
the fixed revisions and provenance contract, and rerun only the acceptance
level affected by the change.

## Suggested Skills

- `superpowers:brainstorming`: invoke before designing new behavior or changing
  the DS4 profiling contract.
- `handoff`: invoke at the end of a future session to compact only new context
  and reference this document instead of repeating it.
