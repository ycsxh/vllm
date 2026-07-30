# Ticket 4 PR review remediation

Status: approved by the repository operator on 2026-07-30.

## Scope

Resolve the final whole-PR review findings without rewriting the accepted
Ticket 4 commits or changing successful serving and measurement behavior.

The remediation has two parts:

1. update the current README and replacement handoff to record the accepted
   Gate D result: implementation `05175de028`, 30/30 valid points, 90/90 valid
   repetitions, and deferred legacy retirement;
2. retain the exact recognized OOM line in `point-failure.json`, `status.json`,
   the raised `UnsupportedPointError`, and therefore the run manifest.

## OOM evidence contract

Point failure classification will inspect the original exception followed by
the P and D logs. The first line containing a recognized OOM marker is the
retained reason. If no marker is found, the original exception remains the
failure reason and behavior is unchanged.

The point-level artifacts preserve the original exception type while using the
retained reason as their `error` value. A recognized OOM is re-raised as
`UnsupportedPointError` with the same retained reason so the run manifest and
point artifacts remain consistent for report auditing.

## Boundaries

- Preserve `05175de02822dde89fa7a6cdbf7f348471237608`,
  `c15aa647b4a470c90d0ad1fbaa76b12fcca4ecb7`, and every accepted ancestor.
- Keep the `origin/main` integration as a merge, not a rebase.
- Do not change the valid-point path, launcher configuration, workload,
  report aggregation, cache accounting, client timing, proxy, or vLLM core.
- Do not modify upstream or execute legacy retirement.
- Do not rerun GPU Gate D: the only runtime change is failure-reason
  preservation for a recognized unsupported OOM, a path absent from the
  accepted 30-point run.

## Verification

- Extend the existing OOM regression to require the exact log reason in both
  point-level artifacts and the propagated exception.
- Run the Ticket 4 replacement-path and completion-client pytest surface.
- Run pre-commit on every changed file.
- Repeat the final Standards and Spec reviews against `origin/main`.
- Require the human submitter to review every changed line and approve the PR
  description before the fork-local PR is created.
