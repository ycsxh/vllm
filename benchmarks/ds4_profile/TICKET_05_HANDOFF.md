# Ticket 05 P-side Prefill Profile Handoff

## Status

- Branch: `codex/ticket-05-p-side-prefill-profile`
- Personal fork: `ycsxh/vllm`
- Upstream: read-only; no upstream interaction is authorized
- Implementation base handoff: `a59db6380383e8efc6db7ee3d18784b59391c199`
- Task 6 implementation commit: `f318f3405d99a363ced68dc40f71522308359287`
- Candidate source commit: `7ed4b9a501dff9ddb47f2239db5389d2072830cc`
- Candidate source dirty state: `false`
- Hardware state: `remote_pending`
- Merge state: not ready for personal-fork merge

Tasks 1–7 are implemented and independently reviewed. The current execution
environment exposes no CUDA device and has no `/mnt/ds4` Ticket 02 or results
mount, so neither smoke nor the full matrix has run. Static, CPU, mock, skip,
and schema checks are not hardware acceptance.

Build and execute exactly the candidate source commit above. The final
docs-only evidence commit changes only this handoff and does not change runtime
code, tests, or configuration. An older hardware result does not validate a
newer code commit.

## Frozen contract

- Config: `benchmarks/ds4_profile/config/p-prefill-profile.json`
- Config SHA-256:
  `719d2bde83fd51ea4cd3b5502324f59e7236caa04d87de669bfd7d1379916418`
- Workloads/points: 34/68
- Smoke: five selectors, ten complete hit/recompute points
- Per-step limits: 4096 scheduled tokens, eight sequences
- Homogeneous prefix/block size: 4096/16 tokens
- GPU/NUMA: GPU0/NUMA0 only
- Expected full-plan checksum: `remote_pending` (requires the pinned Ticket 02
  mounts; record the printed frozen manifest checksum before launching smoke)
- Image ID: `remote_pending`
- Result root: `/mnt/ds4/results/ticket-05/`
- Full-run estimate: 3–8 hours, with long-context recompute and native KV
  capacity pressure as the main runtime risks

## Local and server evidence

The school checkout used for implementation reported:

```text
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
/mnt/ds4 = absent
```

No remote artifact exists yet, so the hardware lifecycle remains
`remote_pending`, not skipped-as-pass, `remote_failed`, or `remote_verified`.
The local hardware/input gate failed. Local acceptance consists only of the
focused pytest
suites, Ticket 04 regressions, pre-commit, CLI parsing/print-plan with test
fixtures, and independent reviews.

Recorded local gates:

- Ticket 05 plus Ticket 04/container regressions:
  `132 passed, 2 skipped` in 65.57 seconds. The skips are hardware-gated.
- Explicit Ticket 05 GPU test: `1 skipped, 65 deselected`; no GPU claim.
- Required four-file command: `133 passed, 2 skipped, 7 failed` in 73.60
  seconds. All seven failures are `test_workloads.py` fixture-input failures:
  the pinned `.scratch/ds4-agent-1p1d-profile` manifest, Ticket 01 Parquet, and
  tokenizer directories are absent. No Ticket 05 assertion failed.
- The complete listed pre-commit file gate passed after formatting.
- Task 3–6 implementation slices and the final Task 6 lifecycle passed
  independent standards/spec reviews with no remaining P0–P2 findings.

The seven missing-input failures remain an environment gate and must be rerun
with the pinned assets. They are not converted to skips or passes.

## School-server procedure

Use a Ticket 05-specific image tag and preserve every attempt in a distinct
result directory. With the existing `DS4_RUN` array:

```bash
"${DS4_RUN[@]}" preflight
"${DS4_RUN[@]}" p-profile --print-plan
"${DS4_RUN[@]}" p-profile --smoke \
  --output-dir /mnt/ds4/results/ticket-05/smoke
"${DS4_RUN[@]}" exec \
  --output /mnt/ds4/results/ticket-05-smoke-validation.json \
  -- /opt/ds4-profile/bin/python -m benchmarks.ds4_profile.prefill_profile \
  validate --result-dir /mnt/ds4/results/ticket-05/smoke
DS4_P_PREFILL_GPU_SMOKE=1 /opt/ds4-profile/bin/python -m pytest \
  tests/benchmarks/ds4_profile/test_prefill_profile.py -k gpu_smoke -v
"${DS4_RUN[@]}" p-profile \
  --output-dir /mnt/ds4/results/ticket-05/full
```

Do not start the full run until smoke, the hardware-gated pytest, and the
independent validator all pass. Record the source SHA, dirty state, image ID,
host invocation, run IDs, runtime, result paths, and SHA-256 checksums.

## Acceptance checklist

- `run_kind` and frozen manifests are exact: smoke has the configured ten IDs,
  full has all 68, and neither is inferred from observed rows.
- Every feasible point has all chunks for three warmups and ten steady turns.
  OOC has only a valid coordinate prefix and one terminal row.
- No empty, partial, preempted, or unrelated SchedulerOutput was timed.
- Only `GPUWorker.execute_model` is inside CUDA-event and wall timing; setup,
  priming, compile, capture, reset, and warmup remain separate.
- Recompute rows prove zero cached tokens.
- Every hit's `prefix_evidence.parquet` rows prove a real GPU0 prime completed
  and synchronized; measured SchedulerOutput tables reuse the same physical
  block IDs; and every ID maps in bounds to recorded live
  `worker.model_runner.kv_caches` `torch.Tensor` values on `cuda:0`, including
  tensor name, device, shape, physical block axis, and block dimension.
- Token IDs, Scheduler IDs, mocks, or configured capacity alone are rejected
  as HBM evidence.
- All measured rows use CUDA Graph mode `FULL` or `PIECEWISE`.
- OOC rows prove authoritative allocator pressure and a clean reset.
- Every fully passed pair has exactly one comparison; OOC pairs have none.
- `provenance.json` finishes as `remote_verified` with
  `hardware_validated: true`.

Retain `remote_failed` and partial artifacts unchanged. Never turn a skip,
mock, bootstrap failure, static check, or CPU-only result into hardware proof.
