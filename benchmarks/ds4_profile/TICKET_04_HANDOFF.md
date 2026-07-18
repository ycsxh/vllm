# Ticket 04 school-server handoff

## Local development status

The implementation and school-server acceptance stages are complete on branch
`codex/ticket-04-ds4-profile-spine`. The exact hardware-accepted implementation
commit is `c783ea0e69138503c6f83a2f6cb41b79b75404e8`. The earlier handoff commit
`c8625b4e9f` and first fix `383cb530b` are retained as invalid attempts and must
not be presented as passing.

Local evidence recorded before handoff:

- 47 DS4 tests passed;
- 1 dual-RTX-3090 test was explicitly skipped by its hardware gate;
- the full pre-commit run passed; and
- the standards and specification review findings were closed.

The developer workstation must not load the Qwen model, set
`DS4_PROFILE_SPINE_GPU_SMOKE=1`, or run the real profile path. The GPU test is
gated by default. Real `GPUWorker` execution, CUDA Graph evidence, and the
dual-GPU end-to-end result belong only to the school-server stage.

## School-server acceptance incident

The first school-server acceptance attempt against `c8625b4e9f` found a
serious correctness defect in the real asynchronous GPU path. This commit is
not hardware accepted and must not be presented as passing.

The retained invalid run is
`ticket-04/ds4-spine-20260718T134929Z-0fd115ed` under the isolated server
evidence root. Preflight was ready and both RTX 3090 workers loaded the pinned
model, compiled the requested ranges, and captured `FULL` and `PIECEWISE` CUDA
Graphs. The run still correctly finalized as invalid:

- Prefill treated `sampled_token_ids=[[]]` as an actual sampled token because
  it checked only the outer list.
- Decode compared the returned sampled token with the `-1` placeholder stored
  in runner state by async scheduling.
- More importantly, the teacher-forcing injection updated CPU-side request
  state but did not replace the GPU-resident `prev_sampled_token_ids` consumed
  by the next model step. Merely accepting the `-1` placeholder could therefore
  make bookkeeping tests pass while the model advanced with its sampled token
  instead of the predetermined replay token.

The invalid artifacts contain six raw rows, zero aggregates, structured
Prefill and Decode failure phases, `hardware_validated: false`, and an Invalid
`result.md`. The independent validator passed structural validation only; that
does not override the invalid hardware result.

Any fix must preserve async scheduling, the low-level
`Worker.execute_model` boundary, torch.compile, and CUDA Graph requirements.
Async mode must replace the request's GPU `prev_sampled_token_ids` entry as
well as CPU state. A focused test must prove that the next step's real GPU
input reads the injected token. Acceptance must be rerun from preflight through
the hardware-gated pytest against the new exact commit and image.

The first fix attempt at `383cb530b` also remains unaccepted. Its retained
invalid run is `ticket-04/ds4-spine-20260718T143314Z-4fad469c` under the
`ticket-04-acceptance-383cb530b` evidence root. It established two additional
fail-closed requirements:

- Updating the GPU async token cache must occur inside
  `torch.inference_mode()` because the cache is an inference tensor.
- CUDA Graph capture sizes must cover both the one-token Decode point and the
  128-token Prefill point. Compile coverage alone is insufficient; capture
  size `[1]` caused the real Prefill point to report runtime mode `NONE`.

That run also finalized with six partial rows, zero aggregates,
`hardware_validated: false`, and an Invalid report. It does not validate any
later commit.

## Accepted school-server result

The rerun against clean commit `c783ea0e69138503c6f83a2f6cb41b79b75404e8`
passed on the documented dual RTX 3090 server. The immutable image ID was
`sha256:3841fe21a5e9b94f10d60d1d6da17a54393cbb99fd0f91bcc27ae77f2aab114f`.
The accepted evidence root is
`/home/lyc/ds4-storage/results/ticket-04-acceptance-c783ea0e6`, and the real
profile result is
`ticket-04/ds4-spine-20260718T144113Z-1c85fd88` beneath that root.

Acceptance evidence:

- Preflight was ready and all seven checks passed; the CPU dry run and printed
  two-worker NUMA plan also passed.
- Both GPU workers passed. The 30 raw rows comprise two startup, two capture,
  six warmup, and 20 steady rows; the two aggregate rows contain ten measured
  samples per point.
- All 26 warmup and steady observations used CUDA Graphs: 13 Decode rows were
  `FULL` and 13 Prefill rows were `PIECEWISE`.
- Every Decode warmup and steady row records a sampled token, a discarded
  sampled-token flag, and an injected token. The harness additionally checked
  after every step that the next real GPU input consumed the previously
  injected token.
- Independent artifact validation returned zero. A separate parquet audit
  asserted row counts, graph modes, teacher-forcing fields, clean source, and
  `hardware_validated: true`.
- The explicitly enabled hardware gate was collected and executed twice. Both
  executions passed; the retained raw pytest log reports
  `1 passed in 63.23s` with no skipped test.

Key SHA-256 checksums:

```text
093c53e75876d95c3cc59a7464401faba9e59ffa194217520d4f28e353248fb8  raw_samples.parquet
fdaea000520d3366d383eb155a165700335b3a49e43151f05268a823e8091b63  aggregates.parquet
1c316450c26a5df63eb53d485421e755580bc5a4f2f4a3344a4f330a7027c22b  provenance.json
ebb47ebc5b980a52ef3567fc2c3f1206b66e6a7943de851072ff346ba5ceb47e  result.md
fec0465d8727b934a4944d59b33e01c43a6aad8c638f12d985880b37576c3ecd  ticket-04-validation-ds4-spine-20260718T144113Z-1c85fd88.json
6e53d063ed5a554b5b4f5ced794ac3fa0e33d2e5a7f30ac10a9324429d335725  ticket-04-artifact-assertions.json
78f11adff68e834ec0627aad5e14b35f33aae6b7cf34305b36273c5621e1bdc9  ticket-04-hardware-pytest-evidence.json
2ccd864365994f3591fa487ef7c3b16705cb80f375d41491f6001ed9624d9bdb  ticket-04-hardware-pytest.log
```

The later documentation-only commit recording these results is not a different
validated implementation. Hardware acceptance remains explicitly tied to
`c783ea0e69138503c6f83a2f6cb41b79b75404e8` and the image ID above.

## Local Codex review handoff

The review PR is the draft
[`ycsxh/vllm#3`](https://github.com/ycsxh/vllm/pull/3), from
`codex/ticket-04-ds4-profile-spine` to the personal fork's `main`. It is not an
upstream PR. Do not create an upstream issue, comment, review, PR, or push while
reviewing this handoff.

The PR base at creation was
`72fe9df32ca636e93a7f2b1f3f4de074806752f3`. Review the seven Ticket 04 commits
from `159a779bb` through the hardware-accepted implementation `c783ea0e6`, then
review the later documentation-only acceptance commits separately. The PR body
records the duplicate-work checks, test results, model run, invalid attempts,
performance boundary, and AI-assistance disclosure required for human review.
Its personal-fork `pre-run-check` remains red by design while the draft has no
`verified` or `ready` label; the dependent pre-commit job is therefore skipped.
Do not add a readiness label until the human review is complete.

All three server attempts are bundled for transfer to the local development
computer:

```text
/home/lyc/ds4-storage/results/ticket-04-review-evidence-c783ea0e6.tar.gz
SHA-256: a3ed24494dced6c8dbbc46e3109329bcdbaf814230dd56696624981eecb7b7db
Size: 3.9 MiB
```

The archive contains both retained invalid evidence roots and the final passed
root. After copying it, verify the archive checksum before extracting it. A
local reviewer must keep the invalid attempts distinct from the accepted
`c783ea0e6` result and must not infer hardware acceptance from a skipped test.

## Transfer the implementation

The branch, including the minimal fixes, has been pushed only to the personal
fork. Do not push to or open a PR against `vllm-project/vllm`.

```bash
git push personal codex/ticket-04-ds4-profile-spine
```

On the school server, fetch the personal-fork branch and detach at the exact
implementation commit:

```bash
git fetch personal codex/ticket-04-ds4-profile-spine
git switch --detach c783ea0e69138503c6f83a2f6cb41b79b75404e8
test "$(git rev-parse HEAD)" = \
  "c783ea0e69138503c6f83a2f6cb41b79b75404e8"
git status --short
```

Start from a clean checkout. If diagnosis requires a patch, retain the dirty
diff in provenance and do not claim that it validates the clean commit.

## Hardware acceptance

Follow [`container/README.md`](container/README.md) to build the image and
define the `DS4_RUN` array with the persistent Ticket 01/02, tokenizer, model,
cache, configuration, and result mounts. Run cheap checks before loading the
model:

```bash
"${DS4_RUN[@]}" preflight
"${DS4_RUN[@]}" cpu-dry-run
"${DS4_RUN[@]}" profile-spine --print-plan
```

Resolve any preflight or mount failure first. Then run the real Ticket 04 path:

```bash
"${DS4_RUN[@]}" profile-spine
```

The command prints the run-specific result directory. It must contain
`run-config.json`, `raw_samples.parquet`, `aggregates.parquet`,
`provenance.json`, and `result.md`. Independently validate it through the same
image:

```bash
"${DS4_RUN[@]}" exec \
  --output /mnt/ds4/results/ticket-04-validation.json \
  -- /opt/ds4-profile/bin/python -m benchmarks.ds4_profile.profile_spine \
  validate --result-dir /mnt/ds4/results/ticket-04/RUN_ID
```

Finally, opt into the hardware-gated pytest only on the documented dual-3090
host by setting `DS4_PROFILE_SPINE_GPU_SMOKE=1`. Do not assert a latency
threshold; acceptance covers execution and artifact invariants.

## Exit gate and iteration

Ticket 04 passes only when preflight is ready, both Prefill and Decode workers
pass, observed CUDA Graph state is recorded for measured samples, artifact
validation succeeds, and `result.md` reports `hardware_validated: true`.

If execution fails, preserve the invalid result directory and completed sample
rows. Record the source SHA, dirty state, image ID, command, result path, and
error phase. Make the smallest fix on the feature branch, push the new commit
to the personal fork, and rerun acceptance against that new exact SHA. Evidence
from `c8625b4e9f` or `383cb530b` does not validate later code changes.

After acceptance, return the result path, validation record, checksums, source
SHA, dirty state, and any diagnostic patch to the local workflow. Only then is
the ticket ready for human review and a PR to the personal fork's `main`.
