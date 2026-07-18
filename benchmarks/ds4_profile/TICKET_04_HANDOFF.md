# Ticket 04 school-server handoff

## Local development status

The local implementation stage is complete on branch
`codex/ticket-04-ds4-profile-spine` at implementation commit
`c8625b4e9f`. Ticket 04 as a whole is not complete until the school-server
hardware acceptance below passes.

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

## Transfer the implementation

The branch has not been pushed as part of this handoff. From the developer
workstation, push only to the personal fork:

```bash
git push origin codex/ticket-04-ds4-profile-spine
```

Do not push to or open a PR against `vllm-project/vllm`.

On the school server, fetch the personal-fork branch and detach at the exact
implementation commit:

```bash
git fetch origin codex/ticket-04-ds4-profile-spine
git switch --detach c8625b4e9f
test "$(git rev-parse --short=10 HEAD)" = "c8625b4e9f"
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
from `c8625b4e9f` does not validate later code changes.

After acceptance, return the result path, validation record, checksums, source
SHA, dirty state, and any diagnostic patch to the local workflow. Only then is
the ticket ready for human review and a PR to the personal fork's `main`.
