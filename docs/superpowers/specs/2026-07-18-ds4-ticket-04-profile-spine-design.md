# DS4 Ticket 04 Profile Spine Design

> [!CAUTION]
> Superseded pre-refactor design retained for historical traceability only.
> It must not guide new profile work. See
> [`../../../benchmarks/ds4_profile/AUTHORITATIVE_SPEC.md`](../../../benchmarks/ds4_profile/AUTHORITATIVE_SPEC.md).

## Scope

Ticket 04 establishes one thin, complete path from a prepared Ticket 02
workload through vLLM's GPU worker/model-runner boundary to versioned raw
samples, aggregates, provenance, and a readable result. It deliberately covers
only one batch-1 chunked-prefill point and one batch-1 teacher-forced decode
step. Tickets 05 and 06 will extend the same harness into full P- and D-side
sweeps.

The implementation is developed and contract-tested locally. Hardware
execution and final acceptance happen later on the school server with two RTX
3090 GPUs. Local completion must not be represented as hardware validation.

## Runtime Architecture

The profile spine has three layers:

1. A contract layer owns versioned PyArrow schemas, stable identifiers,
   validation, aggregation, and Markdown rendering. It has no CUDA dependency.
2. An orchestration layer loads a pinned Ticket 02 exact replay, constructs the
   two thin profile points, separates startup/capture, warmup, and steady-state
   samples, and writes the artifact set.
3. A GPU backend owns version-specific vLLM initialization and execution. It
   initializes the GPUWorker/GPUModelRunner path and calls it with
   SchedulerOutput-shaped work. It does not use HTTP, an online queue, or
   `LLM.generate`.

CPU tests use a deterministic backend that implements the same measurement
interface. This tests the complete artifact path without claiming that mocked
timings validate GPU execution.

### Prefill point

The prefill point uses one selected exact replay, batch size one, chunked
prefill, a 4096-token total scheduling limit, FP16 weights and KV cache, tensor
parallel size one, and prefix caching enabled. The recorded point parameters
distinguish prompt/context length, scheduled tokens, cached tokens, and new
tokens.

### Decode point

The decode point executes one batch-1 decode step after preparing the required
request/KV state outside the measured steady-state sample. The measured slice
includes model execution, logits, and sampling. The sampled token is recorded
and discarded; the predetermined next execution token from the Ticket 02
replay is injected into request state. The artifact records both token values
and confirms that the injected token, rather than the sampled token, advances
the teacher-forced state.

### Compilation and CUDA graphs

The main GPU path keeps torch.compile and CUDA graphs enabled. Model load,
compilation, and graph capture are represented by startup/capture records and
are excluded from steady-state aggregates. Warmups are also recorded but
excluded. The run configuration rejects an eager main path; eager execution is
allowed only for an explicitly labeled diagnostic run outside Ticket 04 main
results.

## Artifact Contract

Every successful run writes the following files into one result directory:

- `run-config.json`: the frozen effective input configuration.
- `raw_samples.parquet`: one row per startup/capture, warmup, or steady sample.
- `aggregates.parquet`: summaries derived only from steady samples.
- `provenance.json`: source, runtime, model, hardware, roles, preflight, and
  invocation metadata.
- `result.md`: a minimal readable summary with artifact references and an
  explicit validation state.

All schemas carry an artifact schema version. Timing columns include units in
their names or schema metadata. Synchronized runner wall time, optional CUDA
event model time, and derived timings remain separate fields.

A run has one `run_id`. Each logical point has a deterministic `point_id`
derived from its role and workload parameters. A `sample_id` combines the run,
point, phase, and sample ordinal. Tests validate uniqueness and cross-artifact
referential integrity. Tests may inject a fixed run ID; normal runs create one
and persist it before execution.

Aggregates contain sample count, median, p90, mean, coefficient of variation,
and a noisy flag. A coefficient of variation above five percent is flagged but
does not silently delete samples.

## Failure Semantics

Hardware prerequisite failure produces `skipped` or `invalid` provenance and a
readable result; it never produces a passed hardware-validation state. Runtime
failure preserves completed rows when possible and records a structured error
with the failed phase and point. Empty or partial Parquet artifacts retain the
same versioned schemas so downstream validation can distinguish an invalid run
from corrupt output.

Artifacts are written through a staging directory and finalized only after
schema and cross-file validation. An interrupted staging directory is not a
successful result and may be retained for diagnosis.

## CLI and Container Integration

The project-local CLI accepts a frozen configuration and a result directory.
It supports:

- a deterministic CPU fixture run for contract validation;
- a real GPU run for the school server;
- schema validation of an existing result directory; and
- plan/dry-run output for container command inspection.

The Ticket 03 container entrypoint exposes the real profile-spine command using
the same CLI, configuration schema, mounts, and artifact schemas. The mounted
result directory is the only durable output location. The existing preflight
is extended rather than duplicated so source revision, dirty state, model
revision, CUDA/driver/PyTorch versions, topology, P2P status, NUMA binding,
role assignment, and all run parameters remain in one provenance record.

## Testing

CPU contract tests exercise the public CLI and observable artifacts. They
cover:

- required columns, types, units, nullability, enums, and schema versions;
- stable and unique run, point, and sample identifiers;
- startup/capture and warmup exclusion from aggregates;
- hand-computable median, p90, mean, coefficient of variation, and noisy flag;
- teacher-forcing state transition and sampled-token discard;
- complete, skipped, invalid, and partial-failure result states;
- Markdown references and validation wording; and
- identical schemas through direct and container-planned execution.

A hardware-gated GPU smoke test checks the two worker roles, low-level runner
boundary, expected row counts, compile/CUDA-graph state, teacher-forced token,
and artifact invariants. It asserts no machine-specific latency threshold.

Tests extend the existing DS4 profile test area and reuse nearby fixtures and
helpers. A new test file is justified only for the profile artifact/harness
contract, which is distinct from Ticket 03 container lifecycle tests.

## DS4 Project Workflow Documentation

`benchmarks/ds4_profile/WORKFLOW.md` documents the two-stage workflow for this
DS4 dual-3090 profile project only. It is not a repository-wide vLLM
contribution rule. The main DS4 README, container runbook, and Ticket 03
handoff link to it instead of copying its contents.

The workflow uses strong language only for result-integrity gates:

- the school server checks out the exact pushed feature-branch commit named in
  the handoff;
- the actual commit and dirty state are recorded;
- local work is not labeled hardware-validated;
- failed or skipped hardware checks are not presented as success; and
- the feature branch is merged into the personal fork's `main` only after
  hardware acceptance.

Branch naming, archive layout, and exact commands are recommendations. This
keeps the workflow useful without making incidental mechanics rigid.

### Local development stage

Inputs are the pinned Ticket 01/02 artifacts, Ticket 03 container contract,
and the personal fork's known `main` baseline. The local stage implements the
harness, passes CPU contract tests, runs the deterministic fixture end to end,
checks container wiring, and writes a handoff containing the exact pushed
commit and verification results.

Its outputs are a pushed feature-branch commit, frozen example configuration,
local artifacts labeled as fixture-only, and the handoff. It makes no GPU or
performance claim.

### School-server stage

The server fetches and checks out the exact handoff commit, verifies source
state, builds a commit-labeled image, then runs preflight and the GPU profile
spine. Its outputs are raw and aggregate Parquet, provenance, Markdown result,
commands, checksums, and an acceptance summary. Evidence updates remain on the
same feature branch. Only validated work proceeds to a personal-fork PR and
merge.

No workflow step opens or mutates an upstream `vllm-project/vllm` issue, PR, or
branch.

## Out of Scope

- Full prefill or decode matrices, which belong to Tickets 05 and 06.
- P-to-D KV transfer, offline cache studies, or policy simulation.
- HTTP, proxy, online scheduler, or queueing measurements.
- Performance thresholds in correctness tests.
- Hardware validation on the local development machine.
- Repository-wide contribution-policy changes.
