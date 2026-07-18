# DS4 Ticket 04 Async Teacher-Forcing Fix Design

## Incident and scope

The first dual-RTX-3090 run against `c8625b4e9f` exposed a serious correctness
defect in the real async GPU path. The retained result
`ds4-spine-20260718T134929Z-0fd115ed` is invalid and contains six raw rows,
zero aggregates, two failed workers, and `hardware_validated: false`.

This fix is limited to interpreting empty sampled-token batches correctly and
making teacher-forced injection authoritative for the next async GPU step. It
does not change the profile points, model, runtime configuration, async
scheduling, compilation, CUDA Graph policy, or low-level runner boundary.

## Considered approaches

### Update both async GPU and CPU state

Keep async scheduling and replace the predetermined token in all state that
can feed the next step: request output state, CPU token staging, and the
request-indexed GPU `prev_sampled_token_ids` cache. Continue recording the
actual sampled token from `ModelRunnerOutput`. This is the selected approach
because it preserves the measurement contract and makes the next model input
unambiguously teacher forced.

### Disable async scheduling

This would avoid the GPU token cache, but it would change the runtime exercised
by Ticket 04 and evade the failing production path. It is rejected.

### Use the high-level scheduler or `LLM.generate`

This would delegate state advancement but move the measurement away from the
required `vllm.v1.worker.gpu_worker.Worker.execute_model` boundary. It is
rejected.

## State transition

For each setup or decode sample, the backend obtains the actual sampled token
from `ModelRunnerOutput`. Sync scheduling may already place that token in the
request state. Async scheduling instead places `-1` in CPU request state and
caches the actual token in `input_batch.prev_sampled_token_ids` on the GPU.

The injection operation will locate the request through
`input_batch.req_id_to_index`, verify that the CPU state contains either the
actual sampled token or the documented async placeholder, and replace it with
the predetermined replay token. When `prev_sampled_token_ids` is present, it
will replace the matching GPU entry too. The input-batch CPU token staging slot
will remain synchronized for non-async and diagnostic paths.

Async mode must replace the next-step input source
`prev_sampled_token_ids`; updating only Python request state is not sufficient.
The next call to `_prepare_input_ids` must therefore copy the predetermined
token, not the sampled token, into the actual GPU input.

Prefill will treat a sampled-token result as non-empty only when at least one
inner per-request token list contains a token. The valid chunked-prefill shape
`[[]]` will not be rejected.

## Failure behavior

Missing request mappings, absent token state, unexpected non-placeholder CPU
state, incompatible GPU cache shape, or failure to replace the expected GPU
entry will remain fail-closed errors. Such a run must preserve partial rows and
finalize as invalid. No path may convert a skipped or invalid run into a pass.

The GPU cache is an inference tensor, so its in-place replacement must execute
inside `torch.inference_mode()`. Ticket 04 configuration validation must also
require both capture sizes and compile sizes to cover Decode token count 1 and
Prefill token count 128. Otherwise the 128-token point can compile successfully
but execute with CUDA Graph runtime mode `NONE`.

## Tests and acceptance

Focused tests will extend the existing Ticket 04 test module. They will cover:

- `[[]]` as an empty Prefill sampled-token result;
- sync request state containing the actual sampled token;
- async CPU request state containing `-1`;
- replacement of the request-indexed GPU `prev_sampled_token_ids` value; and
- the next real input-preparation step reading the injected token into
  `input_ids.gpu`, rather than merely observing mutated bookkeeping objects.

After focused tests and lint pass, the fix will be committed on
`codex/ticket-04-ds4-profile-spine`, pushed only to the personal fork, and
accepted against a new exact SHA and image. The server sequence will rerun
preflight, CPU dry-run, print-plan, the real two-worker profile, independent
artifact validation, and the opted-in hardware pytest. Passing requires two
workers, observed CUDA Graph state for all measured samples, valid artifacts,
and `hardware_validated: true` in the final report.
