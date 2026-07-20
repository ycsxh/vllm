# Ticket 07 CPU Metadata Replay Handoff

## Status

Ticket 07 implementation and school-server acceptance are complete on branch
`codex/ticket-07-kv-cache-manager-replay` under the approved split acceptance.
Keep personal-fork issue #4 open until the human merge decision. No PR was
created.

- Metadata-only validated: **yes**.
- Pilot eviction pressure observed: **no** — zero native evictions.
- Synthetic real-manager LRU conformance: **yes**.
- GPU/HBM validated: **no**.
- Peak GPU memory: **not measured / not applicable**.

Ticket 07 read prompt metadata and prompt token IDs only. It did not read
completion/decode token IDs, allocate KV tensors, load a model, execute Prefill
or Decode, request a GPU, establish HBM residency, or validate GPU behavior.
Zero pilot evictions are a workload observation, not a vLLM manager failure.

## Accepted source and image

The accepted clean source SHA is
`b25fe1c0b6144511b81b97eac269f27afdf0baf1`. The later handoff-only commit does
not change or replace this acceptance boundary.

- final image tag: `local/vllm-ds4-profile:ticket-07`;
- final image ID:
  `sha256:53f0eed60c359c8a01267335778431f960079d69ce4f82746178a831b2788f4a`;
- source dirty label: `false`;
- final image inspect SHA-256:
  `3debaa45c8e348998d5ba7af08808587ba965e954eb6304ba40c8bc64129837a`;
- reused immutable Ticket 03 base ID:
  `sha256:892b612fbcbea36cdfcd567c3f76b4993861e2d36a543ab96b56265839f872a1`.

Ticket 07 changes only Python metadata-replay and container registration
surfaces, not vLLM core or compiled extensions. The standard Ticket 03 base was
therefore reused for the isolated Ticket 07 overlay. An attempted full base
rebuild was canceled during CUDA extension compilation after confirming that
it added no Ticket 07 coverage; Ticket 05's concurrent build and evidence were
not reused or modified.

## Planning and pinned selection

The final clean planning checkpoint was
`b25fe1c0b6144511b81b97eac269f27afdf0baf1` in image
`sha256:53f0eed60c359c8a01267335778431f960079d69ce4f82746178a831b2788f4a`.
The final container planning record is retained at:

```text
/home/lyc/ds4-storage/results/ticket-07-selection-b25fe1c0.json
```

- planning record file SHA-256:
  `cfa4716ac1638a4c11e2cde535e9dec36d11d9709df3abb3fd7d83226b56281d`;
- canonical planning digest:
  `0106d51d3a24352b083608796d8382a5a29c717d44cd3ee21713f84ffc5ba606`;
- candidates / eligible: `20 / 20`;
- every candidate admission status: `passed`;
- every candidate native eviction count: `0`;
- selected trajectory: `astropy__astropy-12907:no_think`;
- reasoning mode: `no_think`;
- selected turns: `15`;
- usable capacity: `730` blocks at block size `16`;
- selected native evictions: `0`;
- input-set digest:
  `08b1760ba60deea1a33b29c925de88bf115b6eb0d4a88ca7999286976f082fea`;
- ordered turn-manifest digest:
  `fdb4b4b0de411493cb181bd9a7f271c96515f9ac5229372a4c26d3b9b3e316db`.

The selected capacity is exactly
`max(ceil(prompt_tokens / 16))` for the complete trajectory. No capacity was
enlarged or lowered, and no prompt was shortened or turn skipped.

## Verified input hashes

The planner, run, and independent validator recomputed the following inputs:

- manifest:
  `c228d3f670b20185a3bde4be08a67b73877179795f1b899ae4f6a23d64371c6a`;
- Ticket 01 data:
  `9b4d82334e9fa429c7cc38ce56eacb59e219cd7c7ccf301a1468d7ec19e2f168`;
- Ticket 01 provenance:
  `3c373acf83837ee69738ec86df58fe4cf037e7812593f49e3a2023e15c302cdf`;
- Ticket 02 data:
  `d47933e19f63ae1df7fc3fe04e228550b41d76e5ba93c09751a307f46ba33be7`;
- Ticket 02 provenance:
  `3b1eecdae4a66ae75f47a0c151f8776f3bf9a4fc927019763a041454f2df165c`;
- tokenizer `.cache/huggingface/.gitignore`:
  `684888c0ebb17f374298b65ee2807526c066094c701bcc7ebbe1c1095f494fc1`;
- tokenizer `.cache/huggingface/CACHEDIR.TAG`:
  `f6572428f6d5e1575e73a1502895a8731f10757dfbb634909c6e154b849af91d`;
- tokenizer `download/tokenizer.json.lock`:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- tokenizer `download/tokenizer.json.metadata`:
  `14e4cdd1bb486faafe835e302d955f44ab7a20683c3e5b33f795889ab64e8dea`;
- tokenizer `download/tokenizer_config.json.lock`:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- tokenizer `download/tokenizer_config.json.metadata`:
  `71da710cabc3c39247fa6b9153cbda3ae7e9096a0a0172be5976d05137b880ce`;
- tokenizer tree record:
  `526fdefe47cd3a4bb95a743d8056eda13a6d11271bb3674a2624ca720be1ad67`;
- tokenizer `tokenizer.json`:
  `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`;
- tokenizer `tokenizer_config.json`:
  `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547`.

## Accepted replay and validation

The accepted result directory is:

```text
/home/lyc/ds4-storage/results/ticket-07/
  ds4-kv-replay-20260720T085917Z-45aef5db
```

The real path was `Request` → `KVCacheManager.get_computed_blocks` →
`allocate_slots` → `take_events` → `free`. Results:

- status: `passed`;
- passed turns: `15 / 15`;
- event rows: `24,297`;
- compulsory misses: `729`;
- capacity misses: `0`;
- prefix-mismatch misses: `0`;
- manager-forced recomputes: `0`;
- native evictions: `0`;
- independent container validator exit: `0`;
- `metadata_only_validated: true`;
- `pilot_eviction_pressure_observed: false`;
- `hardware_validated: false`.

The artifact provenance directly binds the run to the clean source SHA, image
ID, complete host and container invocations, deterministic environment, and
installed package versions. Request hashing was measured for every turn; the
minimum recorded hash duration was `363,363 ns` and the total was
`24,159,107 ns`.

The result says exactly `Metadata only: yes`,
`Pilot eviction pressure observed: no`, `Native eviction count: 0`, and
`GPU/HBM validated: no`.

Artifact SHA-256 values:

- `cache_events.parquet`:
  `93c3893bca4eef0b289a85c3bb9d43430a0590d848598b3682ec9cc2e9c2f16d`;
- `turn_summaries.parquet`:
  `57da3fcbc03554812d19dbf0b8f380cc2ff6f3dbe16af929352384a5156c42be`;
- `run-config.json`:
  `00ed3a177c7c61dc4f425f13fdb58ee98bc62017e28fb9460b591f99f90fe2ca`;
- `provenance.json`:
  `5ab4eb72171dca4adb5572412085ef18b9580270df5ff2744a5424583d336380`;
- `result.md`:
  `3fb1cdefc5c6682891c845c7a1f05788e8e82071f72e6a8a9ba4571457d0cf25`.

## Test and lint evidence

Every Python/vLLM command set:

```bash
export PYTHONHASHSEED=0
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
```

- host Ticket 07 plus container workflow: **57 passed**;
- real upstream manager regressions: **3 passed, 14 Torch deprecation
  warnings**;
- final-image focused Ticket 07 suite: **43 passed, 2 non-functional
  warnings**;
- final-image pytest exec record:
  `/home/lyc/ds4-storage/results/ticket-07-pytest-b25fe1c0.json`;
- final-image pytest exec-record SHA-256:
  `5412c91324518280c1f6966a991994f9094092f8432778dc4ea3446cdd335002`.

The final immutable image packages only the focused Ticket 07 test file. The
suite passed directly in that image without a mounted checkout or ephemeral
dependency installation. It needs no `tblib`; that dependency had only been
pulled in by the root conftest during the superseded mounted-checkout attempt.
The two warnings were an unregistered repository-only pytest mark and an
unwritable pytest cache; no test was skipped.

Selected-file pre-commit passed Ruff check/format, typos, Markdown lint, mypy
3.10, SPDX, import/API/configuration checks, and all other applicable hooks.
The focused real-manager conformance fixture requires native `BlockRemoved`,
all three miss classes, useful-later reuse distance, and manager-defined
operation order. This supplies the LRU semantic gate that the monotonic pilot
does not naturally exercise.

The independent validator now reconstructs active, cached-resident, and free
occupancy from observer/native physical block events and rejects a
self-consistent ledger-only tamper. Focused tests also prove that both
out-of-capacity and pre-event invalid runs finalize schema-stable partial
artifacts with a complete ordered manifest prefix and a non-success provenance
status.

## Preserved failure and partial evidence

No failed or partial evidence was deleted:

- original no-selection diagnostic:
  `ticket-07-selection-failed-e3aa14e29.json`;
- scheme-A host-path planning diagnostic:
  `ticket-07-selection-scheme-a.json`;
- missing run-CLI hash initialization:
  `.ds4-kv-replay-20260720T081339Z-cddb3930.work/`;
- successful replay followed by validator hash-initialization failure:
  `ds4-kv-replay-20260720T081929Z-8a00c486/` and its sibling `.work/`;
- pre-fix image metadata:
  `ticket-07-image-pre-hash-init-failure.json` and
  `ticket-07-image-pre-validator-init-failure.json`;
- missing-`tblib` image pytest record: `ticket-07-pytest-final.json`.

The old accepted image remains tagged
`local/vllm-ds4-profile:ticket-07-pre-spec-review`; its results remain retained
but are superseded by the `b25fe1c0` acceptance above.

Both new-process hashing defects were in Ticket 07 CLI wiring, not vLLM cache
manager semantics. Each was reproduced by a failing test before the fix.

## Merge and repository authority

The branch is eligible for human review and merge into the personal fork after
the final branch review and push complete. The only writable GitHub repository
is `ycsxh/vllm`; `vllm-project/vllm` remains read-only. Push only the named
personal-fork remote. Do not create a PR unless separately authorized, and keep
personal-fork issue #4 open.
