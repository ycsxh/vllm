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
`ec0d408b4044a10986f1d2099b82460d04e19438`. The later handoff-only commit does
not change or replace this acceptance boundary.

- final image tag: `local/vllm-ds4-profile:ticket-07`;
- final image ID:
  `sha256:f0dc7021b3ae479dbcb27e088547ffab39486ed92cf3d8c2678ca04e20e46105`;
- source dirty label: `false`;
- final image inspect SHA-256:
  `9c7407a35fabea64b1b6dd51f92e098d050e95b6c01d1dacef1c95eeb1085917`;
- reused immutable Ticket 03 base ID:
  `sha256:892b612fbcbea36cdfcd567c3f76b4993861e2d36a543ab96b56265839f872a1`.

Ticket 07 changes only Python metadata-replay and container registration
surfaces, not vLLM core or compiled extensions. The standard Ticket 03 base was
therefore reused for the isolated Ticket 07 overlay. An attempted full base
rebuild was canceled during CUDA extension compilation after confirming that
it added no Ticket 07 coverage; Ticket 05's concurrent build and evidence were
not reused or modified.

## Planning and pinned selection

The clean planning checkpoint was
`24176bdeaa35666b4167826892653a288350aea1` in image
`sha256:83f5c2e4088a2f9f8eed95da267ced988e1e03c637bbefcfb4aef16e201bf7b2`.
The container planning record is retained at:

```text
/home/lyc/ds4-storage/results/ticket-07-selection-container.json
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
  ds4-kv-replay-20260720T082747Z-853aec24
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

The result says exactly `Metadata only: yes`,
`Pilot eviction pressure observed: no`, `Native eviction count: 0`, and
`GPU/HBM validated: no`.

Artifact SHA-256 values:

- `cache_events.parquet`:
  `8ed8698b032bc30fd86799c4b95a04dcc13d8780516b20616495b3dfefee9f42`;
- `turn_summaries.parquet`:
  `5514e514c09c19a6716230438cb90e15d097f8e6fb62ed1f14c6cfaba6250242`;
- `run-config.json`:
  `e47bf10891ce8a5ff4f3f283375bae39b834e769933f9bf5f70faa3039724460`;
- `provenance.json`:
  `74eead16742890e76b85a313edc7c9ed2507d574215c60f29ef80ea4168e741a`;
- `result.md`:
  `3fb1cdefc5c6682891c845c7a1f05788e8e82071f72e6a8a9ba4571457d0cf25`.

## Test and lint evidence

Every Python/vLLM command set:

```bash
export PYTHONHASHSEED=0
export VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0
```

- host Ticket 07 plus container workflow: **53 passed**;
- real upstream manager regressions: **3 passed, 14 Torch deprecation
  warnings**;
- final-image focused Ticket 07 suite: **39 passed, 1 unknown-mark warning**;
- final-image pytest exec record:
  `/home/lyc/ds4-storage/results/ticket-07-pytest-with-deps.json`;
- final-image pytest exec-record SHA-256:
  `d63228b4d2396d1e3ba3e1fe1c113c9ae83ab0a18e98ac4443d8ab6758515de7`.

The image does not package repository tests. They were mounted read-only from
the same clean SHA. The first image test attempt stopped at collection because
`tblib` was absent and is retained in `ticket-07-pytest-final.json`. The passing
ephemeral container installed only `tblib==3.2.2` through `uv` before pytest;
the acceptance image itself was not changed or retagged.

Selected-file pre-commit passed Ruff check/format, typos, Markdown lint, mypy
3.10, SPDX, import/API/configuration checks, and all other applicable hooks.
The focused real-manager conformance fixture requires native `BlockRemoved`,
all three miss classes, useful-later reuse distance, and manager-defined
operation order. This supplies the LRU semantic gate that the monotonic pilot
does not naturally exercise.

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

Both new-process hashing defects were in Ticket 07 CLI wiring, not vLLM cache
manager semantics. Each was reproduced by a failing test before the fix.

## Merge and repository authority

The branch is eligible for human review and merge into the personal fork after
the final branch review and push complete. The only writable GitHub repository
is `ycsxh/vllm`; `vllm-project/vllm` remains read-only. Push only the named
personal-fork remote. Do not create a PR unless separately authorized, and keep
personal-fork issue #4 open.
