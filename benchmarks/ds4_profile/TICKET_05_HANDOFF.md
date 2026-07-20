# Ticket 05 P-side Prefill Profile Handoff

## 交接总览

### 这次任务做了什么

- 从交接 SHA `a59db6380383e8efc6db7ee3d18784b59391c199` 继续完成
  Ticket 05 的 Tasks 3-7，实现并验证固定 34 workloads / 68 points 的
  GPU0 P-side chunked-prefill profile。
- 完成真实 prefix-cache priming、GPU 同步、live
  `worker.model_runner.kv_caches` `cuda:0` tensor 检查，以及 prime 与
  measured SchedulerOutput 使用相同物理 block IDs 的证据链。
- 完成 full-batch KV capacity probe、合法 OOC terminal evidence、严格
  partial/preempted SchedulerOutput 拒绝、CUDA Graph runtime-mode 校验，
  并确保只有 `GPUWorker.execute_model` 进入 GPU timing。
- 为超过模型原生 32768 context 的计划 workload 加入并验证固定 YaRN
  配置；同时记录 model/runtime 配置并由 artifact validator 严格复核。
- 修复了完整矩阵才暴露的 mixed prefix-hit batch 问题：
  `cached_tokens == 0` 的请求不再执行伪 priming 或生成伪 HBM evidence，
  但仍 fail-closed 要求 measured computed tokens 为 0。真正有缓存的请求
  继续执行全部 GPU0/HBM/物理 block 检查。
- 将 Scheduler、KV lookup、allocation、cache reset 和 prefix prime 拆成
  互不重叠的 setup timing；OOC cleanup reset 在成功、抛错及后续序列化
  失败路径都会计入该 repetition 唯一的 `cache_reset_time_ms`。
- 全程复用 Ticket 07 base、权重、Torch compile/AOT 与 CUDA Graph 缓存；
  没有重编 vLLM/CUDA 核心。Ticket 05 使用独立镜像 tag、配置目录和结果
  目录，未覆盖 Ticket 04/07 证据。

### 已经完成什么

- Tasks 1-7 的实现、测试、两路独立 review 和最终硬件验收已完成；最终
  Standards/Spec review 均无 P0-P3。
- Runtime candidate 已提交为
  `aed08b145da6ef6ffdbd0028547c40f90a31152e`；正式镜像来源为干净的
  `git archive`，dirty label 为 `false`。
- CPU/static gates 已通过：最终聚焦 suites `148 passed, 2 skipped`，
  DS4 全目录在未挂载输入时 `176 passed, 2 skipped`，缺输入的 7 项在
  frozen snapshot/tokenizer/Ticket 01 只读挂载后全部通过；changed-files
  pre-commit 全通过。skip 没有被当成硬件通过。
- 最终 SHA 的正式 10-point smoke、独立 validator、hardware pytest 和
  68-point full matrix 均已通过；证据路径和精确统计见后文。
- 所有已完成的新 SHA 硬件 artifact 都证明 live `cuda:0` tensors、相同
  物理 block IDs、CUDA Graph `PIECEWISE`、4096 token / 8 sequence 上限，
  且没有 partial/preempted/unrelated GPU timing。

### 现在卡在哪

- Ticket 05 的实现与验收没有剩余 blocker；生命周期为 `remote_verified`。
- 尚未 push、未创建 PR、未关闭个人 fork issue #4。任何 push 都需要新的
  明确授权，并且只能显式使用 personal-fork remote；upstream 保持只读。

### 下一步怎么继续

1. 人工逐行复核所有 changed lines，并确认能端到端解释实现和证据。
2. 如需 push，先复核 remotes，只显式 push 到 `ycsxh/vllm` 对应 remote；
   不得 push 到 upstream。
3. 如以后准备 PR，先执行 AGENTS.md 的 duplicate-work checks，并在描述中
   完整披露 AI assistance、测试命令与 model/hardware evaluation 结果。
4. 不要创建 PR 或关闭 fork issue #4，除非用户另行明确授权。

### 踩过哪些坑

- 共享宿主 config 目录还是旧版，导致正式 smoke 缺少 YaRN overrides；
  后续改用每个 source SHA 独立、三文件齐全并校验 checksum 的配置目录。
- 第一次隔离配置只复制 profile，遗漏 `container-contract.json`，preflight
  fail-closed；该失败已保留，之后完整复制三个合同文件。
- 原始长上下文 workload 超过 Qwen 原生 32768，直接运行会触发 CUDA
  position assert；加入固定 YaRN 后又遇到 legacy/modern rope key 顺序问题，
  最终通过显式 ordering 和 post-EngineArgs 校验解决。
- full matrix 曾在 `high_skew-b2` 报 physical block mismatch。实际不是 r0
  的 786 个 HBM blocks 失效，而是同 batch 的 r1 有 `cached_tokens == 0`，
  旧代码为它生成 shape 不一致的空 evidence。修复 runtime 后，独立 reviewer
  又发现 validator 仍错误要求零缓存 evidence；两边都修复并加回归测试。
- 正式 benchmark 镜像只包含 `benchmarks/ds4_profile`，不含 pytest 文件；
  第一次 hardware pytest 因找不到测试返回 code 4。后续使用只 COPY 测试树、
  不安装不编译的 test overlay，并分别记录正式与测试镜像 ID。
- 自动审批曾因审批服务 `stream disconnected before completion` fail-closed，
  不是代码风险；取得用户明确授权后才继续 GPU 最小复现，没有绕过审批。
- 多次失败、partial、旧 SHA smoke 和 debug artifact 都必须原样保留；不得
  把静态检查、mock、skip、旧 SHA 结果或 validator wrapper 的
  `hardware_validated: false` 当成当前 full 硬件通过。

## Current status

- Branch: `codex/ticket-05-p-side-prefill-profile`
- Personal fork only: `ycsxh/vllm`
- Upstream: read-only; no upstream interaction is authorized
- Original implementation handoff: `a59db6380383e8efc6db7ee3d18784b59391c199`
- Runtime candidate source: `aed08b145da6ef6ffdbd0028547c40f90a31152e`
- Runtime candidate dirty state: `false`
- Runtime commit: `[Benchmarks] Account for OOC cleanup resets`
- Formal image: `local/vllm-ds4-profile:ticket-05-aed08b145da`
- Formal image ID:
  `sha256:6f3cd23c23621d74c130c82e36434df7969b2fd6558f61874a1027c5407ee7d1`
- Base image reused without rebuilding vLLM/CUDA:
  `local/vllm-ds4-profile:ticket-07-base`, ID
  `sha256:892b612fbcbea36cdfcd567c3f76b4993861e2d36a543ab96b56265839f872a1`
- Test overlay image ID:
  `sha256:b8b00939fca0d59d4153053584db4b92693193e809b453db20924027795469cc`
- Hardware lifecycle: `remote_verified`; full artifact has
  `hardware_validated: true` and the separate validator returned zero.
- Merge state: implementation and evidence complete; human review and any
  repository-state mutation remain the submitter's responsibility.
- Fork issue `ycsxh/vllm#4`: keep Open
- Do not create a PR without new explicit user authorization.

The docs-only handoff commit is intentionally separate from the runtime source
above, so it does not invalidate the clean runtime image provenance.

## Final full run

- Image: `local/vllm-ds4-profile:ticket-05-aed08b145da`
- Result directory:
  `/home/lyc/ds4-storage/results/ticket-05-acceptance-aed08b145da/full-v1`
- Independent validator:
  `/home/lyc/ds4-storage/results/ticket-05-acceptance-aed08b145da/full-v1-validation.json`
- Run ID: `ds4-p-prefill-20260720T201938Z-a3a6107f`
- Result: `remote_verified`, `hardware_validated: true`
- Counts: 68 points, 3280 raw rows, 832 turn rows, 64 aggregates,
  32 comparisons, 1108 prefix-evidence rows.
- Statuses: 3276 passed timed rows and four valid OOC terminal rows across
  four OOC points.
- All 836 repetitions have exactly one non-null, finite, non-negative
  `cache_reset_time_ms`; all 418 prefix-hit repetitions have exactly one
  prefix-prime timing.
- Every timed row used `PIECEWISE`; maximum scheduled work was 4096 tokens and
  eight sequences; no partial, preempted, unrelated, empty, or wrong-vector
  output was timed.
- Every evidence row used live `cuda:0` tensors and had identical prime,
  measured, and verified physical block IDs.

Key SHA-256 checksums:

- `image-inspect.json`: `e4f30b1798fd4c0339ebda4693a51f4707cdd96e54a4b3233f29cd5dbfc7684d`
- `test-image-inspect.json`: `fe79ab3df5d303276b83f628735246e2203daa636dbfb5d5018f4ea8a3ec0991`
- `full-v1/raw_samples.parquet`: `d8fb6147418e78434f160e9008828251c82db24bce8b3953421d143608a67af7`
- `full-v1/turn_samples.parquet`: `2bf5152bac463754dfef1e77c70dad111977e6c354ad95154eac20a807e81366`
- `full-v1/aggregates.parquet`: `545ef0d93d672b93c7f6a7b709b5ba43e1b1146f06075ab87a117464423e86f4`
- `full-v1/comparisons.parquet`: `2f6b901c12d702743ae055e6678b7d46d4ae862cc917049ed88b06bf022e3411`
- `full-v1/prefix_evidence.parquet`: `96f55dc7d1e91d99252b7d2824b3c40e7c26dc4a44c89ff9c4fcaf30e18bbac3`
- `full-v1/provenance.json`: `1edef0d834818b1d8b3c7fbbe3f6546e2e121a3864f96673f0654e48f24ecd5e`
- `full-v1/run-config.json`: `5598fc8aeeb87ab1666ee5f593bc95b3510ed0133ebd38a85cd77165c5fe000c`
- `full-v1-validation.json`: `0c073d45d51934398ce735af9176323b62267a610f3f6dc1b5b420dbcb2f2746`
- `hardware-pytest-exec.json`: `b4c4b6a78e45d1dbcd35530d986da285989b809d31501be8c81d1083f9509c65`
- `hardware-pytest-junit.xml`: `9e005498120ff9b7e82ad51e42cf6340c6ef1cb44824c40fc2ad0a420b714b4b`

## Frozen contract

- Config directory:
  `/home/lyc/ds4-storage/config-ticket-05-aed08b145da`
- `p-prefill-profile.json` SHA-256:
  `3875e753cedb86e84aad7388ffcccbb2ea62452c9fcf2881c64a80902dcfccb0`
- `container-contract.json` SHA-256:
  `10c59afae20297b38539a5dc2a0c54bb9063002c9da8fb5fdeec6a05d4fce884`
- `profile-spine.json` SHA-256:
  `1812e0f849a6dde62d9c5831f51d6997ee3b4b11ce81cd64888a2bd16527a3db`
- Workloads/points: 34/68
- Smoke: five selectors, ten hit/recompute points
- Per-step limits: 4096 scheduled tokens, eight sequences
- Homogeneous prefix/block size: 4096/16 tokens
- GPU/NUMA: GPU0/NUMA0, CPU set `0,2,4,6,8,10`
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Model revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`

## Verified new-SHA hardware gates

### Minimal regression

Result:
`/home/lyc/ds4-storage/results/ticket-05-debug-high-skew-f3ddc9502/minimal-v1`

- Selector: `high_skew-b2`, two hit/recompute points
- `remote_verified`, `hardware_validated: true`
- Worker passed; independent validator exit 0
- 91 passed raw rows; 13 evidence rows, all for the genuinely cached `r0`
- All evidence tensors are live `cuda:0` tensors
- Prime/measured/verified physical block IDs are exactly equal
- All timed rows use `PIECEWISE`; no partial/preempted/unrelated timing

Independent validator record:
`/home/lyc/ds4-storage/results/ticket-05-debug-high-skew-f3ddc9502/minimal-v1-validation.json`

### Formal smoke

Result:
`/home/lyc/ds4-storage/results/ticket-05-acceptance-aed08b145da/smoke`

- `remote_verified`, `hardware_validated: true`
- 10 planned points
- 288 raw rows: 286 passed and two valid OOC terminal rows
- 164 prefix evidence rows; all live `cuda:0` tensors
- Prime/measured/verified physical block IDs exactly equal
- All timed rows use `PIECEWISE`
- Maximum 4096 scheduled tokens and eight sequences
- No partial, preempted, unrelated, or vector-mismatched output was timed

Independent validator record:
`/home/lyc/ds4-storage/results/ticket-05-acceptance-aed08b145da/smoke-validation.json`

### Hardware pytest

- Test overlay: `local/vllm-ds4-profile:ticket-05-aed08b145da-tests`
- It only adds `tests/benchmarks/ds4_profile`; it does not install or compile.
- Result: `1 passed, 89 deselected in 505.83s`
- JUnit:
  `/home/lyc/ds4-storage/results/ticket-05-acceptance-aed08b145da/hardware-pytest-junit.xml`
- Exec record:
  `/home/lyc/ds4-storage/results/ticket-05-acceptance-aed08b145da/hardware-pytest-exec.json`

## Failure history -- preserve all artifacts

- Results under
  `/home/lyc/ds4-storage/results/ticket-05-acceptance-aed08b145` are invalid
  diagnostic evidence: their image label used the nonexistent expanded SHA
  `aed08b145d4b841b6b02667657862d1d9d36f9cb`. They were preserved and fully
  rerun under `ticket-05-acceptance-aed08b145da` with the real immutable commit
  `aed08b145da6ef6ffdbd0028547c40f90a31152e`; never cite the former as accepted.
- Old full result:
  `/home/lyc/ds4-storage/results/ticket-05-acceptance-37c7f101f/full-isolated-v2`
  is `remote_failed`. It completed 43/68 points and failed at
  `high_skew-b2 / prefix_hit / warmup 0` with
  `measured physical block IDs do not match the prime`.
- Root cause: mixed prefix-hit batches could contain a request with
  `cached_tokens == 0`. Runtime created group-shape-incompatible empty prime
  evidence for that request, and the validator expected evidence for it.
- Fix in `f3ddc9502`: zero-cached requests are not primed and emit no prefix
  evidence; they still fail closed unless measured computed tokens are zero.
  Nonzero cached requests retain all GPU0 HBM and exact physical-ID checks.
- The fix was test-first and received independent spec and standards reviews.
  The spec reviewer found a P1 validator mismatch; it was fixed and rereviewed.
  Both reviews ended with no P0-P2 findings.
- Earlier failed/debug directories under
  `/home/lyc/ds4-storage/results/ticket-05-*` must also remain unchanged,
  including long-context CUDA assert, YaRN ordering, stale shared config, and
  incomplete isolated-config attempts.

## CPU/static gates passed for `aed08b145da`

- Focused prefill/profile-spine suites: `148 passed, 2 skipped`
- Final prefill suite: `89 passed, 1 skipped`
- DS4 directory without input mounts: `176 passed, 2 skipped`; seven workload
  tests failed only because their fixed `.scratch` inputs were absent.
- The seven input-backed workload tests passed with frozen inputs mounted
  read-only; the remaining source-fallback test passed with the committed
  DeepSeek tokenizer module mounted at its expected repository path.
- Changed-files pre-commit passed, including Ruff check/format, mypy, typos,
  SPDX and repository-specific hooks.
- Commit hooks passed.

The skips are hardware-gated and are not hardware passes. The separate
hardware pytest above is the actual server evidence.

## Remaining repository actions

1. Human submitter reviews every changed line and the evidence above.
2. Push only with explicit authorization and only to the personal-fork remote.
3. Do not create a PR or close fork issue #4 without separate authorization.

## Acceptance reminders

- Every feasible point has all planned chunks for three warmups and ten steady
  repetitions. OOC must have only a valid coordinate prefix and terminal row.
- Empty, partial, preempted, unrelated, or wrong-vector SchedulerOutput is
  never timed.
- Only `GPUWorker.execute_model` is inside CUDA-event/wall timing; setup,
  priming, compile, capture, reset and warmup remain separate.
- Recompute proves zero cached tokens.
- Every nonzero hit prefix proves completed and synchronized GPU0 priming,
  live `worker.model_runner.kv_caches` tensors on `cuda:0`, and exact reuse of
  the same physical block IDs. Token IDs alone are not proof.
- All measured rows use CUDA Graph `FULL` or `PIECEWISE`.
- Every passed pair has exactly one comparison; valid OOC pairs have none.
- Never convert skip, mock, static checks, bootstrap failure or partial output
  into hardware evidence.
