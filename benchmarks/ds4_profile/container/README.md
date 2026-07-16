# DS4 school-server container runbook

This workflow builds the current personal-fork checkout into a local OCI image.
It never downloads vLLM source during the build. Docker is the validated primary
runtime; the Apptainer notes at the end are a conversion route only.

## Host prerequisites

The validated target is Ubuntu 22.04 on x86_64 with two NVIDIA GeForce RTX 3090
GPUs and driver 580.159.03. Install and configure these host components before
building:

- Docker Engine with permission for the experiment user to access its daemon.
- NVIDIA Container Toolkit configured for Docker.
- Enough space in Docker's data root for the CUDA build layers.
- Persistent directories for the raw snapshot, Ticket 01/02 artifacts,
  tokenizers, Hugging Face cache, uv cache, configuration, and results.

NVIDIA's supported Docker configuration command is
`sudo nvidia-ctk runtime configure --runtime=docker`, followed by restarting the
Docker daemon. Follow the current
[NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
instead of copying an old repository bootstrap recipe.

Check the host before proceeding:

```bash
docker version
docker info --format '{{json .Runtimes}}'
nvidia-ctk --version
nvidia-container-cli --version
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi topo -p2p r
docker run --rm --gpus all \
  nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi
```

Host `uv` is optional for the container workflow because the image contains
`uv==0.11.28` and its own `/opt/ds4-profile` virtual environment. It is required
for running repository Python checks directly on the host. Check it with
`command -v uv && uv --version`; if absent, install it through the school's
package manager, environment module, or managed Conda channel. Do not hide an
unpinned remote installer inside build or run commands.

## Persistent layout

Use paths owned by the experiment account. The examples below assume:

```text
/path/to/ds4-storage/
├── cache/
│   ├── huggingface/
│   ├── runtime/
│   └── uv/
├── config/
│   └── container-contract.json
├── results/
├── snapshot/4da61f3d06b48b6817a62b99e9c47035c8e59787/
├── ticket-01/
├── ticket-02/
└── tokenizers/
```

Copy `benchmarks/ds4_profile/config/container-contract.json` into `config/`.
Keep the immutable raw snapshot and Ticket 01/02 directories read-only. The
wrapper mounts them as:

| Host input | Container path | Mode |
| --- | --- | --- |
| Raw snapshot | `/mnt/ds4/raw` | read-only |
| Ticket 01 | `/mnt/ds4/ticket-01` | read-only |
| Ticket 02 | `/mnt/ds4/ticket-02` | read-only |
| Tokenizers | `/mnt/ds4/tokenizers` | read-only |
| Configuration | `/mnt/ds4/config` | read-only |
| Hugging Face cache | `/mnt/ds4/cache/huggingface` | persistent |
| vLLM/Triton/Inductor cache | `/mnt/ds4/cache/runtime` | persistent |
| uv cache | `/mnt/ds4/cache/uv` | persistent |
| Results | `/mnt/ds4/results` | persistent |

## Build from the personal fork

Run from the root of the already-synchronized personal-fork checkout. Confirm
the checkout and dirty state before building; the script records both in image
labels and Docker inspection metadata.

```bash
git rev-parse HEAD
git status --short
bash benchmarks/ds4_profile/container/build.sh \
  --image local/vllm-ds4-profile:ticket-03 \
  --metadata-out /path/to/ds4-storage/results/image-inspect.json
```

The first build compiles the local checkout using the existing vLLM
`docker/Dockerfile`, CUDA 13.0.2, Python 3.12, Ubuntu 22.04, and
`INSTALL_KV_CONNECTORS=true`. The overlay creates the isolated profile virtual
environment. It does not contact or mutate any remote source repository.

Use `tmux` for the first build over VS Code Tunnel. Docker layer caching makes a
restart safe: rerun the same command and completed layers are reused.

## Common run arguments

Every command below uses the same explicit mounts. Replace the placeholder root
once and reuse the shell array:

```bash
STORAGE=/path/to/ds4-storage
DS4_RUN=(
  bash benchmarks/ds4_profile/container/run.sh
  --image local/vllm-ds4-profile:ticket-03
  --snapshot-dir "$STORAGE/snapshot/4da61f3d06b48b6817a62b99e9c47035c8e59787"
  --ticket-01-dir "$STORAGE/ticket-01"
  --ticket-02-dir "$STORAGE/ticket-02"
  --tokenizers-dir "$STORAGE/tokenizers"
  --config-dir "$STORAGE/config"
  --hf-cache-dir "$STORAGE/cache/huggingface"
  --runtime-cache-dir "$STORAGE/cache/runtime"
  --uv-cache-dir "$STORAGE/cache/uv"
  --results-dir "$STORAGE/results"
)
```

GPU-capable commands use both GPUs, host IPC, host networking, unlimited
memlock, and the minimal `SYS_NICE` capability required for NUMA memory
binding. The container is not privileged. `cache-model` and `cpu-dry-run` omit
the NVIDIA runtime entirely. The wrapper records its invocation and image ID.
GPU0 is the Prefill role and is bound to CPU IDs `0,2,4,6,8,10` on NUMA node 0.
GPU1 is the Decode role and is bound to `1,3,5,7,9,11` on NUMA node 1.
Preflight re-detects topology instead of trusting these recorded observations.

## Cache preparation and offline reuse

Allow network access only for the pinned model-cache preparation step:

```bash
"${DS4_RUN[@]}" cache-model
```

This downloads `Qwen/Qwen2.5-Coder-7B-Instruct` at revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242` into the persistent Hugging Face
cache. A partial download is restart-safe through the Hugging Face cache.
All other wrapper commands set `HF_HUB_OFFLINE=1`; replacing the container does
not redownload weights, tokenizers, datasets, or Ticket artifacts.

## Acceptance sequence

Run the cheap checks before loading the model:

```bash
"${DS4_RUN[@]}" preflight
"${DS4_RUN[@]}" cpu-dry-run
"${DS4_RUN[@]}" gpu-smoke
```

`preflight` verifies pinned input hashes, writable results, `uv`, NIXL core and
UCX-linked bindings, CUDA through PyTorch, two visible RTX 3090s, driver,
topology, P2P read status, and `numactl`. It emits `preflight.json` with status
`ready` or `invalid`; readiness alone is not hardware validation.

`cpu-dry-run` regenerates Ticket 01 normalization and Ticket 02 workloads into
`results/cpu-dry-run/`, then compares four generated SHA-256 values to the
pinned artifacts. It never overwrites the mounted Ticket directories.

`gpu-smoke` first reruns preflight. If prerequisites fail, it writes a
`skipped` record instead of a success record. Otherwise it starts two
NUMA-bound workers concurrently, loads the smallest selected exact replay,
loads the pinned Qwen model independently on each GPU, and performs one
`vllm.LLM.generate` model-runner point per role. Only two passing worker records
produce `hardware_validated: true`.

Subsequent profile tickets can run an explicit command through the same image
and mounts while recording its exact exit status:

```bash
"${DS4_RUN[@]}" exec \
  --output /mnt/ds4/results/profile-invocation.json \
  -- /opt/ds4-profile/bin/python -m benchmarks.ds4_profile.PROFILE_MODULE
```

Replace `PROFILE_MODULE` only with the implemented profile entry point from a
later ticket. Ticket 03 does not invent results for profile stages that do not
yet exist.

## Results and restart behavior

Keep these evidence files with the experiment:

- `image-inspect.json`
- `cache-model.json`
- `preflight.json` and `gpu-smoke-preflight.json`
- `cpu-dry-run/provenance.json`
- `gpu-smoke.json`, `gpu-smoke-prefill.json`, and `gpu-smoke-decode.json`
- later profile invocation and artifact files

All expensive state is outside the disposable container. After a disconnect,
restart the same command in `tmux`: Docker reuses build layers, Hugging Face
resumes its cache, and result records are replaced only by a new explicit run.
Use a new results directory when preserving multiple attempts.

## Troubleshooting

- **Docker permission denied:** verify group membership or use the school's
  approved rootless configuration. Do not make the container privileged.
- **GPUs absent in the container:** rerun the fixed CUDA `nvidia-smi` smoke,
  inspect Docker runtimes, and reconfigure NVIDIA Container Toolkit.
- **CUDA initialization or compatibility failure:** compare the recorded host
  driver, `torch.version.cuda`, image CUDA 13.0.2, and device visibility. Do not
  enable forward-compatibility flags blindly on GeForce hardware.
- **NIXL import failure:** inspect the recorded `nixl`, `nixl-cu13`, PyTorch,
  and CUDA versions. Rebuild with `INSTALL_KV_CONNECTORS=true`; do not install a
  moving package interactively into a finished image.
- **UCX transport failure:** retain host networking, inspect NIXL binding load
  errors, and keep GPU P2P disabled for this `SYS`/`CNS` topology. CPU host
  staging is the supported baseline.
- **NUMA binding failure:** compare `nvidia-smi topo -m` with the recorded role
  CPU sets. A topology change must update configuration and provenance rather
  than being ignored.
- **Artifact hash mismatch:** stop. Verify that the mounted snapshot and Ticket
  01/02 directories are the pinned immutable artifacts; do not regenerate them
  in place.
- **Out of memory during smoke:** confirm no other GPU users, preserve FP16,
  and record the failure. Do not silently reduce the intended profile model.

## Apptainer conversion route

If a future school environment forbids Docker, convert the already-built local
image rather than rebuilding from a different source:

```bash
docker save local/vllm-ds4-profile:ticket-03 \
  -o /path/to/ds4-storage/vllm-ds4-profile.tar
apptainer build /path/to/ds4-storage/vllm-ds4-profile.sif \
  docker-archive:/path/to/ds4-storage/vllm-ds4-profile.tar
```

Apptainer also supports `docker-daemon:local/vllm-ds4-profile:ticket-03` when it
can access the Docker daemon. See the
[Apptainer OCI conversion documentation](https://apptainer.org/docs/user/latest/docker_and_oci.html#archives-docker-daemon).
The converted path is not hardware-validated by Ticket 03. Its GPU flags,
environment propagation, writable bind paths, host networking, memlock, and
NUMA semantics must be validated separately before claiming parity with the
Docker workflow.
