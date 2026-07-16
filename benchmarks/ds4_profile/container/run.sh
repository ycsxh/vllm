#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

ORIGINAL_ARGS=("$@")
printf -v HOST_INVOCATION '%q ' "$0" "${ORIGINAL_ARGS[@]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
ARTIFACT_ROOT="${REPO_DIR}/.scratch/ds4-agent-1p1d-profile"

CONFIG_DIR="${REPO_DIR}/benchmarks/ds4_profile/config"
DRY_RUN=false
HF_CACHE_DIR="${ARTIFACT_ROOT}/cache/huggingface"
IMAGE="local/vllm-ds4-profile:dev"
RESULTS_DIR="${ARTIFACT_ROOT}/results"
RUNTIME_CACHE_DIR="${ARTIFACT_ROOT}/cache/runtime"
SNAPSHOT_DIR="${ARTIFACT_ROOT}/snapshot/4da61f3d06b48b6817a62b99e9c47035c8e59787"
TICKET_01_DIR="${ARTIFACT_ROOT}/artifacts/ticket-01"
TICKET_02_DIR="${ARTIFACT_ROOT}/artifacts/ticket-02"
TOKENIZERS_DIR="${ARTIFACT_ROOT}/tokenizers"
UV_CACHE_DIR="${ARTIFACT_ROOT}/cache/uv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --hf-cache-dir)
            HF_CACHE_DIR="$2"
            shift 2
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --runtime-cache-dir)
            RUNTIME_CACHE_DIR="$2"
            shift 2
            ;;
        --snapshot-dir)
            SNAPSHOT_DIR="$2"
            shift 2
            ;;
        --ticket-01-dir)
            TICKET_01_DIR="$2"
            shift 2
            ;;
        --ticket-02-dir)
            TICKET_02_DIR="$2"
            shift 2
            ;;
        --tokenizers-dir)
            TOKENIZERS_DIR="$2"
            shift 2
            ;;
        --uv-cache-dir)
            UV_CACHE_DIR="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "a container command is required" >&2
    exit 2
fi

if [[ "$1" == "cache-model" ]]; then
    HF_HUB_OFFLINE=0
else
    HF_HUB_OFFLINE=1
fi

gpu_args=(--label ai.vllm.ds4.runtime=cpu)
if [[ "$1" != "cache-model" && "$1" != "cpu-dry-run" ]]; then
    gpu_args=(--gpus all --cpuset-cpus=0-11 --cap-add SYS_NICE)
fi

if [[ "${DRY_RUN}" == true ]]; then
    IMAGE_ID=dry-run
else
    if ! command -v docker >/dev/null 2>&1; then
        echo "docker is required to run the DS4 profile image" >&2
        exit 1
    fi
    for input_dir in \
        "${SNAPSHOT_DIR}" \
        "${TICKET_01_DIR}" \
        "${TICKET_02_DIR}" \
        "${TOKENIZERS_DIR}" \
        "${CONFIG_DIR}"; do
        if [[ ! -d "${input_dir}" ]]; then
            echo "required input directory does not exist: ${input_dir}" >&2
            exit 1
        fi
    done
    mkdir -p \
        "${HF_CACHE_DIR}" \
        "${UV_CACHE_DIR}" \
        "${RUNTIME_CACHE_DIR}" \
        "${RESULTS_DIR}"
    IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}")"
fi

run=(
    docker run
    --rm
    "${gpu_args[@]}"
    --ipc=host
    --network=host
    --ulimit memlock=-1:-1
    --user "$(id -u):$(id -g)"
    --env DS4_PREFILL_GPU=0
    --env DS4_DECODE_GPU=1
    --env "DS4_HOST_CONFIG_DIR=${CONFIG_DIR}"
    --env "DS4_HOST_HF_CACHE_DIR=${HF_CACHE_DIR}"
    --env "DS4_HOST_INVOCATION=${HOST_INVOCATION}"
    --env "DS4_HOST_RESULTS_DIR=${RESULTS_DIR}"
    --env "DS4_HOST_RUNTIME_CACHE_DIR=${RUNTIME_CACHE_DIR}"
    --env "DS4_HOST_SNAPSHOT_DIR=${SNAPSHOT_DIR}"
    --env "DS4_HOST_TICKET_01_DIR=${TICKET_01_DIR}"
    --env "DS4_HOST_TICKET_02_DIR=${TICKET_02_DIR}"
    --env "DS4_HOST_TOKENIZERS_DIR=${TOKENIZERS_DIR}"
    --env "DS4_HOST_UV_CACHE_DIR=${UV_CACHE_DIR}"
    --env "DS4_PREFILL_CPUSET=0,2,4,6,8,10"
    --env "DS4_DECODE_CPUSET=1,3,5,7,9,11"
    --env "DS4_IMAGE_ID=${IMAGE_ID}"
    --env HF_HOME=/mnt/ds4/cache/huggingface
    --env HF_HUB_CACHE=/mnt/ds4/cache/huggingface
    --env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
    --env HOME=/mnt/ds4/cache/runtime/home
    --env LOGNAME=ds4-profile
    --env TORCHINDUCTOR_CACHE_DIR=/mnt/ds4/cache/runtime/torchinductor
    --env TRITON_CACHE_DIR=/mnt/ds4/cache/runtime/triton
    --env USER=ds4-profile
    --env UV_CACHE_DIR=/mnt/ds4/cache/uv
    --env VLLM_CACHE_ROOT=/mnt/ds4/cache/runtime/vllm
    --volume "${SNAPSHOT_DIR}:/mnt/ds4/raw:ro"
    --volume "${TICKET_01_DIR}:/mnt/ds4/ticket-01:ro"
    --volume "${TICKET_02_DIR}:/mnt/ds4/ticket-02:ro"
    --volume "${TOKENIZERS_DIR}:/mnt/ds4/tokenizers:ro"
    --volume "${CONFIG_DIR}:/mnt/ds4/config:ro"
    --volume "${HF_CACHE_DIR}:/mnt/ds4/cache/huggingface"
    --volume "${UV_CACHE_DIR}:/mnt/ds4/cache/uv"
    --volume "${RUNTIME_CACHE_DIR}:/mnt/ds4/cache/runtime"
    --volume "${RESULTS_DIR}:/mnt/ds4/results"
    "${IMAGE}"
    "$@"
)

if [[ "${DRY_RUN}" == true ]]; then
    printf '%q ' "${run[@]}"
    printf '\n'
else
    "${run[@]}"
fi
