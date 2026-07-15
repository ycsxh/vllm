#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CUDA_VERSION="13.0.2"
PYTHON_VERSION="3.12"
UBUNTU_VERSION="22.04"
DRY_RUN=false
IMAGE="local/vllm-ds4-profile:dev"
METADATA_OUT="${REPO_DIR}/.scratch/ds4-container-build.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --metadata-out)
            METADATA_OUT="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=normal)" ]]; then
    DIRTY=true
else
    DIRTY=false
fi
BASE_IMAGE="${IMAGE}-base"
BUILD_BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
FINAL_BASE_IMAGE="nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}"

base_build=(
    docker build
    --target vllm-openai-base
    --build-arg "BUILD_BASE_IMAGE=${BUILD_BASE_IMAGE}"
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --build-arg "FINAL_BASE_IMAGE=${FINAL_BASE_IMAGE}"
    --build-arg INSTALL_KV_CONNECTORS=true
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
    --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}"
    --build-arg "VLLM_BUILD_COMMIT=${COMMIT}"
    --build-arg "VLLM_IMAGE_TAG=${BASE_IMAGE}"
    --tag "${BASE_IMAGE}"
    --file "${REPO_DIR}/docker/Dockerfile"
    "${REPO_DIR}"
)
profile_build=(
    docker build
    --build-arg "VLLM_BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
    --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}"
    --build-arg "VLLM_BUILD_COMMIT=${COMMIT}"
    --build-arg "VLLM_BUILD_DIRTY=${DIRTY}"
    --tag "${IMAGE}"
    --file "${SCRIPT_DIR}/Dockerfile"
    "${REPO_DIR}"
)

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if [[ "${DRY_RUN}" == true ]]; then
    print_command "${base_build[@]}"
    print_command "${profile_build[@]}"
    exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to build the DS4 profile image" >&2
    exit 1
fi

"${base_build[@]}"
"${profile_build[@]}"
mkdir -p "$(dirname -- "${METADATA_OUT}")"
docker image inspect "${IMAGE}" >"${METADATA_OUT}"
echo "wrote image metadata to ${METADATA_OUT}"
