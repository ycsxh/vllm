# syntax=docker/dockerfile:1.7

ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION=22.04
ARG VLLM_BUILD_COMMIT=unknown
ARG VLLM_BUILD_DIRTY=unknown

LABEL ai.vllm.ds4.cuda.version="${CUDA_VERSION}" \
      ai.vllm.ds4.python.version="${PYTHON_VERSION}" \
      ai.vllm.ds4.source.commit="${VLLM_BUILD_COMMIT}" \
      ai.vllm.ds4.source.dirty="${VLLM_BUILD_DIRTY}" \
      ai.vllm.ds4.ubuntu.version="${UBUNTU_VERSION}"

ENV VIRTUAL_ENV=/opt/ds4-profile
ENV PATH="/opt/ds4-profile/bin:${PATH}"
ENV DS4_VLLM_COMMIT=${VLLM_BUILD_COMMIT}
ENV DS4_VLLM_DIRTY=${VLLM_BUILD_DIRTY}

WORKDIR /vllm-workspace

COPY benchmarks/ds4_profile /vllm-workspace/benchmarks/ds4_profile

RUN --mount=type=cache,target=/opt/uv/cache \
    uv venv --system-site-packages /opt/ds4-profile && \
    uv pip install \
        --python /opt/ds4-profile/bin/python \
        -r benchmarks/ds4_profile/requirements.txt \
        -r benchmarks/ds4_profile/container/requirements.txt

ENTRYPOINT ["/opt/ds4-profile/bin/python", "-m", \
            "benchmarks.ds4_profile.container.runtime"]
