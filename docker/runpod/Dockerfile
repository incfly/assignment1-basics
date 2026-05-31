# syntax=docker/dockerfile:1

# Keep the RunPod-provided CUDA/PyTorch stack that worked on the L4 pod.
# Do not run `uv sync` in this image; it may replace Torch with a CUDA build
# that requires a newer host driver than RunPod provides.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TRAIN_SETUP_MODE=system-torch \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-transport-https \
        aria2 \
        ca-certificates \
        curl \
        git \
        gnupg \
        rsync \
        tmux \
    && install -m 0755 -d /usr/share/keyrings \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && chmod a+r /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        einops==0.8.2 \
        einx==0.4.3 \
        jaxtyping==0.3.10 \
        tqdm==4.67.3 \
        uv

RUN python3 - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_build", torch.version.cuda)
PY
