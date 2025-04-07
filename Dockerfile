FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        wget \
        libgl1 \
        git-lfs \
        curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3.12 /usr/bin/pip && \
    git lfs install && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir \
        huggingface_hub[hf_transfer] \
        torch \
        torchvision \
        torchaudio \
        xformers \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir \
        runpod \
        requests

COPY . /workspace
WORKDIR /workspace

ARG HUGGINGFACE_ACCESS_TOKEN

# Added huggingface login code
RUN huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN && \
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Comfy-Org/flux1-dev flux1-dev-fp8.safetensors --local-dir /workspace/ComfyUI/models/checkpoints/

COPY ComfyUI/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /tmp/requirements.txt

# COPY start.sh /start.sh
# COPY rp_handler.py /rp_handler.py

# CMD ["bash", "/start.sh"]
CMD ["python", "-u", "rp_handler.py"]