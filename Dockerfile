FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN groupadd -g 1001 appuser && useradd -m -u 1001 -g appuser appuser

WORKDIR /workspace
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN pip install --no-deps -e .

RUN chown -R appuser:appuser /workspace
USER appuser

CMD ["python", "scripts/train.py", "--help"]
