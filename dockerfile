FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and torchaudio with specific CUDA version as requested by original Dockerfile
RUN pip install --no-cache-dir \
    torch==2.8.0+cu128 \
    torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Copy all repository files (including omnivoice package, pyproject.toml, download_models.py, rp_handler.py)
COPY . /app/

# Install runpod SDK and other python requirements
RUN pip install --no-cache-dir runpod requests

# Install the OmniVoice package
RUN pip install -e /app/

# Download and cache HF model weights at build time
RUN python /app/download_models.py

# Start the RunPod handler
CMD ["python", "-u", "/app/rp_handler.py"]
