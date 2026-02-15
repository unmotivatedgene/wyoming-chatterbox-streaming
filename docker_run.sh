#!/bin/bash
set -e

# Default to CPU if CUDA not available
DEVICE="${DEVICE:-cuda}"

# Cache directory for HuggingFace models
HF_CACHE="${HF_CACHE:-/data/huggingface}"

echo "Starting Wyoming Chatterbox..."
echo "Device: $DEVICE"
echo "Model cache: $HF_CACHE"

# Create cache directory
mkdir -p "$HF_CACHE"

# Set HuggingFace cache environment
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE/transformers"

exec python -m wyoming_chatterbox \
    --device "$DEVICE" \
    --hf-token "$HF_TOKEN" \
    --uri "tcp://0.0.0.0:10200" \
    --http-port 5000 \
    --data-dir /data \
    "$@"
