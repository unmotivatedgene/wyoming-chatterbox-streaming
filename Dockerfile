FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python venv
WORKDIR /app
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -U setuptools wheel

# Install torch CPU only
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install from git to get the correct perth module
RUN pip install --no-cache-dir git+https://github.com/resemble-ai/Perth.git

# Install chatterbox-tts
RUN pip install --no-cache-dir chatterbox-tts

# Install wyoming and HTTP server deps
RUN pip install --no-cache-dir wyoming sentence-stream aiohttp

# Copy source and pyproject.toml
COPY src/ /app/src/
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e /app

# Create data directories and set cache env
RUN mkdir -p /data/prompts /data/huggingface
ENV HF_HOME=/data/huggingface
ENV TRANSFORMERS_CACHE=/data/huggingface/transformers

# Expose ports
EXPOSE 10200 5000

COPY docker_run.sh /
RUN chmod +x /docker_run.sh

ENTRYPOINT ["/docker_run.sh"]
