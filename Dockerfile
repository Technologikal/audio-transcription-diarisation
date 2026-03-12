# Crucible-compatible Dockerfile for audio-transcription-diarisation
# Dual-mode: SERVER_MODE=http (Zone 3 pre-processing) or SERVER_MODE=mcp (Zone 5a tool)
#
# Build:
#   docker build --build-arg HF_TOKEN=hf_xxx \
#     -t registry.local:5000/crucible/voice-transcription:1.0.0 .
#
# Run:
#   docker run -e SERVER_MODE=http -p 8001:8001 voice-transcription
#   docker run -e SERVER_MODE=mcp voice-transcription
#
# Models are baked into the image at build time — no internet needed at runtime.
# HF_TOKEN is required at BUILD time for PyAnnote model download.
# At runtime, HF_TOKEN is read from /run/secrets/hf_token (Docker secrets).

FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies — install before copying source for better layer caching.
# Gradio is excluded (not needed for http/mcp server modes).
COPY transcription_project/requirements.txt /tmp/requirements.txt
RUN grep -v '^gradio' /tmp/requirements.txt > /tmp/requirements-server.txt \
    && pip install --no-cache-dir -r /tmp/requirements-server.txt \
    && pip install --no-cache-dir fastapi uvicorn python-multipart \
    && rm /tmp/requirements.txt /tmp/requirements-server.txt

# Bake faster-whisper large-v3 model into image (~3GB download, cached in layer).
# CTranslate2 format is downloaded on first use — trigger it now.
RUN python3 -c "\
from faster_whisper import WhisperModel; \
WhisperModel('large-v3', compute_type='int8')"

# Bake PyAnnote models into image (requires HF_TOKEN at build time).
# This downloads pyannote/speaker-diarisation-3.1 and its dependencies
# (pyannote/segmentation-3.0, wespeaker-voxceleb-resnet34-LM, etc.).
ARG HF_TOKEN
RUN if [ -z "$HF_TOKEN" ]; then \
        echo "ERROR: HF_TOKEN build arg is required to bake PyAnnote models." && \
        echo "Usage: docker build --build-arg HF_TOKEN=hf_xxx ..." && \
        exit 1; \
    fi && \
    python3 -c "\
from pyannote.audio import Pipeline; \
Pipeline.from_pretrained('pyannote/speaker-diarisation-3.1', use_auth_token='${HF_TOKEN}')"

# Copy application source
COPY transcription_project/ /app/

# Copy Crucible integration files
COPY server.py /app/server.py
COPY read_secret.py /app/read_secret.py

# Default to HTTP mode (Zone 3 pre-processing for voice notes)
ENV SERVER_MODE=http

EXPOSE 8001

ENTRYPOINT ["python3", "server.py"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8001/health || exit 1
