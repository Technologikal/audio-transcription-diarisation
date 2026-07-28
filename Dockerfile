# Crucible-compatible Dockerfile for audio-transcription-diarisation
# Dual-mode: SERVER_MODE=http (Zone 3 pre-processing) or SERVER_MODE=mcp (Zone 5a tool)
#
# Build (models copied from host cache — fast):
#   docker build -t registry.local:5000/crucible/voice-transcription:1.0.0 .
#
# Run:
#   docker run -e SERVER_MODE=http -p 8001:8001 voice-transcription
#   docker run -e SERVER_MODE=mcp voice-transcription
#
# Models are baked into the image at build time — no internet needed at runtime.
# Prerequisites: models must be cached on the host in ~/.cache/huggingface/hub/
# (they are downloaded automatically on first use of the CLI tool).

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

# Bake models from host cache into image (avoids slow in-container downloads).
# These are copied from ~/.cache/huggingface/hub/ on the build host.
# The HuggingFace hub library looks for models here at runtime.
COPY hf_cache/models--Systran--faster-whisper-large-v3 /root/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3
COPY hf_cache/models--pyannote--segmentation-3.0 /root/.cache/huggingface/hub/models--pyannote--segmentation-3.0
COPY hf_cache/models--pyannote--speaker-diarization-3.1 /root/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1
COPY hf_cache/models--pyannote--wespeaker-voxceleb-resnet34-LM /root/.cache/huggingface/hub/models--pyannote--wespeaker-voxceleb-resnet34-LM

# Copy application source
COPY transcription_project/ /app/

# Copy Crucible integration files
COPY server.py /app/server.py
COPY http_server.py /app/http_server.py
COPY read_secret.py /app/read_secret.py

# Default to HTTP mode (Zone 3 pre-processing for voice notes)
ENV SERVER_MODE=http

EXPOSE 8001

ENTRYPOINT ["python3", "server.py"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8001/health || exit 1
