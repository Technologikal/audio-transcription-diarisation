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

# Optional faster model, selected with WHISPER_MODEL=large-v3-turbo. Baked
# rather than downloaded because a deployment may run this container with no
# outbound network at all, in which case an unbaked model is not a slow start
# — it is a hard failure at load time.
COPY hf_cache/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo /root/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo

# wav2vec2 alignment weights, for TRANSCRIPTION_BACKEND=whisperx and for
# TRANSCRIPTION_USE_ALIGNMENT=true. These come from torchaudio's own CDN
# rather than HuggingFace, so they need their own cache location and
# TORCH_HOME below to point at it.
COPY torch_cache/hub/checkpoints /root/.cache/torch/hub/checkpoints

# NLTK tokeniser data, required by the whisperx backend's alignment step.
# whisperx calls nltk.download() at runtime; on a network-isolated
# deployment that fails and takes the whole transcription with it.
COPY nltk_data /usr/local/share/nltk_data

# Copy application source
COPY transcription_project/ /app/

# Copy Crucible integration files
COPY server.py /app/server.py
COPY http_server.py /app/http_server.py
COPY read_secret.py /app/read_secret.py

# Default to HTTP mode (Zone 3 pre-processing for voice notes)
ENV SERVER_MODE=http

# Where torchaudio looks for the alignment weights baked in above. Without
# this it defaults elsewhere and tries to download — which fails closed on a
# network-isolated deployment.
ENV TORCH_HOME=/root/.cache/torch

EXPOSE 8001

ENTRYPOINT ["python3", "server.py"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8001/health || exit 1
