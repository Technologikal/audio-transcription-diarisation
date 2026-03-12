"""
HTTP server for Crucible Zone 3 pre-processing (voice note transcription).

Thin FastAPI wrapper around the transcription pipeline. Designed for short
voice notes (typically < 2 minutes, single speaker). The bridge sends audio
via multipart upload and receives plain transcript text.

Endpoints:
    POST /transcribe  — multipart file upload → {"transcript": "..."}
    GET  /health      — returns 200 when ready
"""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from read_secret import read_secret

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="voice-transcription", version="1.0.0")

# Lazy-loaded pipeline components (loaded on first request to keep startup fast)
_pipeline_loaded = False


def _ensure_pipeline():
    """Load HF_TOKEN into environment so the pipeline can access PyAnnote models."""
    global _pipeline_loaded
    if _pipeline_loaded:
        return

    # Read HF_TOKEN from Docker secret or env var
    try:
        hf_token = read_secret("hf_token", fallback_env="HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
    except RuntimeError:
        logger.warning("HF_TOKEN not found — diarisation will fail if models aren't cached")

    _pipeline_loaded = True


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file and return plain transcript text.

    Optimised for short voice notes (< 2 minutes, single speaker).
    Uses the medium model with no wav2vec2 alignment for speed.

    Args:
        file: Audio file (multipart upload, field name "file")

    Returns:
        {"transcript": "transcribed text here"}
    """
    _ensure_pipeline()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        return JSONResponse(
            status_code=500,
            content={"error": "HF_TOKEN not configured"},
        )

    # Write uploaded file to a temporary location
    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from pipeline import transcribe_with_diarisation

        result = transcribe_with_diarisation(
            audio_path=tmp_path,
            hf_token=hf_token,
            whisper_model_name=os.environ.get("WHISPER_MODEL", "medium"),
            language="english",
            chunk_duration_seconds=300,  # Shorter chunks for voice notes
            use_alignment=False,         # Skip wav2vec2 for speed
            backend="faster-whisper",
        )

        # Concatenate all segments into a single transcript string
        transcript = " ".join(
            seg.text.strip() for seg in result.segments if seg.text.strip()
        )

        logger.info(
            "Transcribed %s: %d chars, %.1fs processing time",
            file.filename,
            len(transcript),
            result.elapsed_time,
        )

        return {"transcript": transcript}

    except Exception as exc:
        logger.error("Transcription failed for %s: %s", file.filename, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
