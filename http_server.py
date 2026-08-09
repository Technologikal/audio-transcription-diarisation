"""
HTTP server frontend for the transcription pipeline (voice notes and meetings).

Thin FastAPI wrapper around the transcription pipeline. Designed for short
voice notes (typically < 2 minutes, single speaker). The bridge sends audio
via multipart upload and receives plain transcript text.

Endpoints:
    POST /transcribe  — multipart file upload → {"transcript": "..."}
    GET  /health      — returns 200 when ready

Note: callers commonly impose their own request timeout; long recordings
should be submitted through a caller that tolerates minutes-to-hours runtimes.
With large-v3 on CPU, a 1-minute voice note takes ~2-3 minutes to process.
The bridge timeout will need increasing, or the bridge should poll/stream.
This becomes negligible with GPU (e.g. RTX 3060 upgrade).
"""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
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
        # Secret name is configuration: deployments that mount it under a
        # different name set HF_TOKEN_SECRET. Defaults to the plain name so
        # the service works standalone with no deployment-specific setup.
        hf_token = read_secret(
            os.environ.get("HF_TOKEN_SECRET", "hf_token"), fallback_env="HF_TOKEN"
        )
        os.environ["HF_TOKEN"] = hf_token
    except RuntimeError:
        logger.warning("HF_TOKEN not found — diarisation will fail if models aren't cached")

    _pipeline_loaded = True


def _format_segments_with_speakers(segments) -> str:
    """Collapse a SpeakerSegment list into a diarised plain-text block.

    Returns one line per speaker turn. Consecutive segments from the
    same speaker are merged so the operator sees:

        [Alice] Hi, are you alright?
        [Bob] Yeah, good thanks. Good morning.
        [Alice] It's gorgeous. Nice to be working.

    Falls back to a plain-text concatenation when every segment is
    "UNKNOWN" (i.e. skip_diarisation=True or pyannote wasn't run),
    so voice notes still produce clean output without pointless
    [UNKNOWN] labels.
    """
    cleaned = [s for s in segments if (s.text or "").strip()]
    if not cleaned:
        return ""

    def _label(seg):
        name = getattr(seg, "real_name", None)
        if name:
            return name
        return getattr(seg, "speaker_label", None) or "UNKNOWN"

    all_labels = {_label(s) for s in cleaned}
    if all_labels <= {"UNKNOWN"}:
        return " ".join(s.text.strip() for s in cleaned)

    lines: list[str] = []
    current_label: str | None = None
    current_parts: list[str] = []

    def _flush():
        if current_label is not None and current_parts:
            lines.append(f"[{current_label}] {' '.join(current_parts).strip()}")

    for seg in cleaned:
        label = _label(seg)
        text = seg.text.strip()
        if label == current_label:
            current_parts.append(text)
        else:
            _flush()
            current_label = label
            current_parts = [text]
    _flush()

    return "\n".join(lines)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    skip_diarisation: bool = Form(False),
):
    """Transcribe an uploaded audio file and return plain transcript text.

    Args:
        file: Audio file (multipart upload, field name "file")
        skip_diarisation: If true, skip speaker diarisation (faster, no HF_TOKEN needed)

    Returns:
        {"transcript": "transcribed text here"}
    """
    _ensure_pipeline()

    hf_token = os.environ.get("HF_TOKEN", "")

    # HF_TOKEN is only required when diarisation is enabled
    if not skip_diarisation and not hf_token:
        return JSONResponse(
            status_code=500,
            content={"error": "HF_TOKEN not configured and diarisation is enabled"},
        )

    logger.info(
        "Transcribing %s (skip_diarisation=%s)",
        file.filename, skip_diarisation,
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
            whisper_model_name=os.environ.get("WHISPER_MODEL", "large-v3"),
            language="english",
            chunk_duration_seconds=300,  # Shorter chunks for voice notes
            use_alignment=False,         # Skip wav2vec2 for speed
            backend="faster-whisper",
            skip_diarisation=skip_diarisation,
        )

        # Format segments with speaker labels preserved so callers receive a
        # properly diarised transcript rather than an unlabelled block.
        # Previous behaviour flattened seg.text only, which
        # silently dropped seg.speaker_label and produced a big unlabeled
        # block (fix-49 field test 2026-04-11, Bug 7 triage).
        #
        # Output model:
        # - skip_diarisation=True OR every segment is "UNKNOWN" →
        #   plain concatenated text, no labels (voice notes, single-
        #   speaker recordings)
        # - Otherwise → one line per speaker turn, prefixed with
        #   the mapped real_name if available else speaker_label.
        #   Consecutive segments from the same speaker are merged
        #   into a single turn rather than repeating the label.
        transcript = _format_segments_with_speakers(result.segments)

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
