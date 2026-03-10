#!/usr/bin/env python3
"""
MCP Server for Audio Transcription with Speaker Diarisation.

Exposes the transcription pipeline as tools for Claude via the Model Context
Protocol (MCP). Supports three tools:

1. transcribe_audio - Universal transcription with speaker diarisation
2. map_speakers_from_agenda - Map anonymous speakers to real names using a DOCX agenda
3. check_system_resources - Check available RAM/VRAM for model selection

Usage:
    # Run directly (STDIO transport for Claude Desktop / Claude Code)
    python mcp_server.py

    # Or configure in Claude Desktop's config:
    # {
    #   "mcpServers": {
    #     "audio-transcription": {
    #       "command": "python",
    #       "args": ["path/to/mcp_server.py"],
    #       "env": { "HF_TOKEN": "your_token" }
    #     }
    #   }
    # }
"""

import os
import json
import logging
from typing import Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

from pipeline import (
    transcribe_with_diarisation,
    get_system_resources,
    check_memory_requirements,
    TranscriptionResult,
    WHISPER_MODEL_MEMORY,
    PYANNOTE_MEMORY,
    BACKENDS,
)
from speaker_mapper import SpeakerMapper, SpeakerSegment, SpeakerMapping
from agenda_parser import parse_agenda
from output_formatter import TranscriptFormatter, SummaryFormatter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP("audio-transcription")


@mcp.tool()
def transcribe_audio(
    audio_path: str,
    model: str = "medium",
    language: str = "english",
    compute_type: str = "auto",
    chunk_duration: int = 600,
    backend: str = "faster-whisper",
    use_alignment: bool = True,
    beam_size: Optional[int] = None,
    normalise: bool = False,
    denoise: bool = False,
) -> str:
    """Transcribe an audio file with speaker diarisation.

    Returns JSON with speaker-labelled segments including timestamps and text.
    Supports faster-whisper (default, 4x faster) and WhisperX backends.

    Args:
        audio_path: Path to audio file (m4a, wav, mp3, flac, etc.)
        model: Whisper model size (tiny, base, small, medium, large, large-v3, turbo)
        language: Language code (english, welsh, french, etc.) or "auto" for auto-detection
        compute_type: Model precision (auto, float16, int8_float16, int8, float32)
        chunk_duration: Audio chunk size in seconds for processing (default: 600)
        backend: Transcription backend (faster-whisper or whisperx)
        use_alignment: Refine timestamps with wav2vec2 alignment (default: True)
        beam_size: Beam search width for decoding (higher = more accurate, slower)
        normalise: Apply EBU R128 loudness normalisation before transcription
        denoise: Apply FFT-based noise reduction before transcription
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return json.dumps({
            "status": "error",
            "error": "HF_TOKEN not found. Set it in .env file or environment variable."
        })

    if not os.path.exists(audio_path):
        return json.dumps({
            "status": "error",
            "error": f"Audio file not found: {audio_path}"
        })

    # Handle language
    lang = None if language in ("auto", "None") else language

    # Handle compute type
    ct = None if compute_type == "auto" else compute_type

    try:
        result = transcribe_with_diarisation(
            audio_path=audio_path,
            hf_token=hf_token,
            chunk_duration_seconds=chunk_duration,
            whisper_model_name=model,
            language=lang,
            compute_type=ct,
            backend=backend,
            use_alignment=use_alignment,
            beam_size=beam_size,
            normalise_audio=normalise,
            denoise_audio=denoise,
        )

        output = {
            "status": "success",
            "audio_file": audio_path,
            "model": model,
            "backend": backend,
            "total_duration_seconds": round(result.total_duration, 2),
            "processing_time_seconds": round(result.elapsed_time, 2),
            "segment_count": len(result.segments),
            "segments": [
                {
                    "speaker": seg.speaker_label,
                    "start": round(seg.start_time, 2),
                    "end": round(seg.end_time, 2),
                    "text": seg.text,
                }
                for seg in result.segments
            ],
        }

        return json.dumps(output)

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "error": str(e)
        })


@mcp.tool()
def map_speakers_from_agenda(
    transcript_data: str,
    agenda_path: str,
    output_format: str = "both",
) -> str:
    """Map anonymous speaker labels to real names using a DOCX agenda document.

    Accepts either:
    - JSON output from the transcribe_audio tool (pass the full JSON string)
    - A file path to a previously saved transcription text file

    Returns structured transcript and/or executive summary with named speakers,
    confidence scores, and agenda section markers.

    Args:
        transcript_data: JSON string from transcribe_audio, or path to a saved transcript file
        agenda_path: Path to DOCX agenda file with speaker names and meeting sections
        output_format: Output format - "transcript", "summary", or "both"
    """
    if not os.path.exists(agenda_path):
        return json.dumps({
            "status": "error",
            "error": f"Agenda file not found: {agenda_path}"
        })

    # Parse agenda
    try:
        parsed_agenda = parse_agenda(agenda_path)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"Failed to parse agenda: {e}"
        })

    # Parse transcript data — either JSON or file path
    segments = []
    try:
        if os.path.isfile(transcript_data):
            # Read from file — parse the "Speaker X (time): text" format
            with open(transcript_data, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Parse: "Speaker SPEAKER_00 (0.00s - 1.23s): text here"
                    if line.startswith("Speaker ") and "(" in line and "): " in line:
                        parts = line.split("(", 1)
                        speaker = parts[0].replace("Speaker ", "").strip()
                        time_text = parts[1].split("): ", 1)
                        times = time_text[0].replace("s", "").split(" - ")
                        text = time_text[1] if len(time_text) > 1 else ""
                        segments.append(SpeakerSegment(
                            speaker_label=speaker,
                            start_time=float(times[0]),
                            end_time=float(times[1]),
                            text=text,
                        ))
        else:
            # Parse as JSON from transcribe_audio output
            data = json.loads(transcript_data)
            if data.get("status") != "success":
                return json.dumps({
                    "status": "error",
                    "error": f"Transcript data indicates error: {data.get('error', 'unknown')}"
                })
            for seg in data.get("segments", []):
                segments.append(SpeakerSegment(
                    speaker_label=seg["speaker"],
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"],
                ))
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return json.dumps({
            "status": "error",
            "error": f"Failed to parse transcript data: {e}"
        })

    if not segments:
        return json.dumps({
            "status": "error",
            "error": "No segments found in transcript data"
        })

    # Map speakers to real names
    mapper = SpeakerMapper(parsed_agenda)
    mapped_segments = mapper.map_speakers(segments)
    speaker_mappings = mapper.get_all_mappings()

    # Format output
    output = {
        "status": "success",
        "agenda_sections": len(parsed_agenda.sections),
        "speakers_found": len(parsed_agenda.all_speakers),
        "speaker_mappings": [
            {
                "label": m.speaker_label,
                "real_name": m.real_name,
                "confidence": round(m.confidence, 2),
                "evidence": m.evidence,
            }
            for m in speaker_mappings
        ],
    }

    if output_format in ["transcript", "both"]:
        formatter = TranscriptFormatter(parsed_agenda)
        output["transcript"] = formatter.format(mapped_segments, include_timestamps=True)

    if output_format in ["summary", "both"]:
        formatter = SummaryFormatter(parsed_agenda)
        output["summary"] = formatter.format(mapped_segments, speaker_mappings)

    return json.dumps(output)


@mcp.tool()
def check_system_resources(model: str = "medium") -> str:
    """Check available system resources and whether a given Whisper model will fit.

    Returns RAM/VRAM availability, model memory requirements, and a recommendation
    on whether the selected model can run safely.

    Args:
        model: Whisper model to check requirements for (default: medium)
    """
    resources = get_system_resources()
    model_memory = WHISPER_MODEL_MEMORY.get(model, 2.0)
    total_required = model_memory + PYANNOTE_MEMORY

    # Determine available memory
    if resources['has_cuda'] and resources['available_vram_gb'] is not None:
        available = resources['available_vram_gb']
        memory_type = "VRAM"
    else:
        available = resources['available_ram_gb']
        memory_type = "RAM"

    can_run = available >= total_required * 1.2  # 20% safety margin

    # Find largest model that fits
    recommended = None
    for m in ["large-v3", "large", "large-v3-turbo", "turbo", "medium", "small", "base", "tiny"]:
        req = WHISPER_MODEL_MEMORY.get(m, 5.0) + PYANNOTE_MEMORY
        if req * 1.2 <= available:
            recommended = m
            break

    output = {
        "status": "success",
        "system": {
            "available_memory_gb": round(available, 2),
            "total_ram_gb": round(resources['total_ram_gb'], 2),
            "ram_usage_percent": round(resources['ram_usage_percent'], 1),
            "memory_type": memory_type,
            "has_cuda": resources['has_cuda'],
        },
        "model_check": {
            "requested_model": model,
            "model_memory_gb": model_memory,
            "pyannote_memory_gb": PYANNOTE_MEMORY,
            "total_required_gb": total_required,
            "can_run": can_run,
            "recommended_model": recommended,
        },
        "all_models": {
            name: {
                "memory_gb": mem,
                "total_with_pyannote_gb": mem + PYANNOTE_MEMORY,
                "fits": available >= (mem + PYANNOTE_MEMORY) * 1.2,
            }
            for name, mem in sorted(WHISPER_MODEL_MEMORY.items(), key=lambda x: x[1])
        },
    }

    return json.dumps(output)


if __name__ == "__main__":
    mcp.run()
