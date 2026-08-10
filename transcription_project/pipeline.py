"""
Pipeline Module - Core transcription and diarisation engine.

This module contains the core logic for audio transcription with speaker
diarisation. It is designed to be imported by multiple frontends:
- transcribe.py (CLI)
- mcp_server.py (MCP server for Claude)
- gradio_app.py (Web GUI)

The main function `transcribe_with_diarisation()` returns a TranscriptionResult
dataclass rather than writing files directly, allowing each frontend to handle
output in its own way.

Supports two backends:
- faster-whisper (default): CTranslate2-based Whisper with optional wav2vec2
  alignment via whisperx for more precise word timestamps
- whisperx: Full WhisperX pipeline with integrated diarisation
"""

import warnings
# Suppress noisy deprecation warnings from torchaudio/torchcodec that don't
# affect functionality. These fire on every audio load and clutter verbose output.
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*In 2.9.*torchcodec.*")

import torch
import numpy as np

# #93: subtitle-credit hallucination filter (dependency-free sibling module).
from hallucination_filter import (
    is_subtitle_credit_hallucination,
    filter_hallucination_segments,
)

# PyAnnote 3.x model checkpoints contain many custom types (Specifications,
# Problem, Powerset, Resolution, etc.) that torch.load won't deserialise with
# PyTorch 2.6+'s weights_only=True default. Rather than chasing each type,
# we patch torch.load to default to weights_only=False. This is safe — we only
# load trusted models from HuggingFace (pyannote/segmentation-3.0,
# pyannote/wespeaker-voxceleb-resnet34-LM, etc.).
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from pyannote.audio import Pipeline
import os
import soundfile as sf
import numpy as np
import subprocess
import math
import logging
import time
import psutil
import gc
from dataclasses import dataclass, field
from typing import Optional, List

from agenda_parser import parse_agenda, ParsedAgenda
from speaker_mapper import SpeakerMapper, SpeakerSegment, SpeakerMapping

logger = logging.getLogger(__name__)

# Memory requirements for faster-whisper models (in GB)
# These are significantly lower than openai-whisper due to CTranslate2 optimisation.
# Values shown are approximate for the default compute_type (float16 on GPU, int8 on CPU).
WHISPER_MODEL_MEMORY = {
    "tiny": 0.5,
    "base": 0.5,
    "small": 1.0,
    "medium": 2.0,
    "large": 3.0,
    "large-v2": 3.0,
    "large-v3": 3.0,
    "large-v3-turbo": 2.5,
    "turbo": 2.0,
    "distil-large-v3": 1.5,
}

# PyAnnote diarization pipeline memory requirement (approximate)
PYANNOTE_MEMORY = 2.5  # GB

# Supported backends
BACKENDS = ["faster-whisper", "whisperx"]

# Map full language names to ISO 639-1 codes used by faster-whisper and whisperx
LANGUAGE_CODE_MAP = {
    "english": "en", "welsh": "cy", "french": "fr", "spanish": "es",
    "german": "de", "italian": "it", "portuguese": "pt", "dutch": "nl",
    "japanese": "ja", "chinese": "zh", "korean": "ko", "russian": "ru",
    "arabic": "ar", "hindi": "hi", "turkish": "tr", "polish": "pl",
    "swedish": "sv", "danish": "da", "norwegian": "no", "finnish": "fi",
    "greek": "el", "czech": "cs", "romanian": "ro", "hungarian": "hu",
    "ukrainian": "uk", "thai": "th", "vietnamese": "vi", "indonesian": "id",
    "malay": "ms", "hebrew": "he", "persian": "fa", "tamil": "ta",
    "telugu": "te", "bengali": "bn", "urdu": "ur", "gujarati": "gu",
    "marathi": "mr", "kannada": "kn", "malayalam": "ml", "punjabi": "pa",
    "catalan": "ca", "galician": "gl", "basque": "eu", "slovenian": "sl",
    "slovak": "sk", "croatian": "hr", "serbian": "sr", "bulgarian": "bg",
    "latvian": "lv", "lithuanian": "lt", "estonian": "et",
}


def _to_language_code(language):
    """Convert a language name or code to ISO 639-1 code for faster-whisper.

    Accepts full names ("english"), ISO codes ("en"), or None.
    Returns ISO 639-1 code or None for auto-detection.
    """
    if language is None or language.lower() == "none":
        return None
    lang = language.lower().strip()
    # Already a valid 2-3 letter code
    if len(lang) <= 3:
        return lang
    return LANGUAGE_CODE_MAP.get(lang, lang)


@dataclass
class TranscriptionResult:
    """Result of a transcription run, returned by transcribe_with_diarisation()."""
    segments: List[SpeakerSegment]           # Raw speaker segments
    transcription_lines: List[str]           # Formatted "Speaker X (time): text" lines
    parsed_agenda: Optional[ParsedAgenda] = None    # If agenda was provided
    speaker_mappings: Optional[List[SpeakerMapping]] = None  # If agenda mapping was done
    mapped_segments: Optional[List[SpeakerSegment]] = None  # Segments with real names
    elapsed_time: float = 0.0                # Processing duration in seconds
    total_duration: float = 0.0              # Audio duration in seconds


# ---------------------------------------------------------------------------
# Optional progress reporting and cooperative cancellation
# ---------------------------------------------------------------------------
#
# Both are opt-in and transport-agnostic. The pipeline knows nothing about
# HTTP, job identifiers, or any particular caller: it is handed two plain
# callables and invokes them at stage boundaries. Supply neither and every
# code path behaves exactly as it did before, so the CLI, MCP and Gradio
# frontends are unaffected — and can adopt them later if they want to.


class TranscriptionCancelled(Exception):
    """Raised at a stage boundary when the cancellation predicate returns True.

    A distinct type so callers can tell a deliberate stop from a failure. It
    unwinds through the existing `finally` blocks, so temporary chunk files
    are removed on the way out.
    """


class _ProgressReporter:
    """Advances a monotonic marker and checks for cancellation.

    The marker is the *only* evidence of progress a caller gets, and its
    value is meaningless — what matters is that it changes. A caller
    watching for a stall reads "has it moved", never "how far has it got".

    Emission is rate-limited because PyAnnote's hook fires thousands of
    times per pass, while a watchdog measured in minutes needs nothing like
    that resolution. A step change always emits regardless of the limit, so
    phase transitions are never swallowed.
    """

    MIN_EMIT_INTERVAL_SECONDS = 1.0

    def __init__(self, on_marker=None, should_cancel=None):
        self._on_marker = on_marker
        self._should_cancel = should_cancel
        self._seq = 0
        self._last_emit = 0.0
        self._last_phase = None
        self.total_duration = None

    @property
    def enabled(self):
        return self._on_marker is not None

    def check_cancelled(self):
        """Raise if the caller has asked for cancellation. Cheap; call freely."""
        if self._should_cancel is not None and self._should_cancel():
            raise TranscriptionCancelled("cancellation requested")

    def emit(self, phase, *, chunk_index=None, chunk_total=None,
             audio_seconds_done=None, force=False):
        """Advance the marker and hand it to the caller.

        `audio_seconds_done` is deliberately left None outside the
        transcription phase. Global diarisation processes the whole file in
        one pass, so there is no meaningful position within the recording
        while it runs; reporting one would be a fabricated number, and
        reporting zero would be worse — a caller would render it as 0%
        progress on a job that is 61% of the way through its work.
        """
        self.check_cancelled()
        if self._on_marker is None:
            return

        now = time.time()
        phase_changed = phase != self._last_phase
        if not (force or phase_changed) and (now - self._last_emit) < self.MIN_EMIT_INTERVAL_SECONDS:
            return

        self._seq += 1
        self._last_emit = now
        self._last_phase = phase
        marker = {
            "marker_seq": self._seq,
            "phase": phase,
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "audio_seconds_done": audio_seconds_done,
            "audio_total_seconds": self.total_duration,
        }
        try:
            self._on_marker(marker)
        except Exception:
            # A caller whose reporting breaks must not take the
            # transcription down with it. Progress is an accessory to the
            # job, never a precondition for it.
            logger.debug("on_marker callback raised; ignoring", exc_info=True)


class _PyannoteProgressHook:
    """Bridges PyAnnote's hook protocol onto the marker reporter.

    Implements the protocol rather than subclassing `ProgressHook`, whose
    `__enter__` starts a `rich` progress bar on stdout — unwanted inside a
    service, and it would fight the container's log stream. PyAnnote only
    requires a callable with this signature.

    This exists because global diarisation is ONE opaque pass that runs
    before the chunk loop and accounts for ~61% of total runtime — about
    116 minutes on a three-hour recording. Without it the marker would
    freeze for longer than any usable stall threshold, and the watchdog
    would cancel healthy work on every real meeting.
    """

    def __init__(self, reporter):
        self._reporter = reporter

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        # Reported as the flat phase "diarise" rather than "diarise:<step>":
        # the status contract fixes phase to four values, and a caller that
        # switch-matched on it would silently stop recognising the phase the
        # moment PyAnnote renamed an internal step. The sub-step goes to the
        # log, where a human can use it and nothing depends on it.
        if step_name != getattr(self, "_last_step", None):
            self._last_step = step_name
            logger.debug("Diarisation step: %s", step_name)
        # Cancellation is checked inside emit(), which is what makes a long
        # diarisation pass interruptible rather than a two-hour commitment.
        self._reporter.emit("diarise")


class _HookInjectingPipeline:
    """Forwards to a PyAnnote Pipeline, adding a progress hook to each call.

    Used only by the WhisperX backend, whose wrapper offers no way to pass a
    hook through. `called` records whether the delegation actually happened,
    so an upstream change that routes around this seam is reported rather
    than quietly producing an uninstrumented run.
    """

    def __init__(self, inner, hook):
        self._inner = inner
        self._hook = hook
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True
        kwargs.setdefault("hook", self._hook)
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def get_system_resources():
    """
    Check available system resources (RAM and VRAM).

    Returns:
        dict: Available RAM in GB, available VRAM in GB (if GPU available), and usage percentages
    """
    # Get RAM info
    ram = psutil.virtual_memory()
    available_ram_gb = ram.available / (1024**3)
    total_ram_gb = ram.total / (1024**3)
    ram_usage_percent = ram.percent

    resources = {
        'available_ram_gb': available_ram_gb,
        'total_ram_gb': total_ram_gb,
        'ram_usage_percent': ram_usage_percent,
        'has_cuda': torch.cuda.is_available(),
        'available_vram_gb': None,
        'total_vram_gb': None
    }

    # Get VRAM info if CUDA available
    if torch.cuda.is_available():
        try:
            # Get memory info from primary GPU
            vram_free, vram_total = torch.cuda.mem_get_info()
            resources['available_vram_gb'] = vram_free / (1024**3)
            resources['total_vram_gb'] = vram_total / (1024**3)
        except Exception as e:
            logger.warning(f"Could not get VRAM info: {e}")

    return resources


def check_memory_requirements(whisper_model_name, auto_adjust=False, skip_diarisation=False):
    """
    Check if system has enough memory to run the models.

    Args:
        whisper_model_name: Name of the Whisper model to use
        auto_adjust: If True, automatically suggest a smaller model if insufficient memory
        skip_diarisation: If True, exclude PyAnnote memory from requirements

    Returns:
        tuple: (can_proceed, recommended_model, warning_message)
    """
    resources = get_system_resources()
    pyannote_mem = 0 if skip_diarisation else PYANNOTE_MEMORY
    required_memory = WHISPER_MODEL_MEMORY.get(whisper_model_name, 5.0) + pyannote_mem

    # Determine which memory pool to check (GPU VRAM or CPU RAM)
    if resources['has_cuda'] and resources['available_vram_gb'] is not None:
        available_memory = resources['available_vram_gb']
        memory_type = "VRAM"
        total_memory = resources['total_vram_gb']
    else:
        available_memory = resources['available_ram_gb']
        memory_type = "RAM"
        total_memory = resources['total_ram_gb']

    # Log current resource status
    logger.info(f"System resources: {available_memory:.2f} GB available {memory_type} "
                f"({total_memory:.2f} GB total, {resources['ram_usage_percent']:.1f}% in use)")
    logger.info(f"Required memory for {whisper_model_name} model + PyAnnote: ~{required_memory:.1f} GB")

    # Add safety margin (20% buffer)
    safety_margin = 1.2
    required_with_margin = required_memory * safety_margin

    if available_memory < required_with_margin:
        warning_msg = (
            f"WARNING: Low {memory_type}! Available: {available_memory:.2f} GB, "
            f"Required: ~{required_memory:.1f} GB (with 20% buffer: {required_with_margin:.1f} GB)"
        )

        # Try to find a smaller model that fits
        recommended_model = None
        if auto_adjust:
            for model in ["medium", "small", "base", "tiny"]:
                test_required = WHISPER_MODEL_MEMORY[model] + PYANNOTE_MEMORY
                if test_required * safety_margin <= available_memory:
                    recommended_model = model
                    break

        return False, recommended_model, warning_msg

    return True, whisper_model_name, None


def find_speaker_at_time(time_point, diarisation_segments, word_start=None, word_end=None, tolerance=0.1):
    """
    Find which speaker is active at a given time point.

    Uses the word's midpoint for assignment. Handles overlapping PyAnnote segments
    by picking the speaker with greatest temporal overlap. Falls back with a small
    tolerance to cover gaps between diarisation segments.

    Args:
        time_point: Time in seconds (typically word midpoint) to look up
        diarisation_segments: List of (turn, _, speaker) tuples from PyAnnote
        word_start: Optional word start time for overlap calculation
        word_end: Optional word end time for overlap calculation
        tolerance: Seconds of tolerance for gap handling (default: 0.1)

    Returns:
        Speaker label string, or None if no speaker found
    """
    candidates = []
    for turn, _, speaker in diarisation_segments:
        if turn.start <= time_point <= turn.end:
            candidates.append((turn, speaker))

    if len(candidates) == 1:
        return candidates[0][1]
    elif len(candidates) > 1 and word_start is not None and word_end is not None:
        # Overlapping segments — pick speaker with greatest temporal overlap
        best_speaker = None
        best_overlap = 0
        for turn, speaker in candidates:
            overlap_start = max(turn.start, word_start)
            overlap_end = min(turn.end, word_end)
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        return best_speaker
    elif len(candidates) > 1:
        # Fallback: pick earliest-starting segment
        return min(candidates, key=lambda x: x[0].start)[1]

    # No exact match — try with tolerance for small gaps
    for turn, _, speaker in diarisation_segments:
        if (turn.start - tolerance) <= time_point <= (turn.end + tolerance):
            return speaker

    return None


def fill_none_speakers(assigned_words):
    """
    Replace None speaker assignments using carry-forward/carry-backward.

    Forward pass carries the last known speaker through gaps.
    Backward pass fills any remaining Nones at the start of the list.

    Args:
        assigned_words: List of (word_dict, speaker_label_or_None) tuples

    Returns:
        List of (word_dict, speaker_label) tuples with no None speakers
    """
    filled = list(assigned_words)

    # Forward pass: carry forward last known speaker
    last_speaker = None
    for i, (word, speaker) in enumerate(filled):
        if speaker is not None:
            last_speaker = speaker
        elif last_speaker is not None:
            filled[i] = (word, last_speaker)

    # Backward pass: fill any remaining Nones at the start
    next_speaker = None
    for i in range(len(filled) - 1, -1, -1):
        word, speaker = filled[i]
        if speaker is not None:
            next_speaker = speaker
        elif next_speaker is not None:
            filled[i] = (word, next_speaker)

    return filled


def assign_and_group_words(all_words, diarisation_segments, chunk_start_time,
                           max_gap=2.0, progress=None):
    """
    Assign Whisper words to PyAnnote speakers and group into SpeakerSegments.

    For each word, uses the word's midpoint to find the active speaker. Consecutive
    words from the same speaker are grouped into a single SpeakerSegment. A new
    segment starts when the speaker changes or the gap between words exceeds max_gap.

    Args:
        all_words: List of word dicts (keys: word, start, end)
        diarisation_segments: List of (turn, _, speaker) tuples from PyAnnote
        chunk_start_time: Start time of this chunk in the original audio (seconds)
        max_gap: Maximum gap in seconds between words before starting a new segment
        progress: Optional _ProgressReporter. The assignment loop below is
            O(words x diarisation_segments) — on a long recording that is
            every word in hours of audio scanned against every speaker turn,
            and in global mode it runs ONCE at the end over the whole file
            rather than per chunk. Left uninstrumented it is a silent tail:
            the marker's last movement would be the "assemble" emit before
            this call, and a watchdog would read the wait as a stall on work
            that is finishing normally. Omit it and behaviour is unchanged.

    Returns:
        List[SpeakerSegment] with absolute timestamps
    """
    if not all_words:
        return []

    # Assign each word to a speaker
    assigned = []
    for idx, word in enumerate(all_words):
        # Checked every 500 words rather than every word: the reporter
        # rate-limits emission anyway, and this keeps the hot loop free of a
        # per-word call. 500 words is a few seconds of speech, so the marker
        # still moves far more often than any threshold cares about.
        if progress is not None and idx % 500 == 0:
            progress.emit(
                "assemble",
                audio_seconds_done=chunk_start_time + word["start"],
            )
        midpoint = (word["start"] + word["end"]) / 2
        speaker = find_speaker_at_time(
            midpoint, diarisation_segments,
            word_start=word["start"], word_end=word["end"]
        )
        assigned.append((word, speaker))

    # Fill gaps in speaker assignment
    assigned = fill_none_speakers(assigned)

    # Group consecutive same-speaker words into segments
    segments = []
    current_words = [assigned[0]]
    current_speaker = assigned[0][1]

    for word, speaker in assigned[1:]:
        prev_word = current_words[-1][0]
        gap = word["start"] - prev_word["end"]

        if speaker != current_speaker or gap > max_gap:
            # Flush current group as a SpeakerSegment
            segments.append(_build_speaker_segment(
                current_words, current_speaker, chunk_start_time
            ))
            current_words = [(word, speaker)]
            current_speaker = speaker
        else:
            current_words.append((word, speaker))

    # Flush final group
    if current_words:
        segments.append(_build_speaker_segment(
            current_words, current_speaker, chunk_start_time
        ))

    return segments


def _build_speaker_segment(word_speaker_pairs, speaker_label, chunk_start_time):
    """
    Build a SpeakerSegment from a group of words assigned to the same speaker.

    Args:
        word_speaker_pairs: List of (word_dict, speaker_label) tuples
        speaker_label: The speaker label for this segment
        chunk_start_time: Start time of the chunk in original audio (seconds)

    Returns:
        SpeakerSegment with absolute timestamps
    """
    words = [ws[0] for ws in word_speaker_pairs]
    text = "".join(w["word"] for w in words).strip()
    seg_start = words[0]["start"]
    seg_end = words[-1]["end"]

    return SpeakerSegment(
        speaker_label=speaker_label or "UNKNOWN",
        start_time=chunk_start_time + seg_start,
        end_time=chunk_start_time + seg_end,
        text=text
    )


def _build_agenda_prompt(parsed_agenda):
    """
    Build a Whisper initial_prompt from a parsed agenda to improve transcription accuracy.

    Provides Whisper with expected vocabulary: meeting title, speaker names, and
    agenda topics. This helps Whisper correctly recognise proper nouns and
    domain-specific terms.

    Args:
        parsed_agenda: ParsedAgenda object with metadata, sections, and speakers

    Returns:
        str: A prompt string for Whisper's initial_prompt parameter
    """
    parts = []

    if parsed_agenda.metadata.title:
        parts.append(parsed_agenda.metadata.title)

    if parsed_agenda.all_speakers:
        parts.append("Speakers: " + ", ".join(parsed_agenda.all_speakers))

    section_titles = [s.title for s in parsed_agenda.sections if s.title]
    if section_titles:
        parts.append("Topics: " + ", ".join(section_titles))

    return ". ".join(parts) if parts else ""


def _refine_timestamps_with_wav2vec2(all_words, chunk_file, language, device):
    """
    Refine word timestamps using wav2vec2 forced alignment via whisperx.

    This produces more precise word boundaries (within 20-50ms) compared to
    faster-whisper's native timestamps (which can be off by 100-200ms).

    Args:
        all_words: List of word dicts (keys: word, start, end)
        chunk_file: Path to the audio chunk WAV file
        language: Language code for alignment model selection
        device: Device string ("cuda" or "cpu")

    Returns:
        List of word dicts with refined timestamps
    """
    import whisperx

    # Build segments in the format whisperx.align() expects
    # Group words back into segment-like chunks for alignment
    transcript_segments = []
    current_segment_words = []
    current_start = None

    for word in all_words:
        if current_start is None:
            current_start = word["start"]
        current_segment_words.append(word["word"])

        # Create a new segment every ~30 words or at natural breaks
        if len(current_segment_words) >= 30:
            transcript_segments.append({
                "text": "".join(current_segment_words).strip(),
                "start": current_start,
                "end": word["end"],
            })
            current_segment_words = []
            current_start = None

    # Flush remaining words
    if current_segment_words:
        transcript_segments.append({
            "text": "".join(current_segment_words).strip(),
            "start": current_start,
            "end": all_words[-1]["end"],
        })

    if not transcript_segments:
        return all_words

    # Load alignment model
    lang_code = _to_language_code(language) or "en"

    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=lang_code,
            device=device,
        )

        # whisperx.align() needs numpy audio array
        audio_array = whisperx.load_audio(chunk_file)

        aligned_result = whisperx.align(
            transcript_segments,
            align_model,
            align_metadata,
            audio_array,
            device,
            return_char_alignments=False,
        )

        # Extract refined word timestamps
        # whisperx returns words without leading spaces, but our
        # _build_speaker_segment joins with "".join(w["word"]) expecting
        # the faster-whisper convention of space-prefixed words like " word".
        refined_words = []
        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                if "start" in word_info and "end" in word_info:
                    word_text = word_info["word"]
                    if not word_text.startswith(" "):
                        word_text = " " + word_text
                    refined_words.append({
                        "word": word_text,
                        "start": word_info["start"],
                        "end": word_info["end"],
                    })

        # Clean up alignment model
        del align_model
        gc.collect()

        if refined_words:
            logger.debug(f"wav2vec2 alignment refined {len(refined_words)} words")
            return refined_words
        else:
            logger.warning("wav2vec2 alignment produced no words, falling back to original timestamps")
            return all_words

    except Exception as e:
        logger.warning(f"wav2vec2 alignment failed, using original timestamps: {e}")
        return all_words


def _transcribe_with_whisperx_backend(audio_path, hf_token, chunk_duration_seconds,
                                       whisper_model_name, language, compute_type,
                                       initial_prompt, beam_size, diarisation_kwargs=None,
                                       progress=None):
    """
    Full WhisperX pipeline: transcribe + align + diarise in one pass.

    This is the fallback backend when --backend whisperx is used. WhisperX handles
    the entire pipeline internally using its own integration of faster-whisper,
    wav2vec2 alignment, and pyannote diarisation.

    Args:
        audio_path: Path to the audio file
        hf_token: Hugging Face authentication token
        chunk_duration_seconds: Not used by WhisperX (it handles chunking internally)
        whisper_model_name: Whisper model name
        language: Language code or None for auto-detect
        compute_type: Compute type for the model
        initial_prompt: Initial prompt for Whisper context
        beam_size: Beam search width
        progress: Optional _ProgressReporter. This path is instrumented
            separately and deliberately (T014b): markers added to the
            faster-whisper chunk loop do not exist here, so without this a
            caller switching backend for speed would see the marker freeze
            permanently and a stall watchdog would cancel every healthy run.

    Returns:
        Tuple of (speaker_segments, transcription_lines, total_duration)

    Raises:
        TranscriptionCancelled: cancellation was requested at a boundary.
    """
    import whisperx

    if progress is None:
        progress = _ProgressReporter()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lang_code = _to_language_code(language)

    logger.info(f"Loading WhisperX model: {whisper_model_name} (compute_type={compute_type})")
    model = whisperx.load_model(whisper_model_name, device, compute_type=compute_type,
                                 language=lang_code)

    logger.info("Loading audio for WhisperX")
    audio = whisperx.load_audio(audio_path)
    total_duration = len(audio) / 16000  # WhisperX loads at 16kHz
    progress.total_duration = total_duration
    progress.emit("extract", force=True)

    # Step 1: Transcribe
    logger.info("WhisperX: Transcribing audio")
    progress.emit("transcribe", force=True)
    transcribe_kwargs = {}
    if beam_size is not None:
        transcribe_kwargs["beam_size"] = beam_size
    result = model.transcribe(audio, batch_size=16, **transcribe_kwargs)
    detected_language = result.get("language", lang_code or "en")

    # Step 2: Align
    logger.info("WhisperX: Aligning word timestamps with wav2vec2")
    progress.emit("transcribe", audio_seconds_done=total_duration, force=True)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_metadata,
        audio, device, return_char_alignments=False
    )
    del align_model
    gc.collect()

    # Step 3: Diarise
    logger.info("WhisperX: Running speaker diarisation")
    progress.emit("diarise", force=True)
    # `whisperx.DiarizationPipeline` no longer exists at the package root in
    # the installed version — it lives in whisperx.diarize. The old
    # attribute access raised AttributeError, so this backend could not
    # complete a run at all. Found while instrumenting it (T014b); the
    # import is fixed here because instrumenting a path that cannot execute
    # would be theatre.
    from whisperx.diarize import DiarizationPipeline

    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

    # DiarizationPipeline.__call__ takes no hook argument, but it delegates
    # to the PyAnnote pipeline it holds on `.model`. Wrapping that is the
    # least invasive seam available: no DataFrame conversion is
    # reimplemented, and if a future whisperx stops routing through it, the
    # wrapper simply never fires and says so rather than silently producing
    # an uninstrumented — and therefore watchdog-unsafe — diarisation.
    hook_wrapper = None
    if progress.enabled:
        hook_wrapper = _HookInjectingPipeline(
            diarize_model.model, _PyannoteProgressHook(progress)
        )
        diarize_model.model = hook_wrapper

    diarize_segments = diarize_model(audio_path, **(diarisation_kwargs or {}))

    if hook_wrapper is not None and not hook_wrapper.called:
        # FR-006a: a stage that cannot be instrumented must be reported as
        # uninstrumented, not left silent. A frozen marker here would be
        # read as a stall on the dominant phase of the run.
        logger.warning(
            "WhisperX diarisation ran WITHOUT progress instrumentation — the "
            "marker did not advance for this stage. Stall detection is not "
            "safe against this backend until the hook seam is restored "
            "(FR-006a)."
        )

    result = whisperx.assign_word_speakers(diarize_segments, result)
    del diarize_model
    gc.collect()
    progress.emit("assemble", audio_seconds_done=total_duration, force=True)

    # Convert WhisperX output to SpeakerSegment objects
    speaker_segments = []
    transcription_lines = []

    for segment in result.get("segments", []):
        speaker = segment.get("speaker", "UNKNOWN")
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        text = segment.get("text", "").strip()

        if text:
            seg_obj = SpeakerSegment(
                speaker_label=speaker,
                start_time=start,
                end_time=end,
                text=text,
            )
            speaker_segments.append(seg_obj)

            line = f"Speaker {speaker} ({start:.2f}s - {end:.2f}s): {text}"
            transcription_lines.append(line)

    del model
    gc.collect()

    return speaker_segments, transcription_lines, total_duration


def transcribe_with_diarisation(audio_path, hf_token, chunk_duration_seconds=600,
                                 whisper_model_name="medium", language="english",
                                 agenda_path: Optional[str] = None,
                                 legacy_mode: bool = False,
                                 initial_prompt: Optional[str] = None,
                                 beam_size: Optional[int] = None,
                                 best_of: Optional[int] = None,
                                 compression_ratio_threshold: Optional[float] = None,
                                 logprob_threshold: Optional[float] = None,
                                 no_speech_threshold: Optional[float] = None,
                                 normalise_audio: bool = False,
                                 denoise_audio: bool = False,
                                 compute_type: Optional[str] = None,
                                 backend: str = "faster-whisper",
                                 use_alignment: bool = True,
                                 skip_diarisation: bool = False,
                                 num_speakers: Optional[int] = None,
                                 min_speakers: Optional[int] = None,
                                 max_speakers: Optional[int] = None,
                                 min_cluster_size: Optional[int] = None,
                                 diarise_per_chunk: bool = False,
                                 on_marker=None,
                                 should_cancel=None) -> TranscriptionResult:
    """
    Transcribes an audio file with speaker diarisation, processing in chunks.

    By default, uses faster-whisper with wav2vec2 alignment for the best combination
    of speed and word-level timestamp accuracy. Falls back to full WhisperX pipeline
    if --backend whisperx is specified.

    Args:
        audio_path: Path to the audio file to transcribe
        hf_token: Hugging Face authentication token
        chunk_duration_seconds: Duration of each audio chunk in seconds (default: 600)
        whisper_model_name: Name of the Whisper model to use (default: "medium")
        language: Language code to force (default: "english"). Set to None for auto-detection.
        agenda_path: Optional path to DOCX agenda file for agenda-aware transcription
        legacy_mode: If True, use per-segment transcription instead of full-chunk alignment
        initial_prompt: Optional context prompt for Whisper (domain terms, speaker names)
        beam_size: Beam search width for decoding (default: faster-whisper default of 5)
        best_of: Number of candidate decodings to evaluate (default: 5)
        compression_ratio_threshold: Discard segments above this ratio as hallucinated (default: 2.4)
        logprob_threshold: Discard segments below this log probability (default: -1.0)
        no_speech_threshold: Discard segments above this no-speech probability (default: 0.6)
        normalise_audio: If True, apply EBU R128 loudness normalisation via FFmpeg
        denoise_audio: If True, apply FFT-based noise reduction via FFmpeg
        compute_type: Compute type for faster-whisper (float16, int8_float16, int8, float32).
                      If None, auto-selects float16 for GPU or int8 for CPU.
        backend: Transcription backend ("faster-whisper" or "whisperx")
        use_alignment: If True, refine timestamps with wav2vec2 alignment (default: True)
        skip_diarisation: If True, skip PyAnnote speaker diarisation entirely. All segments
            are labelled SPEAKER_00. Significantly faster on CPU — use for single-speaker
            audio like voice notes. Default: False.
        num_speakers: Exact speaker count hint for PyAnnote. Overrides min/max if set.
        min_speakers: Lower bound for speaker count.
        max_speakers: Upper bound for speaker count.
        min_cluster_size: Override PyAnnote's clustering min_cluster_size (default 12).
            Lower values let short interjections survive instead of being merged.
        diarise_per_chunk: If True, run diarisation independently on each chunk (legacy
            behaviour). Default False — diarise the full file once for global clustering.
        on_marker: Optional callable invoked at each stage boundary with a dict
            carrying marker_seq, phase, chunk_index, chunk_total,
            audio_seconds_done and audio_total_seconds. Transport-agnostic —
            the pipeline neither knows nor cares what the caller does with it.
            Omit it and nothing is reported, which is the default.
        should_cancel: Optional callable returning True when the caller wants
            the run to stop. Checked at the same stage boundaries; when it
            returns True the pipeline raises TranscriptionCancelled, which
            unwinds through the existing cleanup blocks. Omit it and the run
            can never be interrupted, which is the default.

    Returns:
        TranscriptionResult with segments, lines, and optional agenda data

    Raises:
        TranscriptionCancelled: should_cancel() returned True at a boundary.
    """
    from faster_whisper import WhisperModel

    progress = _ProgressReporter(on_marker=on_marker, should_cancel=should_cancel)

    start_time_total = time.time()
    transcription_lines = []
    speaker_segments = []
    parsed_agenda = None

    # Convert language name to ISO 639-1 code (faster-whisper requires codes, not names)
    language = _to_language_code(language)

    # Parse agenda if provided
    if agenda_path:
        if not os.path.exists(agenda_path):
            logger.error(f"Agenda file not found: {agenda_path}")
            raise RuntimeError(f"Agenda file not found: {agenda_path}")

        logger.info(f"Parsing agenda document: {agenda_path}")
        parsed_agenda = parse_agenda(agenda_path)
        logger.info(f"Agenda parsed: {len(parsed_agenda.sections)} sections, "
                   f"{len(parsed_agenda.all_speakers)} speakers listed")

        # Auto-generate initial prompt from agenda if no manual prompt provided
        if not initial_prompt:
            initial_prompt = _build_agenda_prompt(parsed_agenda)
            if initial_prompt:
                logger.info(f"Auto-generated Whisper prompt from agenda: {initial_prompt[:100]}...")

    if initial_prompt:
        logger.info(f"Using initial prompt for Whisper context")
    if beam_size is not None:
        logger.info(f"Whisper beam_size: {beam_size}")
    if best_of is not None:
        logger.info(f"Whisper best_of: {best_of}")

    # Determine compute type
    if compute_type is None:
        compute_type = "float16" if torch.cuda.is_available() else "int8"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build diarisation kwargs once — used at every PyAnnote call site
    diarisation_kwargs = {}
    if num_speakers is not None:
        diarisation_kwargs["num_speakers"] = num_speakers
        logger.info(f"PyAnnote num_speakers hint: {num_speakers}")
    else:
        if min_speakers is not None:
            diarisation_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarisation_kwargs["max_speakers"] = max_speakers
        if min_speakers is not None or max_speakers is not None:
            logger.info(f"PyAnnote speaker bounds: min={min_speakers}, max={max_speakers}")

    # Use WhisperX backend if requested
    if backend == "whisperx":
        logger.info("Using WhisperX backend (full pipeline)")
        speaker_segments, transcription_lines, total_duration = _transcribe_with_whisperx_backend(
            audio_path, hf_token, chunk_duration_seconds,
            whisper_model_name, language, compute_type,
            initial_prompt, beam_size, diarisation_kwargs,
            progress=progress,
        )
    else:
        # Default: faster-whisper backend with manual PyAnnote diarisation
        logger.info(f"Loading faster-whisper model: {whisper_model_name} "
                    f"(device={device}, compute_type={compute_type})")
        whisper_model = WhisperModel(
            whisper_model_name,
            device=device,
            compute_type=compute_type,
        )

        diarisation_pipeline = None
        if skip_diarisation:
            logger.info("Diarisation SKIPPED — all segments will be labelled SPEAKER_00")
        else:
            logger.info("Loading PyAnnote speaker diarisation pipeline")
            diarisation_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )

            # Override clustering min_cluster_size if requested. Lower values
            # prevent short interjections from being absorbed into larger clusters.
            if min_cluster_size is not None:
                try:
                    diarisation_pipeline.clustering.min_cluster_size = min_cluster_size
                    logger.info(f"PyAnnote min_cluster_size override: {min_cluster_size}")
                except AttributeError:
                    logger.warning(
                        "Could not set clustering.min_cluster_size — "
                        "PyAnnote pipeline structure may have changed"
                    )

            # Send pipeline to GPU if available
            if torch.cuda.is_available():
                diarisation_pipeline.to(torch.device("cuda"))
                logger.info("Using GPU acceleration (CUDA)")

        # Get audio duration using ffprobe
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        try:
            duration_str = subprocess.check_output(probe_cmd, stderr=subprocess.PIPE).decode("utf-8").strip()
            total_duration = float(duration_str)
            logger.info(f"Audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            # The first marker of the run. Everything before this point is
            # model loading and probing, which is bounded and fast; from here
            # on the caller has evidence the job is alive.
            progress.total_duration = total_duration
            progress.emit("extract", force=True)
        except FileNotFoundError:
            logger.error("ffprobe not found. Please install FFmpeg.")
            raise RuntimeError("ffprobe not found. Please install FFmpeg.")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e.stderr.decode('utf-8')}")
            raise RuntimeError(f"ffprobe failed: {e.stderr.decode('utf-8')}")
        except ValueError:
            logger.error(f"Could not parse audio duration from file: {audio_path}")
            raise RuntimeError(f"Could not parse audio duration from file: {audio_path}")

        # Run global diarisation on the full file (default) instead of per chunk.
        # Per-chunk clustering can't see speakers who appear in other chunks, and
        # clusters below min_cluster_size get absorbed — so short interjections
        # end up merged into the wrong speaker. Global diarisation fixes both.
        # Skipped if --diarise-per-chunk, --legacy, or --skip-diarisation.
        full_diar_segments = None
        temp_full_path = None
        global_diarisation = (
            not skip_diarisation and not legacy_mode and not diarise_per_chunk
        )
        if global_diarisation:
            temp_full_path = "temp_full_diarise.wav"
            full_ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ac", "1", "-ar", "16000",
            ]
            full_filters = []
            if denoise_audio:
                full_filters.append("afftdn=nf=-25")
            if normalise_audio:
                full_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
            if full_filters:
                full_ffmpeg_cmd.extend(["-af", ",".join(full_filters)])
            full_ffmpeg_cmd.append(temp_full_path)

            logger.info("Preprocessing full audio for global diarisation")
            progress.emit("extract", force=True)
            try:
                subprocess.run(full_ffmpeg_cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg full-file preprocess failed: {e.stderr.decode('utf-8')}")
                raise RuntimeError("ffmpeg full-file preprocess failed")

            logger.info("Running speaker diarisation on full file (global clustering)")
            progress.emit("diarise", force=True)
            diar_start = time.time()
            # T014a: the hook is what keeps the marker moving through this
            # call. It is a single pass over the whole file and, measured on
            # real audio, 61% of total runtime — ~116 minutes on a three-hour
            # recording. Instrumenting only the chunk loop below would leave
            # the marker frozen for the majority of a healthy run.
            diar_hook = _PyannoteProgressHook(progress) if progress.enabled else None
            if diar_hook is not None:
                full_diarisation = diarisation_pipeline(
                    temp_full_path, hook=diar_hook, **diarisation_kwargs
                )
            else:
                full_diarisation = diarisation_pipeline(temp_full_path, **diarisation_kwargs)
            full_diar_segments = list(full_diarisation.itertracks(yield_label=True))
            unique_speakers = {label for _, _, label in full_diar_segments}
            logger.info(
                f"Global diarisation: {len(full_diar_segments)} segments, "
                f"{len(unique_speakers)} unique speakers, "
                f"{time.time() - diar_start:.1f}s"
            )

        # Words accumulated across all chunks with absolute timestamps —
        # used for global diarisation assignment after the chunk loop.
        global_words = []

        # Process in chunks
        num_chunks = math.ceil(total_duration / chunk_duration_seconds)
        logger.info(f"Processing audio in {num_chunks} chunk(s) of {chunk_duration_seconds} seconds each")

        for chunk_idx, start_time in enumerate(range(0, math.ceil(total_duration), chunk_duration_seconds), 1):
            end_time = min(start_time + chunk_duration_seconds, total_duration)
            chunk_file = f"temp_chunk_{start_time}.wav"

            logger.info(f"Processing chunk {chunk_idx}/{num_chunks} ({start_time}s - {end_time}s)")
            progress.emit(
                "transcribe",
                chunk_index=chunk_idx,
                chunk_total=num_chunks,
                # No position on the FIRST chunk: nothing has been decoded
                # yet, and reporting 0.0 makes a caller render "0%" on a job
                # that may be two-thirds finished — global diarisation having
                # already run. Later chunks start at a genuine offset, which
                # is real progress and worth reporting.
                audio_seconds_done=float(start_time) if start_time > 0 else None,
                force=True,
            )

            # Use ffmpeg to create chunk (with optional audio preprocessing)
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i", audio_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-ac", "1",
                "-ar", "16000",
            ]

            # Build audio filter chain if preprocessing is enabled
            audio_filters = []
            if denoise_audio:
                audio_filters.append("afftdn=nf=-25")  # FFT-based denoiser, noise floor -25dB
            if normalise_audio:
                audio_filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")  # EBU R128 normalisation
            if audio_filters:
                ffmpeg_cmd.extend(["-af", ",".join(audio_filters)])
                if chunk_idx == 1:
                    logger.info(f"Audio preprocessing: {', '.join(audio_filters)}")

            ffmpeg_cmd.append(chunk_file)
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            except FileNotFoundError:
                logger.error("ffmpeg not found. Please install FFmpeg.")
                raise RuntimeError("ffmpeg not found. Please install FFmpeg.")
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg failed: {e.stderr.decode('utf-8')}")
                raise RuntimeError(f"ffmpeg failed: {e.stderr.decode('utf-8')}")

            try:
                if legacy_mode:
                    # Legacy mode: transcribe each speaker segment independently
                    logger.debug(f"Running speaker diarisation on chunk {chunk_idx} (legacy mode)")
                    # Same hook as the global pass: per-chunk diarisation is opaque
                    # too, and on a long chunk it is the dominant cost.
                    _chunk_hook = _PyannoteProgressHook(progress) if progress.enabled else None
                    diarisation = (
                        diarisation_pipeline(chunk_file, hook=_chunk_hook, **diarisation_kwargs)
                        if _chunk_hook is not None
                        else diarisation_pipeline(chunk_file, **diarisation_kwargs)
                    )

                    info = sf.info(chunk_file)
                    sample_rate = info.samplerate
                    chunk_audio, _ = sf.read(chunk_file)

                    segments = list(diarisation.itertracks(yield_label=True))
                    logger.debug(f"Found {len(segments)} speaker segments in chunk {chunk_idx}")

                    for seg_idx, (turn, _, speaker) in enumerate(segments, 1):
                        # Legacy mode transcribes each speaker turn
                        # separately, so this loop — not the chunk loop — is
                        # where a long recording actually spends its time.
                        progress.emit(
                            "transcribe",
                            chunk_index=chunk_idx,
                            chunk_total=num_chunks,
                            audio_seconds_done=float(start_time) + float(turn.start),
                        )
                        start_frame = int(turn.start * sample_rate)
                        end_frame = int(turn.end * sample_rate)
                        segment_audio_data = chunk_audio[start_frame:end_frame].astype(np.float32)

                        logger.debug(f"Transcribing segment {seg_idx}/{len(segments)} - Speaker {speaker}")

                        # Build transcription params for faster-whisper
                        transcribe_kwargs = {
                            "word_timestamps": False,
                            # Layered hallucination mitigation, from two
                            # independently-developed fixes kept together:
                            #  - condition_on_previous_text=False stops Whisper
                            #    drifting from one hallucinated phrase into
                            #    fabricated follow-on sentences (#93).
                            #  - vad_filter=True drops silence before decoding,
                            #    removing the conditions that trigger it at all.
                            "condition_on_previous_text": False,
                            "vad_filter": True,
                        }
                        if language:
                            transcribe_kwargs["language"] = language
                        if initial_prompt:
                            transcribe_kwargs["initial_prompt"] = initial_prompt
                        if beam_size is not None:
                            transcribe_kwargs["beam_size"] = beam_size
                        if best_of is not None:
                            transcribe_kwargs["best_of"] = best_of
                        if compression_ratio_threshold is not None:
                            transcribe_kwargs["compression_ratio_threshold"] = compression_ratio_threshold
                        if logprob_threshold is not None:
                            transcribe_kwargs["log_prob_threshold"] = logprob_threshold
                        if no_speech_threshold is not None:
                            transcribe_kwargs["no_speech_threshold"] = no_speech_threshold

                        seg_result, _ = whisper_model.transcribe(segment_audio_data, **transcribe_kwargs)
                        # #93: drop subtitle-credit hallucination segments before joining.
                        _kept, _ = filter_hallucination_segments(seg.text for seg in seg_result)
                        transcription = " ".join(_kept).strip()

                        original_start = start_time + turn.start
                        original_end = start_time + turn.end

                        line = f"Speaker {speaker} ({original_start:.2f}s - {original_end:.2f}s): {transcription}"
                        transcription_lines.append(line)

                        segment_obj = SpeakerSegment(
                            speaker_label=speaker,
                            start_time=original_start,
                            end_time=original_end,
                            text=transcription
                        )
                        speaker_segments.append(segment_obj)

                        logger.debug(f"Completed segment {seg_idx}/{len(segments)}")
                else:
                    # Default mode: transcribe full chunk first, then align words to speakers
                    # Step 1: Transcribe the full chunk with word-level timestamps
                    logger.debug(f"Transcribing full chunk {chunk_idx} with word-level timestamps")

                    # Build transcription params for faster-whisper
                    # condition_on_previous_text=False prevents prompt-loop hallucination
                    # where initial_prompt content gets echoed during silent regions.
                    # vad_filter=True drops silence before decoding, eliminating the
                    # silent-prefix attack surface entirely.
                    transcribe_kwargs = {
                        "word_timestamps": True,
                        # See the note above: prompt-loop drift is prevented by
                        # condition_on_previous_text=False, and the silent-prefix
                        # attack surface removed by vad_filter=True.
                        "condition_on_previous_text": False,
                        "vad_filter": True,
                    }
                    if language:
                        transcribe_kwargs["language"] = language
                    if initial_prompt:
                        transcribe_kwargs["initial_prompt"] = initial_prompt
                    if beam_size is not None:
                        transcribe_kwargs["beam_size"] = beam_size
                    if best_of is not None:
                        transcribe_kwargs["best_of"] = best_of
                    if compression_ratio_threshold is not None:
                        transcribe_kwargs["compression_ratio_threshold"] = compression_ratio_threshold
                    if logprob_threshold is not None:
                        transcribe_kwargs["log_prob_threshold"] = logprob_threshold
                    if no_speech_threshold is not None:
                        transcribe_kwargs["no_speech_threshold"] = no_speech_threshold

                    # faster-whisper accepts file path directly — no need to read into memory
                    segments_gen, transcription_info = whisper_model.transcribe(
                        chunk_file, **transcribe_kwargs
                    )

                    # Collect all words with timestamps (convert from objects to dicts
                    # for compatibility with assign_and_group_words)
                    all_words = []
                    for segment in segments_gen:
                        # faster-whisper decodes lazily, so the real work of a
                        # chunk happens during this iteration rather than in
                        # the call above. Emitting here — rate-limited to 1/s
                        # inside the reporter — keeps the marker moving through
                        # a long chunk, and is also where a cancellation
                        # request takes effect mid-chunk.
                        progress.emit(
                            "transcribe",
                            chunk_index=chunk_idx,
                            chunk_total=num_chunks,
                            audio_seconds_done=float(start_time) + float(segment.end or 0.0),
                        )
                        # #93: skip subtitle-credit hallucination segments entirely.
                        if is_subtitle_credit_hallucination(segment.text):
                            logger.warning(
                                "Dropped Whisper subtitle-credit hallucination (#93): %r",
                                segment.text.strip(),
                            )
                            continue
                        if segment.words:
                            for word in segment.words:
                                all_words.append({
                                    "word": word.word,
                                    "start": word.start,
                                    "end": word.end,
                                })

                    logger.debug(f"faster-whisper produced {len(all_words)} words for chunk {chunk_idx}")

                    # Optional: refine timestamps with wav2vec2 alignment
                    if use_alignment and all_words and not skip_diarisation:
                        logger.debug(f"Refining timestamps with wav2vec2 alignment for chunk {chunk_idx}")
                        progress.emit(
                            "transcribe",
                            chunk_index=chunk_idx,
                            chunk_total=num_chunks,
                            audio_seconds_done=float(end_time),
                            force=True,
                        )
                        all_words = _refine_timestamps_with_wav2vec2(
                            all_words, chunk_file, language, device
                        )

                    if skip_diarisation:
                        # Skip PyAnnote — group all words into SPEAKER_00 segments
                        if all_words:
                            text = "".join(w["word"] for w in all_words).strip()
                            chunk_speaker_segments = [SpeakerSegment(
                                speaker_label="SPEAKER_00",
                                start_time=start_time + all_words[0]["start"],
                                end_time=start_time + all_words[-1]["end"],
                                text=text
                            )]
                        else:
                            chunk_speaker_segments = []
                        # Append per-chunk for skip_diarisation
                        for seg_obj in chunk_speaker_segments:
                            line = f"Speaker {seg_obj.speaker_label} ({seg_obj.start_time:.2f}s - {seg_obj.end_time:.2f}s): {seg_obj.text}"
                            transcription_lines.append(line)
                            speaker_segments.append(seg_obj)
                        logger.debug(f"Produced {len(chunk_speaker_segments)} segments for chunk {chunk_idx}")
                    elif global_diarisation:
                        # Global mode: accumulate words with absolute timestamps;
                        # speaker assignment happens after the chunk loop.
                        for w in all_words:
                            global_words.append({
                                "word": w["word"],
                                "start": w["start"] + start_time,
                                "end": w["end"] + start_time,
                            })
                        logger.debug(
                            f"Chunk {chunk_idx}: collected {len(all_words)} words "
                            f"(global total: {len(global_words)})"
                        )
                    else:
                        # Per-chunk diarisation (--diarise-per-chunk path)
                        logger.debug(f"Running speaker diarisation on chunk {chunk_idx}")
                        # Same hook as the global pass: per-chunk diarisation is opaque
                        # too, and on a long chunk it is the dominant cost.
                        _chunk_hook = _PyannoteProgressHook(progress) if progress.enabled else None
                        diarisation = (
                            diarisation_pipeline(chunk_file, hook=_chunk_hook, **diarisation_kwargs)
                            if _chunk_hook is not None
                            else diarisation_pipeline(chunk_file, **diarisation_kwargs)
                        )

                        diar_segments = list(diarisation.itertracks(yield_label=True))
                        logger.debug(f"Found {len(diar_segments)} diarisation segments in chunk {chunk_idx}")

                        if all_words and diar_segments:
                            chunk_speaker_segments = assign_and_group_words(
                                all_words, diar_segments, start_time, max_gap=2.0,
                                progress=progress,
                            )
                        elif all_words:
                            text = "".join(w["word"] for w in all_words).strip()
                            chunk_speaker_segments = [SpeakerSegment(
                                speaker_label="UNKNOWN",
                                start_time=start_time + all_words[0]["start"],
                                end_time=start_time + all_words[-1]["end"],
                                text=text
                            )]
                        else:
                            chunk_speaker_segments = []

                        for seg_obj in chunk_speaker_segments:
                            line = f"Speaker {seg_obj.speaker_label} ({seg_obj.start_time:.2f}s - {seg_obj.end_time:.2f}s): {seg_obj.text}"
                            transcription_lines.append(line)
                            speaker_segments.append(seg_obj)

                        logger.debug(f"Produced {len(chunk_speaker_segments)} aligned speaker segments for chunk {chunk_idx}")

            finally:
                # Clean up the temporary chunk file
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)

        # Post-loop: assign accumulated global words to global diarisation segments.
        if global_diarisation:
            progress.emit("assemble", audio_seconds_done=total_duration, force=True)
            try:
                if global_words and full_diar_segments:
                    logger.info(
                        f"Aligning {len(global_words)} words to "
                        f"{len(full_diar_segments)} global diarisation segments"
                    )
                    # The global case is the one that matters: this runs ONCE
                    # over every word in the whole recording, after the chunk
                    # loop has finished, so nothing else is emitting markers
                    # while it works.
                    final_segments = assign_and_group_words(
                        global_words, full_diar_segments, 0, max_gap=2.0,
                        progress=progress,
                    )
                elif global_words:
                    text = "".join(w["word"] for w in global_words).strip()
                    final_segments = [SpeakerSegment(
                        speaker_label="UNKNOWN",
                        start_time=global_words[0]["start"],
                        end_time=global_words[-1]["end"],
                        text=text
                    )]
                else:
                    final_segments = []

                for seg_obj in final_segments:
                    line = f"Speaker {seg_obj.speaker_label} ({seg_obj.start_time:.2f}s - {seg_obj.end_time:.2f}s): {seg_obj.text}"
                    transcription_lines.append(line)
                    speaker_segments.append(seg_obj)

                logger.info(f"Global assignment produced {len(final_segments)} speaker segments")
            finally:
                if temp_full_path and os.path.exists(temp_full_path):
                    os.remove(temp_full_path)

    # Build result
    elapsed_time = time.time() - start_time_total
    logger.info(f"Transcription complete. Total segments: {len(transcription_lines)}")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    result = TranscriptionResult(
        segments=speaker_segments,
        transcription_lines=transcription_lines,
        parsed_agenda=parsed_agenda,
        elapsed_time=elapsed_time,
        total_duration=total_duration,
    )

    # Apply agenda-aware speaker mapping if agenda was provided
    if parsed_agenda and speaker_segments:
        logger.info("Applying speaker mapping and generating agenda-aware output")
        mapper = SpeakerMapper(parsed_agenda)
        mapped_segments = mapper.map_speakers(speaker_segments)
        speaker_mappings = mapper.get_all_mappings()
        result.mapped_segments = mapped_segments
        result.speaker_mappings = speaker_mappings

    return result
