import whisper
import torch
from pyannote.audio import Pipeline
import os
import soundfile as sf
import numpy as np
import io
import subprocess
import json
import math
import argparse
import logging
import time
import psutil
import gc
from dotenv import load_dotenv
from typing import Optional

from agenda_parser import parse_agenda, ParsedAgenda
from speaker_mapper import SpeakerMapper, SpeakerSegment
from output_formatter import TranscriptFormatter, SummaryFormatter, save_outputs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Memory requirements for Whisper models (in GB)
# These are approximate CPU RAM requirements. GPU VRAM requirements are similar.
WHISPER_MODEL_MEMORY = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 2.5,
    "medium": 5.0,
    "large": 10.0,
    "large-v3-turbo": 8.0,
    "turbo": 6.0
}

# PyAnnote diarization pipeline memory requirement (approximate)
PYANNOTE_MEMORY = 2.5  # GB

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

def check_memory_requirements(whisper_model_name, auto_adjust=False):
    """
    Check if system has enough memory to run the models.

    Args:
        whisper_model_name: Name of the Whisper model to use
        auto_adjust: If True, automatically suggest a smaller model if insufficient memory

    Returns:
        tuple: (can_proceed, recommended_model, warning_message)
    """
    resources = get_system_resources()
    required_memory = WHISPER_MODEL_MEMORY.get(whisper_model_name, 5.0) + PYANNOTE_MEMORY

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

def transcribe_with_diarisation(audio_path, hf_token, chunk_duration_seconds=600, whisper_model_name="medium", output_file=None, language="english", agenda_path: Optional[str] = None, output_format: str = 'transcript'):
    """
    Transcribes an audio file with speaker diarisation, processing in chunks.

    Args:
        audio_path: Path to the audio file to transcribe
        hf_token: Hugging Face authentication token
        chunk_duration_seconds: Duration of each audio chunk in seconds (default: 600)
        whisper_model_name: Name of the Whisper model to use (default: "medium")
        output_file: Optional path to output file. If None, prints to console.
        language: Language code to force (default: "english"). Set to None for auto-detection.
        agenda_path: Optional path to DOCX agenda file for agenda-aware transcription
        output_format: Output format: 'transcript', 'summary', or 'both' (default: 'transcript')
    """
    start_time_total = time.time()
    transcription_lines = []
    speaker_segments = []  # Store as SpeakerSegment objects for agenda-aware processing
    parsed_agenda = None

    # Parse agenda if provided
    if agenda_path:
        if not os.path.exists(agenda_path):
            logger.error(f"Agenda file not found: {agenda_path}")
            raise RuntimeError(f"Agenda file not found: {agenda_path}")

        logger.info(f"Parsing agenda document: {agenda_path}")
        parsed_agenda = parse_agenda(agenda_path)
        logger.info(f"Agenda parsed: {len(parsed_agenda.sections)} sections, "
                   f"{len(parsed_agenda.all_speakers)} speakers listed")

    logger.info(f"Loading Whisper model: {whisper_model_name}")
    whisper_model = whisper.load_model(whisper_model_name)

    logger.info("Loading PyAnnote speaker diarisation pipeline")
    diarisation_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
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
    except FileNotFoundError:
        logger.error("ffprobe not found. Please install FFmpeg.")
        raise RuntimeError("ffprobe not found. Please install FFmpeg.")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed: {e.stderr.decode('utf-8')}")
        raise RuntimeError(f"ffprobe failed: {e.stderr.decode('utf-8')}")
    except ValueError:
        logger.error(f"Could not parse audio duration from file: {audio_path}")
        raise RuntimeError(f"Could not parse audio duration from file: {audio_path}")

    # Process in chunks
    num_chunks = math.ceil(total_duration / chunk_duration_seconds)
    logger.info(f"Processing audio in {num_chunks} chunk(s) of {chunk_duration_seconds} seconds each")

    for chunk_idx, start_time in enumerate(range(0, math.ceil(total_duration), chunk_duration_seconds), 1):
        end_time = min(start_time + chunk_duration_seconds, total_duration)
        chunk_file = f"temp_chunk_{start_time}.wav"

        logger.info(f"Processing chunk {chunk_idx}/{num_chunks} ({start_time}s - {end_time}s)")

        # Use ffmpeg to create chunk
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-i", audio_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-ac", "1",
            "-ar", "16000",
            chunk_file
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install FFmpeg.")
            raise RuntimeError("ffmpeg not found. Please install FFmpeg.")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr.decode('utf-8')}")
            raise RuntimeError(f"ffmpeg failed: {e.stderr.decode('utf-8')}")

        try:
            # Perform speaker diarisation on the chunk
            logger.debug(f"Running speaker diarisation on chunk {chunk_idx}")
            diarisation = diarisation_pipeline(chunk_file)

            # Get the sample rate of the audio file
            info = sf.info(chunk_file)
            sample_rate = info.samplerate

            # Read entire chunk into memory for reliable slicing
            chunk_audio, _ = sf.read(chunk_file)

            # Count speaker segments for progress tracking
            segments = list(diarisation.speaker_diarization.itertracks(yield_label=True))
            logger.debug(f"Found {len(segments)} speaker segments in chunk {chunk_idx}")

            # For each speaker segment, transcribe the audio
            for seg_idx, (turn, _, speaker) in enumerate(segments, 1):
                # Extract the audio segment for the current speaker
                start_frame = int(turn.start * sample_rate)
                end_frame = int(turn.end * sample_rate)

                # Slice the audio data from memory
                segment_audio_data = chunk_audio[start_frame:end_frame]

                # Convert the audio data to float32
                segment_audio_data = segment_audio_data.astype(np.float32)

                # Transcribe the segment using Whisper
                logger.debug(f"Transcribing segment {seg_idx}/{len(segments)} - Speaker {speaker}")

                # Build transcribe parameters
                transcribe_params = {"audio": segment_audio_data}
                if language:
                    transcribe_params["language"] = language

                result = whisper_model.transcribe(**transcribe_params)
                transcription = result["text"]

                # Adjust timestamps for the original file
                original_start = start_time + turn.start
                original_end = start_time + turn.end

                line = f"Speaker {speaker} ({original_start:.2f}s - {original_end:.2f}s): {transcription}"
                transcription_lines.append(line)

                # Also create SpeakerSegment object for agenda-aware processing
                segment_obj = SpeakerSegment(
                    speaker_label=speaker,
                    start_time=original_start,
                    end_time=original_end,
                    text=transcription
                )
                speaker_segments.append(segment_obj)

                logger.debug(f"Completed segment {seg_idx}/{len(segments)}")

        finally:
            # Clean up the temporary chunk file
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

    # Output results
    elapsed_time = time.time() - start_time_total
    logger.info(f"Transcription complete. Total segments: {len(transcription_lines)}")
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Handle agenda-aware output if agenda was provided
    if parsed_agenda and speaker_segments:
        logger.info("Applying speaker mapping and generating agenda-aware output")

        # Map speakers to real names
        mapper = SpeakerMapper(parsed_agenda)
        mapped_segments = mapper.map_speakers(speaker_segments)
        speaker_mappings = mapper.get_all_mappings()

        # Generate formatted output
        if output_file:
            save_outputs(mapped_segments, parsed_agenda, speaker_mappings,
                        output_file, format_type=output_format)
        else:
            # Print to console
            if output_format in ['transcript', 'both']:
                transcript_formatter = TranscriptFormatter(parsed_agenda)
                transcript = transcript_formatter.format(mapped_segments, include_timestamps=True)
                print("\n" + transcript)

            if output_format in ['summary', 'both']:
                summary_formatter = SummaryFormatter(parsed_agenda)
                summary = summary_formatter.format(mapped_segments, speaker_mappings)
                print("\n" + summary)
    else:
        # Original output format (no agenda)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(transcription_lines))
            logger.info(f"Transcription saved to: {output_file}")
        else:
            print("\n--- Transcription Results ---")
            for line in transcription_lines:
                print(line)

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with speaker diarisation using Whisper and PyAnnote."
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe"
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=600,
        help="Duration of audio chunks in seconds (default: 600)"
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo", "turbo"],
        help="Whisper model to use (default: medium)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save transcription (default: print to console)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output"
    )
    parser.add_argument(
        "--language",
        "-l",
        default="english",
        help="Force specific language for transcription (default: english). Set to None for auto-detection. Supported: english, welsh, french, etc."
    )
    parser.add_argument(
        "--auto-adjust",
        action="store_true",
        help="Automatically adjust model size if insufficient memory detected"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if insufficient memory is detected (may cause OOM errors)"
    )
    parser.add_argument(
        "--agenda",
        help="Path to DOCX agenda file for agenda-aware transcription with speaker name mapping"
    )
    parser.add_argument(
        "--output-format",
        default="transcript",
        choices=["transcript", "summary", "both"],
        help="Output format (default: transcript). Requires --agenda. 'transcript' generates full verbatim with named speakers, 'summary' generates executive summary, 'both' generates separate files for each."
    )

    args = parser.parse_args()

    # Adjust logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate audio file exists
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        exit(1)

    if not os.path.isfile(args.audio_file):
        logger.error(f"Path is not a file: {args.audio_file}")
        exit(1)

    # Get the Hugging Face token from environment or .env file
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not found. Please either:")
        logger.error("  1. Create a .env file with: HF_TOKEN=your_token_here")
        logger.error("  2. Set environment variable: export HF_TOKEN=your_token_here")
        exit(1)

    # Check memory requirements before proceeding
    can_proceed, recommended_model, warning_msg = check_memory_requirements(
        args.model,
        auto_adjust=args.auto_adjust
    )

    if not can_proceed:
        logger.warning(warning_msg)

        if recommended_model:
            logger.warning(f"Recommended model for your system: {recommended_model}")

            if args.auto_adjust:
                logger.info(f"Auto-adjusting from {args.model} to {recommended_model} due to memory constraints")
                args.model = recommended_model
            else:
                logger.warning("Consider using --auto-adjust flag to automatically select appropriate model")
                logger.warning(f"Or manually specify a smaller model with: --model {recommended_model}")

        if not args.force and not args.auto_adjust:
            logger.error("Insufficient memory detected. Use --force to proceed anyway (risk of OOM error)")
            logger.error("Or use --auto-adjust to automatically select a suitable model size")
            exit(1)
        elif not args.auto_adjust:
            logger.warning("Proceeding with --force flag. System may run out of memory!")

    # Validate agenda-related arguments
    if args.output_format != "transcript" and not args.agenda:
        logger.warning("--output-format requires --agenda to be specified. Using default transcript format.")
        args.output_format = "transcript"

    if args.agenda and not os.path.exists(args.agenda):
        logger.error(f"Agenda file not found: {args.agenda}")
        exit(1)

    if args.language:
        logger.info(f"Starting transcription of: {args.audio_file} (forced language: {args.language})")
    else:
        logger.info(f"Starting transcription of: {args.audio_file} (auto-detect language)")

    if args.agenda:
        logger.info(f"Agenda-aware mode enabled with output format: {args.output_format}")

    transcribe_with_diarisation(
        args.audio_file,
        hf_token,
        args.chunk_duration,
        args.model,
        args.output,
        args.language,
        agenda_path=args.agenda,
        output_format=args.output_format
    )