#!/usr/bin/env python3
"""
CLI wrapper for the audio transcription and diarisation pipeline.

This is the command-line interface for the transcription engine. The core logic
lives in pipeline.py — this file handles argument parsing, validation, and output.

Usage:
    python3 transcribe.py audio_file.m4a -o output.txt
    python3 transcribe.py meeting.m4a --model small --agenda agenda.docx --output-format both -o meeting.txt -v
"""

import os
import sys
import argparse
import logging

from dotenv import load_dotenv

from pipeline import (
    transcribe_with_diarisation,
    check_memory_requirements,
    TranscriptionResult,
    WHISPER_MODEL_MEMORY,
    PYANNOTE_MEMORY,
    BACKENDS,
)
from output_formatter import TranscriptFormatter, SummaryFormatter, save_outputs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def handle_output(result: TranscriptionResult, output_file=None, output_format='transcript'):
    """
    Handle output from a transcription result — write to file or print to console.

    Args:
        result: TranscriptionResult from the pipeline
        output_file: Optional path to output file. If None, prints to console.
        output_format: Output format: 'transcript', 'summary', or 'both'
    """
    if result.parsed_agenda and result.mapped_segments:
        # Agenda-aware output with named speakers
        if output_file:
            save_outputs(result.mapped_segments, result.parsed_agenda,
                        result.speaker_mappings, output_file, format_type=output_format)
        else:
            if output_format in ['transcript', 'both']:
                transcript_formatter = TranscriptFormatter(result.parsed_agenda)
                transcript = transcript_formatter.format(result.mapped_segments, include_timestamps=True)
                print("\n" + transcript)

            if output_format in ['summary', 'both']:
                summary_formatter = SummaryFormatter(result.parsed_agenda)
                summary = summary_formatter.format(result.mapped_segments, result.speaker_mappings)
                print("\n" + summary)
    else:
        # Original output format (no agenda)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(result.transcription_lines))
            logger.info(f"Transcription saved to: {output_file}")
        else:
            print("\n--- Transcription Results ---")
            for line in result.transcription_lines:
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
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3",
                 "large-v3-turbo", "turbo", "distil-large-v3"],
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
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy per-segment transcription instead of full-chunk alignment (slower, less accurate)"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Initial prompt to provide context for Whisper (e.g. domain terms, speaker names). Auto-generated from agenda if --agenda is provided and no manual prompt given."
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam search width for Whisper decoding (default: Whisper's default of 5). Higher values improve accuracy at the cost of speed."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Number of candidate decodings to evaluate (default: Whisper's default of 5). Higher values improve accuracy at the cost of speed."
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        default=None,
        help="Discard segments with compression ratio above this value as likely hallucinated (default: Whisper's default of 2.4)."
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        default=None,
        help="Discard segments with average log probability below this value (default: Whisper's default of -1.0)."
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=None,
        help="Discard segments where no-speech probability exceeds this value (default: Whisper's default of 0.6)."
    )
    parser.add_argument(
        "--normalise",
        action="store_true",
        help="Apply EBU R128 loudness normalisation to audio before transcription. Ensures consistent volume levels."
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply FFT-based noise reduction to audio before transcription. Helps with room noise and hum."
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        choices=["float16", "int8_float16", "int8", "float32"],
        help="Compute type for faster-whisper model (default: auto-select float16 for GPU, int8 for CPU). Lower precision = faster + less memory."
    )
    parser.add_argument(
        "--backend",
        default="faster-whisper",
        choices=BACKENDS,
        help="Transcription backend (default: faster-whisper). Use 'whisperx' for full WhisperX pipeline with integrated diarisation."
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip wav2vec2 timestamp alignment (use faster-whisper's native word timestamps instead). Faster but slightly less precise word boundaries."
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers in the recording. Hint passed to PyAnnote diarisation — useful when speakers are acoustically similar. Overrides --min-speakers/--max-speakers."
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Lower bound on speaker count for PyAnnote diarisation."
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Upper bound on speaker count for PyAnnote diarisation."
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=None,
        help="Override PyAnnote's clustering min_cluster_size (default 12). Lower values (e.g. 6) help short interjections survive instead of being merged into similar-sounding clusters."
    )
    parser.add_argument(
        "--diarise-per-chunk",
        action="store_true",
        help="Diarise each 10-min chunk independently (legacy behaviour). Default runs diarisation once on the full file for global clustering — fixes per-chunk speaker-merging at no time cost."
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

    if args.legacy:
        logger.info("Using legacy per-segment transcription mode")
    else:
        logger.info("Using full-chunk transcription with word-level alignment")

    if args.backend != "faster-whisper":
        logger.info(f"Using backend: {args.backend}")

    # Run the pipeline
    result = transcribe_with_diarisation(
        args.audio_file,
        hf_token,
        args.chunk_duration,
        args.model,
        args.language,
        agenda_path=args.agenda,
        legacy_mode=args.legacy,
        initial_prompt=args.prompt,
        beam_size=args.beam_size,
        best_of=args.best_of,
        compression_ratio_threshold=args.compression_ratio_threshold,
        logprob_threshold=args.logprob_threshold,
        no_speech_threshold=args.no_speech_threshold,
        normalise_audio=args.normalise,
        denoise_audio=args.denoise,
        compute_type=args.compute_type,
        backend=args.backend,
        use_alignment=not args.no_align,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        min_cluster_size=args.min_cluster_size,
        diarise_per_chunk=args.diarise_per_chunk,
    )

    # Handle output
    handle_output(result, output_file=args.output, output_format=args.output_format)
