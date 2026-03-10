#!/usr/bin/env python3
"""
Gradio GUI for Audio Transcription Tool

A web-based interface for the audio transcription pipeline with speaker
diarization and agenda-aware features.

Usage:
    python3 gradio_app.py

Then open browser to: http://localhost:7860
"""

import gradio as gr
import os
import logging
from typing import Optional, Tuple
import psutil

from pipeline import (
    transcribe_with_diarisation,
    check_memory_requirements,
    TranscriptionResult,
    WHISPER_MODEL_MEMORY,
    PYANNOTE_MEMORY,
)
from output_formatter import TranscriptFormatter, SummaryFormatter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_system_info() -> str:
    """Get current system resource information as formatted string."""
    ram = psutil.virtual_memory()
    available_gb = ram.available / (1024**3)
    total_gb = ram.total / (1024**3)
    used_percent = ram.percent

    return f"💻 **System Resources**: {available_gb:.1f} GB available / {total_gb:.1f} GB total ({used_percent:.0f}% used)"


def estimate_memory(model: str) -> str:
    """Estimate memory requirements for selected model."""
    model_mem = WHISPER_MODEL_MEMORY.get(model, 5.0)
    total_mem = model_mem + PYANNOTE_MEMORY

    return f"📊 **Estimated Memory**: ~{total_mem:.1f} GB ({model_mem:.1f} GB model + {PYANNOTE_MEMORY:.1f} GB PyAnnote)"


import asyncio
import threading

def transcribe_audio(
    audio_file,
    agenda_file,
    model: str,
    language: str,
    output_format: str,
    chunk_duration: int,
    auto_adjust: bool,
    force: bool,
    verbose: bool,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """
    Wrapper function for Gradio interface.

    NOTE: This runs transcription synchronously. For very long audio files,
    consider using the CLI instead to avoid browser timeout issues.

    Returns:
        Tuple of (transcript_output, summary_output, status_message)
    """
    try:
        # Validate inputs
        if audio_file is None:
            return "", "", "❌ Error: Please upload an audio file"

        # Handle both string paths and file objects
        audio_path = audio_file if isinstance(audio_file, str) else audio_file.name
        agenda_path = None
        if agenda_file:
            agenda_path = agenda_file if isinstance(agenda_file, str) else agenda_file.name

        # Check file size and warn about processing time
        import os.path
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

        if file_size_mb > 50:
            warning = f"⚠️ **Large file detected ({file_size_mb:.1f} MB)**\n\n"
            warning += "This may take a very long time to process and could cause the browser to timeout.\n"
            warning += "For files larger than 50MB, we recommend using the CLI instead:\n\n"
            warning += f"```bash\npython3 transcribe.py \"{audio_path}\" -o output.txt\n```\n\n"
            warning += "Continue anyway in GUI? This page may freeze for several hours..."
            # For now, we'll still proceed but with warning
            progress(0, desc=f"⚠️ Large file - may take hours...")

        # Check for HF_TOKEN
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            return "", "", "❌ Error: HF_TOKEN not found. Please set it in .env file"

        # Update progress
        progress(0, desc="Validating inputs...")

        # Check memory requirements
        can_proceed, recommended_model, warning_msg = check_memory_requirements(
            model, auto_adjust=auto_adjust
        )

        if not can_proceed and not force:
            if auto_adjust and recommended_model:
                model = recommended_model
                status = f"⚠️ Auto-adjusted to {model} model due to memory constraints\n\n"
            else:
                return "", "", f"❌ {warning_msg}\n\nEnable 'Force execution' to proceed anyway (may crash)"
        else:
            status = ""

        progress(0.05, desc="Initializing...")

        # Handle language
        if language == "Auto-detect":
            language = None

        # Update progress
        progress(0.1, desc="Loading models...")

        # Run transcription
        logger.info(f"Starting transcription via Gradio: {audio_path}")

        result = transcribe_with_diarisation(
            audio_path=audio_path,
            hf_token=hf_token,
            chunk_duration_seconds=chunk_duration,
            whisper_model_name=model,
            language=language,
            agenda_path=agenda_path,
        )

        progress(1.0, desc="Complete!")

        # Format results
        transcript_text = ""
        summary_text = ""
        effective_format = output_format if agenda_path else 'transcript'

        if result.parsed_agenda and result.mapped_segments:
            # Agenda-aware output with named speakers
            if effective_format in ['transcript', 'both']:
                formatter = TranscriptFormatter(result.parsed_agenda)
                transcript_text = formatter.format(result.mapped_segments, include_timestamps=True)

            if effective_format in ['summary', 'both']:
                formatter = SummaryFormatter(result.parsed_agenda)
                summary_text = formatter.format(result.mapped_segments, result.speaker_mappings)

            if effective_format == 'both':
                status += "✅ **Transcription Complete!**\n\nBoth transcript and summary generated successfully."
            elif effective_format == 'summary':
                status += "✅ **Summary Generated!**\n\nExecutive summary created successfully."
            else:
                status += "✅ **Transcription Complete!**\n\nFull transcript generated successfully."
        else:
            # Plain transcript output
            transcript_text = '\n'.join(result.transcription_lines)
            status += "✅ **Transcription Complete!**\n\nFull transcript generated successfully."

        return transcript_text, summary_text, status

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return "", "", f"❌ **Error**: {str(e)}"


# Create Gradio interface
with gr.Blocks(
    title="Audio Transcription Tool",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {max-width: 1200px !important}
        #status-box {min-height: 100px}
    """
) as demo:

    gr.Markdown("""
    # 🎤 Audio Transcription with Speaker Diarization

    Upload an audio file to transcribe with automatic speaker identification.
    Optionally provide a meeting agenda to map speakers to real names and generate structured summaries.

    ⚠️ **Important**: This GUI is best for **short audio clips (< 10 minutes) for testing**.
    For long recordings, the browser may timeout. Use the CLI instead:
    ```bash
    python3 transcribe.py audio.m4a --agenda agenda.docx -o output.txt
    ```
    """)

    # System info display
    system_info = gr.Markdown(get_system_info())

    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Input Files")

            audio_input = gr.Audio(
                label="Audio File",
                type="filepath",
                sources=["upload"]
            )

            agenda_input = gr.File(
                label="Agenda Document (optional - DOCX)",
                file_types=[".docx", ".doc"]
            )

            gr.Markdown("### ⚙️ Configuration")

            model_select = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large", "large-v3-turbo", "turbo"],
                value="medium",
                label="Whisper Model",
                info="Larger models are more accurate but require more memory"
            )

            # Update memory estimate when model changes
            memory_estimate = gr.Markdown(estimate_memory("medium"))
            model_select.change(
                fn=estimate_memory,
                inputs=[model_select],
                outputs=[memory_estimate]
            )

            language_select = gr.Dropdown(
                choices=["english", "welsh", "french", "spanish", "german", "Auto-detect"],
                value="english",
                label="Language",
                info="Force specific language or auto-detect"
            )

            output_format_radio = gr.Radio(
                choices=["transcript", "summary", "both"],
                value="transcript",
                label="Output Format (requires agenda)",
                info="Transcript: full verbatim | Summary: executive summary | Both: separate files"
            )

            with gr.Accordion("Advanced Options", open=False):
                chunk_duration_slider = gr.Slider(
                    minimum=300,
                    maximum=900,
                    value=600,
                    step=60,
                    label="Chunk Duration (seconds)",
                    info="Process audio in chunks to manage memory"
                )

                auto_adjust_check = gr.Checkbox(
                    label="Auto-adjust model size",
                    value=True,
                    info="Automatically select smaller model if insufficient memory"
                )

                force_check = gr.Checkbox(
                    label="Force execution",
                    value=False,
                    info="⚠️ Proceed even if insufficient memory (may crash)"
                )

                verbose_check = gr.Checkbox(
                    label="Verbose logging",
                    value=False,
                    info="Show detailed progress in console"
                )

            transcribe_btn = gr.Button(
                "🚀 Start Transcription",
                variant="primary",
                size="lg"
            )

        # Right column - Outputs
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Status")

            status_output = gr.Markdown(
                "Ready to transcribe. Upload an audio file and click 'Start Transcription'.",
                elem_id="status-box"
            )

            gr.Markdown("### 📄 Results")

            with gr.Tabs():
                with gr.Tab("Transcript"):
                    transcript_output = gr.Textbox(
                        label="Full Transcript",
                        lines=20,
                        max_lines=30,
                        show_copy_button=True,
                        placeholder="Transcript will appear here..."
                    )

                    transcript_download = gr.File(
                        label="Download Transcript",
                        visible=False
                    )

                with gr.Tab("Summary"):
                    summary_output = gr.Textbox(
                        label="Executive Summary",
                        lines=20,
                        max_lines=30,
                        show_copy_button=True,
                        placeholder="Summary will appear here (requires agenda)..."
                    )

                    summary_download = gr.File(
                        label="Download Summary",
                        visible=False
                    )

    # Footer
    gr.Markdown("""
    ---
    ### 💡 Tips
    - **Memory**: Larger models need more RAM/VRAM. Use auto-adjust to prevent crashes.
    - **Agenda**: Upload a DOCX agenda to map speaker labels (SPEAKER_00, SPEAKER_01) to real names.
    - **Performance**: Processing time varies by model size and audio length. Medium model takes ~3 hours for 60min audio on CPU.
    - **Language**: Forcing language (e.g., English) is ~2x faster than auto-detect.

    📖 [Documentation](https://github.com/Technologikal/audio-transcription-diarisation) |
    🐛 [Report Issues](https://github.com/Technologikal/audio-transcription-diarisation/issues)
    """)

    # Connect the transcribe button
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[
            audio_input,
            agenda_input,
            model_select,
            language_select,
            output_format_radio,
            chunk_duration_slider,
            auto_adjust_check,
            force_check,
            verbose_check
        ],
        outputs=[
            transcript_output,
            summary_output,
            status_output
        ]
    )

    # Note: Auto-refresh removed due to Gradio API changes
    # System info will update on page load


if __name__ == "__main__":
    logger.info("Starting Gradio interface...")

    # Check for HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN not found! Please set it in .env file")

    # Launch the interface
    demo.launch(
        server_name="127.0.0.1",  # Only accessible locally by default
        server_port=7860,
        share=False,  # Set to True to create public link for testing
        show_error=True
    )
