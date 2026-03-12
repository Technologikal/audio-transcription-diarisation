# Audio Transcription with Speaker Diarisation

A Python-based audio transcription pipeline that combines faster-whisper (CTranslate2-based, ~4x faster than openai-whisper) with PyAnnote.audio (speaker identification) and optional wav2vec2 alignment to transcribe meeting recordings and attribute text to specific speakers.

## Features

- **Speaker Diarisation**: Automatically identifies and labels different speakers
- **Multi-Language Support**: Transcribe in 99 languages or auto-detect
- **Smart Memory Management**: Automatic resource checking prevents OOM crashes
- **GPU Acceleration**: Automatically uses CUDA when available
- **Chunked Processing**: Handles large audio files by processing in segments
- **Agenda-Aware Mode**: Map anonymous speakers to real names using a DOCX agenda
- **MCP Server**: Expose transcription as tools for Claude via FastMCP
- **Dual Backend**: faster-whisper (default) or WhisperX
- **Docker/Crucible Support**: Dockerfile for containerised deployment

## Quick Start

```bash
# Setup
cd transcription_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure (add your Hugging Face token)
cp .env.example .env
# Edit .env and add your HF_TOKEN

# Run
python3 transcribe.py audio_file.m4a -o output.txt
```

## Memory Requirements

The script automatically checks available RAM/VRAM before loading models.
faster-whisper uses CTranslate2 optimisation, requiring significantly less memory than openai-whisper:

| Model | Total Memory | Recommended RAM/VRAM |
|-------|-------------|----------------------|
| tiny | 3.0 GB | 4+ GB |
| small | 3.5 GB | 5+ GB |
| medium (default) | 4.5 GB | 6+ GB |
| large-v3 | 5.5 GB | 7+ GB |

**Pro tip**: Use `--auto-adjust` to automatically select the best model for your system:

```bash
python3 transcribe.py audio.m4a --model large-v3 --auto-adjust -o output.txt
```

## Usage Examples

```bash
# Basic usage (medium model, English)
python3 transcribe.py meeting.m4a -o transcription.txt

# Use large-v3 for highest accuracy
python3 transcribe.py meeting.m4a --model large-v3 --auto-adjust -o output.txt

# Agenda-aware transcription with speaker names
python3 transcribe.py meeting.m4a --agenda agenda.docx --output-format both -o meeting.txt

# Audio pre-processing (denoise + normalise)
python3 transcribe.py meeting.m4a --denoise --normalise -o output.txt

# MCP server mode (for Claude integration)
python3 mcp_server.py
```

## Requirements

- Python 3.12+
- FFmpeg (for audio processing)
- 4-7GB RAM (depending on model choice)
- Optional: CUDA-capable GPU for acceleration
- Hugging Face account with access to PyAnnote models

## Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Architecture details
- Setup instructions
- Performance benchmarks
- Development conventions

## Output Format

```
Speaker SPEAKER_00 (0.50s - 3.25s): Hello everyone, welcome to the meeting.
Speaker SPEAKER_01 (3.80s - 8.15s): Thank you for having me today.
Speaker SPEAKER_00 (8.90s - 12.45s): Let's start with the first agenda item.
```

## Licence

[Your chosen licence]

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
