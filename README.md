# Audio Transcription with Speaker Diarisation

A Python-based audio transcription pipeline that combines OpenAI Whisper (speech-to-text) with PyAnnote.audio (speaker identification) to transcribe meeting recordings and attribute text to specific speakers.

## Features

- 🎙️ **Speaker Diarisation**: Automatically identifies and labels different speakers
- 🗣️ **Multi-Language Support**: Transcribe in 99 languages or auto-detect
- 🧠 **Smart Memory Management**: Automatic resource checking prevents OOM crashes
- ⚡ **GPU Acceleration**: Automatically uses CUDA when available
- 📦 **Chunked Processing**: Handles large audio files by processing in segments
- 🔒 **Secure**: Token management via `.env` files
- 🎛️ **Flexible Models**: Choose from 7 Whisper models (tiny to large)

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

The script automatically checks available RAM/VRAM before loading models:

| Model | Memory Needed | Recommended RAM |
|-------|---------------|-----------------|
| tiny | 3.5 GB | 4+ GB |
| base | 4.0 GB | 5+ GB |
| small | 5.0 GB | 6+ GB |
| medium (default) | 7.5 GB | 9+ GB |
| large | 12.5 GB | 15+ GB |

**Pro tip**: Use `--auto-adjust` to automatically select the best model for your system:

```bash
python3 transcribe.py audio.m4a --model large --auto-adjust -o output.txt
```

## Usage Examples

```bash
# Basic usage (medium model, English)
python3 transcribe.py meeting.m4a -o transcription.txt

# Use smaller model for faster processing
python3 transcribe.py meeting.m4a --model small -o output.txt

# Auto-adjust model based on available memory
python3 transcribe.py meeting.m4a --model large --auto-adjust -o output.txt

# Welsh language transcription
python3 transcribe.py meeting.m4a --language welsh -o output.txt

# Auto-detect language
python3 transcribe.py meeting.m4a --language None -o output.txt

# Verbose logging for detailed progress
python3 transcribe.py meeting.m4a -v -o output.txt
```

## Requirements

- Python 3.13.7+
- FFmpeg (for audio processing)
- 4-15GB RAM (depending on model choice)
- Optional: CUDA-capable GPU for acceleration
- Hugging Face account with access to PyAnnote models

## Documentation

See [CLAUDE.md](CLAUDE.md) for comprehensive documentation including:
- Architecture details
- Setup instructions
- Performance benchmarks
- Development conventions
- Troubleshooting guide

## Output Format

```
Speaker SPEAKER_00 (0.50s - 3.25s): Hello everyone, welcome to the meeting.
Speaker SPEAKER_01 (3.80s - 8.15s): Thank you for having me today.
Speaker SPEAKER_00 (8.90s - 12.45s): Let's start with the first agenda item.
```

## License

[Your chosen license]

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.
