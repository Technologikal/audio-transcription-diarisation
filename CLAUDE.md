# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based audio transcription and speaker diarisation pipeline that combines OpenAI Whisper (speech-to-text) with PyAnnote.audio (speaker identification) to transcribe meeting recordings and attribute text to specific speakers.

## Architecture

The system uses a **chunked processing pipeline** to handle large audio files:

```
Audio Input → FFmpeg (chunk to 10min WAV segments, 16kHz mono)
  → PyAnnote (identify speaker segments)
  → For each speaker segment: extract audio → Whisper (transcribe)
  → Output: Speaker + Timestamps + Text
```

**Key Design Pattern**: Large audio files are processed in 10-minute chunks to prevent memory overflow. Each chunk is independently diarized, then individual speaker segments within the chunk are transcribed. Timestamps are adjusted to maintain continuity across chunks.

**Main Components**:
- `transcription_project/transcribe.py` - Main orchestration script containing `transcribe_with_diarisation()` function
- `transcription_project/agenda_parser.py` - Parses DOCX agenda files to extract metadata, sections, and speakers
- `transcription_project/speaker_mapper.py` - Maps anonymous speaker labels to real names using hybrid approach
- `transcription_project/output_formatter.py` - Generates structured transcripts and executive summaries
- `transcription_project/debug_pyannote.py` - Utility to test PyAnnote authentication and pipeline loading
- `transcription_project/.env` - Secure storage for HF_TOKEN (git-ignored)
- `transcription_project/.env.example` - Template for environment variables

## Setup and Dependencies

### Repository Information

**GitHub Repository**: https://github.com/Technologikal/audio-transcription-diarisation

This project is version-controlled with Git and hosted on GitHub (private repository). This enables:
- Multi-machine development
- Version history tracking
- Collaboration capabilities
- Automatic backup

### Initial Setup

**Option 1: Clone from GitHub (Recommended)**

```bash
# Clone the repository
git clone https://github.com/Technologikal/audio-transcription-diarisation.git
cd audio-transcription-diarisation/transcription_project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure authentication
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

**Option 2: Local Setup (if already have the files)**

```bash
cd transcription_project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The active virtual environment is at `transcription_project/venv/` (not the root-level venv which is empty).

### Hugging Face Authentication (Required)

PyAnnote.audio requires authentication to download pre-trained models. The project supports two authentication methods:

#### Method 1: .env File (Recommended - Secure)

1. Create Hugging Face account at [huggingface.co](https://huggingface.co/)
2. Generate access token with "read" permissions in settings
3. Accept user conditions for these models on Hugging Face:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarisation-3.1](https://huggingface.co/pyannote/speaker-diarisation-3.1)
   - [pyannote/speaker-diarisation-community-1](https://huggingface.co/pyannote/speaker-diarisation-community-1)
4. Create `.env` file in `transcription_project/` directory:
   ```bash
   # Copy the example file
   cp .env.example .env

   # Edit .env and add your token
   echo "HF_TOKEN=your_hugging_face_token_here" > .env
   ```

The script automatically loads the token from `.env` file using `python-dotenv`. The `.env` file is git-ignored for security.

#### Method 2: Environment Variable (Alternative)

```bash
export HF_TOKEN="your_hugging_face_access_token"
```

**Note**: The .env method is preferred as it persists across sessions and keeps tokens out of shell history.

### System Dependencies

- **FFmpeg** (n8.0 or compatible) - Required for audio conversion and chunking
- **Python 3.13.7** - Runtime environment
- **CUDA-capable GPU** - Optional but recommended; pipeline automatically uses GPU if available

### Model Memory Requirements

The script requires sufficient RAM (or VRAM if using GPU) to load both the Whisper model and PyAnnote diarisation pipeline simultaneously. **Resource checking is performed automatically before model loading** to prevent out-of-memory (OOM) errors.

**Memory Requirements by Model** (approximate GB needed):

| Model | Whisper | PyAnnote | Total | Recommended RAM/VRAM |
|-------|---------|----------|-------|----------------------|
| `tiny` | 1.0 GB | 2.5 GB | 3.5 GB | 4+ GB |
| `base` | 1.5 GB | 2.5 GB | 4.0 GB | 5+ GB |
| `small` | 2.5 GB | 2.5 GB | 5.0 GB | 6+ GB |
| `medium` | 5.0 GB | 2.5 GB | 7.5 GB | 9+ GB |
| `turbo` | 6.0 GB | 2.5 GB | 8.5 GB | 10+ GB |
| `large-v3-turbo` | 8.0 GB | 2.5 GB | 10.5 GB | 13+ GB |
| `large` | 10.0 GB | 2.5 GB | 12.5 GB | 15+ GB |

**Automatic Resource Management**:

The script automatically checks available system resources before loading models and will:
1. Report available RAM/VRAM and required memory
2. Warn if insufficient memory detected
3. Suggest appropriate model size for your system
4. Optionally auto-adjust to a smaller model with `--auto-adjust` flag
5. Prevent execution unless `--force` flag is used

**Examples**:
```bash
# Let script automatically select appropriate model size
python3 transcribe.py audio.m4a --model large --auto-adjust

# Force execution despite memory warnings (may cause OOM crash)
python3 transcribe.py audio.m4a --model large --force

# Check memory requirements without running
python3 transcribe.py audio.m4a --model large --verbose
```

The script adds a 20% safety buffer to prevent edge-case OOM errors. Memory is checked against GPU VRAM if CUDA is available, otherwise against system RAM.

## Common Commands

### Run Transcription

The script supports a full command-line interface with sensible defaults optimized for accuracy:

```bash
# Activate environment (token auto-loaded from .env)
source transcription_project/venv/bin/activate

# Basic usage with defaults (medium model, English language, console output)
python3 transcribe.py audio_file.m4a

# Save to output file
python3 transcribe.py audio_file.m4a -o transcription.txt

# Use different model
python3 transcribe.py audio_file.m4a --model small

# Force different language or auto-detect
python3 transcribe.py audio_file.m4a --language welsh
python3 transcribe.py audio_file.m4a --language None  # auto-detect

# Custom chunk size for memory-constrained systems
python3 transcribe.py audio_file.m4a --chunk-duration 300

# Enable verbose logging for detailed progress
python3 transcribe.py audio_file.m4a --verbose -o output.txt

# Auto-adjust model size based on available memory (recommended for large models)
python3 transcribe.py audio_file.m4a --model large --auto-adjust -o output.txt

# Force execution even if insufficient memory detected (use with caution)
python3 transcribe.py audio_file.m4a --model large --force -o output.txt

# Combine multiple options
python3 transcribe.py meeting.m4a --model small -l english -o meeting.txt -v

# Get help
python3 transcribe.py --help
```

**Available Options**:
- `audio_file` (required): Path to audio file to transcribe
- `--model`: Whisper model (tiny, base, small, medium, large, large-v3-turbo, turbo) [default: **medium**]
- `--language` / `-l`: Language code for transcription [default: **english**]
- `--chunk-duration`: Chunk size in seconds [default: 600]
- `--output` / `-o`: Output file path (prints to console if not specified)
- `--verbose` / `-v`: Enable detailed debug logging
- `--auto-adjust`: Automatically select smaller model if insufficient memory detected
- `--force`: Force execution even if insufficient memory (may cause OOM errors)
- `--agenda`: Path to DOCX agenda file for agenda-aware transcription with speaker name mapping
- `--output-format`: Output format (transcript, summary, or both) [default: **transcript**]. Requires `--agenda`.

### Debug PyAnnote Setup

```bash
source transcription_project/venv/bin/activate
python3 debug_pyannote.py
```

This validates Hugging Face authentication and tests model loading with debug-level logging.

### Agenda-Aware Transcription (NEW)

The system now supports **agenda-aware transcription** for community meetings. When you provide a meeting agenda (in DOCX format), the system will:

1. **Parse the agenda** to extract meeting metadata (date, time, location), sections/topics, and speaker assignments
2. **Map anonymous speaker labels** (SPEAKER_00, SPEAKER_01, etc.) to real names using a hybrid approach
3. **Generate structured output** with named speakers organized by agenda sections
4. **Create executive summaries** with key discussion points cross-referenced to agenda topics

**Key Features**:
- Extracts speaker names from agenda documents
- Maps diarization speakers to real names using temporal analysis and agenda context
- Provides confidence scores for speaker mappings
- Generates two output formats:
  - **Structured Transcript**: Full verbatim with real names, timestamps, and section markers
  - **Executive Summary**: Meeting overview with key points organized by agenda topics

**Usage Examples**:

```bash
# Basic agenda-aware transcription (generates structured transcript)
python3 transcribe.py meeting.m4a --agenda agenda.docx -o output.txt

# Generate executive summary instead of full transcript
python3 transcribe.py meeting.m4a --agenda agenda.docx --output-format summary -o summary.txt

# Generate both transcript and summary (creates two files)
python3 transcribe.py meeting.m4a --agenda agenda.docx --output-format both -o meeting.txt
# This creates: meeting_transcript.txt and meeting_summary.txt

# Combine with other options
python3 transcribe.py meeting.m4a --agenda agenda.docx --model small --output-format both -o meeting.txt -v
```

**Agenda Document Requirements**:

The agenda parser is designed to handle semi-structured DOCX files. It looks for:

1. **Metadata** (typically at the top of the document):
   - Date: "Date: 15 November 2025" or "Meeting Date: 15/11/2025"
   - Time: "Time: 14:00" or "2:00 PM"
   - Location: "Location: Community Centre" or "Venue: Main Hall"
   - Title: "Meeting: Monthly Board Meeting"

2. **Sections/Topics** (identified by):
   - Heading styles in the document
   - Section numbers (1., 1.1, A., etc.)
   - Bold text with short length (< 100 chars)

3. **Speaker Names** (extracted from patterns like):
   - "Presenter: John Smith"
   - "Led by: Jane Doe"
   - "Speaker: John Smith"
   - "(John Smith)" in section descriptions
   - "- John Smith" at start of line

**Speaker Mapping Algorithm** (Hybrid Approach):

The system uses a sophisticated hybrid approach to map anonymous speaker labels to real names:

1. **Temporal Alignment**: Divides audio into sections corresponding to agenda topics
2. **Dominance Analysis**: Identifies which speaker is most active in each section
3. **Confidence Scoring**: Assigns confidence based on multiple factors:
   - Dominance ratio in primary section (> 60% = high confidence)
   - Exclusive speaker assignment (only one speaker listed for section)
   - Overall speaking time ratio (> 20% = significant contribution)
4. **Name Assignment**: Maps speakers to names from agenda using evidence-based matching

**Output Format Examples**:

**Structured Transcript**:
```
================================================================================
MEETING TRANSCRIPT
================================================================================

Meeting: Monthly Community Board Meeting
Date: 15 November 2025
Time: 14:00 - 16:00
Location: Community Centre Main Hall

--------------------------------------------------------------------------------

================================================================================
SECTION: Welcome and Introductions
================================================================================

[00:00 - 01:23] Sarah Johnson: Welcome everyone to our monthly board meeting...
[01:24 - 02:45] Mark Williams: Thank you Sarah. I'd like to start by...
```

**Executive Summary**:
```
================================================================================
MEETING SUMMARY
================================================================================

Meeting: Monthly Community Board Meeting
Date: 15 November 2025

--------------------------------------------------------------------------------
PARTICIPANTS
--------------------------------------------------------------------------------

  • Sarah Johnson
  • Mark Williams
  • Emily Chen (confidence: 65%)

--------------------------------------------------------------------------------
AGENDA AND DISCUSSION
--------------------------------------------------------------------------------

1. Welcome and Introductions
   Led by: Sarah Johnson

   Discussion highlights:

     • Sarah Johnson: Welcome everyone to our monthly board meeting.
       Today we'll be discussing the budget proposal and upcoming
       community events...

     • Mark Williams: Thank you Sarah. I'd like to start by
       addressing the concerns raised in the last meeting...
```

**Architecture Components**:

The agenda-aware feature is built using three new modules:

- **`agenda_parser.py`**: Parses DOCX agenda files to extract metadata, sections, and speakers
  - `AgendaParser` class handles document parsing with flexible pattern matching
  - `ParsedAgenda` dataclass stores extracted information

- **`speaker_mapper.py`**: Maps anonymous speaker labels to real names
  - `SpeakerMapper` class implements hybrid mapping algorithm
  - `SpeakerSegment` dataclass represents individual speech segments
  - Confidence scoring based on temporal analysis and dominance ratios

- **`output_formatter.py`**: Generates formatted transcripts and summaries
  - `TranscriptFormatter` creates structured transcripts with named speakers
  - `SummaryFormatter` generates executive summaries organized by agenda topics

**Dependencies**:

The agenda feature requires the `python-docx` library for parsing DOCX files. This is included in `requirements.txt` and installed automatically with:

```bash
pip install -r requirements.txt
```

### Gradio Web GUI (EXPERIMENTAL)

⚠️ **Status: Proof-of-Concept / Prototype**

An experimental web-based GUI is available for testing and demos. **This is NOT recommended for production use** due to known limitations.

**Files:**
- `transcription_project/gradio_app.py` - Full-featured web interface
- `transcription_project/gradio_simple.py` - Minimal test interface

**Launch the GUI:**

```bash
# Activate environment
source transcription_project/venv/bin/activate

# Launch web interface
python3 gradio_app.py

# Opens browser to http://localhost:7860
```

**Features:**
- 🎤 Audio file upload with drag-and-drop
- 📄 Optional agenda document upload
- 🎛️ Model selection and configuration
- 💻 Real-time system resource monitoring
- 📊 Tabbed results viewer (Transcript / Summary)

**Known Limitations:**

1. **Browser Timeout for Long Files** ⚠️
   - The GUI runs transcription synchronously
   - For audio files > 10-30 minutes, the browser will timeout after 2-5 minutes
   - The page freezes during processing and may crash
   - **Workaround**: Use GUI only for short test clips (< 10 minutes)
   - **Production**: Use CLI for real meeting recordings
   - **Tracked in**: [Issue #7](https://github.com/Technologikal/audio-transcription-diarisation/issues/7)

2. **Table-Based Agendas Not Supported** ⚠️
   - The agenda parser only reads paragraphs, not tables
   - Many real agendas use table layouts (columns for speakers, topics, etc.)
   - Results in 0 speakers found, no name mapping
   - **Tracked in**: [Issue #2](https://github.com/Technologikal/audio-transcription-diarisation/issues/2)

**When to Use GUI:**
- ✅ Testing with short audio clips (< 10 minutes)
- ✅ Experimenting with different models/settings
- ✅ Demos and quick trials
- ✅ Learning the interface

**When to Use CLI:**
- ✅ **Production transcription** of meeting recordings
- ✅ Long audio files (> 10 minutes)
- ✅ Batch processing
- ✅ Reliable, non-interactive processing

**Future Improvements:**

See GitHub issues for planned enhancements:
- [Issue #7](https://github.com/Technologikal/audio-transcription-diarisation/issues/7) - Background task processing
- [Issue #3](https://github.com/Technologikal/audio-transcription-diarisation/issues/3) - Full Gradio implementation
- [Issue #4](https://github.com/Technologikal/audio-transcription-diarisation/issues/4) - NiceGUI alternative
- [Issue #5](https://github.com/Technologikal/audio-transcription-diarisation/issues/5) - PyQt6 desktop app
- [Issue #6](https://github.com/Technologikal/audio-transcription-diarisation/issues/6) - Streamlit dashboard

**Lessons Learned:**

1. **Synchronous vs Async**: Long-running tasks need background processing to avoid browser timeouts
2. **File Handling**: Gradio returns both string paths and file objects - handle both cases
3. **Progress Updates**: Real-time progress requires proper async architecture
4. **User Expectations**: Clear warnings about limitations prevent frustration
5. **Use Case Matching**: Web GUIs work best for short, interactive tasks - CLI is better for long batch jobs

## Key Configuration

**Default Settings** (optimized for production use):
- Whisper model: **`medium`** (best balance of accuracy and speed) - use `--model` to change
- Language: **`english`** (forced language detection) - use `--language` to change
- PyAnnote model: `pyannote/speaker-diarisation-3.1` (hardcoded in source)
- Chunk duration: 600 seconds (10 minutes) - use `--chunk-duration` to change
- Audio processing: 16kHz mono WAV conversion (hardcoded in FFmpeg command)
- Logging level: INFO - use `--verbose` for DEBUG level

**Available Whisper Models** (smallest to largest):
`tiny`, `base`, `small`, `medium`, `large`, `large-v3-turbo`, `turbo`

**Model Selection Guidance**:
- `tiny`: Fastest, lowest accuracy - good for quick tests (requires ~4GB RAM, not recommended for production)
- `base`: Better than tiny, still fast (requires ~5GB RAM)
- `small`: Good balance of speed and accuracy - recommended for faster processing (requires ~6GB RAM)
- `medium`: **Default** - Best balance of accuracy and speed for production use (requires ~9GB RAM)
- `turbo`: Faster than large, good accuracy (requires ~10GB RAM)
- `large-v3-turbo`: Latest optimized large model (requires ~13GB RAM)
- `large`: Highest accuracy, significantly slower (requires ~15GB RAM)

**Choosing Based on Available Memory**:
- Less than 5GB available: Use `tiny` with `--auto-adjust`
- 5-6GB available: Use `base` or `small`
- 6-9GB available: Use `small` or `medium`
- 9-13GB available: Use `medium` (default) or `turbo`
- 13-15GB available: Use `large-v3-turbo`
- 15GB+ available: Use `large` for maximum accuracy

**Pro tip**: Always use `--auto-adjust` when requesting larger models to ensure safe execution

**Language Options**:
- `english` (default): Forces English transcription, avoids language confusion
- `welsh`, `french`, `spanish`, etc.: Force specific language
- `None`: Auto-detect language (slower, may misidentify languages)

**Performance Estimates** (on CPU, per 60-minute audio):
- `small` + english: ~3.5 hours processing time
- `medium` + english: ~3 hours processing time
- Forcing language vs auto-detect: ~2x faster

## Important Implementation Details

### Chunking Strategy

The `transcribe_with_diarisation()` function processes long audio files in configurable chunks (default 600s) to avoid memory issues. For each chunk:

1. FFmpeg extracts chunk to temporary WAV file (`temp_chunk_{start_time}.wav`)
2. PyAnnote performs diarisation on chunk
3. **Entire chunk is read into memory** (avoids file seeking issues)
4. Each speaker segment within chunk is individually transcribed
5. Timestamps are adjusted to reflect position in original file: `original_start = start_time + turn.start`
6. Temporary chunk file is deleted (guaranteed by try/finally block)

### GPU Acceleration

The pipeline automatically detects and uses CUDA if available:
```python
if torch.cuda.is_available():
    diarisation_pipeline.to(torch.device("cuda"))
```

GPU usage is logged at INFO level when detected.

### Logging and Progress Tracking

The script uses Python's `logging` module with two verbosity levels:

**INFO Level (default)**:
- **System resource status** (available RAM/VRAM and memory requirements)
- **Memory warnings** if insufficient resources detected
- Model loading status
- Audio duration and chunk count
- Current chunk progress (e.g., "Processing chunk 2/5")
- GPU acceleration status
- Language detection mode
- Total segments and processing time
- File save confirmation

**DEBUG Level (`--verbose` flag)**:
- Speaker segment counts per chunk
- Individual segment transcription progress
- Detailed step-by-step execution

**Example Output (INFO level)**:
```
2025-10-30 15:20:15 - INFO - System resources: 5.49 GB available RAM (7.44 GB total, 26.2% in use)
2025-10-30 15:20:15 - INFO - Required memory for medium model + PyAnnote: ~7.5 GB
2025-10-30 15:20:15 - INFO - Starting transcription of: meeting.m4a (forced language: english)
2025-10-30 15:20:15 - INFO - Loading Whisper model: medium
2025-10-30 15:20:17 - INFO - Loading PyAnnote speaker diarisation pipeline
2025-10-30 15:20:19 - INFO - Audio duration: 1234.56 seconds (20.58 minutes)
2025-10-30 15:20:19 - INFO - Processing audio in 3 chunk(s) of 600 seconds each
2025-10-30 15:20:19 - INFO - Processing chunk 1/3 (0s - 600s)
...
2025-10-30 15:35:47 - INFO - Transcription complete. Total segments: 87
2025-10-30 15:35:47 - INFO - Total processing time: 328.42 seconds (5.47 minutes)
2025-10-30 15:35:47 - INFO - Transcription saved to: meeting.txt
```

**Example Output with Auto-Adjust**:
```
2025-10-30 15:20:15 - INFO - System resources: 5.49 GB available RAM (7.44 GB total, 26.2% in use)
2025-10-30 15:20:15 - INFO - Required memory for large model + PyAnnote: ~12.5 GB
2025-10-30 15:20:15 - WARNING - WARNING: Low RAM! Available: 5.49 GB, Required: ~12.5 GB (with 20% buffer: 15.0 GB)
2025-10-30 15:20:15 - WARNING - Recommended model for your system: base
2025-10-30 15:20:15 - INFO - Auto-adjusting from large to base due to memory constraints
2025-10-30 15:20:15 - INFO - Starting transcription of: meeting.m4a (forced language: english)
2025-10-30 15:20:15 - INFO - Loading Whisper model: base
...
```

### Error Handling

The script includes comprehensive error handling:
- **Resource Checking**: Validates sufficient RAM/VRAM before loading models to prevent OOM crashes
- **Input Validation**: Checks if audio file exists and is a valid file
- **FFmpeg Errors**: Catches and reports ffmpeg/ffprobe failures with stderr output
- **Missing Dependencies**: Clear error messages if FFmpeg not installed
- **Temp File Cleanup**: `try/finally` blocks ensure temporary chunk files are deleted even on errors
- **Environment Variables**: Validates HF_TOKEN is set (from .env or environment) before processing
- **Graceful Degradation**: Provides helpful error messages with solutions

### Language Detection

Whisper supports 99 languages. By default, the script forces English detection which:
- Improves accuracy for English audio
- Speeds up processing (~2x faster than auto-detect)
- Prevents misidentification of accented English as other languages

Override with `--language` flag when needed:
```bash
# Welsh language meeting
python3 transcribe.py meeting.m4a --language welsh

# Mixed language - auto-detect
python3 transcribe.py meeting.m4a --language None
```

### Sample Data

The `transcription_project/audio/LibriSpeech/` directory contains the dev-clean dataset for testing with known multi-speaker samples.

**Note**: The original `dev-clean.tar.gz` archive (322 MB) is not tracked in Git due to GitHub's 100 MB file size limit. Only the extracted files are included in the repository.

## Git Workflow and Multi-Machine Development

This project is hosted on GitHub and follows standard Git workflows for version control and multi-machine development.

### Basic Git Workflow

```bash
# Check current status
git status

# Stage changes
git add .

# Commit with descriptive message
git commit -m "Brief description of changes"

# Push to GitHub
git push

# Pull latest changes (on another machine)
git pull
```

### Multi-Machine Development

**Setting up on a new machine:**

```bash
# 1. Clone repository
git clone https://github.com/Technologikal/audio-transcription-diarisation.git
cd audio-transcription-diarisation/transcription_project

# 2. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure authentication (not tracked in Git)
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 4. Ready to work!
python3 transcribe.py --help
```

**Syncing changes between machines:**

```bash
# Before starting work (pull latest changes)
git pull

# After making changes (push to GitHub)
git add .
git commit -m "Description of what changed"
git push

# On other machine (get updates)
git pull
```

### Git Configuration

**User Identity** (one-time setup):
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

**GitHub Authentication**:
- Uses GitHub CLI (`gh`) for authentication
- Run `gh auth login` once per machine
- Credentials stored securely by GitHub CLI

### Important Git Conventions

**Files NOT tracked in Git** (via `.gitignore`):
- `.env` (contains secrets)
- `venv/` (virtual environment)
- `*.m4a`, `*.wav`, `*.mp3`, `*.flac` (audio files)
- `*.tar.gz`, `*.zip` (large archives)
- `temp_chunk_*.wav` (temporary processing files)
- `*_transcription.txt` (output files)
- `*.doc`, `*.docx`, `*.pdf` (documents)

**Files tracked in Git**:
- Source code (`.py` files)
- Documentation (`.md` files)
- Configuration templates (`.env.example`)
- Requirements (`requirements.txt`)
- Sample data (LibriSpeech extracted text files only)

### Branch Strategy

**Current**: Single `main` branch (simple workflow for solo development)

**For collaboration** (future consideration):
- `main` - stable production code
- `feature/feature-name` - new features
- `bugfix/issue-description` - bug fixes

### Commit Message Guidelines

Use clear, descriptive commit messages:

```bash
# Good examples
git commit -m "Add batch processing support for multiple audio files"
git commit -m "Fix memory leak in chunk processing"
git commit -m "Update documentation for new --auto-adjust flag"

# Avoid vague messages
git commit -m "fix stuff"
git commit -m "updates"
```

## Current Limitations

1. **Single File Processing**: No batch processing support - must run script multiple times for multiple files
2. **Output Format**: Only plain text output. No support for structured formats (JSON, SRT, VTT) or speaker identification by name
3. **PyAnnote Model**: Diarization model is hardcoded to `pyannote/speaker-diarisation-3.1` in source code
4. **Audio Format Settings**: FFmpeg conversion parameters (16kHz, mono) are hardcoded and not configurable via CLI
5. **No Package Structure**: All code in single script file - not installable as a Python package
6. **Speaker Labels**: Speakers identified as generic labels (SPEAKER_00, SPEAKER_01) rather than by name

## Development Conventions

- **Version Control**: Git for version control, GitHub for hosting (private repository)
- **Virtual Environment**: All dependencies managed within `transcription_project/venv/`
- **Code Style**: Standard Python (PEP 8) - no configured linters/formatters
- **Spelling**: UK English throughout codebase (e.g., "diarisation" not "diarization")
- **Logging**: Uses Python `logging` module with configurable verbosity (INFO/DEBUG levels)
- **Error Handling**: Comprehensive try/except blocks with meaningful error messages
- **Argument Parsing**: Uses `argparse` for CLI interface with help documentation
- **Security**: HF_TOKEN stored in `.env` file (git-ignored), managed by `python-dotenv`
- **Modular Design**: Transcription and diarisation logic in separate functions (though all in single file)
- **Hugging Face Hub**: Pre-trained models downloaded and cached locally on first run
- **Memory Management**: Large files processed in chunks; entire chunks read into memory to avoid seeking issues
- **Resource Monitoring**: Pre-flight memory checks using `psutil` to prevent OOM crashes
- **Resource Cleanup**: `try/finally` blocks ensure temporary files are cleaned up
- **Multi-Machine Support**: Repository designed for seamless development across multiple machines

## Recent Improvements

The following improvements were made to the codebase (October 2025):

**Critical Fixes**:
- Fixed incomplete `requirements.txt` - added `torch`, `torchaudio`, `soundfile`, `numpy`, `python-dotenv`, `psutil`
- Fixed audio file seeking bug - changed to read entire chunk into memory for reliable slicing
- Maintained PyAnnote API compatibility with v4.0.1 - uses `.speaker_diarisation.itertracks()`
- Added automatic resource checking to prevent OOM (out-of-memory) crashes

**Infrastructure Enhancements**:
- Set up Git version control with GitHub hosting (private repository)
- Implemented multi-machine development workflow
- Standardised UK English spelling throughout codebase
- Configured comprehensive `.gitignore` for security and file management
- Removed large files (>100 MB) to comply with GitHub limits

**Security Enhancements**:
- Implemented secure `.env` file storage for HF_TOKEN using `python-dotenv`
- Created `.env.example` template for easy setup
- Added comprehensive `.gitignore` to prevent token exposure
- Token automatically loaded on script startup (no manual export needed)

**Feature Additions**:
- **Resource Monitoring**: Automatic RAM/VRAM checking before model loading with 20% safety buffer
- **Auto-Adjust Mode**: `--auto-adjust` flag automatically selects appropriate model for available memory
- **Force Mode**: `--force` flag allows bypassing memory warnings (use with caution)
- Added full CLI argument parsing with `argparse`
- Added `--language` / `-l` flag to force language detection (default: english)
- Changed default model from `tiny` to `medium` for better accuracy
- Changed default language from auto-detect to `english` for speed and accuracy
- Added input validation for audio files
- Added comprehensive error handling for subprocess calls
- Added output file option (`--output` / `-o`)
- Implemented proper logging with INFO and DEBUG levels
- Added progress tracking and timing information
- Added `--verbose` / `-v` flag for detailed execution logs
- Improved temporary file cleanup with try/finally blocks

**Performance Optimizations**:
- Forcing language detection speeds up processing by ~2x vs auto-detect
- Medium model provides better accuracy than small model with similar processing time
- Memory-efficient chunk processing prevents overflow on large files
- Pre-flight resource checks prevent system crashes and wasted processing time

## Production Usage Recommendations

For transcribing meeting recordings:

1. **Use the defaults** - optimized for English meetings (requires ~9GB RAM):
   ```bash
   python3 transcribe.py meeting.m4a -o meeting.txt
   ```

2. **For systems with limited memory** - let script auto-select model size:
   ```bash
   python3 transcribe.py meeting.m4a --auto-adjust -o meeting.txt
   ```

3. **For highest accuracy** - use large model with auto-adjust safety:
   ```bash
   python3 transcribe.py meeting.m4a --model large --auto-adjust -o meeting.txt
   ```

4. **For faster processing** (with slight accuracy trade-off):
   ```bash
   python3 transcribe.py meeting.m4a --model small -o meeting.txt
   ```

5. **For bilingual meetings**:
   ```bash
   python3 transcribe.py meeting.m4a --language None -o meeting.txt
   ```

6. **Monitor progress and resource usage** with verbose mode:
   ```bash
   python3 transcribe.py meeting.m4a -v -o meeting.txt
   ```

**Memory Considerations**:
- The script automatically checks available memory before loading models
- Use `--auto-adjust` when targeting larger models to prevent OOM crashes
- If you have less than 9GB available RAM/VRAM, consider using `small` or `base` model directly
- Memory requirements are logged at startup for transparency
