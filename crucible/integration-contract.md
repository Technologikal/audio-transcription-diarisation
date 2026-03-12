# Integration Contract: audio-transcription-diarisation

**Version**: 1.0.0
**Last updated**: 2026-03-12

This document declares the stable interfaces that Crucible depends on.
Changes to these interfaces follow the breaking change policy below.

---

## Stable Interfaces

### HTTP (Zone 3 — pre-processing, voice notes)

Entry point: `SERVER_MODE=http` → `server.py` → `http_server.py`

#### `POST /transcribe`

Accepts a multipart file upload and returns a plain transcript.

- **Content-Type**: `multipart/form-data`
- **Field name**: `file` (audio bytes — m4a, ogg, wav, mp3, flac, etc.)
- **Success response** (200):
  ```json
  {"transcript": "transcribed text here"}
  ```
- **Error response** (500):
  ```json
  {"error": "description of what went wrong"}
  ```
- **Behaviour**:
  - Uses faster-whisper large-v3 model (override via `WHISPER_MODEL` env var)
  - Language forced to English (single-speaker voice note assumption)
  - Diarisation runs but output is concatenated into a single string
  - wav2vec2 alignment skipped for speed

#### `GET /health`

- **Success response** (200):
  ```json
  {"status": "ok"}
  ```

### MCP Tools (Zone 5a — on-demand transcription)

Entry point: `SERVER_MODE=mcp` → `server.py` → `mcp_server.py`
Transport: STDIO (FastMCP)

#### `transcribe_audio`

Transcribe an audio file with speaker diarisation.

- **Parameters**:
  | Name | Type | Required | Default | Description |
  |------|------|----------|---------|-------------|
  | `audio_path` | string | yes | — | Path to audio file |
  | `model` | string | no | `"medium"` | Whisper model size |
  | `language` | string | no | `"english"` | Language or `"auto"` |
  | `compute_type` | string | no | `"auto"` | Model precision (float16, int8, etc.) |
  | `chunk_duration` | int | no | `600` | Audio chunk size in seconds |
  | `backend` | string | no | `"faster-whisper"` | Backend (`faster-whisper` or `whisperx`) |
  | `use_alignment` | bool | no | `true` | Refine timestamps with wav2vec2 |
  | `beam_size` | int | no | `null` | Beam search width |
  | `normalise` | bool | no | `false` | Apply EBU R128 loudness normalisation |
  | `denoise` | bool | no | `false` | Apply FFT-based noise reduction |

- **Returns**: JSON string
  ```json
  {
    "status": "success",
    "audio_file": "/path/to/audio.m4a",
    "model": "large-v3",
    "backend": "faster-whisper",
    "total_duration_seconds": 1234.56,
    "processing_time_seconds": 328.42,
    "segment_count": 87,
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "start": 0.5,
        "end": 3.25,
        "text": "Hello everyone, welcome to the meeting."
      }
    ]
  }
  ```
- **Error return**:
  ```json
  {"status": "error", "error": "description"}
  ```

#### `map_speakers_from_agenda`

Map anonymous speaker labels to real names using a DOCX agenda.

- **Parameters**:
  | Name | Type | Required | Default | Description |
  |------|------|----------|---------|-------------|
  | `transcript_data` | string | yes | — | JSON from `transcribe_audio` or path to transcript file |
  | `agenda_path` | string | yes | — | Path to DOCX agenda file |
  | `output_format` | string | no | `"both"` | `"transcript"`, `"summary"`, or `"both"` |

- **Returns**: JSON string with `speaker_mappings`, optional `transcript` and `summary` fields

#### `check_system_resources`

Check available RAM/VRAM and whether a given model will fit.

- **Parameters**:
  | Name | Type | Required | Default | Description |
  |------|------|----------|---------|-------------|
  | `model` | string | no | `"medium"` | Whisper model to check requirements for |

- **Returns**: JSON string with system info, model requirements, and recommendation

---

## Breaking Change Policy

- **Removing or renaming** an interface listed above = **major** version bump
- **Adding** new interfaces or optional parameters = **minor** version bump
- **Internal changes** (performance, refactoring, bug fixes) = **patch** version bump

Crucible's contract tests are written against the interfaces above. A major version
bump requires updating Crucible's `tools/voice-transcription/manifest.json` and
`tools/voice-transcription/tests/test_contract.py`.

---

## Secrets Required

| Secret | Mount path | Purpose |
|--------|-----------|---------|
| `hf_token` | `/run/secrets/hf_token` | HuggingFace API token for PyAnnote model access |

Fallback: `HF_TOKEN` environment variable (for local development only).

At **build time**, `HF_TOKEN` is required as a `--build-arg` to bake PyAnnote models
into the Docker image. At **runtime**, it is only needed if models aren't cached in
the image (which shouldn't happen with a properly built image).

---

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 6 GB | 8+ GB |
| Disk | ~4 GB (baked models) | ~4 GB |
| CPU | Any x86_64 | Multi-core for faster chunked processing |
| GPU | Not required | CUDA-capable GPU significantly speeds up processing |

Processing time (CPU, large-v3 model): approximately 2-3x real-time audio duration.
With GPU: near real-time.

---

## Docker Image

- **Registry**: `registry.local:5000/crucible/voice-transcription`
- **Tag format**: semver (e.g., `1.0.0`)
- **Base**: `python:3.12-slim`
- **Baked models**: faster-whisper large-v3, PyAnnote speaker-diarisation-3.1
- **Exposed port**: 8001 (HTTP mode)
- **Entry point**: `python3 server.py`
- **Mode selection**: `SERVER_MODE` env var (`http` or `mcp`)
