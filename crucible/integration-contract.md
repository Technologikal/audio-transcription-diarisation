# Integration Contract: audio-transcription-diarisation

**Version**: 1.1.0
**Last updated**: 2026-08-09

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
- **Optional field**: `job_id` (string) — *added 1.1.0.* An opaque identifier
  chosen by the caller. Supply it and the run becomes observable and stoppable
  through the two endpoints below. Omit it and the run reports nothing, which
  is exactly the pre-1.1.0 behaviour. No format is imposed and none should be:
  the identifier means nothing to this service.
- **Success response** (200):
  ```json
  {"transcript": "transcribed text here"}
  ```
- **Cancelled response** (409) — *added 1.1.0*:
  ```json
  {"error": "transcription cancelled", "cancelled": true}
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
  - **One transcription at a time.** A second request blocks until the first
    finishes, as it always has — before 1.1.0 that was an accident of the
    handler occupying the event loop; it is now explicit.
  - **Runs in a worker child process** (1.1.0). The service itself never exits
    to stop a job, so it needs no external supervisor to survive a
    cancellation.

#### `GET /transcribe/{job_id}/status`

*Added 1.1.0.* Reports how far the job currently running has got.

- **Success response** (200) — fields after `state` are **omitted until the
  first stage boundary is reached**, because "has not reported yet" and
  "reported an unknown value" are different claims:
  ```json
  {
    "job_id": "…", "state": "running",
    "marker_seq": 14, "phase": "diarise",
    "chunk_index": 5, "chunk_total": 18,
    "audio_seconds_done": 1500.0, "audio_total_seconds": 5400.0
  }
  ```
- **Not-this-job response** (404): `{"error": "not running here"}`. The service
  is single-worker and answers only about the job in flight. A 404 means the
  job is finished, never started, or lost — **on its own it is not evidence of
  failure**, and this endpoint does not guess which.
- `marker_seq` is monotonic. Its value carries no meaning; only that it
  *changes* does. A caller watching for a stall reads "has it moved", never
  "how far has it got".
- `phase` is one of `extract`, `transcribe`, `diarise`, `assemble`.
- `audio_seconds_done` is **absent during global diarisation**, which is one
  pass over the whole file with no position within the recording. Reporting
  zero there would read as 0% progress on a job that is ~61% through its work.
- **No transcript content.** Position and stage only — status is not an egress
  path for recorded material.
- No client state and no schedule: the caller polls at whatever cadence suits
  it.

#### `POST /transcribe/{job_id}/cancel`

*Added 1.1.0.* Asks the running job to stop.

- **Success response** (202): `{"status": "cancelling", "job_id": "…"}` —
  returned **immediately**, never blocking on the worker.
- **Not-this-job response** (404): `{"error": "not running here"}`. A caller
  whose goal is "that work is not running" may treat this as success.
- Idempotent: repeat requests are accepted and logged.
- **Behaviour**: the worker stops at its next stage boundary and unwinds
  cleanly, removing its temporary files. If it is wedged and never reaches a
  boundary, its process group is signalled and then killed once
  `TRANSCRIPTION_CANCEL_GRACE_SECONDS` (default 90) has elapsed. The service
  stays up throughout and serves the next request with a fresh child.

#### `GET /health`

- **Success response** (200):
  ```json
  {"status": "ok"}
  ```
- *Since 1.1.0 this answers **during** a transcription.* Before, the pipeline
  occupied the single event loop, so the endpoint could not respond while the
  service was working — it reported unhealthy roughly 90 seconds into every
  successful job and could never have indicated a genuine hang.

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
| `hf_token` | `/run/secrets/$HF_TOKEN_SECRET` | HuggingFace API token for PyAnnote model access |

Fallback: `HF_TOKEN` environment variable (for local development only).

**The secret's name is configuration**, not a fixed string. The service reads
`/run/secrets/$HF_TOKEN_SECRET`, defaulting to the plain name `hf_token` so it
runs standalone with no deployment-specific setup. A deployment that mounts it
under a namespaced name **must** set `HF_TOKEN_SECRET` to match.

> Getting this wrong fails in a way that is easy to miss: the service starts
> healthy, and only *diarised* requests fail (`HTTP 500 — HF_TOKEN not
> configured`). Requests passing `skip_diarisation=true` need no token and keep
> working, so a smoke test using one will report the service as fine. Crucible
> shipped exactly this fault on 2026-08-09 and it survived a merge review.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVER_MODE` | — | `http` or `mcp` |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model name |
| `HF_TOKEN_SECRET` | `hf_token` | Name of the mounted HF token secret |
| `TRANSCRIPTION_CANCEL_GRACE_SECONDS` | `90` | *Added 1.1.0.* How long a cancelled worker is given to unwind before its process group is killed |

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

Processing time (CPU, large-v3 model): **approximately 1.05x real-time audio
duration** — measured 2026-08-09 on a 20-minute multi-speaker slice (1263s
wallclock for 1200s of audio) and corroborated by a 3-minute slice (172.6s for
180s). The earlier "2-3x" figure in this document was an estimate, never a
measurement.

The split matters more than the total, because it is wildly uneven:

| Phase | Share of runtime |
|-------|------------------|
| Global diarisation — **one pass, before the chunk loop** | **~61%** |
| Chunked transcription | ~37% |

A three-hour recording therefore spends roughly **two hours inside a single
diarisation call**. Any caller timing out, or watching for progress, has to
accommodate that phase specifically.

With GPU: substantially faster; not measured here.

---

## Docker Image

- **Registry**: `registry.local:5000/crucible/voice-transcription`
- **Tag format**: semver (e.g., `1.0.0`)
- **Base**: `python:3.12-slim`
- **Baked models**: faster-whisper large-v3, PyAnnote speaker-diarisation-3.1
- **Exposed port**: 8001 (HTTP mode)
- **Entry point**: `python3 server.py`
- **Mode selection**: `SERVER_MODE` env var (`http` or `mcp`)
