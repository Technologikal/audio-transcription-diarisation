"""
HTTP server frontend for the transcription pipeline (voice notes and meetings).

Thin FastAPI wrapper around the transcription pipeline. Designed for short
voice notes (typically < 2 minutes, single speaker). The bridge sends audio
via multipart upload and receives plain transcript text.

Endpoints:
    POST /transcribe                    — multipart file upload → {"transcript": "..."}
    GET  /transcribe/{job_id}/status    — progress of the job currently running
    POST /transcribe/{job_id}/cancel    — ask that job to stop
    GET  /health                        — returns 200 when ready

The transcription itself runs in a **worker child process**, not in this one.
A caller cannot otherwise stop a wedged run: Python cannot kill a thread, so
forced cancellation needs a killable unit. The service never exits to cancel
a job — a standalone deployment must not need an external supervisor to
survive a cancellation, and borrowing one from a particular deployment's
restart policy would make this component depend on how it happens to be run.

Progress and cancellation are generic capability: `job_id` is an opaque
string chosen by the caller, absent means "run without reporting", and
nothing here knows anything about who is calling or why.

Note: callers commonly impose their own request timeout; long recordings
should be submitted through a caller that tolerates minutes-to-hours runtimes.
With large-v3 on CPU, a 1-minute voice note takes ~2-3 minutes to process.
The bridge timeout will need increasing, or the bridge should poll/stream.
This becomes negligible with GPU (e.g. RTX 3060 upgrade).
"""

import logging
import multiprocessing as mp
import os
import queue as queue_mod
import shutil
import signal
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from read_secret import read_secret

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="voice-transcription", version="1.0.0")

# Lazy-loaded pipeline components (loaded on first request to keep startup fast)
_pipeline_loaded = False

# One transcription at a time.
#
# While `transcribe` was `async def` this was implicit: the synchronous
# pipeline occupied the event loop, so a second request could not even be
# dispatched until the first returned. Moving the handler to the threadpool
# frees the loop — and would, without this lock, let the threadpool run
# several CPU-bound transcriptions at once on a machine sized for one.
#
# A blocking acquire, not a 429: it preserves exactly what a caller saw
# before (a second request waits, then runs), while the event loop stays
# free to answer /health, status and cancel throughout.
_transcription_lock = threading.Lock()

# How long a cancelled worker is given to unwind cleanly before it is killed.
# Long enough for a responsive worker to reach its next stage boundary and
# unwind through its cleanup blocks; short enough that reclaiming a genuinely
# wedged slot is still prompt.
CANCEL_GRACE_SECONDS = float(os.environ.get("TRANSCRIPTION_CANCEL_GRACE_SECONDS", "90"))

# How often the parent drains markers and re-checks the worker.
_PUMP_INTERVAL_SECONDS = 0.5


class _JobState:
    """What the parent knows about the job its child is running.

    Single-worker, so one record is enough. Guarded by `lock` because it is
    written from the threadpool thread running the transcription and read
    from the event loop serving status and cancel.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.job_id = None
        self.running = False
        self.marker_seq = None
        self.phase = None
        self.chunk_index = None
        self.chunk_total = None
        self.audio_seconds_done = None
        self.audio_total_seconds = None
        self.cancel_requested = False
        self.cancel_requested_at = None
        self.cancel_event = None
        self.worker_pid = None

    def snapshot(self):
        with self.lock:
            return {
                "job_id": self.job_id,
                "running": self.running,
                "marker_seq": self.marker_seq,
                "phase": self.phase,
                "chunk_index": self.chunk_index,
                "chunk_total": self.chunk_total,
                "audio_seconds_done": self.audio_seconds_done,
                "audio_total_seconds": self.audio_total_seconds,
            }

    def apply_marker(self, marker):
        """Record a marker from the worker.

        Monotonic by construction — the worker only ever counts up — but
        enforced here anyway, because a marker that could go backwards would
        let a caller read a stalled job as freshly alive.
        """
        with self.lock:
            seq = marker.get("marker_seq")
            if not isinstance(seq, int):
                return
            if self.marker_seq is not None and seq <= self.marker_seq:
                return
            self.marker_seq = seq
            self.phase = marker.get("phase")
            self.chunk_index = marker.get("chunk_index")
            self.chunk_total = marker.get("chunk_total")
            if marker.get("audio_seconds_done") is not None:
                self.audio_seconds_done = marker["audio_seconds_done"]
            if marker.get("audio_total_seconds") is not None:
                self.audio_total_seconds = marker["audio_total_seconds"]


_state = _JobState()


def _worker(audio_path, skip_diarisation, whisper_model, marker_q, result_q, cancel_event, workdir):
    """Run one transcription. Executes in the child process.

    Everything heavy is imported here rather than at module scope, so the
    parent process never loads torch. That keeps the parent small and its
    event loop responsive, and it is why forking is safe.
    """
    try:
        # Own process group, so a forced kill can take the worker's ffmpeg
        # and model subprocesses with it instead of orphaning them.
        os.setsid()
    except OSError:
        pass

    # Temporary chunk files are written relative to the working directory,
    # so give the job its own. Cleanup then means removing one directory,
    # rather than globbing for `temp_chunk_*.wav` and hoping the pattern
    # still matches whatever the pipeline names them (FR-016).
    os.chdir(workdir)

    from pipeline import TranscriptionCancelled, transcribe_with_diarisation

    def on_marker(marker):
        try:
            marker_q.put_nowait(marker)
        except Exception:
            # A full or broken queue must not stop the transcription.
            pass

    def should_cancel():
        return cancel_event.is_set()

    try:
        result = transcribe_with_diarisation(
            audio_path=audio_path,
            hf_token=os.environ.get("HF_TOKEN", ""),
            whisper_model_name=whisper_model,
            language="english",
            chunk_duration_seconds=300,  # Shorter chunks for voice notes
            use_alignment=False,         # Skip wav2vec2 for speed
            backend="faster-whisper",
            skip_diarisation=skip_diarisation,
            on_marker=on_marker,
            should_cancel=should_cancel,
        )
        # Formatted in the child so only plain data crosses the process
        # boundary — SpeakerSegment objects would have to be picklable, and
        # making the return shape depend on that is a needless coupling.
        result_q.put((
            "ok",
            {
                "transcript": _format_segments_with_speakers(result.segments),
                "elapsed_time": result.elapsed_time,
            },
        ))
    except TranscriptionCancelled:
        result_q.put(("cancelled", None))
    except Exception as exc:
        result_q.put(("error", str(exc)))


def _kill_worker(proc, reason):
    """Terminate a worker that will not stop on its own.

    SIGTERM to the group first, then SIGKILL if it is still there. The group
    is what matters: a bare kill of the worker leaves its ffmpeg children
    running, holding CPU and the temporary files we are about to remove.

    `reason` is required rather than assumed. An earlier version hard-coded
    "cancellation grace period expired" into this function, which then
    announced a grace-period expiry on every ordinary completion — because
    the cleanup path calls it too. Operator-facing output that reports a
    timeout on healthy work sends someone hunting a fault that is not there.
    """
    if proc.pid is None or not proc.is_alive():
        return
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None

    logger.warning("Terminating worker %s: %s", proc.pid, reason)
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except OSError:
        pass

    proc.join(timeout=10)
    if proc.is_alive():
        logger.warning("Worker %s ignored SIGTERM; killing", proc.pid)
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass
        proc.join(timeout=10)


def _pump(proc, marker_q, result_q):
    """Relay markers from the worker until it produces a result.

    Returns the worker's (outcome, payload).
    """
    outcome = None
    while outcome is None:
        # Drain every marker waiting, so a burst does not leave the recorded
        # position lagging behind the work.
        while True:
            try:
                _state.apply_marker(marker_q.get_nowait())
            except queue_mod.Empty:
                break
            except Exception:
                break

        try:
            outcome = result_q.get(timeout=_PUMP_INTERVAL_SECONDS)
            break
        except queue_mod.Empty:
            pass

        with _state.lock:
            cancelled_at = _state.cancel_requested_at
        if cancelled_at is not None and (time.monotonic() - cancelled_at) > CANCEL_GRACE_SECONDS:
            _kill_worker(
                proc,
                f"cancellation grace period of {CANCEL_GRACE_SECONDS:.0f}s expired "
                "without the worker reaching a stage boundary",
            )
            # Fall through: the worker may still have queued a result before
            # dying, and the death check below reports it if not.

        if not proc.is_alive():
            # The child is gone. It may have put a result on the queue an
            # instant before exiting, so look once more before concluding it
            # died — otherwise a clean run that finished between polls would
            # be reported as a crash.
            try:
                outcome = result_q.get(timeout=2)
            except queue_mod.Empty:
                exitcode = proc.exitcode
                if _state.cancel_requested:
                    outcome = ("cancelled", None)
                else:
                    outcome = ("error", f"transcription worker exited unexpectedly (code {exitcode})")
            break

    return outcome


def _ensure_pipeline():
    """Load HF_TOKEN into environment so the pipeline can access PyAnnote models."""
    global _pipeline_loaded
    if _pipeline_loaded:
        return

    # Read HF_TOKEN from Docker secret or env var
    try:
        # Secret name is configuration: deployments that mount it under a
        # different name set HF_TOKEN_SECRET. Defaults to the plain name so
        # the service works standalone with no deployment-specific setup.
        hf_token = read_secret(
            os.environ.get("HF_TOKEN_SECRET", "hf_token"), fallback_env="HF_TOKEN"
        )
        os.environ["HF_TOKEN"] = hf_token
    except RuntimeError:
        logger.warning("HF_TOKEN not found — diarisation will fail if models aren't cached")

    _pipeline_loaded = True


def _format_segments_with_speakers(segments) -> str:
    """Collapse a SpeakerSegment list into a diarised plain-text block.

    Returns one line per speaker turn. Consecutive segments from the
    same speaker are merged so the operator sees:

        [Alice] Hi, are you alright?
        [Bob] Yeah, good thanks. Good morning.
        [Alice] It's gorgeous. Nice to be working.

    Falls back to a plain-text concatenation whenever the whole recording
    resolves to a single speaker — skip_diarisation=True (every segment
    "SPEAKER_00"), pyannote not run ("UNKNOWN"), or a real diarisation run
    that found one voice. Labelling a monologue adds nothing.
    """
    cleaned = [s for s in segments if (s.text or "").strip()]
    if not cleaned:
        return ""

    def _label(seg):
        name = getattr(seg, "real_name", None)
        if name:
            return name
        return getattr(seg, "speaker_label", None) or "UNKNOWN"

    all_labels = {_label(s) for s in cleaned}
    # Single-speaker output carries no useful label, whatever that speaker is
    # called. Test the COUNT, not a magic name: skip_diarisation labels every
    # segment "SPEAKER_00", other paths use "UNKNOWN", and a genuine diarisation
    # run that finds one speaker should also come back as plain text.
    if len(all_labels) <= 1:
        return " ".join(s.text.strip() for s in cleaned)

    lines: list[str] = []
    current_label: str | None = None
    current_parts: list[str] = []

    def _flush():
        if current_label is not None and current_parts:
            lines.append(f"[{current_label}] {' '.join(current_parts).strip()}")

    for seg in cleaned:
        label = _label(seg)
        text = seg.text.strip()
        if label == current_label:
            current_parts.append(text)
        else:
            _flush()
            current_label = label
            current_parts = [text]
    _flush()

    return "\n".join(lines)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    skip_diarisation: bool = Form(False),
    job_id: str | None = Form(None),
):
    """Transcribe an uploaded audio file and return plain transcript text.

    Declared `def`, not `async def`, deliberately. The pipeline below is
    synchronous and CPU-bound for minutes to hours. On an `async def`
    handler it runs *on* the single uvicorn event loop, so the server can
    answer nothing at all — not `/health`, not a status query, not a
    cancellation — for the whole duration of a transcription. As a plain
    `def`, FastAPI offloads it to its threadpool and the loop stays free.

    That is why `/health` used to flip to unhealthy about 90 seconds into
    every *successful* job: the healthcheck was measuring event-loop
    availability, not service health, and could never have indicated a
    genuine hang.

    Args:
        file: Audio file (multipart upload, field name "file")
        skip_diarisation: If true, skip speaker diarisation (faster, no HF_TOKEN needed)
        job_id: Optional opaque identifier chosen by the caller. When given,
            the job's progress becomes readable at
            `GET /transcribe/{job_id}/status` and it can be stopped at
            `POST /transcribe/{job_id}/cancel`. No format is imposed and none
            should be — it means nothing here. Omit it and the run reports
            nothing, which is exactly what callers got before this existed.

    Returns:
        {"transcript": "transcribed text here"}
    """
    _ensure_pipeline()

    hf_token = os.environ.get("HF_TOKEN", "")

    # HF_TOKEN is only required when diarisation is enabled
    if not skip_diarisation and not hf_token:
        return JSONResponse(
            status_code=500,
            content={"error": "HF_TOKEN not configured and diarisation is enabled"},
        )

    logger.info(
        "Transcribing %s (skip_diarisation=%s)",
        file.filename, skip_diarisation,
    )

    # Write uploaded file to a temporary location
    suffix = Path(file.filename or "audio.ogg").suffix or ".ogg"
    # `file.file` is the underlying spooled file object — the synchronous
    # read that pairs with a synchronous handler. `await file.read()` is
    # unavailable here and would need the handler back on the event loop.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    workdir = None
    try:
        with _transcription_lock:
            # Each job gets its own working directory, so the temporary
            # chunk files the pipeline writes relative to CWD are scoped to
            # it and removed with it — including when the worker is killed
            # and never reaches its own cleanup (FR-016).
            workdir = tempfile.mkdtemp(prefix="transcribe-job-")

            # `fork`, chosen explicitly rather than left to the platform
            # default. It is safe here precisely because the parent never
            # imports torch: every heavy import lives inside the worker.
            ctx = mp.get_context("fork")
            marker_q = ctx.Queue()
            result_q = ctx.Queue()
            cancel_event = ctx.Event()

            proc = ctx.Process(
                target=_worker,
                args=(
                    tmp_path,
                    skip_diarisation,
                    os.environ.get("WHISPER_MODEL", "large-v3"),
                    marker_q,
                    result_q,
                    cancel_event,
                    workdir,
                ),
                daemon=True,
            )

            with _state.lock:
                _state.reset()
                _state.job_id = job_id or None
                _state.running = True
                _state.cancel_event = cancel_event

            proc.start()
            with _state.lock:
                _state.worker_pid = proc.pid
            logger.info(
                "Worker %s started for %s (job_id=%s, skip_diarisation=%s)",
                proc.pid, file.filename, job_id or "-", skip_diarisation,
            )

            try:
                outcome, payload = _pump(proc, marker_q, result_q)
            finally:
                # A worker that has just delivered its result is usually
                # still alive for a moment while its queue feeder thread
                # flushes, so give it a chance to exit on its own before
                # reaching for signals. Killing it here worked — the result
                # was already in hand — but it terminated healthy processes
                # as a matter of routine, which is not something to do
                # casually just because it happens to be harmless.
                proc.join(timeout=5)
                if proc.is_alive():
                    _kill_worker(proc, "still running after the request completed")
                with _state.lock:
                    _state.reset()

        if outcome == "cancelled":
            logger.info("Transcription of %s was cancelled", file.filename)
            return JSONResponse(
                status_code=409,
                content={"error": "transcription cancelled", "cancelled": True},
            )
        if outcome == "error":
            logger.error("Transcription failed for %s: %s", file.filename, payload)
            return JSONResponse(status_code=500, content={"error": payload})

        # The transcript arrives already formatted by the worker. Speaker
        # labels are preserved rather than flattened — the old behaviour
        # dropped seg.speaker_label and produced one unlabelled block
        # (fix-49 field test 2026-04-11, Bug 7 triage).
        transcript = payload["transcript"]
        logger.info(
            "Transcribed %s: %d chars, %.1fs processing time",
            file.filename,
            len(transcript),
            payload["elapsed_time"],
        )
        return {"transcript": transcript}

    except Exception as exc:
        logger.error("Transcription failed for %s: %s", file.filename, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


@app.get("/transcribe/{job_id}/status")
async def transcription_status(job_id: str):
    """Report how far the job currently running has got.

    Single-worker, so this answers only about the job in flight. A `404`
    means "not running here" — which the caller may read as finished, never
    started, or lost. On its own it is not evidence of failure, and this
    endpoint deliberately does not guess which it is.

    Carries position and stage only. No transcript content: status is not
    an egress path for recorded material.
    """
    snap = _state.snapshot()
    if not snap["running"] or snap["job_id"] is None or snap["job_id"] != job_id:
        return JSONResponse(status_code=404, content={"error": "not running here"})

    body = {"job_id": job_id, "state": "running"}
    # Fields are omitted rather than sent null before the first stage
    # boundary: "no marker yet" and "marker of unknown value" are different
    # claims, and a caller distinguishing them is how it avoids judging a
    # job that has never reported.
    for key in (
        "marker_seq", "phase", "chunk_index", "chunk_total",
        "audio_seconds_done", "audio_total_seconds",
    ):
        if snap[key] is not None:
            body[key] = snap[key]
    return body


@app.post("/transcribe/{job_id}/cancel")
async def cancel_transcription(job_id: str):
    """Ask the running job to stop.

    Returns `202` immediately and never blocks on the worker: the caller is
    entitled to a prompt answer, and how long the worker takes to notice is
    not its problem. The worker stops at its next stage boundary; if it is
    wedged and never reaches one, it is killed when the grace period
    expires. Either way this process stays up and serves the next request
    with a fresh child.
    """
    with _state.lock:
        if not _state.running or _state.job_id is None or _state.job_id != job_id:
            return JSONResponse(status_code=404, content={"error": "not running here"})
        already = _state.cancel_requested
        _state.cancel_requested = True
        if _state.cancel_requested_at is None:
            _state.cancel_requested_at = time.monotonic()
        if _state.cancel_event is not None:
            _state.cancel_event.set()
        pid = _state.worker_pid

    logger.info(
        "Cancellation requested for job %s (worker %s)%s",
        job_id, pid, " [repeat]" if already else "",
    )
    return JSONResponse(
        status_code=202,
        content={"status": "cancelling", "job_id": job_id},
    )
