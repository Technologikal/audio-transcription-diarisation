"""Every Signal voice note lost its last ~5-8% (found 2026-10-06).

Signal stores voice notes as raw ADTS `.aac`. For that format ffprobe's
`format=duration` is an ESTIMATE from the bitrate — it says so on stderr — and
on variable-bitrate speech it runs short (a 231.5s note reported 214.1s). The
pipeline cut audio at that figure, so the sign-off of every voice note was
never transcribed. These tests build that kind of file synthetically and pin
the measured duration to the real one.

Run inside the transcription image (pipeline imports torch/pyannote):
    python -m pytest transcription_project/test_audio_duration.py -q
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import decoded_audio_duration, probe_audio_duration  # noqa: E402


def _make(tmp_path, name, *codec_args, seconds=60):
    """Speech-like bursts: loud for 3s, near-silent for 4s — which is what
    makes the encoder's bitrate vary, and the header estimate wrong."""
    out = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"anoisesrc=d={seconds}:c=pink:a=0.3,"
               "volume='if(lt(mod(t,7),3),1,0.02)':eval=frame",
         "-ac", "1", "-ar", "44100", *codec_args, str(out)],
        check=True,
    )
    return out


def _header_duration(path):
    return float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    ))


def test_the_fault_is_real_for_raw_aac(tmp_path):
    """Guards the premise: if ffprobe ever stops estimating, this says so."""
    f = _make(tmp_path, "note.aac", "-c:a", "aac", "-q:a", "0.6", "-f", "adts")
    assert _header_duration(f) < 59.0


def test_a_raw_aac_voice_note_is_measured_in_full(tmp_path):
    f = _make(tmp_path, "note.aac", "-c:a", "aac", "-q:a", "0.6", "-f", "adts")
    assert probe_audio_duration(str(f)) == pytest.approx(60.0, abs=0.1)


@pytest.mark.parametrize("name,args", [
    ("meeting.m4a", ("-c:a", "aac", "-b:a", "64k")),
    ("youtube.mp3", ("-c:a", "libmp3lame", "-q:a", "5")),
    ("plain.wav", ()),
])
def test_container_formats_keep_their_exact_header(tmp_path, name, args):
    f = _make(tmp_path, name, *args)
    assert probe_audio_duration(str(f)) == pytest.approx(60.0, abs=0.1)


def test_decoded_duration_matches_a_known_length(tmp_path):
    f = _make(tmp_path, "x.wav", seconds=12)
    assert decoded_audio_duration(str(f)) == pytest.approx(12.0, abs=0.05)


def test_a_file_that_cannot_be_read_raises(tmp_path):
    bad = tmp_path / "not-audio.aac"
    bad.write_bytes(b"this is not audio")
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        probe_audio_duration(str(bad))
