"""Unit tests for the Whisper subtitle-credit hallucination filter (#93).

Dependency-free (stdlib only) so it runs on the host without torch /
faster-whisper. The filter lives in its own module for exactly this reason —
`pipeline.py` imports torch at module load and cannot be imported here.

Run:
    cd transcription_project && python3 -m pytest test_hallucination_filter.py -v
"""

from __future__ import annotations

import pytest

from hallucination_filter import (
    is_subtitle_credit_hallucination,
    filter_hallucination_segments,
    _HALLUCINATION_PHRASES,
)


@pytest.mark.parametrize("phrase", _HALLUCINATION_PHRASES)
def test_every_blocklist_phrase_is_caught(phrase):
    # Guards against a phrase being added to the list but normalising to
    # something the matcher never actually catches (whole-segment policy means
    # any such drift is a silent miss, not a false drop).
    assert is_subtitle_credit_hallucination(phrase) is True


# --- phrases that MUST be dropped (unambiguous subtitle-credit hallucinations) ---

def test_amara_credit_dropped():
    assert is_subtitle_credit_hallucination("Subtitles by the Amara.org community") is True

def test_amara_credit_case_insensitive():
    assert is_subtitle_credit_hallucination("subtitles by the amara.org community") is True

def test_bare_amara_domain_segment_dropped():
    assert is_subtitle_credit_hallucination("Amara.org") is True

def test_thanks_for_watching_with_bang():
    assert is_subtitle_credit_hallucination("Thanks for watching!") is True

def test_thank_you_for_watching_with_period():
    assert is_subtitle_credit_hallucination("Thank you for watching.") is True

def test_subscribe_apostrophe_normalised():
    assert is_subtitle_credit_hallucination("Don't forget to subscribe") is True

def test_please_subscribe_channel():
    assert is_subtitle_credit_hallucination("Please subscribe to my channel") is True


# --- phrases that MUST be kept (genuine or ambiguous — conservative policy) ---

def test_real_speech_kept():
    assert is_subtitle_credit_hallucination(
        "So I fixed the switch and rebooted the firewall"
    ) is False

def test_ambiguous_thank_you_kept():
    # Isolated "Thank you." is a hallucination only sometimes — too risky to strip.
    assert is_subtitle_credit_hallucination("Thank you.") is False

def test_ambiguous_bye_kept():
    assert is_subtitle_credit_hallucination("Bye.") is False

def test_amara_mid_sentence_kept():
    # Whole-segment match only — a real sentence that mentions the domain stays.
    assert is_subtitle_credit_hallucination(
        "The client mentioned amara.org as a subtitle vendor"
    ) is False

def test_empty_kept():
    assert is_subtitle_credit_hallucination("") is False
    assert is_subtitle_credit_hallucination("   ") is False


# --- segment-list filter helper ---

def test_filter_drops_only_hallucinations_and_counts():
    segs = [
        "Right, so the UniFi controller needed a firmware bump.",
        "Subtitles by the Amara.org community",
        "Then I swapped the failed PSU.",
    ]
    kept, dropped = filter_hallucination_segments(segs)
    assert kept == [
        "Right, so the UniFi controller needed a firmware bump.",
        "Then I swapped the failed PSU.",
    ]
    assert dropped == 1

def test_filter_no_hallucinations_keeps_all():
    segs = ["one", "two", "three"]
    kept, dropped = filter_hallucination_segments(segs)
    assert kept == segs
    assert dropped == 0
