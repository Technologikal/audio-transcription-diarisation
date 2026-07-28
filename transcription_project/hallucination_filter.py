"""Whisper subtitle-credit hallucination filter (#93).

faster-whisper emits fabricated "subtitle credit" phrases (learned from the
YouTube/TED subtitle corpora it was trained on) as standalone segments during
silence or low-confidence audio — e.g. "Subtitles by the Amara.org community",
"Thanks for watching". Combined with ``condition_on_previous_text=True`` these
can seed multi-sentence drift into plausible fabricated content (see the
2026-06-02 Alpha Recruitment incident). `pipeline.py` sets
``condition_on_previous_text=False`` AND drops the segments this module flags.

Deliberately kept dependency-free (stdlib only) so it is unit-testable on the
host — `pipeline.py` imports torch at module load and cannot be imported in a
plain test environment.

**Conservative by design:** a segment is dropped only when its *whole*
normalised text equals a known credit phrase. Mid-sentence occurrences and
ambiguous trailing phrases ("Thank you.", "Bye.") are never dropped, so genuine
speech is preserved. Multi-sentence drift beyond the credit phrase is handled by
``condition_on_previous_text=False`` plus operator flagging at report-generation.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)

# Raw phrases; normalised once at import via _normalise so matching is consistent
# with how incoming segment text is normalised.
_HALLUCINATION_PHRASES = [
    "Subtitles by the Amara.org community",
    "Subtitles by Amara.org",
    "Amara.org community",
    "Amara.org",
    "Thanks for watching",
    "Thank you for watching",
    "Don't forget to subscribe",
    "Please subscribe to my channel",
    "Please subscribe",
]


def _normalise(text: str) -> str:
    """Lowercase; drop punctuation except internal dots (to preserve
    'amara.org'); collapse whitespace; strip leading/trailing dots."""
    t = text.strip().lower()
    t = re.sub(r"[^a-z0-9.\s]", "", t)      # drop punctuation/apostrophes ("don't" -> "dont")
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip(".")


_BLOCKLIST = {_normalise(p) for p in _HALLUCINATION_PHRASES}


def is_subtitle_credit_hallucination(text: str) -> bool:
    """True if ``text`` is (as a whole segment) a known subtitle-credit
    hallucination safe to drop. Whole-segment match only — never a substring."""
    norm = _normalise(text)
    return bool(norm) and norm in _BLOCKLIST


def filter_hallucination_segments(segment_texts: Iterable[str]) -> tuple[list[str], int]:
    """Return ``(kept_texts, dropped_count)``, logging each dropped segment.
    Used by pipeline.py to strip hallucination segments before assembling a
    transcript."""
    kept: list[str] = []
    dropped = 0
    for t in segment_texts:
        if is_subtitle_credit_hallucination(t):
            logger.warning("Dropped Whisper subtitle-credit hallucination (#93): %r", t.strip())
            dropped += 1
        else:
            kept.append(t)
    return kept, dropped
