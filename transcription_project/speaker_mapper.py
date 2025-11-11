"""
Speaker Mapper Module

Maps anonymous speaker labels from diarization (SPEAKER_00, SPEAKER_01, etc.)
to real names from the agenda using a hybrid approach that combines:
- Temporal alignment with agenda sections
- Speaker frequency analysis
- Confidence scoring
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict, Counter

from agenda_parser import ParsedAgenda, AgendaSection

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Individual speech segment from transcription"""
    speaker_label: str  # e.g., "SPEAKER_00"
    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    real_name: Optional[str] = None  # Mapped name
    confidence: float = 0.0  # Mapping confidence (0-1)
    agenda_section: Optional[str] = None  # Which agenda section this belongs to


@dataclass
class SpeakerMapping:
    """Mapping from diarization label to real name"""
    speaker_label: str  # e.g., "SPEAKER_00"
    real_name: str  # e.g., "John Smith"
    confidence: float  # 0-1 confidence score
    evidence: List[str] = field(default_factory=list)  # Reasons for mapping


class SpeakerMapper:
    """
    Maps speaker labels to real names using agenda information and
    temporal analysis of the transcription.
    """

    def __init__(self, parsed_agenda: ParsedAgenda):
        """
        Initialize mapper with parsed agenda.

        Args:
            parsed_agenda: ParsedAgenda object with sections and speakers
        """
        self.agenda = parsed_agenda
        self.mappings: Dict[str, SpeakerMapping] = {}

    def map_speakers(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Map speaker labels to real names for all segments.

        This uses a hybrid approach:
        1. Analyze temporal distribution of speakers
        2. Match speakers to agenda sections based on timing
        3. Use speaker frequency to infer primary vs secondary speakers
        4. Assign confidence scores

        Args:
            segments: List of SpeakerSegment objects with speaker_label set

        Returns:
            List of SpeakerSegment objects with real_name and confidence set
        """
        logger.info(f"Mapping speakers for {len(segments)} segments")

        # Step 1: Assign agenda sections to segments based on timing
        self._assign_sections_to_segments(segments)

        # Step 2: Analyze speaker patterns
        speaker_stats = self._analyze_speaker_patterns(segments)

        # Step 3: Create mappings based on agenda and patterns
        self._create_speaker_mappings(segments, speaker_stats)

        # Step 4: Apply mappings to segments
        mapped_segments = self._apply_mappings(segments)

        logger.info(f"Created {len(self.mappings)} speaker mappings")
        for mapping in self.mappings.values():
            logger.info(f"  {mapping.speaker_label} -> {mapping.real_name} "
                       f"(confidence: {mapping.confidence:.2f})")

        return mapped_segments

    def _assign_sections_to_segments(self, segments: List[SpeakerSegment]):
        """
        Assign agenda sections to segments based on timing.

        This is a simple approach: divide audio duration equally among sections.
        In production, you might use more sophisticated methods (e.g., topic modeling).

        Args:
            segments: List of speaker segments (modified in place)
        """
        if not segments or not self.agenda.sections:
            return

        # Calculate total duration
        total_duration = max(seg.end_time for seg in segments)

        # Estimate duration per section (equal division for now)
        section_duration = total_duration / len(self.agenda.sections)

        for segment in segments:
            # Determine which section this segment falls into
            section_idx = int(segment.start_time // section_duration)
            section_idx = min(section_idx, len(self.agenda.sections) - 1)

            section = self.agenda.sections[section_idx]
            segment.agenda_section = section.title
            logger.debug(f"Segment at {segment.start_time:.1f}s assigned to section: {section.title}")

    def _analyze_speaker_patterns(self, segments: List[SpeakerSegment]) -> Dict[str, Dict]:
        """
        Analyze speaking patterns for each speaker label.

        Returns:
            Dict mapping speaker_label to statistics (total_time, segment_count, sections_active)
        """
        stats = defaultdict(lambda: {
            'total_time': 0.0,
            'segment_count': 0,
            'sections_active': Counter(),
            'segments': []
        })

        for segment in segments:
            speaker = segment.speaker_label
            stats[speaker]['total_time'] += (segment.end_time - segment.start_time)
            stats[speaker]['segment_count'] += 1
            stats[speaker]['segments'].append(segment)
            if segment.agenda_section:
                stats[speaker]['sections_active'][segment.agenda_section] += 1

        return stats

    def _create_speaker_mappings(self, segments: List[SpeakerSegment],
                                  speaker_stats: Dict[str, Dict]):
        """
        Create mappings from speaker labels to real names.

        Uses a hybrid approach:
        - If a section has only one speaker in agenda, map dominant speaker to that name
        - If a section has multiple speakers, use frequency to infer primary/secondary
        - Calculate confidence based on evidence strength

        Args:
            segments: List of speaker segments
            speaker_stats: Statistics for each speaker label
        """
        # Track which names have been assigned
        assigned_names = set()

        # Sort speakers by total speaking time (descending)
        sorted_speakers = sorted(speaker_stats.items(),
                                key=lambda x: x[1]['total_time'],
                                reverse=True)

        for speaker_label, stats in sorted_speakers:
            # Find the section where this speaker is most active
            if stats['sections_active']:
                most_active_section = stats['sections_active'].most_common(1)[0][0]

                # Find the agenda section object
                agenda_section = None
                for section in self.agenda.sections:
                    if section.title == most_active_section:
                        agenda_section = section
                        break

                if agenda_section and agenda_section.speakers:
                    # Get unassigned speakers from this section
                    available_speakers = [s for s in agenda_section.speakers
                                         if s not in assigned_names]

                    if available_speakers:
                        # Assign the first available speaker
                        real_name = available_speakers[0]
                        assigned_names.add(real_name)

                        # Calculate confidence based on evidence
                        evidence = []
                        confidence = 0.0

                        # Evidence: speaker is dominant in their primary section
                        total_segments_in_section = sum(
                            1 for seg in segments if seg.agenda_section == most_active_section
                        )
                        speaker_segments_in_section = stats['sections_active'][most_active_section]
                        dominance_ratio = speaker_segments_in_section / total_segments_in_section

                        if dominance_ratio > 0.6:
                            evidence.append(f"Dominant in section '{most_active_section}' ({dominance_ratio:.0%})")
                            confidence += 0.5

                        # Evidence: only one speaker listed for section
                        if len(agenda_section.speakers) == 1:
                            evidence.append(f"Only speaker listed for section '{most_active_section}'")
                            confidence += 0.3

                        # Evidence: significant speaking time
                        total_time = sum(s['total_time'] for s in speaker_stats.values())
                        time_ratio = stats['total_time'] / total_time

                        if time_ratio > 0.2:
                            evidence.append(f"Significant speaking time ({time_ratio:.0%})")
                            confidence += 0.2

                        confidence = min(confidence, 1.0)  # Cap at 1.0

                        self.mappings[speaker_label] = SpeakerMapping(
                            speaker_label=speaker_label,
                            real_name=real_name,
                            confidence=confidence,
                            evidence=evidence
                        )

                        logger.debug(f"Mapped {speaker_label} -> {real_name} (confidence: {confidence:.2f})")
                        continue

            # If no mapping created, use fallback
            if speaker_label not in self.mappings:
                # Try to find any unassigned speaker from agenda
                unassigned = [s for s in self.agenda.all_speakers if s not in assigned_names]

                if unassigned:
                    real_name = unassigned[0]
                    assigned_names.add(real_name)
                    self.mappings[speaker_label] = SpeakerMapping(
                        speaker_label=speaker_label,
                        real_name=real_name,
                        confidence=0.3,
                        evidence=["Assigned by process of elimination"]
                    )
                else:
                    # No names left - use speaker label as fallback
                    self.mappings[speaker_label] = SpeakerMapping(
                        speaker_label=speaker_label,
                        real_name=speaker_label,  # Keep original label
                        confidence=0.0,
                        evidence=["No matching name in agenda"]
                    )

    def _apply_mappings(self, segments: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """
        Apply speaker mappings to all segments.

        Args:
            segments: List of speaker segments

        Returns:
            Updated list with real_name and confidence set
        """
        for segment in segments:
            if segment.speaker_label in self.mappings:
                mapping = self.mappings[segment.speaker_label]
                segment.real_name = mapping.real_name
                segment.confidence = mapping.confidence

        return segments

    def get_mapping(self, speaker_label: str) -> Optional[SpeakerMapping]:
        """
        Get the mapping for a specific speaker label.

        Args:
            speaker_label: Speaker label (e.g., "SPEAKER_00")

        Returns:
            SpeakerMapping object or None
        """
        return self.mappings.get(speaker_label)

    def get_all_mappings(self) -> List[SpeakerMapping]:
        """
        Get all speaker mappings.

        Returns:
            List of SpeakerMapping objects
        """
        return list(self.mappings.values())
