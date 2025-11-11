"""
Output Formatter Module

Generates formatted outputs for transcription results:
- Structured transcript with named speakers and agenda sections
- Executive summary with key points organized by agenda topics
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import textwrap
import logging

from agenda_parser import ParsedAgenda, AgendaSection
from speaker_mapper import SpeakerSegment, SpeakerMapping

logger = logging.getLogger(__name__)


class TranscriptFormatter:
    """
    Formats transcription segments into a structured transcript with
    named speakers and agenda section markers.
    """

    def __init__(self, parsed_agenda: ParsedAgenda):
        """
        Initialize formatter with parsed agenda.

        Args:
            parsed_agenda: ParsedAgenda object
        """
        self.agenda = parsed_agenda

    def format(self, segments: List[SpeakerSegment],
               include_timestamps: bool = True,
               include_confidence: bool = False) -> str:
        """
        Format segments into a structured transcript.

        Args:
            segments: List of SpeakerSegment objects with mapped names
            include_timestamps: Include timestamp markers
            include_confidence: Include confidence scores for speaker mappings

        Returns:
            Formatted transcript as string
        """
        logger.info("Generating structured transcript")

        lines = []

        # Add header with meeting metadata
        lines.append("=" * 80)
        lines.append("MEETING TRANSCRIPT")
        lines.append("=" * 80)
        lines.append("")

        if self.agenda.metadata.title:
            lines.append(f"Meeting: {self.agenda.metadata.title}")
        if self.agenda.metadata.date:
            lines.append(f"Date: {self.agenda.metadata.date}")
        if self.agenda.metadata.time:
            lines.append(f"Time: {self.agenda.metadata.time}")
        if self.agenda.metadata.location:
            lines.append(f"Location: {self.agenda.metadata.location}")

        lines.append("")
        lines.append("-" * 80)
        lines.append("")

        # Group segments by agenda section
        current_section = None

        for segment in segments:
            # Add section marker when entering new section
            if segment.agenda_section != current_section:
                current_section = segment.agenda_section
                lines.append("")
                lines.append("=" * 80)
                lines.append(f"SECTION: {current_section or 'Unknown'}")
                lines.append("=" * 80)
                lines.append("")

            # Format speaker name
            speaker_name = segment.real_name or segment.speaker_label

            # Add confidence indicator if requested and confidence is low
            if include_confidence and segment.confidence < 0.7:
                speaker_name += f" [confidence: {segment.confidence:.0%}]"

            # Format timestamp
            timestamp_str = ""
            if include_timestamps:
                timestamp_str = f"[{self._format_timestamp(segment.start_time)} - " \
                              f"{self._format_timestamp(segment.end_time)}] "

            # Add the transcription line
            lines.append(f"{timestamp_str}{speaker_name}: {segment.text}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF TRANSCRIPT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds into HH:MM:SS or MM:SS format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


class SummaryFormatter:
    """
    Generates an executive summary of the meeting organized by
    agenda topics with key points extracted from the transcription.
    """

    def __init__(self, parsed_agenda: ParsedAgenda):
        """
        Initialize formatter with parsed agenda.

        Args:
            parsed_agenda: ParsedAgenda object
        """
        self.agenda = parsed_agenda

    def format(self, segments: List[SpeakerSegment],
               speaker_mappings: List[SpeakerMapping]) -> str:
        """
        Generate executive summary organized by agenda sections.

        Args:
            segments: List of SpeakerSegment objects with mapped names
            speaker_mappings: List of SpeakerMapping objects

        Returns:
            Formatted executive summary as string
        """
        logger.info("Generating executive summary")

        lines = []

        # Add header
        lines.append("=" * 80)
        lines.append("MEETING SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        # Meeting metadata
        if self.agenda.metadata.title:
            lines.append(f"Meeting: {self.agenda.metadata.title}")
        if self.agenda.metadata.date:
            lines.append(f"Date: {self.agenda.metadata.date}")
        if self.agenda.metadata.time:
            lines.append(f"Time: {self.agenda.metadata.time}")
        if self.agenda.metadata.location:
            lines.append(f"Location: {self.agenda.metadata.location}")

        lines.append("")

        # Participants section
        lines.append("-" * 80)
        lines.append("PARTICIPANTS")
        lines.append("-" * 80)
        lines.append("")

        for mapping in speaker_mappings:
            confidence_str = f" (confidence: {mapping.confidence:.0%})" if mapping.confidence < 0.7 else ""
            lines.append(f"  • {mapping.real_name}{confidence_str}")

        lines.append("")

        # Group segments by section
        sections_content = self._group_segments_by_section(segments)

        # Generate summary for each agenda section
        lines.append("-" * 80)
        lines.append("AGENDA AND DISCUSSION")
        lines.append("-" * 80)
        lines.append("")

        for section in self.agenda.sections:
            section_segments = sections_content.get(section.title, [])

            lines.append(f"\n{section.section_number or ''} {section.title}".strip())
            if section.speakers:
                lines.append(f"   Led by: {', '.join(section.speakers)}")
            lines.append("")

            if section.description:
                # Wrap description
                wrapped = textwrap.fill(section.description, width=76, initial_indent="   ",
                                       subsequent_indent="   ")
                lines.append(wrapped)
                lines.append("")

            # Add key points from transcription
            if section_segments:
                lines.append("   Discussion highlights:")
                lines.append("")

                # Get unique speakers in this section
                speakers_in_section = defaultdict(list)
                for seg in section_segments:
                    speaker_name = seg.real_name or seg.speaker_label
                    speakers_in_section[speaker_name].append(seg.text)

                # Summarize each speaker's contribution
                for speaker, texts in speakers_in_section.items():
                    combined_text = " ".join(texts)

                    # Extract first sentence or first 150 chars as summary
                    summary = self._extract_summary(combined_text, max_length=150)

                    wrapped = textwrap.fill(f"• {speaker}: {summary}",
                                          width=76,
                                          initial_indent="     ",
                                          subsequent_indent="       ")
                    lines.append(wrapped)

                lines.append("")
            else:
                lines.append("   [No discussion recorded for this section]")
                lines.append("")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF SUMMARY")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _group_segments_by_section(self, segments: List[SpeakerSegment]) -> Dict[str, List[SpeakerSegment]]:
        """
        Group segments by agenda section.

        Args:
            segments: List of SpeakerSegment objects

        Returns:
            Dict mapping section title to list of segments
        """
        sections = defaultdict(list)
        for segment in segments:
            if segment.agenda_section:
                sections[segment.agenda_section].append(segment)
        return sections

    def _extract_summary(self, text: str, max_length: int = 150) -> str:
        """
        Extract a summary from text (first sentence or truncate to max_length).

        Args:
            text: Input text
            max_length: Maximum length of summary

        Returns:
            Summary string
        """
        # Try to find first sentence
        sentences = text.split('. ')
        if sentences and len(sentences[0]) < max_length:
            return sentences[0] + '.'

        # Otherwise truncate to max_length
        if len(text) <= max_length:
            return text
        else:
            return text[:max_length].rsplit(' ', 1)[0] + '...'


def save_outputs(segments: List[SpeakerSegment],
                 parsed_agenda: ParsedAgenda,
                 speaker_mappings: List[SpeakerMapping],
                 output_file: str,
                 format_type: str = 'both'):
    """
    Save formatted outputs to file(s).

    Args:
        segments: List of SpeakerSegment objects
        parsed_agenda: ParsedAgenda object
        speaker_mappings: List of SpeakerMapping objects
        output_file: Base output file path
        format_type: 'transcript', 'summary', or 'both'
    """
    if format_type in ['transcript', 'both']:
        # Generate transcript
        transcript_formatter = TranscriptFormatter(parsed_agenda)
        transcript = transcript_formatter.format(segments, include_timestamps=True)

        # Save transcript
        transcript_file = output_file if format_type == 'transcript' else output_file.replace('.txt', '_transcript.txt')
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(transcript)
        logger.info(f"Transcript saved to: {transcript_file}")

    if format_type in ['summary', 'both']:
        # Generate summary
        summary_formatter = SummaryFormatter(parsed_agenda)
        summary = summary_formatter.format(segments, speaker_mappings)

        # Save summary
        summary_file = output_file.replace('.txt', '_summary.txt') if format_type == 'both' else output_file
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary saved to: {summary_file}")
