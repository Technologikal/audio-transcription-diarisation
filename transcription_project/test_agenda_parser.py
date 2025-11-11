#!/usr/bin/env python3
"""
Test script for agenda parser.

Tests the agenda parsing functionality independently from the full
transcription pipeline. Shows extracted metadata, sections, and speakers.

Usage:
    python3 test_agenda_parser.py path/to/agenda.docx
"""

import sys
import logging
from agenda_parser import parse_agenda

# Configure logging to show detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def test_agenda_parser(agenda_path: str):
    """
    Test the agenda parser and display all extracted information.

    Args:
        agenda_path: Path to DOCX agenda file
    """
    print("=" * 80)
    print("AGENDA PARSER TEST")
    print("=" * 80)
    print(f"\nParsing: {agenda_path}\n")

    try:
        # Parse the agenda
        parsed_agenda = parse_agenda(agenda_path)

        # Display metadata
        print("-" * 80)
        print("MEETING METADATA")
        print("-" * 80)
        metadata = parsed_agenda.metadata
        print(f"Title:    {metadata.title or 'Not found'}")
        print(f"Date:     {metadata.date or 'Not found'}")
        print(f"Time:     {metadata.time or 'Not found'}")
        print(f"Location: {metadata.location or 'Not found'}")

        if metadata.raw_metadata:
            print("\nRaw metadata fields:")
            for key, value in metadata.raw_metadata.items():
                print(f"  {key}: {value}")

        # Display sections
        print("\n" + "-" * 80)
        print(f"AGENDA SECTIONS ({len(parsed_agenda.sections)} found)")
        print("-" * 80)

        for i, section in enumerate(parsed_agenda.sections, 1):
            print(f"\nSection {i}:")
            if section.section_number:
                print(f"  Number:      {section.section_number}")
            print(f"  Title:       {section.title}")

            if section.speakers:
                print(f"  Speakers:    {', '.join(section.speakers)}")
            else:
                print(f"  Speakers:    None identified")

            if section.description:
                # Truncate long descriptions
                desc = section.description[:200]
                if len(section.description) > 200:
                    desc += "..."
                print(f"  Description: {desc}")

        # Display all unique speakers
        print("\n" + "-" * 80)
        print(f"ALL SPEAKERS ({len(parsed_agenda.all_speakers)} unique)")
        print("-" * 80)

        if parsed_agenda.all_speakers:
            for speaker in parsed_agenda.all_speakers:
                print(f"  • {speaker}")
        else:
            print("  No speakers identified in agenda")

        print("\n" + "=" * 80)
        print("PARSING COMPLETE")
        print("=" * 80)

        # Summary statistics
        print(f"\nSummary:")
        print(f"  - {len(parsed_agenda.sections)} sections parsed")
        print(f"  - {len(parsed_agenda.all_speakers)} unique speakers identified")
        print(f"  - Metadata fields: {len(metadata.raw_metadata)}")

        return parsed_agenda

    except FileNotFoundError:
        logger.error(f"Agenda file not found: {agenda_path}")
        print(f"\n❌ ERROR: File not found: {agenda_path}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to parse agenda: {e}", exc_info=True)
        print(f"\n❌ ERROR: Failed to parse agenda: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_agenda_parser.py <path_to_agenda.docx>")
        print("\nExample:")
        print("  python3 test_agenda_parser.py sample_agenda.docx")
        sys.exit(1)

    agenda_file = sys.argv[1]
    test_agenda_parser(agenda_file)
