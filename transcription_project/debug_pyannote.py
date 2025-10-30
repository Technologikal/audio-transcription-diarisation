
import os
import logging
from pyannote.audio import Pipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('pyannote-debug')

log.debug("Starting pyannote debug script.")

# Get Hugging Face token
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    log.error("HF_TOKEN environment variable not set.")
else:
    log.debug("HF_TOKEN found.")
    try:
        log.debug("Attempting to load pyannote.audio pipeline...")
        diarisation_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        log.debug("Successfully loaded pyannote.audio pipeline.")
    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)

log.debug("Pyannote debug script finished.")
