# Gemini Project: Transcription and Diarization

## Project Overview

This project is a Python-based tool for transcribing audio files and performing speaker diarisation. It uses the `openai-whisper` library for audio transcription and `pyannote.audio` for identifying different speakers in an audio file.

The goal of this project is to create a tool that can take an audio recording of a meeting, transcribe the conversation, and attribute each part of the transcription to the correct speaker. This will help streamline the process of creating meeting minutes and extracting actionable insights.

The project is currently capable of:

*   Transcribing an audio file using the `openai-whisper` library.
*   Performing speaker diarisation using the `pyannote.audio` library to identify different speakers.
*   Associating the transcribed text with the corresponding speaker and providing timestamps for each segment.

## Building and Running

### 1. Setup a Virtual Environment

It is recommended to use a virtual environment to manage the project's dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

The project's dependencies are listed in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
```

### 3. Hugging Face Authentication

The `pyannote.audio` library requires authentication with the Hugging Face Hub to download the pre-trained speaker diarisation models. You will need to:

1.  **Create a Hugging Face Account:** If you don't have one, create an account at [huggingface.co](https://huggingface.co/).
2.  **Get an Access Token:** Go to your Hugging Face settings and create an access token with "read" permissions.
3.  **Accept User Conditions:** You will need to accept the user conditions for the following models on the Hugging Face website:
    *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
    *   [pyannote/speaker-diarisation-3.1](https://huggingface.co/pyannote/speaker-diarisation-3.1)
    *   [pyannote/speaker-diarisation-community-1](https://huggingface.co/pyannote/speaker-diarisation-community-1)
4.  **Set the Environment Variable:** Before running the script, you need to set the `HF_TOKEN` environment variable to your Hugging Face access token:

    ```bash
    export HF_TOKEN="your_hugging_face_token"
    ```

### 4. Running the Transcription Script

The main script for transcribing audio is `transcribe.py`. You can run it from the root of the project as follows:

```bash
python3 transcription_project/transcribe.py
```

This will transcribe the sample audio file located at `transcription_project/audio/LibriSpeech/dev-clean/251/136532/251-136532-0000.flac` and print the transcription with speaker labels and timestamps to the console.

## Development Conventions

*   **Virtual Environment:** All dependencies are managed within a Python virtual environment.
*   **Code Style:** The project follows the standard Python PEP 8 style guide.
*   **Modular Design:** The transcription and diarisation functionalities are separated into different modules to promote code reusability and maintainability.
*   **Hugging Face Hub:** The project relies on the Hugging Face Hub for downloading pre-trained models. All models are downloaded and cached locally.