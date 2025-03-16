# bongovaad (বঙ্গবাদ) [![PyPI version](https://badge.fury.io/py/bongovaad.svg)](https://badge.fury.io/py/bongovaad)

bongovaad is a Python package for transcribing Bengali audio from YouTube videos. It uses the Hugging Face Inference API with Whisper models for high-quality speech recognition.

## Features

- **Cloud-based Transcription**: Uses Hugging Face Inference API with state-of-the-art Whisper models.
- **SRT Subtitle Generation**: Automatically creates SRT subtitle files for video players.
- **Efficient Audio Processing**: Handles audio segmentation for longer videos with progress tracking.
- **Concurrent Processing**: Uses asynchronous requests for faster transcription of multiple segments.
- **Temporary File Management**: Uses temporary directories for clean processing.
- **Robust Error Handling**: Comprehensive error handling and logging.
- **Command-line Interface**: Easy-to-use CLI with multiple options.

## Requirements

- Python 3.8 or higher
- Hugging Face API key
- ffmpeg

## Installation

### 1. Install ffmpeg

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg -y
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

### 2. Install bongovaad

```bash
pip install bongovaad
```

### 3. Get a Hugging Face API Key

1. Create an account on [Hugging Face](https://huggingface.co/join)
2. Generate an API key at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set it as an environment variable:

```bash
export HF_API_KEY="your_api_key_here"
```

## Usage

### Basic Usage

```bash
bongovaad --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Advanced Options

```bash
bongovaad --url "https://www.youtube.com/watch?v=VIDEO_ID" \
          --segment-length 10 \
          --output-format both \
          --model-id "openai/whisper-large-v3-turbo" \
          --api-key "your_api_key_here" \
          --verbose
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--url` | YouTube URL for audio transcription | (Required) |
| `--segment-length` | Length of each audio segment in seconds | 8 |
| `--output-format` | Output format (txt, srt, or both) | both |
| `--api-key` | Hugging Face API key | HF_API_KEY env var |
| `--model-id` | Model ID to use for transcription | openai/whisper-large-v3-turbo |
| `--verbose` | Enable verbose logging | False |

## Output Files

The tool generates two types of output files:

1. **Text File** (`VIDEO_ID.txt`): Contains the full transcription text.
2. **SRT File** (`VIDEO_ID.srt`): Contains time-coded subtitles compatible with video players.

## Python API

You can also use bongovaad as a Python library:

```python
import os
from bongovaad import BongoVaadTranscriber

# Get API key from environment variable or set it directly
api_key = os.environ.get("HF_API_KEY", "your_api_key_here")

# Initialize the transcriber
transcriber = BongoVaadTranscriber(
    api_key=api_key,
    model_id="openai/whisper-large-v3-turbo"
)

# Transcribe a YouTube video
output_files = transcriber.transcribe(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    segment_length_seconds=10,
    output_format="both"
)

# Print output file paths
print(f"Text file: {output_files['txt']}")
print(f"SRT file: {output_files['srt']}")
```

## Performance Considerations

- Processing time depends on video length, internet connection, and Hugging Face API response times.
- Longer segment lengths may improve speed but could reduce accuracy for complex audio.
- The API has rate limits, so be mindful of how many requests you make.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the Inference API
- [Whisper](https://github.com/openai/whisper) for the state-of-the-art ASR model
- [OpenAI](https://openai.com/) for developing the Whisper model
