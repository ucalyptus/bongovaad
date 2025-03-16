# bongovaad (বঙ্গবাদ) [![PyPI version](https://badge.fury.io/py/bongovaad.svg)](https://badge.fury.io/py/bongovaad)

bongovaad is a Python package for transcribing Bengali audio from YouTube videos. It uses a fine-tuned Whisper model optimized for Bengali speech recognition.

## Features

- **Fine-tuned Model**: We've [LoRA](https://arxiv.org/abs/2106.09685)-tuned the whisper-large-v2 model on the 'bn' subset of Mozilla Common Voice 13, achieving a Word-Error-Rate(WER) of 57, compared to the WER of 103.4 in the original OpenAI [paper](https://cdn.openai.com/papers/whisper.pdf) (Page 23).
- **SRT Subtitle Generation**: Automatically creates SRT subtitle files for video players.
- **Efficient Audio Processing**: Handles audio segmentation for longer videos with progress tracking.
- **Temporary File Management**: Uses temporary directories for clean processing.
- **Robust Error Handling**: Comprehensive error handling and logging.
- **Command-line Interface**: Easy-to-use CLI with multiple options.

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
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
          --device cuda \
          --verbose
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--url` | YouTube URL for audio transcription | (Required) |
| `--segment-length` | Length of each audio segment in seconds | 8 |
| `--output-format` | Output format (txt, srt, or both) | both |
| `--use-8bit` | Use 8-bit quantization for the model | True |
| `--device` | Device to use for inference (auto, cuda, cpu) | auto |
| `--verbose` | Enable verbose logging | False |

## Output Files

The tool generates two types of output files:

1. **Text File** (`VIDEO_ID.txt`): Contains the full transcription text.
2. **SRT File** (`VIDEO_ID.srt`): Contains time-coded subtitles compatible with video players.

## Python API

You can also use bongovaad as a Python library:

```python
from bongovaad import BongoVaadTranscriber

# Initialize the transcriber
transcriber = BongoVaadTranscriber(use_8bit=True, device="cuda")

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

- Processing time depends on video length and hardware capabilities.
- Using a GPU significantly improves performance.
- Longer segment lengths may improve speed but could reduce accuracy for complex audio.

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

- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
- [Whisper](https://github.com/openai/whisper) for the base ASR model
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for the training dataset
