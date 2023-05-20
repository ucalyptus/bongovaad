# bongovaad

bongovaad is a Python package that provides functionality for transcribing audio from YouTube videos. It utilizes the powerful ASR (Automatic Speech Recognition) model provided by the Whisper library from Hugging Face.

## Features

- We have already LoRA-tuned the whisper-large-v2 model on the 'bn' subset of Mozilla Common Voice 13, obtaining a Word-Error-Rate(WER) of 57, compared to the WER of 101 obtained by the original OpenAI paper. More information available [here](https://huggingface.co/ucalyptus/whisper-large-v2-bengali-100steps).
- Handles audio segmentation for longer videos using AudioSegment

### ToDo
-  Automatic SRT file creation

## Installation

Before using bongovaad, you need to install ffmpeg:

```
sudo apt install ffmpeg -y
```

To install bongovaad, you can use pip:
`pip install bongovaad`


# Usage
bongovaad provides a command-line interface (CLI) that allows you to transcribe audio from YouTube videos. Here's how to use it:

`bongovaad --url <youtube_url>`

Replace <youtube_url> with the actual YouTube URL of the video you want to transcribe. The output will be written to text files containing the transcriptions of the audio segments.

# Example
`bongovaad --url https://www.youtube.com/watch?v=ABC12345`

This command transcribes the audio from the YouTube video with the specified URL (https://www.youtube.com/watch?v=ABC12345).

# License
This project is licensed under the MIT License. See the LICENSE file for more information.

# Contributing
Contributions are welcome! Please refer to the contributing guidelines for more information.
If you encounter any issues or have suggestions for improvements, please create a new issue on the GitHub repository.

# Acknowledgements
- [PEFT](https://github.com/huggingface/peft)
