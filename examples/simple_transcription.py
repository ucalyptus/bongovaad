#!/usr/bin/env python3
"""
Simple example of using BongoVaad as a library.
"""

import os
from bongovaad import BongoVaadTranscriber

def main():
    # Get API key from environment variable or set it directly
    api_key = os.environ.get("HF_API_KEY", "your_api_key_here")
    
    # Initialize the transcriber with Hugging Face API
    transcriber = BongoVaadTranscriber(
        api_key=api_key,
        model_id="openai/whisper-large-v3-turbo"
    )
    
    # YouTube URL to transcribe
    youtube_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
    
    # Transcribe the video
    output_files = transcriber.transcribe(
        url=youtube_url,
        segment_length_seconds=10,
        output_format="both"
    )
    
    # Print output file paths
    print(f"Transcription completed!")
    print(f"Text file: {output_files.get('txt')}")
    print(f"SRT file: {output_files.get('srt')}")

if __name__ == "__main__":
    main() 