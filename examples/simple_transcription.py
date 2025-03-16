#!/usr/bin/env python3
"""
Simple example of using BongoVaad as a library.
"""

from bongovaad import BongoVaadTranscriber

def main():
    # Initialize the transcriber
    transcriber = BongoVaadTranscriber(use_8bit=True, device="auto")
    
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