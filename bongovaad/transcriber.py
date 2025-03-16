#!/usr/bin/env python3
"""
BongoVaad - Bengali Speech Recognition Tool
A tool for transcribing Bengali audio from YouTube videos using Hugging Face Inference API.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import ffmpeg
import srt
from pydub import AudioSegment
from pytube import YouTube
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bongovaad")


class BongoVaadTranscriber:
    """Main class for handling Bengali speech transcription."""

    def __init__(self, api_key: Optional[str] = None, model_id: str = "openai/whisper-large-v3-turbo"):
        """
        Initialize the transcriber with Hugging Face Inference API.
        
        Args:
            api_key: Hugging Face API key (if None, will look for HF_API_KEY environment variable)
            model_id: Model ID to use for transcription
        """
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Set HF_API_KEY environment variable or pass api_key parameter.")
        
        self.model_id = model_id
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        logger.info(f"Using model: {model_id}")

    async def _transcribe_audio_segment(self, segment_file: str) -> Dict:
        """
        Transcribe an audio segment using Hugging Face Inference API.
        
        Args:
            segment_file: Path to the audio segment file
            
        Returns:
            Transcription result
        """
        try:
            with open(segment_file, "rb") as f:
                audio_data = f.read()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/octet-stream"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    data=audio_data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    
                    result = await response.json()
                    return result
                    
        except Exception as e:
            logger.error(f"Failed to transcribe segment: {str(e)}")
            raise

    def download_audio(self, url: str) -> str:
        """
        Download audio from a YouTube video.
        
        Args:
            url: YouTube URL
            
        Returns:
            Path to the downloaded audio file
        """
        logger.info(f"Downloading audio from: {url}")
        try:
            # Create a temporary directory for downloads
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, "audio.mp4")
            
            # Download the audio
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).first()
            if not stream:
                raise ValueError("No audio stream found for this YouTube video")
            
            stream.download(output_path=temp_dir, filename="audio.mp4")
            
            # Convert to MP3
            output_path = os.path.join(temp_dir, "audio.mp3")
            ffmpeg.input(temp_audio_path).output(output_path).run(quiet=True, overwrite_output=True)
            
            # Remove the original file
            os.remove(temp_audio_path)
            
            logger.info(f"Audio downloaded and converted successfully to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to download audio: {str(e)}")
            raise

    def get_youtube_video_id(self, url: str) -> str:
        """
        Extract the video ID from a YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            YouTube video ID
        """
        try:
            yt = YouTube(url)
            return yt.video_id
        except Exception as e:
            logger.error(f"Failed to extract YouTube video ID: {str(e)}")
            raise

    async def transcribe_async(self, url: str, segment_length_seconds: int = 8, 
                  output_format: str = "both") -> Dict[str, str]:
        """
        Transcribe audio from a YouTube video asynchronously.
        
        Args:
            url: YouTube URL
            segment_length_seconds: Length of each audio segment in seconds
            output_format: Output format ("txt", "srt", or "both")
            
        Returns:
            Dictionary with paths to output files
        """
        if output_format not in ["txt", "srt", "both"]:
            raise ValueError("Output format must be 'txt', 'srt', or 'both'")
            
        try:
            # Download the audio
            audio_path = self.download_audio(url)
            video_id = self.get_youtube_video_id(url)
            
            # Load the audio
            song = AudioSegment.from_mp3(audio_path)
            total_duration_ms = len(song)
            segment_length_ms = segment_length_seconds * 1000
            
            logger.info(f"Starting transcription of {total_duration_ms/1000:.2f} seconds of audio...")
            
            # Prepare for transcription
            segments = []
            transcriptions = []
            segment_files = []
            
            # Split audio into segments
            for i, segment_start in enumerate(tqdm(
                range(0, total_duration_ms, segment_length_ms),
                desc="Preparing segments",
                unit="segment"
            )):
                segment_end = min(segment_start + segment_length_ms, total_duration_ms)
                segment = song[segment_start:segment_end]
                
                # Export segment to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    segment_file = temp_file.name
                
                segment.export(segment_file, format="mp3")
                segment_files.append({
                    "file": segment_file,
                    "start": segment_start,
                    "end": segment_end
                })
            
            # Transcribe segments concurrently
            logger.info("Transcribing segments...")
            tasks = []
            for segment_info in segment_files:
                task = asyncio.create_task(self._transcribe_audio_segment(segment_info["file"]))
                tasks.append((task, segment_info))
            
            # Process results as they complete
            for i, (task, segment_info) in enumerate(tqdm(
                tasks, 
                desc="Transcribing segments",
                unit="segment"
            )):
                result = await task
                
                # Extract text from result
                text = result.get("text", "").strip()
                
                # Store segment information for SRT
                segments.append({
                    "start": segment_info["start"],
                    "end": segment_info["end"],
                    "text": text
                })
                
                transcriptions.append(text)
                
                # Delete temporary file
                os.remove(segment_info["file"])
            
            # Create output files
            output_files = {}
            
            # Create TXT file if requested
            if output_format in ["txt", "both"]:
                txt_output = f"{video_id}.txt"
                with open(txt_output, "w", encoding="utf-8") as f:
                    f.write("\n".join(transcriptions))
                output_files["txt"] = txt_output
                logger.info(f"Text transcription saved to: {txt_output}")
            
            # Create SRT file if requested
            if output_format in ["srt", "both"]:
                srt_output = f"{video_id}.srt"
                srt_subtitles = self._create_srt(segments)
                with open(srt_output, "w", encoding="utf-8") as f:
                    f.write(srt_subtitles)
                output_files["srt"] = srt_output
                logger.info(f"SRT subtitles saved to: {srt_output}")
            
            # Clean up
            os.remove(audio_path)
            if os.path.exists(os.path.dirname(audio_path)):
                os.rmdir(os.path.dirname(audio_path))
            
            logger.info("Transcription completed successfully!")
            return output_files
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def transcribe(self, url: str, segment_length_seconds: int = 8, 
                  output_format: str = "both") -> Dict[str, str]:
        """
        Transcribe audio from a YouTube video (synchronous wrapper).
        
        Args:
            url: YouTube URL
            segment_length_seconds: Length of each audio segment in seconds
            output_format: Output format ("txt", "srt", or "both")
            
        Returns:
            Dictionary with paths to output files
        """
        return asyncio.run(self.transcribe_async(
            url=url,
            segment_length_seconds=segment_length_seconds,
            output_format=output_format
        ))

    def _create_srt(self, segments: List[Dict[str, Union[int, str]]]) -> str:
        """
        Create SRT subtitles from transcription segments.
        
        Args:
            segments: List of dictionaries with segment information
            
        Returns:
            SRT formatted string
        """
        srt_segments = []
        
        for i, segment in enumerate(segments):
            start_time = timedelta(milliseconds=segment["start"])
            end_time = timedelta(milliseconds=segment["end"])
            
            srt_segment = srt.Subtitle(
                index=i+1,
                start=start_time,
                end=end_time,
                content=segment["text"]
            )
            srt_segments.append(srt_segment)
        
        return srt.compose(srt_segments)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="BongoVaad - Bengali Speech Recognition Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--url", 
        type=str, 
        required=True,
        help="YouTube URL for audio transcription"
    )
    parser.add_argument(
        "--segment-length", 
        type=int, 
        default=8,
        help="Length of each audio segment in seconds"
    )
    parser.add_argument(
        "--output-format", 
        type=str, 
        choices=["txt", "srt", "both"], 
        default="both",
        help="Output format (txt, srt, or both)"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        default=None,
        help="Hugging Face API key (if not provided, will use HF_API_KEY environment variable)"
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="openai/whisper-large-v3-turbo",
        help="Model ID to use for transcription"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize transcriber
        transcriber = BongoVaadTranscriber(
            api_key=args.api_key,
            model_id=args.model_id
        )
        
        # Transcribe audio
        output_files = transcriber.transcribe(
            url=args.url,
            segment_length_seconds=args.segment_length,
            output_format=args.output_format
        )
        
        # Print output file paths
        for fmt, path in output_files.items():
            print(f"{fmt.upper()} output: {path}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 