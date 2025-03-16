#!/usr/bin/env python3
"""
BongoVaad - Bengali Speech Recognition Tool
A tool for transcribing Bengali audio from YouTube videos using fine-tuned Whisper models.
"""

import argparse
import logging
import os
import sys
import tempfile
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Union

import ffmpeg
import srt
import torch
from peft import PeftConfig, PeftModel
from pydub import AudioSegment
from pytube import YouTube
from tqdm import tqdm
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bongovaad")


class BongoVaadTranscriber:
    """Main class for handling Bengali speech transcription."""

    def __init__(self, use_8bit: bool = True, device: str = "auto"):
        """
        Initialize the transcriber with the fine-tuned model.
        
        Args:
            use_8bit: Whether to use 8-bit quantization for the model
            device: Device to use for inference ("auto", "cuda", "cpu")
        """
        self.peft_model_id = "ucalyptus/whisper-large-v2-bengali-100steps"
        self.language = "bn"
        self.task = "transcribe"
        self.use_8bit = use_8bit
        self.device = device
        self.pipeline = None
        self.forced_decoder_ids = None
        
        # Initialize the model and pipeline
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Set up the ASR pipeline with the fine-tuned model."""
        logger.info("Loading model and setting up pipeline...")
        try:
            peft_config = PeftConfig.from_pretrained(self.peft_model_id)
            model = WhisperForConditionalGeneration.from_pretrained(
                peft_config.base_model_name_or_path, 
                load_in_8bit=self.use_8bit, 
                device_map=self.device
            )
            model = PeftModel.from_pretrained(model, self.peft_model_id)
            
            tokenizer = WhisperTokenizer.from_pretrained(
                peft_config.base_model_name_or_path, 
                language=self.language, 
                task=self.task
            )
            
            processor = WhisperProcessor.from_pretrained(
                peft_config.base_model_name_or_path, 
                language=self.language, 
                task=self.task
            )
            
            feature_extractor = processor.feature_extractor
            self.forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=self.language, 
                task=self.task
            )
            
            self.pipeline = AutomaticSpeechRecognitionPipeline(
                model=model, 
                tokenizer=tokenizer, 
                feature_extractor=feature_extractor
            )
            logger.info("Pipeline setup complete.")
        except Exception as e:
            logger.error(f"Failed to set up pipeline: {str(e)}")
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

    def transcribe(self, url: str, segment_length_seconds: int = 8, 
                  output_format: str = "both") -> Dict[str, str]:
        """
        Transcribe audio from a YouTube video.
        
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
            
            # Process each segment
            for i, segment_start in enumerate(tqdm(
                range(0, total_duration_ms, segment_length_ms),
                desc="Transcribing segments",
                unit="segment"
            )):
                segment_end = min(segment_start + segment_length_ms, total_duration_ms)
                segment = song[segment_start:segment_end]
                
                # Export segment to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    segment_file = temp_file.name
                
                segment.export(segment_file, format="mp3")
                
                # Transcribe the segment
                with torch.cuda.amp.autocast():
                    transcript = self.pipeline(
                        segment_file, 
                        generate_kwargs={"forced_decoder_ids": self.forced_decoder_ids}, 
                        max_new_tokens=448
                    )
                
                # Store segment information for SRT
                segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": transcript["text"].strip()
                })
                
                transcriptions.append(transcript["text"])
                
                # Delete temporary file
                os.remove(segment_file)
            
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
        "--use-8bit", 
        action="store_true", 
        default=True,
        help="Use 8-bit quantization for the model"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
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
            use_8bit=args.use_8bit,
            device=args.device
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