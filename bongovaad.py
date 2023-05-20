import argparse
from peft import PeftModel, PeftConfig
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from pydub import AudioSegment
from pytube import YouTube
import os
import torch
import ffmpeg

def setup_pipeline():
    peft_model_id = "ucalyptus/whisper-large-v2-bengali-100steps"
    language = "bn"
    task = "transcribe"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
    feature_extractor = processor.feature_extractor
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    pipeline = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
    
    return pipeline, forced_decoder_ids

def get_youtube_video_id(url):
    # Create a YouTube object
    yt = YouTube(url)
    
    # Extract the video ID
    video_id = yt.video_id
    
    return video_id

def BongoVaad(url, pipeline, forced_decoder_ids):
    try:
        # Download the audio from YouTube
        YouTube(url).streams.filter(only_audio=True).first().download(output_path=".", filename="audio")
        audio_path = "audio.mp4"
        audio_ = "audio.mp3"
        ffmpeg.input(audio_path).output(audio_).run()
        os.remove(audio_path)

        # Load the audio as an AudioSegment
        song = AudioSegment.from_mp3(audio_)
        print("Starting transcription...")
        segment_length = 8 * 1000  # 8 seconds in milliseconds

        for i, segment_start in enumerate(range(0, len(song), segment_length)):
            segment_end = segment_start + segment_length
            segment = song[segment_start:segment_end]

            # Export segment to a temporary file
            segment_file = f"segment_{i}.mp3"
            segment.export(segment_file, format="mp3")

            with torch.cuda.amp.autocast():
                transcript = pipeline(segment_file, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=448)

            # Save transcription to a text file
            output_file = f"{get_youtube_video_id(url)}_{i}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcript["text"])

            # Delete temporary file
            os.remove(segment_file)

        print("Transcription completed successfully!")

    finally:
        # Remove downloaded audio file
        if os.path.exists(audio_):
            os.remove(audio_)


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="BongoVaad")
    parser.add_argument("youtube_url", type=str, help="YouTube URL for audio transcription")
    args = parser.parse_args()

    # Setup the pipeline
    pipeline, forced_decoder_ids = setup_pipeline()

    # Transcribe YouTube audio
    BongoVaad(args.youtube_url, pipeline, forced_decoder_ids)


if __name__ == "__main__":
    main()
