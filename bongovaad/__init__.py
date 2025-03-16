"""
BongoVaad - Bengali Speech Recognition Tool
A tool for transcribing Bengali audio from YouTube videos using fine-tuned Whisper models.
"""

__version__ = "0.4.0"

from .transcriber import BongoVaadTranscriber, main

__all__ = ["BongoVaadTranscriber", "main"] 