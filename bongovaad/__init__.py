"""
BongoVaad - Bengali Speech Recognition Tool
A tool for transcribing Bengali audio from YouTube videos using Hugging Face Inference API.
"""

__version__ = "0.5.0"

from .transcriber import BongoVaadTranscriber, main

__all__ = ["BongoVaadTranscriber", "main"] 