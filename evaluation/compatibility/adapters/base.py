# evaluation/compatibility/adapters/base.py
"""
Abstract base class for TTS model adapters.

Each model adapter must implement:
    synthesize(text, script_variant) -> SynthResult

A SynthResult with success=False and an error string is valid output
(used when a model is not installed or errors on a specific input).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Shared reference audio for voice cloning across all models
REF_AUDIO_PATH = Path(__file__).parents[1] / "assets" / "hindi_ref.wav"


@dataclass
class SynthResult:
    success: bool
    latency_s: float = 0.0
    sample_rate: int = 22050
    audio: Optional[np.ndarray] = None   # float32, shape (N,)
    error: Optional[str] = None
    warnings: list = field(default_factory=list)


class TTSAdapter(ABC):
    """Common interface for all TTS model adapters."""

    name: str = "base"
    supported_scripts: list = ["roman", "devanagari", "mixed"]

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the model weights/repo are installed and loadable."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called once before synthesize()."""

    @abstractmethod
    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        """
        Synthesize speech for the given text.

        Args:
            text:           Input text (Roman, Devanagari, or mixed script).
            script_variant: One of "roman", "devanagari", "mixed".

        Returns:
            SynthResult with audio array or error message.
        """

    def unload(self) -> None:
        """Free model from memory (optional — override if needed)."""
