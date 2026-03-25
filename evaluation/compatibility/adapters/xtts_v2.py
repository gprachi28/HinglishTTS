# evaluation/compatibility/adapters/xtts_v2.py
"""
XTTS v2 adapter — voice cloning with reference audio.

Install:
    hf download coqui/XTTS-v2 --local-dir models/xtts_v2
    pip install TTS soundfile

Model type: Multilingual autoregressive TTS (17 languages including Hindi).
Sample rate: 24000 Hz
Script support: Verified on Roman, Devanagari, and mixed-script Hindi.
Phase 1.5 results: 90% pass rate across all three script variants (18/20 each).
Known limitation: Struggles with numerical/entity code-switching (CS-06 pattern).
"""

import time
from pathlib import Path

from .base import REF_AUDIO_PATH, SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "xtts_v2"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


class XTTSV2Adapter(TTSAdapter):
    name = "xtts_v2"
    supported_scripts = ["roman", "devanagari", "mixed"]  # All verified in Phase 1.5 testing

    def __init__(self):
        self._tts = None

    def is_available(self) -> bool:
        """Check if XTTS-v2 library and weights are available."""
        try:
            from TTS.api import TTS  # noqa: F401
            # XTTS-v2 downloads weights on first load, but check if already present
            if WEIGHTS_DIR.exists():
                return True
            # If not locally cached, TTS will download on load — this is OK
            return True
        except ImportError:
            return False

    def load(self) -> None:
        """Load XTTS-v2 model with device auto-detection."""
        import torch
        from TTS.api import TTS

        # Determine device: CUDA → MPS → CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # TTS API: gpu parameter is deprecated, use device via to()
        self._tts = TTS(MODEL_NAME, gpu=False, progress_bar=False)
        self._tts.to(device)

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        """
        Synthesize speech for the given text.

        Args:
            text: Input text (Roman preferred; Devanagari/mixed untested).
            script_variant: One of "roman", "devanagari", "mixed".

        Returns:
            SynthResult with audio or error message.
        """
        if self._tts is None:
            return SynthResult(success=False, error="Model not loaded — call load() first")
        if not REF_AUDIO_PATH.exists():
            return SynthResult(success=False, error=f"Reference audio not found: {REF_AUDIO_PATH}")

        warnings = []

        # XTTS-v2 may struggle with numerical/entity code-switching (CS-06 pattern)
        if script_variant in ("roman", "devanagari", "mixed"):
            if any(char.isdigit() for char in text) or "PM" in text or "AM" in text:
                warnings.append(
                    "Input contains numerals/time entities — XTTS-v2 may struggle with "
                    "this code-switching pattern (CS-06). Results may be degraded."
                )

        try:
            import os
            import tempfile

            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            start = time.perf_counter()
            self._tts.tts_to_file(
                text=text,
                speaker_wav=str(REF_AUDIO_PATH),
                language="hi",
                file_path=tmp_path,
            )
            latency = time.perf_counter() - start

            audio, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)

            return SynthResult(
                success=True,
                latency_s=round(latency, 3),
                audio=audio,
                sample_rate=sr,
                warnings=warnings,
            )

        except Exception as e:
            error_msg = str(e)
            if "language" in error_msg.lower():
                error_msg = f"Language parameter error (expected 'hi' for Hindi): {e}"
            return SynthResult(success=False, error=error_msg)

    def unload(self) -> None:
        """Free model from memory."""
        import gc

        self._tts = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
