# evaluation/compatibility/adapters/xtts_v2.py
"""
XTTS v2 adapter — voice cloning with reference audio.

Install:
    hf download coqui/XTTS-v2 --local-dir models/xtts_v2
    pip install TTS soundfile

Model type: Multilingual autoregressive TTS (17 languages including Hindi).
Sample rate: 24000 Hz
Language support: Expects Roman script for Hindi — Devanagari handling is untested.
"""

import time
from pathlib import Path

from .base import REF_AUDIO_PATH, SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "xtts_v2"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


class XTTSV2Adapter(TTSAdapter):
    name = "xtts_v2"
    supported_scripts = ["roman"]  # Only Roman is verified; Devanagari/mixed untested

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

        # XTTS-v2 is trained on Roman script for all languages — Devanagari handling is untested
        if script_variant in ("devanagari", "mixed"):
            warnings.append(
                f"XTTS-v2 not tested with {script_variant} script — model expects Roman. "
                "Results may be degraded or fail."
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
