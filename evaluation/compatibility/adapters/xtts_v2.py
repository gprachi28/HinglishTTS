# evaluation/compatibility/adapters/xtts_v2.py
"""
XTTS v2 adapter (Coqui TTS).

Install:
    pip install TTS
    # weights download automatically on first use via TTS API

Repo: https://github.com/coqui-ai/TTS
Supported scripts: Roman (primary), Devanagari (limited — not officially supported)
Reference speaker: uses a short Hindi speaker wav for voice conditioning.
"""

import time
from pathlib import Path

from .base import SynthResult, TTSAdapter

# Path to a short (~10s) reference speaker wav for voice cloning.
# Replace with a real Hindi speaker recording.
REFERENCE_WAV = Path(__file__).parent.parent / "assets" / "reference_speaker.wav"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


class XTTSV2Adapter(TTSAdapter):
    name = "xtts_v2"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._tts = None

    def is_available(self) -> bool:
        try:
            from TTS.api import TTS  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        from TTS.api import TTS
        self._tts = TTS(MODEL_NAME, gpu=True)

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if self._tts is None:
            return SynthResult(success=False, error="Model not loaded — call load() first")

        if not REFERENCE_WAV.exists():
            return SynthResult(
                success=False,
                error=f"Reference speaker wav not found: {REFERENCE_WAV}"
            )

        try:
            import numpy as np
            import soundfile as sf
            import tempfile, os

            start = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            self._tts.tts_to_file(
                text=text,
                speaker_wav=str(REFERENCE_WAV),
                language="hi" if script_variant in ("devanagari", "mixed") else "hi",
                file_path=tmp_path,
            )
            latency = time.perf_counter() - start

            audio, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)
            return SynthResult(success=True, latency_s=latency, audio=audio, sample_rate=sr)

        except Exception as e:
            return SynthResult(success=False, error=str(e))

    def unload(self) -> None:
        self._tts = None
