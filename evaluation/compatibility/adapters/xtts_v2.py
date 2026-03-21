# evaluation/compatibility/adapters/xtts_v2.py
"""
XTTS v2 adapter — voice cloning with reference audio.

Install:
    hf download coqui/XTTS-v2 --local-dir models/xtts_v2
    pip install TTS soundfile
"""

import time
from pathlib import Path

from .base import REF_AUDIO_PATH, SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "xtts_v2"
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
        if not REF_AUDIO_PATH.exists():
            return SynthResult(success=False, error=f"Reference audio not found: {REF_AUDIO_PATH}")

        try:
            import os
            import tempfile

            import numpy as np
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
            return SynthResult(success=True, latency_s=round(latency, 3), audio=audio, sample_rate=sr)

        except Exception as e:
            return SynthResult(success=False, error=str(e))

    def unload(self) -> None:
        self._tts = None
