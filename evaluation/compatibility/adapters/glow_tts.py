# evaluation/compatibility/adapters/glow_tts.py
"""
Glow-TTS baseline adapter.

This is the locally-trained baseline (LJSpeech + IndicVoices-R fine-tune).
Training script: training/glow_tts_baseline.py (not yet implemented).

Install:
    pip install TTS   # Coqui TTS includes Glow-TTS
    # After training, checkpoint will be at:
    # models/glow_tts/checkpoint_XXXXX.pth + config.json

Status: Pending — training/glow_tts_baseline.py not yet built.
        Adapter is scaffolded; will work once checkpoint exists.
"""

import time
from pathlib import Path

import numpy as np

from .base import SynthResult, TTSAdapter

CHECKPOINT_DIR = Path(__file__).parents[3] / "models" / "glow_tts"


class GlowTTSAdapter(TTSAdapter):
    name = "glow_tts"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._synthesizer = None

    def is_available(self) -> bool:
        checkpoints = list(CHECKPOINT_DIR.glob("*.pth")) if CHECKPOINT_DIR.exists() else []
        if not checkpoints:
            return False
        try:
            from TTS.utils.synthesizer import Synthesizer  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        from TTS.utils.synthesizer import Synthesizer

        checkpoints = sorted(CHECKPOINT_DIR.glob("*.pth"))
        config_path = CHECKPOINT_DIR / "config.json"

        if not checkpoints:
            raise FileNotFoundError(f"No .pth checkpoint found in {CHECKPOINT_DIR}")
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {CHECKPOINT_DIR}")

        self._synthesizer = Synthesizer(
            tts_checkpoint=str(checkpoints[-1]),
            tts_config_path=str(config_path),
            use_cuda=True,
        )

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if self._synthesizer is None:
            return SynthResult(success=False, error="Model not loaded — call load() first")

        try:
            start = time.perf_counter()
            wav = self._synthesizer.tts(text)
            latency = time.perf_counter() - start

            audio = np.array(wav, dtype=np.float32)
            sr = self._synthesizer.tts_config.audio["sample_rate"]
            return SynthResult(success=True, latency_s=latency, audio=audio, sample_rate=sr)

        except Exception as e:
            return SynthResult(success=False, error=str(e))

    def unload(self) -> None:
        self._synthesizer = None
