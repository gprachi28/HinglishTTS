# evaluation/compatibility/adapters/cosyvoice2.py
"""
CosyVoice 2 adapter.

Install:
    cd models/cosyvoice2
    pip install -r requirements.txt
    # Download weights:
    huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir models/cosyvoice2/weights

Repo: https://github.com/FunAudioLLM/CosyVoice
Supported scripts: Roman (reported), Devanagari (unverified — test purpose)
Note: CosyVoice 2 is primarily trained on Chinese/English — Hindi support
      is community-reported. Devanagari rendering likely unsupported.
"""

import time
from pathlib import Path

import numpy as np

from .base import SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "cosyvoice2" / "weights"
REPO_DIR = Path(__file__).parents[3] / "models" / "cosyvoice2"


class CosyVoice2Adapter(TTSAdapter):
    name = "cosyvoice2"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        if not WEIGHTS_DIR.exists():
            return False
        try:
            import sys
            sys.path.insert(0, str(REPO_DIR))
            from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        import sys
        sys.path.insert(0, str(REPO_DIR))
        from cosyvoice.cli.cosyvoice import CosyVoice2
        self._model = CosyVoice2(str(WEIGHTS_DIR))

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if self._model is None:
            return SynthResult(success=False, error="Model not loaded — call load() first")

        try:
            start = time.perf_counter()
            # Zero-shot inference (no reference speaker)
            output = list(self._model.inference_sft(
                tts_text=text,
                spk_id="中文女",   # default speaker; Hindi needs a custom speaker
            ))
            latency = time.perf_counter() - start

            if not output:
                return SynthResult(success=False, error="Model returned empty output")

            audio = output[0]["tts_speech"].numpy().squeeze()
            sr = self._model.sample_rate
            return SynthResult(success=True, latency_s=latency, audio=audio, sample_rate=sr)

        except Exception as e:
            return SynthResult(success=False, error=str(e))

    def unload(self) -> None:
        self._model = None
