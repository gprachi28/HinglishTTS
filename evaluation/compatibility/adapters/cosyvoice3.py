# evaluation/compatibility/adapters/cosyvoice3.py
"""
CosyVoice 3 adapter.

Install:
    hf download FunAudioLLM/CosyVoice3 --local-dir models/cosyvoice3/weights
    pip install cosyvoice   # update package name if different

Note: HF repo name is a placeholder — verify at huggingface.co/FunAudioLLM
"""

import time
from pathlib import Path

from .base import SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "cosyvoice3" / "weights"
REPO_DIR = Path(__file__).parents[3] / "models" / "cosyvoice3"


class CosyVoice3Adapter(TTSAdapter):
    name = "cosyvoice3"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        return WEIGHTS_DIR.exists()

    def load(self) -> None:
        import sys
        sys.path.insert(0, str(REPO_DIR))
        from cosyvoice.cli.cosyvoice import CosyVoice3
        self._model = CosyVoice3(str(WEIGHTS_DIR))

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if self._model is None:
            return SynthResult(success=False, error="Model not loaded — call load() first")

        try:
            start = time.perf_counter()
            output = list(self._model.inference_sft(tts_text=text))
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
