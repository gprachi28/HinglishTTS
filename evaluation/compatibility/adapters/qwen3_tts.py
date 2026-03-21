# evaluation/compatibility/adapters/qwen3_tts.py
"""
Qwen3-TTS adapter.

Install:
    pip install transformers accelerate
    # Download weights:
    huggingface-cli download Qwen/Qwen3-TTS --local-dir models/qwen3_tts/weights

HuggingFace: https://huggingface.co/Qwen/Qwen3-TTS
Supported scripts: Roman, Devanagari (self-reported), Mixed
Note: Qwen3-TTS supports multilingual including Hindi/Devanagari per docs.
"""

import time
from pathlib import Path

import numpy as np

from .base import SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "qwen3_tts" / "weights"


class Qwen3TTSAdapter(TTSAdapter):
    name = "qwen3_tts"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        if not WEIGHTS_DIR.exists():
            return False
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        self._processor = AutoProcessor.from_pretrained(str(WEIGHTS_DIR))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(WEIGHTS_DIR),
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if self._model is None:
            return SynthResult(success=False, error="Model not loaded — call load() first")

        try:
            import torch

            inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)

            start = time.perf_counter()
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, max_new_tokens=2048)
            latency = time.perf_counter() - start

            # Decode audio tokens to waveform
            audio = self._processor.decode_audio(output_ids)
            sr = self._processor.audio_sample_rate

            return SynthResult(
                success=True,
                latency_s=latency,
                audio=audio.cpu().numpy().squeeze(),
                sample_rate=sr,
            )

        except Exception as e:
            return SynthResult(success=False, error=str(e))

    def unload(self) -> None:
        import gc
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
