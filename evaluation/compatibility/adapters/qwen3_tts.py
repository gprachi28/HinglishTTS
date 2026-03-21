# evaluation/compatibility/adapters/qwen3_tts.py
"""
Qwen3-TTS Base adapter — voice cloning mode.

Install:
    hf download Qwen/Qwen3-TTS-12Hz-1.7B --local-dir models/qwen3_tts/weights
    pip install -U qwen-tts
    # Model is gated — run `huggingface-cli login` first

Model type: Base (supports generate_voice_clone() with reference audio).
            Do NOT use the VoiceDesign variant here — it lacks voice cloning.
Sample rate: 24000 Hz
"""

import time
from pathlib import Path

from .base import REF_AUDIO_PATH, SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "qwen3_tts" / "weights"


class Qwen3TTSAdapter(TTSAdapter):
    name = "qwen3_tts"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        if not WEIGHTS_DIR.exists():
            return False
        try:
            from qwen_tts.inference import qwen3_tts_model  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        import torch
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(
            str(WEIGHTS_DIR),
            device_map="auto",
            dtype=torch.float16,
        )

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if self._model is None:
            return SynthResult(success=False, error="Model not loaded — call load() first")
        if not REF_AUDIO_PATH.exists():
            return SynthResult(success=False, error=f"Reference audio not found: {REF_AUDIO_PATH}")

        # Hindi has no native language ID — use "english" for Roman,
        # None for Devanagari/mixed to expose the gap explicitly.
        language = "english" if script_variant == "roman" else None

        warnings = []
        if language is None:
            warnings.append(
                f"Hindi not in codec_language_id — {script_variant} uses language=None"
            )

        try:
            prompt = self._model.create_voice_clone_prompt(str(REF_AUDIO_PATH))

            start = time.perf_counter()
            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                language=language,
            )
            latency = time.perf_counter() - start

            if not wavs:
                return SynthResult(success=False, error="Model returned empty audio")

            return SynthResult(
                success=True,
                latency_s=round(latency, 3),
                audio=wavs[0],
                sample_rate=sample_rate,
                warnings=warnings,
            )

        except Exception as e:
            return SynthResult(success=False, error=str(e))

    def unload(self) -> None:
        import gc
        self._model = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
