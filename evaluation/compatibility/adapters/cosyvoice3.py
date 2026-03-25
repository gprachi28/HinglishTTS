# evaluation/compatibility/adapters/cosyvoice3.py
"""
CosyVoice 3 adapter — voice cloning with reference audio.

Install:
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git models/cosyvoice3/repo
    cd models/cosyvoice3/repo && pip install -r requirements.txt
    huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
        --local-dir models/cosyvoice3/weights
    # Also download ttsfrd (optional, Linux only — for text normalisation):
    huggingface-cli download FunAudioLLM/CosyVoice-ttsfrd \
        --local-dir pretrained_models/CosyVoice-ttsfrd

Model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
Sample rate: 22050 Hz (read from cosyvoice.sample_rate)
"""

import sys
import time
from pathlib import Path

from .base import REF_AUDIO_PATH, SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "cosyvoice3" / "weights"
REPO_DIR = Path(__file__).parents[3] / "models" / "cosyvoice3" / "repo"
MATCHA_DIR = REPO_DIR / "third_party" / "Matcha-TTS"


class CosyVoice3Adapter(TTSAdapter):
    name = "cosyvoice3"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._model = None

    def _patch_sys_path(self) -> None:
        for p in (str(REPO_DIR), str(MATCHA_DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)

    def is_available(self) -> bool:
        if not WEIGHTS_DIR.exists():
            return False
        try:
            self._patch_sys_path()
            from cosyvoice.cli.cosyvoice import AutoModel  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        self._patch_sys_path()
        from cosyvoice.cli.cosyvoice import AutoModel
        self._model = AutoModel(model_dir=str(WEIGHTS_DIR))

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if self._model is None:
            return SynthResult(success=False, error="Model not loaded — call load() first")
        if not REF_AUDIO_PATH.exists():
            return SynthResult(success=False, error=f"Reference audio not found: {REF_AUDIO_PATH}")

        # CosyVoice3 requires prompt_text = "<system><|endofprompt|><ref_transcript>"
        REF_TRANSCRIPT = (
            "Gas connection band hone ki khabaron par sarkar ki safai: "
            "Kaha - sabhi ko e-KYC ki zaroorat nahin; "
            "keval ve log karaen jinaka record adhoora"
        )
        PROMPT_TEXT = f"You are a helpful assistant.<|endofprompt|>{REF_TRANSCRIPT}"

        try:
            start = time.perf_counter()
            output = list(self._model.inference_zero_shot(
                tts_text=text,
                prompt_text=PROMPT_TEXT,
                prompt_wav=str(REF_AUDIO_PATH),
                stream=False,
            ))
            latency = time.perf_counter() - start

            if not output:
                return SynthResult(success=False, error="Model returned empty output")

            audio = output[0]["tts_speech"].numpy().squeeze()
            sr = self._model.sample_rate
            return SynthResult(
                success=True,
                latency_s=round(latency, 3),
                audio=audio,
                sample_rate=sr,
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
