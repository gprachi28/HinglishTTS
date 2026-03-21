# evaluation/compatibility/adapters/fish_audio_s2.py
"""
Fish Audio S2 Pro adapter.

Install:
    hf download fishaudio/fish-audio-s2-pro --local-dir models/fish_audio_s2/weights
    pip install fish-audio-sdk   # update package name if different

Note: HF repo name is a placeholder — verify at huggingface.co/fishaudio
"""

import subprocess
import tempfile
import time
from pathlib import Path

from .base import SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "fish_audio_s2" / "weights"
REPO_DIR = Path(__file__).parents[3] / "models" / "fish_audio_s2"


class FishAudioS2Adapter(TTSAdapter):
    name = "fish_audio_s2"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        return WEIGHTS_DIR.exists()

    def load(self) -> None:
        pass  # lazy-loaded at synthesis time via subprocess

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if not WEIGHTS_DIR.exists():
            return SynthResult(
                success=False,
                error=f"Fish Audio S2 Pro weights not found at {WEIGHTS_DIR}"
            )

        try:
            import os
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "python", "-m", "tools.inference",
                "--text", text,
                "--output", tmp_path,
                "--checkpoint-path", str(WEIGHTS_DIR),
            ]

            start = time.perf_counter()
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(REPO_DIR), timeout=120,
            )
            latency = time.perf_counter() - start

            if result.returncode != 0:
                return SynthResult(
                    success=False,
                    error=f"Subprocess failed: {result.stderr[:500]}",
                )

            audio, sr = sf.read(tmp_path, dtype="float32")
            os.unlink(tmp_path)
            return SynthResult(success=True, latency_s=latency, audio=audio, sample_rate=sr)

        except subprocess.TimeoutExpired:
            return SynthResult(success=False, error="Inference timed out (>120s)")
        except Exception as e:
            return SynthResult(success=False, error=str(e))
