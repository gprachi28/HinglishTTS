# evaluation/compatibility/adapters/fish_speech.py
"""
Fish-Speech 1.5 adapter.

Install:
    cd models/fish_speech
    pip install -e ".[stable]"
    # Download weights:
    huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish_speech/weights

Repo: https://github.com/fishaudio/fish-speech
Supported scripts: Roman, Devanagari (reported), Mixed
"""

import time
from pathlib import Path

from .base import SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "fish_speech" / "weights"
REPO_DIR = Path(__file__).parents[3] / "models" / "fish_speech"


class FishSpeechAdapter(TTSAdapter):
    name = "fish_speech"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        if not WEIGHTS_DIR.exists():
            return False
        try:
            import fish_speech  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        # Fish-Speech uses its own inference CLI; we call it via subprocess
        # or via the Python API if available.
        # Import deferred — only available after pip install -e .
        pass

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        import subprocess
        import tempfile
        import os
        import numpy as np
        import soundfile as sf

        if not WEIGHTS_DIR.exists():
            return SynthResult(
                success=False,
                error=f"Fish-Speech weights not found at {WEIGHTS_DIR}"
            )

        try:
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
                cmd,
                capture_output=True,
                text=True,
                cwd=str(REPO_DIR),
                timeout=120,
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
