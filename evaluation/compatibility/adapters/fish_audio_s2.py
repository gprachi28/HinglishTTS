# evaluation/compatibility/adapters/fish_audio_s2.py
"""
Fish Audio S2 Pro adapter — voice cloning with reference audio.

Inference uses a 3-step pipeline:
  1. Extract VQ codes from reference audio (DAC codec)
  2. Generate semantic tokens from text + prompt (text2semantic)
  3. Decode semantic tokens back to audio (DAC decoder)

Device detection:
  - macOS with MPS: use --device mps
  - CUDA: use --device cuda
  - CPU fallback: use --device cpu

Install:
    hf download fishaudio/s2-pro --local-dir models/fish_audio_s2/weights
    pip install fish-speech (after commenting out pyaudio in pyproject.toml)
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .base import REF_AUDIO_PATH, SynthResult, TTSAdapter

WEIGHTS_DIR = Path(__file__).parents[3] / "models" / "fish_audio_s2" / "weights"
REPO_DIR = Path(__file__).parents[3] / "models" / "fish_audio_s2" / "repo"

# Reference transcript for voice cloning (from base.py context)
REF_TRANSCRIPT = "Gas connection band hone ki khabaron sun kar bahut chinta ho gayi mujhe."


def get_device() -> str:
    """Detect best device: mps (macOS) > cuda > cpu."""
    try:
        if sys.platform == "darwin":
            import torch
            if torch.backends.mps.is_available():
                return "mps"
    except (ImportError, AttributeError):
        pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, AttributeError):
        pass
    return "cpu"


class FishAudioS2Adapter(TTSAdapter):
    name = "fish_audio_s2"
    supported_scripts = ["roman", "devanagari", "mixed"]

    def __init__(self):
        self._device = get_device()

    def is_available(self) -> bool:
        if not WEIGHTS_DIR.exists() or not REPO_DIR.exists():
            return False
        try:
            import fish_speech  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> None:
        """No eager loading — inference is subprocess-based."""
        pass

    def synthesize(self, text: str, script_variant: str) -> SynthResult:
        if not WEIGHTS_DIR.exists():
            return SynthResult(success=False, error=f"Weights not found: {WEIGHTS_DIR}")
        if not REPO_DIR.exists():
            return SynthResult(success=False, error=f"Repo not found: {REPO_DIR}")
        if not REF_AUDIO_PATH.exists():
            return SynthResult(success=False, error=f"Reference audio not found: {REF_AUDIO_PATH}")

        try:
            import soundfile as sf

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                output_path = tmpdir_path / "output.wav"

                # Fish Speech text2semantic inference
                cmd = [
                    sys.executable, "-m", "fish_speech.models.text2semantic.inference",
                    "--text", text,
                    "--prompt-text", REF_TRANSCRIPT,
                    "--prompt-audio", str(REF_AUDIO_PATH),
                    "--checkpoint-path", str(WEIGHTS_DIR),
                    "--device", self._device,
                    "--output", str(output_path),
                ]

                start = time.perf_counter()
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=str(REPO_DIR), timeout=120, env={**os.environ, "PYTHONPATH": str(REPO_DIR)}
                )
                latency = time.perf_counter() - start

                if result.returncode != 0:
                    stderr = result.stderr[:500] if result.stderr else result.stdout[:500]
                    return SynthResult(success=False, error=f"Inference failed: {stderr}")

                if not output_path.exists():
                    return SynthResult(success=False, error="Output file not created")

                audio, sr = sf.read(str(output_path), dtype="float32")
                return SynthResult(success=True, latency_s=round(latency, 3), audio=audio, sample_rate=sr)

        except subprocess.TimeoutExpired:
            return SynthResult(success=False, error="Inference timed out (>120s)")
        except Exception as e:
            return SynthResult(success=False, error=str(e))

    def unload(self) -> None:
        """Clean up GPU memory if needed."""
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass
