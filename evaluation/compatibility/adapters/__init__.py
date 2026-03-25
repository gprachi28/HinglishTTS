from .base import SynthResult, TTSAdapter
from .fish_audio_s2 import FishAudioS2Adapter
from .qwen3_tts import Qwen3TTSAdapter

__all__ = [
    "TTSAdapter", "SynthResult",
    "FishAudioS2Adapter", "Qwen3TTSAdapter",
]
