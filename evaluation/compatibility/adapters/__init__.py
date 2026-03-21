from .base import SynthResult, TTSAdapter
from .cosyvoice3 import CosyVoice3Adapter
from .fish_audio_s2 import FishAudioS2Adapter
from .glow_tts import GlowTTSAdapter
from .qwen3_tts import Qwen3TTSAdapter
from .xtts_v2 import XTTSV2Adapter

__all__ = [
    "TTSAdapter", "SynthResult",
    "CosyVoice3Adapter", "FishAudioS2Adapter",
    "GlowTTSAdapter", "Qwen3TTSAdapter", "XTTSV2Adapter",
]
