# evaluation/compatibility/compute_word_timestamps.py
"""
Whisper word-level timestamp extraction for Boundary Penalty alignment.

Runs faster-whisper with word_timestamps=True on all synthesized audio files
and caches per-word start/end times to:

    results/{model}/word_timestamps.json

Format:
    {
      "T01_roman": [
        {"word": "aaj",     "start": 0.00, "end": 0.32},
        {"word": "ka",      "start": 0.32, "end": 0.48},
        ...
      ],
      "T01_mixed": [ ... ],
      ...
    }

compute_boundary_penalty.py reads this file to get exact word boundaries
instead of the uniform-duration fallback.

Usage:
    python -m evaluation.compatibility.compute_word_timestamps --model sarvam_tts
    python -m evaluation.compatibility.compute_word_timestamps --model qwen3_tts
    python -m evaluation.compatibility.compute_word_timestamps --model all
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

WHISPER_MODEL = "medium"
KNOWN_MODELS = ["sarvam_tts", "qwen3_tts", "fish_audio_s2", "xtts_v2", "cosyvoice3"]


def extract_word_timestamps(audio_dir: Path, whisper_model_name: str) -> dict:
    """
    Transcribe all WAVs in audio_dir with word_timestamps=True.

    Returns {stem: [{"word": str, "start": float, "end": float}, ...]}
    """
    from faster_whisper import WhisperModel

    print(f"  Loading faster-whisper '{whisper_model_name}'...")
    model = WhisperModel(whisper_model_name, device="auto", compute_type="auto")

    result: dict[str, list] = {}
    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"  Transcribing {len(wav_files)} files with word timestamps...")

    for wav in wav_files:
        segments, _ = model.transcribe(
            str(wav),
            language=None,
            task="transcribe",
            word_timestamps=True,
        )
        words = []
        for seg in segments:
            if seg.words is None:
                continue
            for w in seg.words:
                words.append(
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 4),
                        "end": round(w.end, 4),
                    }
                )
        result[wav.stem] = words
        preview = " ".join(w["word"] for w in words[:6])
        print(f"    {wav.stem}: [{preview}...] ({len(words)} words)")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Whisper word timestamps for boundary penalty"
    )
    parser.add_argument("--model", default="all", help="Model name or 'all'")
    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help=f"Whisper model size (default: {WHISPER_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if word_timestamps.json already exists",
    )
    args = parser.parse_args()

    models = KNOWN_MODELS if args.model == "all" else [args.model]

    for model in models:
        audio_dir = RESULTS_DIR / model / "audio"
        if not audio_dir.exists():
            print(f"Skipping {model} — no audio at {audio_dir}")
            continue

        out_path = RESULTS_DIR / model / "word_timestamps.json"
        if out_path.exists() and not args.force:
            existing = json.loads(out_path.read_text())
            print(
                f"{model}: word_timestamps.json already exists "
                f"({len(existing)} entries) — use --force to re-run"
            )
            continue

        print(f"\n{'='*60}")
        print(f"Word timestamps — {model}")
        print("=" * 60)

        timestamps = extract_word_timestamps(audio_dir, args.whisper_model)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(timestamps, f, indent=2, ensure_ascii=False)

        print(f"\n  Saved {len(timestamps)} entries → {out_path}")


if __name__ == "__main__":
    main()
