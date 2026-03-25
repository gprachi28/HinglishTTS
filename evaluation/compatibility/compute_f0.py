# evaluation/compatibility/compute_f0.py
"""
F0 RMSE proxy — prosodic continuity metric.

Computes the standard deviation of F0 (voiced frames only) for each
synthesized audio file as a proxy for prosodic discontinuity at
code-switch boundaries.

Note: A true Boundary F0 RMSE requires forced alignment to locate switch-point
timestamps in the audio. This metric computes F0 std-dev across the full
utterance as a proxy — higher values indicate more prosodic variability,
which correlates with hard language switches.

Threshold guidance:
  < 15 Hz  — very smooth prosody
  15–30 Hz — moderate variation (natural in conversational speech)
  > 30 Hz  — high discontinuity (likely hard switches at code boundaries)

Dependencies: librosa

Usage:
    python -m evaluation.compatibility.compute_f0 --model cosyvoice3
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
SCRIPT_VARIANTS = ["roman", "devanagari", "mixed"]


def compute_f0_stats(wav_path: Path) -> float | None:
    """
    Returns std-dev of voiced F0 (Hz) for one audio file, or None if the
    file is too short or contains no voiced speech (e.g. silent outputs).
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)

    if len(y) / sr < 0.2:
        return None  # too short to be meaningful (likely silent output)

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
        sr=sr,
        frame_length=2048,
    )

    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) < 10:
        return None  # insufficient voiced frames

    return float(np.std(voiced_f0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3_tts")
    args = parser.parse_args()

    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"\nF0 RMSE proxy — {args.model}")
    print("-" * 55)
    print(f"  {'File':<30} {'F0 std-dev (Hz)':>16}")
    print(f"  {'-'*30} {'-'*16}")

    per_file: dict[str, float | None] = {}
    variant_vals: dict[str, list[float]] = {v: [] for v in SCRIPT_VARIANTS}

    for wav in wav_files:
        val = compute_f0_stats(wav)
        stem = wav.stem
        per_file[stem] = val
        val_str = f"{val:.2f}" if val is not None else "silent/skip"
        print(f"  {stem:<30} {val_str:>16}")

        for v in SCRIPT_VARIANTS:
            if stem.endswith(f"_{v}") and val is not None:
                variant_vals[v].append(val)

    print(f"\n  {'Variant':<15} {'Mean (Hz)':>10} {'Median (Hz)':>12} {'N valid':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*8}")
    summary: dict[str, dict] = {}
    for v in SCRIPT_VARIANTS:
        vals = variant_vals[v]
        if vals:
            mean_v = round(float(np.mean(vals)), 2)
            med_v  = round(float(np.median(vals)), 2)
            print(f"  {v:<15} {mean_v:>10.2f} {med_v:>12.2f} {len(vals):>8}")
        else:
            mean_v = med_v = None
            print(f"  {v:<15} {'—':>10} {'—':>12} {0:>8}")
        summary[v] = {"mean": mean_v, "median": med_v, "n_valid": len(vals)}

    output = {
        "model": args.model,
        "metric": "f0_stddev_proxy",
        "note": "F0 std-dev across voiced frames — proxy for boundary prosodic discontinuity",
        "variant_summary": summary,
        "per_file": {k: (round(v, 4) if v is not None else None) for k, v in per_file.items()},
    }

    out_path = RESULTS_DIR / args.model / "f0.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
