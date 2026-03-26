# evaluation/compatibility/compute_hnr.py
"""
HNR (Harmonics-to-Noise Ratio) — perceptual voice quality metric.

Measures the ratio of periodic (harmonic) energy to aperiodic (noise) energy
in the synthesized speech. HNR captures absolute voice quality — breathiness,
hoarseness, and synthesis artifacts that make a voice sound unnatural regardless
of how consistent it is frame-to-frame.

HNR interpretation (Praat standard):
  > 20 dB  — excellent quality (clean, natural voice)
  15–20 dB — good quality
  10–15 dB — moderate (some breathiness / noise)
  <  10 dB — poor quality (heavy noise, synthesis artifacts)

Why HNR over MFCC-continuity:
  MFCC stability is self-normalizing and relative — two models with very
  different absolute quality can score identically. HNR is an absolute dB
  measure that reflects the true signal-to-noise quality of the synthesis.

Dependencies: praat-parselmouth, librosa, numpy

Usage:
    python -m evaluation.compatibility.compute_hnr --model sarvam_tts
    python -m evaluation.compatibility.compute_hnr --model qwen3_tts
    python -m evaluation.compatibility.compute_hnr --model sarvam_tts --limit 20
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
SCRIPT_VARIANTS = ["roman", "mixed"]


def compute_hnr(wav_path: Path) -> dict | None:
    """
    Compute HNR for a single WAV file using Praat via parselmouth.

    Returns dict with:
      - mean_hnr_db:   mean HNR across voiced frames (dB)
      - median_hnr_db: median HNR (more robust to outliers)
      - min_hnr_db:    worst frame (captures synthesis artifacts)
      - voiced_fraction: fraction of frames that are voiced
    Returns None if audio is too short or entirely unvoiced.
    """
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        raise ImportError("praat-parselmouth required: pip install praat-parselmouth")

    snd = parselmouth.Sound(str(wav_path))

    if snd.duration < 0.1:
        return None

    # Praat HNR via autocorrelation — standard Praat settings for speech
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)

    # Extract HNR values per frame; Praat returns --undefined-- (nan) for unvoiced
    n_frames = call(harmonicity, "Get number of frames")
    all_values = []
    voiced_values = []
    for i in range(1, n_frames + 1):
        v = call(harmonicity, "Get value in frame", i)
        if not (v != v):  # exclude Python nan
            all_values.append(v)
            if v > -150:  # Praat uses -200 dB as noise floor marker for silence/unvoiced
                voiced_values.append(v)

    if len(voiced_values) < 3:
        return None

    hnr_arr = np.array(voiced_values)
    voiced_fraction = round(len(voiced_values) / n_frames, 4)

    return {
        "mean_hnr_db":    round(float(np.mean(hnr_arr)), 3),
        "median_hnr_db":  round(float(np.median(hnr_arr)), 3),
        "min_hnr_db":     round(float(np.min(hnr_arr)), 3),
        "voiced_fraction": voiced_fraction,
        "n_voiced_frames": len(voiced_values),
        "n_total_frames":  n_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute HNR voice quality metric")
    parser.add_argument("--model", default="sarvam_tts")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N sentences")
    args = parser.parse_args()

    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    wav_files = sorted(audio_dir.glob("*.wav"))
    if args.limit:
        wav_files = [w for w in wav_files
                     if any(w.stem.startswith(f"T{i:02d}_") for i in range(1, args.limit + 1))]

    print(f"\nHNR (Harmonics-to-Noise Ratio) — {args.model}")
    print(f"{'':55} Higher = better voice quality (dB)")
    print("-" * 85)
    print(f"  {'File':<28} {'Mean HNR':>10} {'Median HNR':>12} {'Min HNR':>10} {'Voiced%':>9}")
    print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*10} {'-'*9}")

    per_file: dict[str, dict] = {}
    variant_vals: dict[str, list] = {v: [] for v in SCRIPT_VARIANTS}

    for wav in wav_files:
        result = compute_hnr(wav)
        stem = wav.stem

        if result:
            per_file[stem] = result
            print(
                f"  {stem:<28} {result['mean_hnr_db']:>10.2f} "
                f"{result['median_hnr_db']:>12.2f} "
                f"{result['min_hnr_db']:>10.2f} "
                f"{result['voiced_fraction']*100:>8.1f}%"
            )
            for v in SCRIPT_VARIANTS:
                if stem.endswith(f"_{v}"):
                    variant_vals[v].append(result["mean_hnr_db"])
        else:
            print(f"  {stem:<28} {'too short / unvoiced':>10}")

    print(f"\n  {'Variant':<15} {'Mean HNR (dB)':>15} {'Median HNR (dB)':>17} {'Std':>8} {'N':>5}")
    print(f"  {'-'*15} {'-'*15} {'-'*17} {'-'*8} {'-'*5}")

    summary: dict[str, dict] = {}
    for v in ["roman", "mixed"]:
        vals = variant_vals[v]
        if vals:
            arr = np.array(vals)
            mean_hnr   = round(float(np.mean(arr)), 3)
            median_hnr = round(float(np.median(arr)), 3)
            std_hnr    = round(float(np.std(arr)), 3)
            print(f"  {v:<15} {mean_hnr:>15.2f} {median_hnr:>17.2f} {std_hnr:>8.2f} {len(vals):>5}")
        else:
            mean_hnr = median_hnr = std_hnr = None
            print(f"  {v:<15} {'—':>15} {'—':>17} {'—':>8} {'0':>5}  (no audio)")
        summary[v] = {
            "mean_hnr_db":   mean_hnr,
            "median_hnr_db": median_hnr,
            "std_hnr_db":    std_hnr,
            "n": len(vals),
        }

    print(f"""
  Interpretation (Praat standard):
    > 20 dB  — excellent (clean, natural voice)
    15–20 dB — good quality
    10–15 dB — moderate (some breathiness / artifacts)
    <  10 dB — poor (heavy noise / synthesis breakdown)
""")

    output = {
        "model": args.model,
        "metric": "harmonics_to_noise_ratio",
        "variant_summary": summary,
        "per_file": per_file,
    }

    out_path = RESULTS_DIR / args.model / "hnr.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Full results → {out_path}")


if __name__ == "__main__":
    main()
