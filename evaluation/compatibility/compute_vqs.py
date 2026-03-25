"""
Voice Quality Stability (VQS) — MFCC-based formant continuity metric.

Computes formant continuity via MFCC analysis as a proxy for voice quality
consistency and accent stability. Tracks frame-to-frame MFCC distance to
identify synthesis artifacts or code-switch boundaries.

Metrics:
  - stability_score: [0, 1] where 1 = perfect continuity (no discontinuities)
  - max_discontinuity: Maximum frame-to-frame MFCC distance (in units of std)
  - num_peaks: Count of anomalous discontinuities (>1.5x typical)
  - code_switch_boundaries: Detected boundaries via discontinuity peaks

Interpretation:
  < 0.7  — poor voice quality (many artifacts or hard discontinuities)
  0.7-0.85 — moderate quality (some unnatural transitions)
  > 0.85 — good quality (smooth, consistent formants)

Dependencies: librosa, scipy, numpy

Usage:
    python -m evaluation.compatibility.compute_vqs --model qwen3_tts
    python -m evaluation.compatibility.compute_vqs --model fish_audio_s2
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
SCRIPT_VARIANTS = ["roman"]  # Roman script only


def compute_mfcc_formant_continuity(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_mfcc: int = 13,
    window_size: int = 3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute voice quality stability via MFCC formant continuity.

    Returns:
    - stability_score: [0, 1] where 1 = perfect continuity
    - mfcc: (n_mfcc, time_steps)
    - frame_discontinuities: per-frame measure of discontinuity
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Apply median filter to smooth noise (formants are slow-moving)
    mfcc_smooth = np.array([medfilt(row, kernel_size=window_size) for row in mfcc])

    # Compute frame-to-frame euclidean distances (discontinuities)
    distances = np.array([
        euclidean(mfcc_smooth[:, i], mfcc_smooth[:, i+1])
        for i in range(mfcc_smooth.shape[1] - 1)
    ])

    # Normalize by typical distance (use 75th percentile as "normal")
    typical_distance = np.percentile(distances, 75)
    if typical_distance == 0:
        typical_distance = 1.0

    # Frame discontinuities: ratio to typical (>1 = unusual)
    frame_discontinuities = distances / typical_distance

    # Stability score: fraction of frames with "normal" continuity
    # (discontinuity < 1.5x typical)
    stability_score = float(np.mean(frame_discontinuities < 1.5))

    return stability_score, mfcc, frame_discontinuities


def detect_code_switch_peaks(
    frame_discontinuities: np.ndarray,
    sr: int,
    hop_length: int = 512,
    discontinuity_threshold: float = 2.0,
) -> list[dict]:
    """
    Detect code-switch-like boundaries as anomalous discontinuity peaks.

    Returns list of detected peaks with (time, magnitude).
    """
    peaks = []
    for i, disc in enumerate(frame_discontinuities):
        if disc > discontinuity_threshold:
            time = i * hop_length / sr
            peaks.append({
                "time": round(time, 3),
                "magnitude": round(float(disc), 3)
            })
    return peaks


def compute_vqs_stats(wav_path: Path) -> dict | None:
    """
    Compute voice quality stability metrics for one audio file.

    Returns dict with stability_score, max_discontinuity, num_peaks, code_switches,
    or None if the file is too short or silent.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)

    if len(y) / sr < 0.2:
        return None  # too short to be meaningful

    try:
        stability_score, mfcc, frame_discontinuities = compute_mfcc_formant_continuity(
            y, sr
        )

        # Detect peaks
        peaks = detect_code_switch_peaks(frame_discontinuities, sr)

        return {
            "stability_score": round(stability_score, 4),
            "max_discontinuity": round(float(np.max(frame_discontinuities)), 3),
            "mean_discontinuity": round(float(np.mean(frame_discontinuities)), 3),
            "num_peaks": len(peaks),
            "detected_boundaries": peaks,
        }
    except Exception as e:
        print(f"  ⚠ Error processing {wav_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compute Voice Quality Stability metric")
    parser.add_argument("--model", default="qwen3_tts")
    args = parser.parse_args()

    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"\nVoice Quality Stability (VQS) — {args.model}")
    print("-" * 80)
    print(f"  {'File':<30} {'Stability':>12} {'Max Disc':>12} {'Peaks':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")

    per_file: dict[str, dict] = {}
    variant_vals: dict[str, list[float]] = {v: [] for v in SCRIPT_VARIANTS}
    variant_peaks: dict[str, list[int]] = {v: [] for v in SCRIPT_VARIANTS}

    for wav in wav_files:
        val = compute_vqs_stats(wav)
        stem = wav.stem

        if val is not None:
            per_file[stem] = val
            stability = val["stability_score"]
            max_disc = val["max_discontinuity"]
            num_peaks = val["num_peaks"]

            print(f"  {stem:<30} {stability:>12.4f} {max_disc:>12.3f} {num_peaks:>8}")

            # Aggregate by variant
            for v in SCRIPT_VARIANTS:
                if stem.endswith(f"_{v}"):
                    variant_vals[v].append(stability)
                    variant_peaks[v].append(num_peaks)
        else:
            print(f"  {stem:<30} {'silent/skip':>12} {'-':>12} {'-':>8}")

    print(f"\n  {'Variant':<15} {'Mean Stability':>16} {'Median':>10} {'Peaks (mean)':>14} {'N valid':>8}")
    print(f"  {'-'*15} {'-'*16} {'-'*10} {'-'*14} {'-'*8}")

    summary: dict[str, dict] = {}
    for v in SCRIPT_VARIANTS:
        vals = variant_vals[v]
        peaks = variant_peaks[v]
        if vals:
            mean_val = round(float(np.mean(vals)), 4)
            med_val  = round(float(np.median(vals)), 4)
            mean_peaks = round(float(np.mean(peaks)), 2) if peaks else 0.0
            print(f"  {v:<15} {mean_val:>16.4f} {med_val:>10.4f} {mean_peaks:>14.2f} {len(vals):>8}")
        else:
            print(f"  {v:<15} {'—':>16} {'—':>10} {'—':>14} {0:>8}")
            mean_val = med_val = mean_peaks = None

        summary[v] = {
            "mean_stability": mean_val,
            "median_stability": med_val,
            "mean_peaks": mean_peaks,
            "n_valid": len(vals)
        }

    output = {
        "model": args.model,
        "metric": "voice_quality_stability",
        "description": "MFCC-based formant continuity (proxy for accent/quality consistency)",
        "interpretation": {
            "score_lt_0.7": "poor quality (many artifacts/discontinuities)",
            "score_0.7_0.85": "moderate quality (some unnatural transitions)",
            "score_gt_0.85": "good quality (smooth, consistent formants)",
        },
        "variant_summary": summary,
        "per_file": per_file,
    }

    out_path = RESULTS_DIR / args.model / "vqs.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
