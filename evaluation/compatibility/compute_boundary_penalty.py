# evaluation/compatibility/compute_boundary_penalty.py
"""
Boundary Penalty (BP) — code-switch transition smoothness metric.

Measures how much more acoustically disruptive code-switch boundaries are
compared to within-language transitions. Uses MFCC frame-to-frame distances
but compares two zones:

  - boundary frames: frames estimated to coincide with a language switch
  - within frames:   all other frames

  BP = mean_discontinuity(boundary) / mean_discontinuity(within)

Interpretation:
  BP ≈ 1.0  — boundary transitions as smooth as within-language (ideal)
  BP > 1.0  — boundaries are rougher than within-language segments
  BP >> 1.5 — model struggles noticeably at code-switch points

BP explicitly compares the boundary zone to the within-language baseline,
isolating the code-switching difficulty independent of the model's overall style.

Word timestamp estimation:
  We don't have forced alignment, so word boundaries are estimated by
  assuming uniform speaking rate: word i spans
  [i/N * duration, (i+1)/N * duration] seconds, where N = total words.
  This is a valid approximation for short, fluent Hinglish sentences.

Dependencies: librosa, scipy, numpy

Usage:
    python -m evaluation.compatibility.compute_boundary_penalty --model sarvam_tts
    python -m evaluation.compatibility.compute_boundary_penalty --model qwen3_tts
    python -m evaluation.compatibility.compute_boundary_penalty --model sarvam_tts --limit 20
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean

HERE = Path(__file__).parent
TEST_SET_PATH = HERE / "test_set.csv"
RESULTS_DIR = HERE / "results"

HOP_LENGTH = 512
N_MFCC = 13
BOUNDARY_WINDOW = 2   # frames either side of estimated boundary to include


def mfcc_discontinuities(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return raw (un-normalised) frame-to-frame MFCC Euclidean distances."""
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_smooth = np.array([medfilt(row, kernel_size=3) for row in mfcc])
    distances = np.array([
        euclidean(mfcc_smooth[:, i], mfcc_smooth[:, i + 1])
        for i in range(mfcc_smooth.shape[1] - 1)
    ])
    return distances


def boundary_frames(n_words: int, switch_positions: list[int],
                    n_frames: int, sr: int, duration: float) -> set[int]:
    """
    Estimate which MFCC frames correspond to code-switch word boundaries.

    switch_positions: word indices (0-based) where language changes
    e.g. tags = [HI, HI, EN, EN] → switch at position 2 (HI→EN boundary)
    """
    frames = set()
    for pos in switch_positions:
        # Time of the boundary between word (pos-1) and word (pos)
        t = (pos / n_words) * duration
        center = int(t * sr / HOP_LENGTH)
        for delta in range(-BOUNDARY_WINDOW, BOUNDARY_WINDOW + 1):
            f = center + delta
            if 0 <= f < n_frames:
                frames.add(f)
    return frames


def compute_boundary_penalty(wav_path: Path, tags: list[str]) -> dict | None:
    """
    Compute boundary penalty for one audio file given its language tags.

    Returns None if audio is too short or there are no code switches.
    """
    import librosa

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration = len(y) / sr

    if duration < 0.3:
        return None

    # Find switch positions: index of first word of each new language run
    switch_positions = [
        i for i in range(1, len(tags))
        if tags[i] != tags[i - 1]
    ]
    if not switch_positions:
        return None  # no code switches — sentence is monolingual

    distances = mfcc_discontinuities(y, sr)
    n_frames = len(distances)
    if n_frames < 5:
        return None

    b_frames = boundary_frames(len(tags), switch_positions, n_frames, sr, duration)
    w_frames = set(range(n_frames)) - b_frames

    if not b_frames or not w_frames:
        return None

    boundary_disc = float(np.mean(distances[sorted(b_frames)]))
    within_disc   = float(np.mean(distances[sorted(w_frames)]))

    if within_disc == 0:
        return None

    bp = boundary_disc / within_disc

    return {
        "boundary_penalty":      round(bp, 4),
        "boundary_disc_mean":    round(boundary_disc, 4),
        "within_disc_mean":      round(within_disc, 4),
        "n_switches":            len(switch_positions),
        "n_boundary_frames":     len(b_frames),
        "n_within_frames":       len(w_frames),
        "duration_s":            round(duration, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute code-switch boundary penalty")
    parser.add_argument("--model", default="sarvam_tts")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N sentences")
    args = parser.parse_args()

    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    # Load test set → {test_id: language_tags list}
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    tag_map = {row["test_id"]: row["language_tags"].split() for row in rows}

    wav_files = sorted(audio_dir.glob("*.wav"))
    if args.limit:
        wav_files = [w for w in wav_files
                     if any(w.stem.startswith(f"T{i:02d}_") for i in range(1, args.limit + 1))]

    print(f"\nBoundary Penalty (BP) — {args.model}")
    print(f"  BP ≈ 1.0 = boundary as smooth as within-language  |  BP > 1 = rougher at switches")
    print("-" * 80)
    print(f"  {'File':<28} {'BP':>8} {'B-disc':>10} {'W-disc':>10} {'Switches':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    per_file: dict[str, dict] = {}
    variant_bp: dict[str, list[float]] = {"roman": [], "mixed": []}

    for wav in wav_files:
        stem = wav.stem
        # stem format: T01_roman or T01_mixed
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        test_id, variant = parts[0], parts[1]

        tags = tag_map.get(test_id)
        if tags is None:
            continue

        result = compute_boundary_penalty(wav, tags)

        if result:
            per_file[stem] = result
            bp = result["boundary_penalty"]
            print(
                f"  {stem:<28} {bp:>8.3f}"
                f" {result['boundary_disc_mean']:>10.4f}"
                f" {result['within_disc_mean']:>10.4f}"
                f" {result['n_switches']:>10}"
            )
            if variant in variant_bp:
                variant_bp[variant].append(bp)
        else:
            print(f"  {stem:<28} {'—':>8}  (monolingual / too short)")

    # Summary by variant
    print(f"\n  {'Variant':<15} {'Mean BP':>10} {'Median BP':>12} {'Std':>8} {'N':>5}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*8} {'-'*5}")

    summary: dict[str, dict] = {}
    for v in ["roman", "mixed"]:
        vals = variant_bp[v]
        if vals:
            arr = np.array(vals)
            mean_bp   = round(float(np.mean(arr)), 4)
            median_bp = round(float(np.median(arr)), 4)
            std_bp    = round(float(np.std(arr)), 4)
            print(f"  {v:<15} {mean_bp:>10.3f} {median_bp:>12.3f} {std_bp:>8.3f} {len(vals):>5}")
        else:
            mean_bp = median_bp = std_bp = None
            print(f"  {v:<15} {'—':>10} {'—':>12} {'—':>8} {'0':>5}  (no audio)")
        summary[v] = {
            "mean_bp": mean_bp, "median_bp": median_bp,
            "std_bp": std_bp, "n": len(vals),
        }

    print(f"""
  Interpretation:
    BP ≈ 1.0  — boundary transitions as smooth as within-language (ideal)
    BP 1.0–1.5 — mild roughness at code-switch points
    BP > 1.5  — model noticeably struggles at language boundaries
""")

    output = {
        "model": args.model,
        "metric": "boundary_penalty",
        "boundary_window_frames": BOUNDARY_WINDOW,
        "variant_summary": summary,
        "per_file": per_file,
    }
    out_path = RESULTS_DIR / args.model / "boundary_penalty.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Full results → {out_path}")


if __name__ == "__main__":
    main()
