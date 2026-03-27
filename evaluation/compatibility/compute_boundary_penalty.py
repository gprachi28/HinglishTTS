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

Word timestamp sources (in priority order):
  1. Whisper word timestamps — per-word start/end times from faster-whisper
     (word_timestamps=True). Run compute_word_timestamps.py first to cache
     results in results/{model}/word_timestamps.json.
  2. Uniform partition — fallback when no cache is available.
     Assumes uniform speaking rate: word i spans [i/N * T, (i+1)/N * T].

  When Whisper word count differs from the reference tag count (due to
  insertions/deletions), switch times are interpolated proportionally
  across Whisper's actual word timeline.

Dependencies: librosa, scipy, numpy

Usage:
    # Pre-compute timestamps once (recommended):
    python -m evaluation.compatibility.compute_word_timestamps --model sarvam_tts

    # Then run BP (auto-loads timestamps):
    python -m evaluation.compatibility.compute_boundary_penalty --model sarvam_tts
    python -m evaluation.compatibility.compute_boundary_penalty --model qwen3_tts
    python -m evaluation.compatibility.compute_boundary_penalty --model sarvam_tts --limit 20
    python -m evaluation.compatibility.compute_boundary_penalty --model sarvam_tts --no-whisper
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
BOUNDARY_WINDOW = 2  # frames either side of boundary to include


# ---------------------------------------------------------------------------
# Whisper word-timestamp loader
# ---------------------------------------------------------------------------


def load_whisper_switch_times(
    word_timestamps: list[dict],
    tags: list[str],
    audio_duration: float,
) -> list[float] | None:
    """
    Map language-tag switch positions to real times using Whisper word timestamps.

    word_timestamps: [{"word": str, "start": float, "end": float}, ...]
    tags:            language tag per reference word (e.g. ["HI","HI","EN","EN"])

    Switch time = end of the last word before each language change.

    When Whisper word count != reference tag count, switch positions are
    interpolated proportionally across Whisper's actual word timeline so
    that the relative position within the utterance is preserved.

    Returns None if word_timestamps is empty or there are no switches.
    """
    if not word_timestamps:
        return None

    switch_positions = [i for i in range(1, len(tags)) if tags[i] != tags[i - 1]]
    if not switch_positions:
        return None

    n_ref = len(tags)
    n_asr = len(word_timestamps)
    switch_times = []

    if n_asr == n_ref:
        # Perfect count match — use exact word boundaries
        for pos in switch_positions:
            switch_times.append(word_timestamps[pos - 1]["end"])
    else:
        # Count mismatch — map proportionally through Whisper's word timeline
        for pos in switch_positions:
            # Proportion of reference utterance at this switch
            proportion = (pos - 0.5) / n_ref
            # Corresponding Whisper word index
            asr_idx = min(int(proportion * n_asr), n_asr - 1)
            switch_times.append(word_timestamps[asr_idx]["end"])

    return switch_times if switch_times else None


# ---------------------------------------------------------------------------
# MFCC discontinuity helpers
# ---------------------------------------------------------------------------


def mfcc_discontinuities(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return raw (un-normalised) frame-to-frame MFCC Euclidean distances."""
    import librosa

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_smooth = np.array([medfilt(row, kernel_size=3) for row in mfcc])
    distances = np.array(
        [
            euclidean(mfcc_smooth[:, i], mfcc_smooth[:, i + 1])
            for i in range(mfcc_smooth.shape[1] - 1)
        ]
    )
    return distances


def boundary_frames_from_times(
    switch_times: list[float],
    n_frames: int,
    sr: int,
) -> set[int]:
    """Convert switch times (seconds) to MFCC frame indices ± BOUNDARY_WINDOW."""
    frames: set[int] = set()
    for t in switch_times:
        center = int(t * sr / HOP_LENGTH)
        for delta in range(-BOUNDARY_WINDOW, BOUNDARY_WINDOW + 1):
            f = center + delta
            if 0 <= f < n_frames:
                frames.add(f)
    return frames


def boundary_frames_uniform(
    n_words: int,
    switch_positions: list[int],
    n_frames: int,
    sr: int,
    duration: float,
) -> set[int]:
    """
    Fallback: estimate boundary frames assuming uniform word duration.

    switch_positions: word indices (0-based) where language changes.
    """
    frames: set[int] = set()
    for pos in switch_positions:
        t = (pos / n_words) * duration
        center = int(t * sr / HOP_LENGTH)
        for delta in range(-BOUNDARY_WINDOW, BOUNDARY_WINDOW + 1):
            f = center + delta
            if 0 <= f < n_frames:
                frames.add(f)
    return frames


# ---------------------------------------------------------------------------
# Per-file BP computation
# ---------------------------------------------------------------------------


def compute_boundary_penalty(
    wav_path: Path,
    tags: list[str],
    word_timestamps: list[dict] | None = None,
) -> dict | None:
    """
    Compute boundary penalty for one audio file.

    word_timestamps: optional list of {"word", "start", "end"} dicts from
        Whisper (compute_word_timestamps.py). When provided, real word
        boundaries are used instead of uniform estimation.

    Returns None if audio is too short or there are no code switches.
    """
    import librosa

    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration = len(y) / sr

    if duration < 0.3:
        return None

    switch_positions = [i for i in range(1, len(tags)) if tags[i] != tags[i - 1]]
    if not switch_positions:
        return None

    distances = mfcc_discontinuities(y, sr)
    n_frames = len(distances)
    if n_frames < 5:
        return None

    # Resolve boundary frames — Whisper timestamps preferred
    timestamp_source = "uniform"
    b_frames: set[int] = set()

    if word_timestamps:
        switch_times = load_whisper_switch_times(word_timestamps, tags, duration)
        if switch_times:
            b_frames = boundary_frames_from_times(switch_times, n_frames, sr)
            n_asr = len(word_timestamps)
            timestamp_source = (
                "whisper_exact" if n_asr == len(tags) else "whisper_interp"
            )

    if not b_frames:
        b_frames = boundary_frames_uniform(
            len(tags), switch_positions, n_frames, sr, duration
        )

    w_frames = set(range(n_frames)) - b_frames

    if not b_frames or not w_frames:
        return None

    boundary_disc = float(np.mean(distances[sorted(b_frames)]))
    within_disc = float(np.mean(distances[sorted(w_frames)]))

    if within_disc == 0:
        return None

    bp = boundary_disc / within_disc

    return {
        "boundary_penalty": round(bp, 4),
        "boundary_disc_mean": round(boundary_disc, 4),
        "within_disc_mean": round(within_disc, 4),
        "n_switches": len(switch_positions),
        "n_boundary_frames": len(b_frames),
        "n_within_frames": len(w_frames),
        "duration_s": round(duration, 3),
        "timestamp_source": timestamp_source,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute code-switch boundary penalty")
    parser.add_argument("--model", default="sarvam_tts")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only first N sentences"
    )
    parser.add_argument(
        "--no-whisper",
        action="store_true",
        help="Force uniform timestamp estimation (ignore word_timestamps.json)",
    )
    args = parser.parse_args()

    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    # Load test set → {test_id: language_tags list}
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    tag_map = {row["test_id"]: row["language_tags"].split() for row in rows}

    wav_files = sorted(audio_dir.glob("*.wav"))
    if args.limit:
        wav_files = [
            w
            for w in wav_files
            if any(w.stem.startswith(f"T{i:02d}_") for i in range(1, args.limit + 1))
        ]

    print(f"\nBoundary Penalty (BP) — {args.model}")
    print(
        "  BP ≈ 1.0 = boundary as smooth as within-language  |  BP > 1 = rougher at switches"
    )
    print("-" * 90)
    cols = ("File", "BP", "B-disc", "W-disc", "Switches", "Timestamps")
    header = f"  {cols[0]:<28} {cols[1]:>8} {cols[2]:>10} {cols[3]:>10} {cols[4]:>10} {cols[5]:>12}"
    print(header)
    sep = f"  {'-'*28} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*12}"
    print(sep)

    per_file: dict[str, dict] = {}
    variant_bp: dict[str, list[float]] = {"roman": [], "mixed": []}

    # Load Whisper word timestamps cache if available
    use_whisper = not args.no_whisper
    wts_path = RESULTS_DIR / args.model / "word_timestamps.json"
    word_ts_cache: dict = {}

    if use_whisper and wts_path.exists():
        with open(wts_path) as f:
            word_ts_cache = json.load(f)
        print(f"  Using Whisper word timestamps ({len(word_ts_cache)} entries)")
    elif use_whisper:
        print("  No word_timestamps.json found — using uniform estimation")
        print(
            f"  (Run: python -m evaluation.compatibility.compute_word_timestamps"
            f" --model {args.model})"
        )

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

        word_timestamps = word_ts_cache.get(stem) if use_whisper else None
        result = compute_boundary_penalty(wav, tags, word_timestamps=word_timestamps)

        if result:
            per_file[stem] = result
            bp = result["boundary_penalty"]
            src = result.get("timestamp_source", "uniform")
            print(
                f"  {stem:<28} {bp:>8.3f}"
                f" {result['boundary_disc_mean']:>10.4f}"
                f" {result['within_disc_mean']:>10.4f}"
                f" {result['n_switches']:>10}"
                f" {src:>12}"
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
            mean_bp = round(float(np.mean(arr)), 4)
            median_bp = round(float(np.median(arr)), 4)
            std_bp = round(float(np.std(arr)), 4)
            print(
                f"  {v:<15} {mean_bp:>10.3f} {median_bp:>12.3f} {std_bp:>8.3f} {len(vals):>5}"
            )
        else:
            mean_bp = median_bp = std_bp = None
            print(f"  {v:<15} {'—':>10} {'—':>12} {'—':>8} {'0':>5}  (no audio)")
        summary[v] = {
            "mean_bp": mean_bp,
            "median_bp": median_bp,
            "std_bp": std_bp,
            "n": len(vals),
        }

    print("""
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
