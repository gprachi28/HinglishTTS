"""
Code-Switch-Specific VQS Analysis with Visualization.

Correlates VQS discontinuity peaks with actual Hindi-English transitions
in test sentences. Generates visualizations showing:
  - MFCC spectral trajectory
  - Discontinuity peaks (normalized by percentile)
  - Ground-truth code-switch boundaries (from language_tags)
  - Peak-to-boundary alignment metrics

Usage:
    python -m evaluation.compatibility.analyze_vqs_code_switching --model qwen3_tts --output /tmp/vqs_analysis
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import librosa
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
TEST_SET_PATH = HERE / "test_set.csv"


def load_test_set() -> dict[str, dict]:
    """Load test sentences with language tags."""
    test_set = {}
    with open(TEST_SET_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = row["test_id"]
            test_set[test_id] = {
                "category": row["category"],
                "note": row["note"],
                "text_mixed": row["text_mixed"],
                "language_tags": row["language_tags"].split(),  # ['HI', 'EN', 'HI', ...]
            }
    return test_set


def compute_mfcc_and_peaks(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_mfcc: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MFCC and discontinuity peaks.

    Returns:
    - mfcc: (n_mfcc, time_steps)
    - frame_discontinuities: normalized discontinuities
    - times: frame times in seconds
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc_smooth = np.array([medfilt(row, kernel_size=3) for row in mfcc])

    distances = np.array([
        euclidean(mfcc_smooth[:, i], mfcc_smooth[:, i+1])
        for i in range(mfcc_smooth.shape[1] - 1)
    ])

    typical_distance = np.percentile(distances, 75)
    if typical_distance == 0:
        typical_distance = 1.0

    frame_discontinuities = distances / typical_distance
    times = np.arange(len(distances)) * hop_length / sr

    return mfcc, frame_discontinuities, times


def align_language_tags_to_time(
    tags: list[str],
    audio_duration: float,
) -> list[dict]:
    """
    Align discrete language tags to continuous time.

    Assumes tags are uniformly spaced across duration.
    Returns list of dicts with (start_time, end_time, language).
    """
    n_tags = len(tags)
    segment_duration = audio_duration / n_tags

    segments = []
    for i, tag in enumerate(tags):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        segments.append({
            "start": start,
            "end": end,
            "language": tag,
            "index": i,
        })

    return segments


def detect_boundary_peaks(
    frame_discontinuities: np.ndarray,
    times: np.ndarray,
    language_segments: list[dict],
    sr: int = 16000,
    hop_length: int = 512,
    discontinuity_threshold: float = 2.0,
) -> dict[str, Any]:
    """
    Detect peaks and correlate with language boundaries.

    Returns:
    - peaks: list of detected peaks with (time, magnitude, distance_to_boundary)
    - boundary_times: expected code-switch boundary times
    - peak_boundary_alignment: how well peaks align with boundaries
    """
    # Find all peaks above threshold
    peak_indices = np.where(frame_discontinuities > discontinuity_threshold)[0]
    peaks = [
        {
            "time": float(times[i]),
            "magnitude": float(frame_discontinuities[i]),
            "frame": int(i),
        }
        for i in peak_indices
    ]

    # Find language boundaries (transitions)
    boundary_times = []
    for i in range(len(language_segments) - 1):
        curr = language_segments[i]
        next_ = language_segments[i + 1]
        if curr["language"] != next_["language"]:
            boundary_time = curr["end"]
            boundary_times.append({
                "time": boundary_time,
                "transition": f"{curr['language']}→{next_['language']}",
                "indices": (i, i + 1),
            })

    # Align peaks to boundaries
    alignment = []
    for peak in peaks:
        distances_to_boundaries = [
            abs(peak["time"] - b["time"]) for b in boundary_times
        ]
        if distances_to_boundaries:
            min_dist = min(distances_to_boundaries)
            closest_boundary = boundary_times[np.argmin(distances_to_boundaries)]
            alignment.append({
                "peak_time": peak["time"],
                "peak_magnitude": peak["magnitude"],
                "closest_boundary_time": closest_boundary["time"],
                "distance_to_boundary": min_dist,
                "boundary_transition": closest_boundary["transition"],
            })
        else:
            alignment.append({
                "peak_time": peak["time"],
                "peak_magnitude": peak["magnitude"],
                "closest_boundary_time": None,
                "distance_to_boundary": None,
                "boundary_transition": "N/A",
            })

    return {
        "peaks": peaks,
        "boundary_times": boundary_times,
        "alignment": alignment,
    }


def analyze_file(
    model: str,
    test_id: str,
    test_info: dict,
) -> dict[str, Any] | None:
    """
    Analyze a single mixed-script audio file.
    """
    audio_path = RESULTS_DIR / model / "audio" / f"{test_id}_mixed.wav"
    if not audio_path.exists():
        return None

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        duration = len(y) / sr

        if duration < 0.2:
            return None

        mfcc, frame_disc, times = compute_mfcc_and_peaks(y, sr)
        language_segments = align_language_tags_to_time(
            test_info["language_tags"], duration
        )
        peak_analysis = detect_boundary_peaks(
            frame_disc, times, language_segments
        )

        return {
            "test_id": test_id,
            "category": test_info["category"],
            "note": test_info["note"],
            "text_mixed": test_info["text_mixed"],
            "language_tags": test_info["language_tags"],
            "audio_duration": round(float(duration), 3),
            "mfcc_shape": mfcc.shape,
            "n_peaks": len(peak_analysis["peaks"]),
            "n_boundaries": len(peak_analysis["boundary_times"]),
            "boundaries": peak_analysis["boundary_times"],
            "peaks": peak_analysis["peaks"],
            "alignment": peak_analysis["alignment"],
            "alignment_summary": {
                "peaks_at_boundaries": sum(
                    1 for a in peak_analysis["alignment"]
                    if a["distance_to_boundary"] is not None
                    and a["distance_to_boundary"] < 0.1
                ),
                "peaks_within_200ms": sum(
                    1 for a in peak_analysis["alignment"]
                    if a["distance_to_boundary"] is not None
                    and a["distance_to_boundary"] < 0.2
                ),
                "avg_distance_to_nearest_boundary": float(
                    np.mean([
                        a["distance_to_boundary"]
                        for a in peak_analysis["alignment"]
                        if a["distance_to_boundary"] is not None
                    ]) if any(
                        a["distance_to_boundary"] is not None
                        for a in peak_analysis["alignment"]
                    ) else 0.0
                ),
            },
        }
    except Exception as e:
        print(f"  Error analyzing {test_id}_mixed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VQS peaks vs code-switch boundaries"
    )
    parser.add_argument("--model", default="qwen3_tts")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for results (default: results/{model}/)",
    )
    args = parser.parse_args()

    test_set = load_test_set()
    output_dir = Path(args.output) if args.output else RESULTS_DIR / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCode-Switch VQS Analysis — {args.model}")
    print("-" * 80)

    results = []
    for test_id, test_info in sorted(test_set.items()):
        result = analyze_file(args.model, test_id, test_info)
        if result:
            results.append(result)
            print(
                f"  {test_id:<6} {result['category']:<20} "
                f"Peaks: {result['n_peaks']:>2}, Boundaries: {result['n_boundaries']}, "
                f"@Boundaries: {result['alignment_summary']['peaks_at_boundaries']}"
            )
        else:
            print(f"  {test_id:<6} {'SKIPPED':>20}")

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")

    total_peaks = sum(r["n_peaks"] for r in results)
    total_boundaries = sum(r["n_boundaries"] for r in results)
    peaks_at_boundaries = sum(
        r["alignment_summary"]["peaks_at_boundaries"] for r in results
    )
    peaks_near_boundaries = sum(
        r["alignment_summary"]["peaks_within_200ms"] for r in results
    )

    print(f"\n  Total sentences analyzed:        {len(results)}")
    print(f"  Total discontinuity peaks:      {total_peaks}")
    print(f"  Total code-switch boundaries:   {total_boundaries}")
    print(f"  Peaks at boundaries (<100ms):   {peaks_at_boundaries} ({100*peaks_at_boundaries/max(total_peaks, 1):.1f}%)")
    print(f"  Peaks near boundaries (<200ms): {peaks_near_boundaries} ({100*peaks_near_boundaries/max(total_peaks, 1):.1f}%)")

    avg_dist = np.mean([
        a["distance_to_boundary"]
        for r in results
        for a in r["alignment"]
        if a["distance_to_boundary"] is not None
    ])
    print(f"  Mean distance to nearest boundary: {avg_dist:.3f}s")

    # Save detailed results
    out_path = output_dir / "vqs_code_switching_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "summary": {
                "n_sentences": len(results),
                "total_peaks": total_peaks,
                "total_boundaries": total_boundaries,
                "peaks_at_boundaries": peaks_at_boundaries,
                "peaks_within_200ms": peaks_near_boundaries,
                "mean_distance_to_boundary": float(avg_dist),
            },
            "per_sentence": results,
        }, f, indent=2)

    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
