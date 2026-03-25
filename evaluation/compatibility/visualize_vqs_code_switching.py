"""
Generate visualizations of VQS peaks vs code-switch boundaries.

Creates plots showing:
  - MFCC spectrogram
  - Discontinuity peaks with threshold
  - Code-switch boundaries
  - Peak-to-boundary alignment

Usage:
    python -m evaluation.compatibility.visualize_vqs_code_switching \
        --model qwen3_tts --examples T06 T16 T18 --output /tmp/plots
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import cm
except ImportError:
    raise ImportError("matplotlib is required: pip install matplotlib")

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
                "language_tags": row["language_tags"].split(),
            }
    return test_set


def compute_vqs_metrics(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_mfcc: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MFCC and normalized discontinuities."""
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


def align_language_tags_to_time(tags: list[str], duration: float) -> list[dict]:
    """Align discrete language tags to continuous time."""
    n_tags = len(tags)
    segment_duration = duration / n_tags

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


def create_visualization(
    model: str,
    test_id: str,
    test_info: dict,
    output_dir: Path,
) -> Path | None:
    """
    Create a detailed visualization for one utterance.

    Plots:
      1. MFCC spectrogram
      2. Discontinuity peaks (raw and threshold)
      3. Code-switch boundaries highlighted
    """
    audio_path = RESULTS_DIR / model / "audio" / f"{test_id}_mixed.wav"
    if not audio_path.exists():
        return None

    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        duration = len(y) / sr

        if duration < 0.2:
            return None

        hop_length = 512
        mfcc, frame_disc, times = compute_vqs_metrics(y, sr, hop_length=hop_length)
        language_segments = align_language_tags_to_time(
            test_info["language_tags"], duration
        )

        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1.5], hspace=0.35)

        # ─────────────────────────────────────────────────────────────
        # Plot 1: MFCC Spectrogram
        # ─────────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])

        # MFCC as image
        mfcc_db = librosa.power_to_db(np.abs(mfcc), ref=np.max)
        im = ax1.imshow(
            mfcc_db,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, duration, 0, mfcc.shape[0]],
        )

        # Overlay language segments as vertical bands
        for segment in language_segments:
            color = "blue" if segment["language"] == "HI" else "red"
            alpha = 0.15
            ax1.axvspan(
                segment["start"],
                segment["end"],
                alpha=alpha,
                color=color,
                label=segment["language"] if segment["index"] == 0 else "",
            )

        # Mark code-switch boundaries
        for i in range(len(language_segments) - 1):
            if language_segments[i]["language"] != language_segments[i + 1]["language"]:
                boundary_time = language_segments[i]["end"]
                ax1.axvline(boundary_time, color="yellow", linestyle="--", linewidth=2, alpha=0.8)

        ax1.set_ylabel("MFCC Coefficient")
        ax1.set_title(
            f"{test_id} — {test_info['category']}\n{test_info['note']}\n"
            f"Text: {test_info['text_mixed']}",
            fontsize=10,
            fontweight="bold",
        )
        ax1.set_xlim(0, duration)
        cbar = plt.colorbar(im, ax=ax1, label="Power (dB)")

        # Legend for language segments
        blue_patch = mpatches.Patch(color="blue", alpha=0.2, label="Hindi (HI)")
        red_patch = mpatches.Patch(color="red", alpha=0.2, label="English (EN)")
        yellow_line = mpatches.Patch(color="yellow", label="Code-Switch Boundary")
        ax1.legend(handles=[blue_patch, red_patch, yellow_line], loc="upper right", fontsize=9)

        # ─────────────────────────────────────────────────────────────
        # Plot 2: Language Segments with Tag Sequence
        # ─────────────────────────────────────────────────────────────
        ax_seg = fig.add_subplot(gs[1])
        ax_seg.set_xlim(0, duration)
        ax_seg.set_ylim(-0.5, 1.5)

        for segment in language_segments:
            color = "blue" if segment["language"] == "HI" else "red"
            ax_seg.barh(
                0.5,
                segment["end"] - segment["start"],
                left=segment["start"],
                height=0.8,
                color=color,
                alpha=0.6,
                edgecolor="black",
                linewidth=1,
            )
            # Label in middle of segment
            mid = (segment["start"] + segment["end"]) / 2
            ax_seg.text(mid, 0.5, segment["language"], ha="center", va="center", fontweight="bold")

        ax_seg.set_ylabel("Language")
        ax_seg.set_yticks([])
        ax_seg.set_xlabel("Time (s)")
        ax_seg.grid(axis="x", alpha=0.3)

        # ─────────────────────────────────────────────────────────────
        # Plot 3: Discontinuity Peaks
        # ─────────────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2])

        # Plot discontinuity curve
        ax2.plot(times, frame_disc, color="black", alpha=0.6, linewidth=1.5, label="Discontinuity (normalized)")

        # Plot threshold line
        threshold = 2.0
        ax2.axhline(threshold, color="red", linestyle=":", linewidth=2, label=f"Peak threshold ({threshold}×)")

        # Highlight peaks
        peak_indices = np.where(frame_disc > threshold)[0]
        if len(peak_indices) > 0:
            ax2.scatter(
                times[peak_indices],
                frame_disc[peak_indices],
                color="red",
                s=100,
                marker="x",
                linewidths=2,
                label="Detected peaks",
                zorder=5,
            )

        # Mark code-switch boundaries
        for i in range(len(language_segments) - 1):
            if language_segments[i]["language"] != language_segments[i + 1]["language"]:
                boundary_time = language_segments[i]["end"]
                ax2.axvline(boundary_time, color="yellow", linestyle="--", linewidth=2, alpha=0.8)

        # Overlay language segments
        for segment in language_segments:
            color = "blue" if segment["language"] == "HI" else "red"
            ax2.axvspan(
                segment["start"],
                segment["end"],
                alpha=0.1,
                color=color,
            )

        ax2.set_ylabel("Discontinuity Magnitude")
        ax2.set_xlabel("Time (s)")
        ax2.set_xlim(0, duration)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Save figure
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{test_id}_vqs_analysis.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        return out_path

    except Exception as e:
        print(f"  Error visualizing {test_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Visualize VQS code-switch analysis")
    parser.add_argument("--model", default="qwen3_tts")
    parser.add_argument(
        "--examples",
        nargs="+",
        default=["T06", "T16", "T18"],
        help="Test IDs to visualize (default: T06 T16 T18)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for plots (default: results/{model}/vqs_plots/)",
    )
    args = parser.parse_args()

    test_set = load_test_set()
    output_dir = Path(args.output) if args.output else RESULTS_DIR / args.model / "vqs_plots"

    print(f"\nGenerating VQS Visualizations — {args.model}")
    print("-" * 80)

    for test_id in args.examples:
        if test_id not in test_set:
            print(f"  {test_id:<6} NOT FOUND in test set")
            continue

        test_info = test_set[test_id]
        out_path = create_visualization(args.model, test_id, test_info, output_dir)

        if out_path:
            print(f"  {test_id:<6} → {out_path}")
        else:
            print(f"  {test_id:<6} SKIPPED (audio not found or too short)")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
