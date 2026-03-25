# evaluation/compatibility/compute_cspi.py
"""
CSPI (Code-Switching Phonetic Index) — Step 3 Final Metric.

Combines four complementary dimensions into a single balanced metric:
  - H-Index: % Hindi tokens correctly recognized
  - E-Index: % English tokens correctly recognized
  - H-Phoneme-Accuracy: % Hindi phonemes correctly pronounced
  - E-Phoneme-Accuracy: % English phonemes correctly pronounced

CSPI = α·H-Index + β·E-Index + γ·H-Phoneme + δ·E-Phoneme

Default weighting (equal): α=β=γ=δ=0.25

This metric captures both token-level recognition AND phoneme-level accuracy,
addressing the evaluation gap where Fish Audio "meeting"→"making" error is visible
but hidden in token-only metrics.

Usage:
    python -m evaluation.compatibility.compute_cspi --model qwen3_tts
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"


def compute_cspi(model: str, weights: dict = None) -> dict:
    """
    Compute CSPI and comparison metrics for a model.

    Args:
        model: Model name (qwen3_tts, fish_audio_s2, xtts_v2, cosyvoice3)
        weights: Dict with keys 'h_index', 'e_index', 'h_phoneme', 'e_phoneme'
                 Default: equal weights (0.25 each)
    """
    if weights is None:
        weights = {
            'h_index': 0.25,
            'e_index': 0.25,
            'h_phoneme': 0.25,
            'e_phoneme': 0.25,
        }

    model_dir = RESULTS_DIR / model

    # Load individual metrics
    try:
        with open(model_dir / "hindex.json") as f:
            hindex_data = json.load(f)
        h_index = hindex_data.get("weighted_hindex", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        h_index = None

    try:
        with open(model_dir / "eindex.json") as f:
            eindex_data = json.load(f)
        e_index = eindex_data.get("weighted_eindex", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        e_index = None

    try:
        with open(model_dir / "phoneme_accuracy.json") as f:
            phoneme_data = json.load(f)
        h_phoneme = phoneme_data.get("h_phoneme_accuracy", 0)
        e_phoneme = phoneme_data.get("e_phoneme_accuracy", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        h_phoneme = None
        e_phoneme = None

    # Compute CSPI
    if all(x is not None for x in [h_index, e_index, h_phoneme, e_phoneme]):
        cspi = (
            weights['h_index'] * h_index +
            weights['e_index'] * e_index +
            weights['h_phoneme'] * h_phoneme +
            weights['e_phoneme'] * e_phoneme
        )
    else:
        cspi = None

    return {
        "model": model,
        "h_index": h_index,
        "e_index": e_index,
        "h_phoneme_accuracy": h_phoneme,
        "e_phoneme_accuracy": e_phoneme,
        "cspi": cspi,
        "weights": weights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--h-weight", type=float, default=0.25)
    parser.add_argument("--e-weight", type=float, default=0.25)
    parser.add_argument("--h-phoneme-weight", type=float, default=0.25)
    parser.add_argument("--e-phoneme-weight", type=float, default=0.25)
    args = parser.parse_args()

    weights = {
        'h_index': args.h_weight,
        'e_index': args.e_weight,
        'h_phoneme': args.h_phoneme_weight,
        'e_phoneme': args.e_phoneme_weight,
    }

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    if args.model:
        models = [args.model]
    else:
        models = ["qwen3_tts", "fish_audio_s2", "xtts_v2", "cosyvoice3"]

    print("\n" + "=" * 100)
    print("CSPI (Code-Switching Phonetic Index) — Complete Metric Comparison")
    print("=" * 100)
    print(f"\nWeighting: H-Index={weights['h_index']:.2f}, E-Index={weights['e_index']:.2f}, "
          f"H-Phoneme={weights['h_phoneme']:.2f}, E-Phoneme={weights['e_phoneme']:.2f}")
    print()

    all_results = []
    for model in models:
        result = compute_cspi(model, weights)
        all_results.append(result)

    # Display results
    print(f"{'Model':<18} {'H-Index':>10} {'E-Index':>10} {'H-Phoneme':>12} {'E-Phoneme':>12} {'CSPI':>10}")
    print("-" * 100)

    for result in all_results:
        h_idx = f"{result['h_index']:.4f}" if result['h_index'] is not None else "—"
        e_idx = f"{result['e_index']:.4f}" if result['e_index'] is not None else "—"
        h_phon = f"{result['h_phoneme_accuracy']:.4f}" if result['h_phoneme_accuracy'] is not None else "—"
        e_phon = f"{result['e_phoneme_accuracy']:.4f}" if result['e_phoneme_accuracy'] is not None else "—"
        cspi = f"{result['cspi']:.4f}" if result['cspi'] is not None else "—"

        print(f"{result['model']:<18} {h_idx:>10} {e_idx:>10} {h_phon:>12} {e_phon:>12} {cspi:>10}")

    print()
    print("=" * 100)

    # Ranking by CSPI
    ranked = sorted(
        [r for r in all_results if r['cspi'] is not None],
        key=lambda x: x['cspi'],
        reverse=True
    )

    print("\nCSPI Ranking (Higher is Better):\n")
    for i, result in enumerate(ranked, 1):
        cspi_score = result['cspi']
        bar_len = int(cspi_score * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  {i}. {result['model']:<20} {cspi_score:.4f}  [{bar}]")

    print("\n" + "=" * 100)
    print("Interpretation:")
    print("=" * 100)
    print("""
CSPI combines four dimensions:
  1. H-Index (0.25): Do Hindi tokens get recognized correctly?
  2. E-Index (0.25): Do English tokens get recognized correctly?
  3. H-Phoneme (0.25): Are Hindi phonemes pronounced correctly?
  4. E-Phoneme (0.25): Are English phonemes pronounced correctly?

Higher CSPI = better for balanced Hindi-English code-switching.

Key findings:
  - A model with high H-Index but low E-Index (or vice versa) will have lower CSPI
  - A model with high token recognition but low phoneme accuracy will score lower
  - Example: Fish Audio's "meeting"→"making" error shows as high E-Index but lower E-Phoneme
    """)

    # Save results
    output = {
        "metric": "CSPI",
        "timestamp": "2026-03-25",
        "weights": weights,
        "results": all_results,
        "ranking": [
            {
                "rank": i,
                "model": result['model'],
                "cspi": result['cspi']
            }
            for i, result in enumerate(ranked, 1)
        ]
    }

    out_path = RESULTS_DIR / "cspi_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
