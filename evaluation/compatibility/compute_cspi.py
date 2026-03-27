# evaluation/compatibility/compute_cspi.py
"""
CSPI (Code-Switching Phonetic Index) — Step 3 Final Metric.

Combines four complementary dimensions into a single balanced metric:
  - L1-Index: % L1 (matrix language) tokens correctly recognized
  - L2-Index: % L2 (embedded language) tokens correctly recognized
  - L1-Phoneme-Accuracy: % L1 phonemes correctly pronounced
  - L2-Phoneme-Accuracy: % L2 phonemes correctly pronounced

CSPI = α·L1-Index + β·L2-Index + γ·L1-Phoneme + δ·L2-Phoneme

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
        weights: Dict with keys 'l1_index', 'l2_index', 'l1_phoneme', 'l2_phoneme'
                 Default: equal weights (0.25 each)
    """
    if weights is None:
        weights = {
            'l1_index': 0.25,
            'l2_index': 0.25,
            'l1_phoneme': 0.25,
            'l2_phoneme': 0.25,
        }

    model_dir = RESULTS_DIR / model

    def _load_variant(filename_stem: str, key: str):
        """Load metric averaging across roman and mixed variants if both exist."""
        vals = []
        for v in ["roman", "mixed"]:
            path = model_dir / f"{filename_stem}_{v}.json"
            if path.exists():
                with open(path) as f:
                    vals.append(json.load(f).get(key, 0))
        if not vals:
            # Fallback to legacy single-variant file
            path = model_dir / f"{filename_stem}.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f).get(key)
            return None
        return round(sum(vals) / len(vals), 4)

    # Load individual metrics (averaged across variants)
    l1_index = _load_variant("l1index", "weighted_l1index")
    l2_index = _load_variant("l2index", "weighted_l2index")
    l1_phoneme = _load_variant("phoneme_accuracy", "l1_phoneme_accuracy")
    l2_phoneme = _load_variant("phoneme_accuracy", "l2_phoneme_accuracy")

    # Compute CSPI
    if all(x is not None for x in [l1_index, l2_index, l1_phoneme, l2_phoneme]):
        cspi = (
            weights["l1_index"] * l1_index +
            weights["l2_index"] * l2_index +
            weights["l1_phoneme"] * l1_phoneme +
            weights["l2_phoneme"] * l2_phoneme
        )
    else:
        cspi = None

    return {
        "model": model,
        "l1_index": l1_index,
        "l2_index": l2_index,
        "l1_phoneme_accuracy": l1_phoneme,
        "l2_phoneme_accuracy": l2_phoneme,
        "cspi": cspi,
        "weights": weights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--l1-weight", type=float, default=0.25)
    parser.add_argument("--l2-weight", type=float, default=0.25)
    parser.add_argument("--l1-phoneme-weight", type=float, default=0.25)
    parser.add_argument("--l2-phoneme-weight", type=float, default=0.25)
    args = parser.parse_args()

    weights = {
        'l1_index': args.l1_weight,
        'l2_index': args.l2_weight,
        'l1_phoneme': args.l1_phoneme_weight,
        'l2_phoneme': args.l2_phoneme_weight,
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
    print(f"\nWeighting: L1-Index={weights['l1_index']:.2f}, L2-Index={weights['l2_index']:.2f}, "
          f"H-Phoneme={weights['l1_phoneme']:.2f}, E-Phoneme={weights['l2_phoneme']:.2f}")
    print()

    all_results = []
    for model in models:
        result = compute_cspi(model, weights)
        all_results.append(result)

    # Display results
    print(f"{'Model':<18} {'L1-Index':>10} {'L2-Index':>10} {'L1-Phoneme':>12} {'L2-Phoneme':>12} {'CSPI':>10}")
    print("-" * 100)

    for result in all_results:
        h_idx = f"{result['l1_index']:.4f}" if result['l1_index'] is not None else "—"
        e_idx = f"{result['l2_index']:.4f}" if result['l2_index'] is not None else "—"
        h_phon = f"{result['l1_phoneme_accuracy']:.4f}" if result['l1_phoneme_accuracy'] is not None else "—"
        e_phon = f"{result['l2_phoneme_accuracy']:.4f}" if result['l2_phoneme_accuracy'] is not None else "—"
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
  1. L1-Index (0.25): Do L1 (matrix language) tokens get recognized correctly?
  2. L2-Index (0.25): Do L2 (embedded language) tokens get recognized correctly?
  3. L1-Phoneme (0.25): Are L1 phonemes pronounced correctly?
  4. L2-Phoneme (0.25): Are L2 phonemes pronounced correctly?

Higher CSPI = better for balanced code-switching.

Key findings:
  - A model with high L1-Index but low L2-Index (or vice versa) will have lower CSPI
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
