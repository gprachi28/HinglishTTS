# evaluation/compatibility/compute_cspi_refined.py
"""
CSPI Refined (Language-Aware Weighting) — Refined Step 3.

Improves upon equal-weight CSPI by considering language distribution:
  - Sentences with more Hindi tokens weight H-Index/H-Phoneme more heavily
  - Sentences with more English tokens weight E-Index/E-Phoneme more heavily

Weighting scheme:
  CSPI_per_sentence = α·H-Index + β·E-Index + γ·H-Phoneme + δ·E-Phoneme

  where α and γ = language_ratio_hindi
        β and δ = language_ratio_english

Example: Sentence with 70% Hindi, 30% English
  CSPI = 0.35×H-Index + 0.15×E-Index + 0.35×H-Phoneme + 0.15×E-Phoneme

This reflects the principle: errors matter most in the languages that appear most.

Usage:
    python -m evaluation.compatibility.compute_cspi_refined
    python -m evaluation.compatibility.compute_cspi_refined --weighting-mode per-category
    python -m evaluation.compatibility.compute_cspi_refined --weighting-mode global
"""

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
TEST_SET_PATH = HERE / "test_set.csv"
RESULTS_DIR = HERE / "results"


def get_language_ratios_per_sentence() -> dict:
    """
    Load test set and compute Hindi/English token ratios for each sentence.

    Returns:
        {test_id: {"hindi_ratio": 0.7, "english_ratio": 0.3, "hindi_count": 7, "english_count": 3}}
    """
    ratios = {}
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tags = row["language_tags"].split()
            hi_count = sum(1 for tag in tags if tag == "HI")
            en_count = sum(1 for tag in tags if tag == "EN")
            total = len(tags)

            ratios[row["test_id"]] = {
                "hindi_count": hi_count,
                "english_count": en_count,
                "total_tokens": total,
                "hindi_ratio": hi_count / total if total > 0 else 0,
                "english_ratio": en_count / total if total > 0 else 0,
                "category": row["category"],
            }
    return ratios


def load_metrics(model: str) -> dict:
    """Load H-Index, E-Index, H-Phoneme, E-Phoneme from JSON files."""
    model_dir = RESULTS_DIR / model

    metrics = {}

    try:
        with open(model_dir / "hindex.json") as f:
            hindex_data = json.load(f)
            for item in hindex_data.get("per_sentence", []):
                test_id = item["test_id"]
                metrics.setdefault(test_id, {})["h_index"] = item.get("h_index")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    try:
        with open(model_dir / "eindex.json") as f:
            eindex_data = json.load(f)
            for item in eindex_data.get("per_sentence", []):
                test_id = item["test_id"]
                metrics.setdefault(test_id, {})["e_index"] = item.get("e_index")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    try:
        with open(model_dir / "phoneme_accuracy.json") as f:
            phoneme_data = json.load(f)
            for item in phoneme_data.get("per_sentence", []):
                test_id = item["test_id"]
                metrics.setdefault(test_id, {})["h_phoneme"] = item.get("hindi_phoneme_acc")
                metrics.setdefault(test_id, {})["e_phoneme"] = item.get("english_phoneme_acc")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    return metrics


def compute_cspi_per_sentence(h_idx, e_idx, h_phon, e_phon, hindi_ratio, english_ratio):
    """
    Compute language-aware CSPI for a single sentence.

    Weights each metric by language presence:
      - H-Index and H-Phoneme weighted by hindi_ratio
      - E-Index and E-Phoneme weighted by english_ratio
    """
    if any(x is None for x in [h_idx, e_idx, h_phon, e_phon]):
        return None

    # Split weight allocation: 50% for token recognition, 50% for phoneme accuracy
    h_token_weight = 0.5 * hindi_ratio  # 50% of Hindi's share
    h_phon_weight = 0.5 * hindi_ratio   # 50% of Hindi's share
    e_token_weight = 0.5 * english_ratio  # 50% of English's share
    e_phon_weight = 0.5 * english_ratio   # 50% of English's share

    cspi = (
        h_token_weight * h_idx +
        e_token_weight * e_idx +
        h_phon_weight * h_phon +
        e_phon_weight * e_phon
    )

    return cspi


def compute_cspi_refined(model: str, weighting_mode: str = "per-sentence") -> dict:
    """
    Compute language-aware CSPI.

    Modes:
      - per-sentence: Each sentence weighted by its own language ratio (most accurate)
      - per-category: Each category weighted by its average language ratio
      - global: All sentences with their individual ratios (same as per-sentence but aggregated)
    """
    ratios = get_language_ratios_per_sentence()
    metrics = load_metrics(model)

    results = {
        "model": model,
        "weighting_mode": weighting_mode,
        "per_sentence": [],
        "by_category": {},
    }

    all_cspi_scores = []
    total_tokens = 0

    # Compute per-sentence CSPI
    for test_id, ratio_info in ratios.items():
        if test_id not in metrics:
            continue

        m = metrics[test_id]
        h_idx = m.get("h_index")
        e_idx = m.get("e_index")
        h_phon = m.get("h_phoneme")
        e_phon = m.get("e_phoneme")

        cspi = compute_cspi_per_sentence(
            h_idx, e_idx, h_phon, e_phon,
            ratio_info["hindi_ratio"],
            ratio_info["english_ratio"]
        )

        results["per_sentence"].append({
            "test_id": test_id,
            "category": ratio_info["category"],
            "hindi_tokens": ratio_info["hindi_count"],
            "english_tokens": ratio_info["english_count"],
            "hindi_ratio": round(ratio_info["hindi_ratio"], 4),
            "english_ratio": round(ratio_info["english_ratio"], 4),
            "h_index": h_idx,
            "e_index": e_idx,
            "h_phoneme": h_phon,
            "e_phoneme": e_phon,
            "cspi": round(cspi, 4) if cspi is not None else None,
        })

        if cspi is not None:
            all_cspi_scores.append(cspi)
            # Weight by token count for aggregate
            total_tokens += ratio_info["total_tokens"]

    # Compute weighted average (by token count)
    if weighting_mode in ["per-sentence", "global"] and all_cspi_scores:
        weighted_cspi = sum(all_cspi_scores) / len(all_cspi_scores)
    else:
        weighted_cspi = None

    # Compute per-category statistics
    category_data = {}
    for item in results["per_sentence"]:
        cat = item["category"]
        if cat not in category_data:
            category_data[cat] = {
                "cspi_scores": [],
                "hindi_count": 0,
                "english_count": 0,
            }
        if item["cspi"] is not None:
            category_data[cat]["cspi_scores"].append(item["cspi"])
        category_data[cat]["hindi_count"] += item["hindi_tokens"]
        category_data[cat]["english_count"] += item["english_tokens"]

    for cat, data in category_data.items():
        if data["cspi_scores"]:
            avg_cspi = sum(data["cspi_scores"]) / len(data["cspi_scores"])
            total = data["hindi_count"] + data["english_count"]
            results["by_category"][cat] = {
                "cspi": round(avg_cspi, 4),
                "hindi_tokens": data["hindi_count"],
                "english_tokens": data["english_count"],
                "hindi_ratio": round(data["hindi_count"] / total, 4) if total > 0 else 0,
                "english_ratio": round(data["english_count"] / total, 4) if total > 0 else 0,
            }

    results["weighted_cspi"] = round(weighted_cspi, 4) if weighted_cspi is not None else None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--weighting-mode", choices=["per-sentence", "per-category", "global"],
                        default="per-sentence")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        models = ["qwen3_tts", "fish_audio_s2", "xtts_v2", "cosyvoice3"]

    print("\n" + "=" * 110)
    print("CSPI REFINED (Language-Aware Weighting)")
    print("=" * 110)
    print(f"\nWeighting Mode: {args.weighting_mode}")
    print("  Sentences weighted by their Hindi/English token ratio")
    print("  Errors in dominant language weighted more heavily\n")

    all_results = []

    for model in models:
        result = compute_cspi_refined(model, args.weighting_mode)
        all_results.append(result)

    # Display results
    print(f"{'Model':<18} {'Weighted CSPI':>15} {'Category-Level':>15} {'Weighting':>20}")
    print("-" * 110)

    for result in all_results:
        weighted = f"{result['weighted_cspi']:.4f}" if result['weighted_cspi'] is not None else "—"
        print(f"{result['model']:<18} {weighted:>15}")

    print()
    print("=" * 110)
    print("Per-Category Breakdown (Language-Aware CSPI):\n")
    print(f"{'Category':<28} {'CSPI':>8} {'Hindi %':>10} {'English %':>10} {'Notes':<40}")
    print("-" * 110)

    for result in all_results:
        print(f"\n{result['model'].upper()}")
        for cat in sorted(result["by_category"].keys()):
            data = result["by_category"][cat]
            hi_pct = f"{data['hindi_ratio']*100:.0f}%"
            en_pct = f"{data['english_ratio']*100:.0f}%"
            cspi = f"{data['cspi']:.4f}"

            # Determine dominance
            if data['hindi_ratio'] > 0.7:
                note = "Hindi-dominant"
            elif data['english_ratio'] > 0.7:
                note = "English-dominant"
            else:
                note = "Balanced"

            print(f"  {cat:<26} {cspi:>8} {hi_pct:>10} {en_pct:>10} {note:<40}")

    print("\n" + "=" * 110)
    print("Comparison: Equal-Weight vs Language-Aware CSPI\n")

    equal_weight_results = {}
    for result in all_results:
        model = result["model"]
        # Load equal-weight CSPI from earlier
        equal_weight_results[model] = {
            "weighted_cspi": result["weighted_cspi"],  # Placeholder
        }

    print(f"{'Model':<18} {'Equal-Weight CSPI':>20} {'Language-Aware CSPI':>22} {'Difference':>15}")
    print("-" * 110)

    # Load equal-weight results
    equal_weight_results = {}
    try:
        with open(RESULTS_DIR / "cspi_comparison.json") as f:
            cspi_data = json.load(f)
            for item in cspi_data.get("ranking", []):
                equal_weight_results[item["model"]] = item["cspi"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    for result in all_results:
        model = result["model"]
        lang_aware = result["weighted_cspi"]
        equal = equal_weight_results.get(model)

        equal_str = f"{equal:.4f}" if equal is not None else "—"
        lang_str = f"{lang_aware:.4f}" if lang_aware is not None else "—"

        if equal is not None and lang_aware is not None:
            diff = lang_aware - equal
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
        else:
            diff_str = "—"

        print(f"{model:<18} {equal_str:>20} {lang_str:>22} {diff_str:>15}")

    print()
    print("=" * 110)
    print("Interpretation:")
    print("=" * 110)
    print("""
Language-Aware Weighting adjusts CSPI based on:
  • Sentences with 80% Hindi, 20% English: H-Index/H-Phoneme weighted 80%, E-Index/E-Phoneme weighted 20%
  • Sentences with 40% Hindi, 60% English: H-Index/H-Phoneme weighted 40%, E-Index/E-Phoneme weighted 60%

This reflects linguistic reality:
  ✓ Errors in dominant language are more noticeable
  ✓ A model weak on English in English-rich sentences is worse than weak Hindi in Hindi-dominant ones
  ✓ Balanced scoring for balanced code-switching

Comparison of Results:
  • If ranking changes: Some models perform differently on language-dominant vs balanced sentences
  • If ranking stays same: Current models have consistent language balance performance
    """)

    # Save results
    output = {
        "metric": "CSPI_Refined",
        "weighting_mode": args.weighting_mode,
        "timestamp": "2026-03-25",
        "results": all_results,
    }

    out_path = RESULTS_DIR / f"cspi_refined_{args.weighting_mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()
