# evaluation/compatibility/run_metrics.py
"""
Code-Switching Phonetic Index (CSPI) Pipeline — Master metrics runner.

Executes custom evaluation metrics for code-switched speech synthesis:

  1. H-Index    — Hindi token recognition (via ASR)
  2. E-Index    — English token recognition (via ASR)
  3. Phoneme-Accuracy — Language-specific phoneme fidelity
  4. CSPI       — Code-Switching Phonetic Index (composite metric)

Results saved to: evaluation/compatibility/results/{model}/
  - hindex.json
  - eindex.json
  - phoneme_accuracy.json
  - cspi_comparison.json (equal-weight)
  - cspi_refined_per-sentence.json (language-aware)

Usage:
    python -m evaluation.compatibility.run_metrics --model qwen3_tts
    python -m evaluation.compatibility.run_metrics --model fish_audio_s2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parents[1]
RESULTS_DIR = HERE / "results"


def run_step(label: str, module: str, extra_args: list) -> int:
    """Run a metric computation step."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print("=" * 70)
    cmd = [sys.executable, "-m", module] + extra_args
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  ⚠ {label} exited with code {result.returncode}")
    return result.returncode


def load_json(path: Path) -> dict | None:
    """Load JSON results file."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def print_summary(model: str, results_dir: Path) -> None:
    """Print metrics summary."""
    hindex = load_json(results_dir / "hindex.json")
    eindex = load_json(results_dir / "eindex.json")
    phoneme = load_json(results_dir / "phoneme_accuracy.json")
    cspi = load_json(results_dir / "cspi_comparison.json")
    cspi_refined = load_json(results_dir / "cspi_refined_per-sentence.json")

    print(f"\n{'='*70}")
    print(f"CSPI METRICS SUMMARY — {model}")
    print("=" * 70)

    # H-Index
    if hindex:
        h_idx = hindex.get("weighted_hindex", "—")
        print(f"\n  H-Index (Hindi tokens):     {h_idx}")

    # E-Index
    if eindex:
        e_idx = eindex.get("weighted_eindex", "—")
        print(f"  E-Index (English tokens):   {e_idx}")

    # Phoneme Accuracy
    if phoneme:
        h_phon = phoneme.get("h_phoneme_accuracy", "—")
        e_phon = phoneme.get("e_phoneme_accuracy", "—")
        print(f"  H-Phoneme Accuracy:         {h_phon}")
        print(f"  E-Phoneme Accuracy:         {e_phon}")

    # CSPI
    if cspi and "ranking" in cspi:
        for item in cspi["ranking"]:
            if item["model"] == model:
                cspi_score = item.get("cspi", "—")
                print(f"  CSPI (Equal-Weight):        {cspi_score}")
                break

    # CSPI Refined
    if cspi_refined and "results" in cspi_refined:
        for result in cspi_refined["results"]:
            if result["model"] == model:
                cspi_refined_score = result.get("weighted_cspi", "—")
                print(f"  CSPI (Language-Aware):      {cspi_refined_score}")
                break

    print()


def main():
    parser = argparse.ArgumentParser(description="Run CSPI evaluation pipeline")
    parser.add_argument("--model", required=True, help="Model name (qwen3_tts, fish_audio_s2)")
    args = parser.parse_args()

    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"ERROR: No audio found at {audio_dir}")
        print("Run 'python -m evaluation.compatibility.run_tests' first.")
        sys.exit(1)

    model_results = RESULTS_DIR / args.model

    # Run CSPI pipeline steps
    run_step("[1/5] H-Index", "evaluation.compatibility.compute_hindex", ["--model", args.model])
    run_step("[2/5] E-Index", "evaluation.compatibility.compute_eindex", ["--model", args.model])
    run_step("[3/5] Phoneme Accuracy", "evaluation.compatibility.compute_phoneme_accuracy", ["--model", args.model])
    run_step("[4/5] CSPI (Equal-Weight)", "evaluation.compatibility.compute_cspi", ["--model", args.model])
    run_step("[5/5] CSPI (Language-Aware)", "evaluation.compatibility.compute_cspi_refined", ["--model", args.model, "--weighting-mode", "per-sentence"])

    # Print summary
    print_summary(args.model, model_results)

    # Save combined summary
    combined = {
        "model": args.model,
        "hindex": load_json(model_results / "hindex.json"),
        "eindex": load_json(model_results / "eindex.json"),
        "phoneme_accuracy": load_json(model_results / "phoneme_accuracy.json"),
        "cspi": load_json(model_results / "cspi_comparison.json"),
        "cspi_refined": load_json(model_results / "cspi_refined_per-sentence.json"),
    }
    out = model_results / "cspi_summary.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Summary saved → {out}")


if __name__ == "__main__":
    main()
