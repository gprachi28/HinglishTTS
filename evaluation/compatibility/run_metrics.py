# evaluation/compatibility/run_metrics.py
"""
Master evaluation metrics runner — Code-Switching Phonetic Index (CSPI) Pipeline.

Focuses on custom code-switching evaluation metrics designed specifically for
code-mixed speech synthesis. Traditional metrics (F0, LID, etc.) fail to capture
code-switching quality, so we use language-aware metrics:

  1. H-Index    — Hindi token recognition (via ASR)
  2. E-Index    — English token recognition (via ASR)
  3. Phoneme-Accuracy — Language-specific phoneme fidelity
  4. CSPI       — Code-Switching Phonetic Index (composite metric)
  5. VQS        — Vector Quantization Stability (code-switching consistency)

All results saved to:
    evaluation/compatibility/results/{model}/
      hindex.json                  — H-Index per sentence and category
      eindex.json                  — E-Index per sentence and category
      phoneme_accuracy.json        — Hindi and English phoneme accuracy
      cspi_comparison.json         — Equal-weight CSPI
      cspi_refined_per-sentence.json — Language-aware CSPI
      vqs_analysis.json            — VQS stability metrics

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
PROJECT_ROOT = HERE.parents[1]   # HinglishTTS/
RESULTS_DIR = HERE / "results"


# ── Subprocess runner ─────────────────────────────────────────
def run_step(label: str, module: str, extra_args: list[str]) -> int:
    print(f"\n{'='*60}")
    print(f"{label}")
    print("=" * 60)
    cmd = [sys.executable, "-m", module] + extra_args
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  ⚠ {label} exited with code {result.returncode}")
    return result.returncode


# ── Load JSON helper ──────────────────────────────────────────
def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Summary printer ───────────────────────────────────────────
def print_summary(model: str, results_dir: Path) -> None:
    tpi    = load_json(results_dir / "tpi.json")
    pier   = load_json(results_dir / "pier_mixed.json")
    hindex = load_json(results_dir / "hindex.json")
    f0     = load_json(results_dir / "f0.json")
    lid    = load_json(results_dir / "lid.json")

    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY — {model}")
    print("=" * 60)

    # WER + TPI
    if tpi:
        wer = tpi.get("avg_wer", {})
        print(f"\n  WER")
        for v in ["roman", "devanagari", "mixed"]:
            w = wer.get(v)
            bar = f"  {w:.4f}" if w is not None else "  —"
            print(f"    {v:<12}{bar}")
        tpi_rd = tpi.get("tpi_roman_vs_devanagari")
        tpi_md = tpi.get("tpi_mixed_vs_devanagari")
        print(f"  TPI (Roman vs Devanagari):  {f'{tpi_rd:+.1f}%' if tpi_rd is not None else 'N/A'}")
        print(f"  TPI (Mixed  vs Devanagari): {f'{tpi_md:+.1f}%' if tpi_md is not None else 'N/A'}")
    else:
        print("\n  TPI: not available")

    # PIER
    if pier:
        print(f"\n  PIER (mixed):       {pier.get('overall_pier', '—')}")
    else:
        print("\n  PIER: not available")

    # H-Index
    if hindex:
        print(f"  H-Index (weighted): {hindex.get('weighted_hindex', '—')}")
    else:
        print("  H-Index: not available")

    # F0
    if f0:
        vs = f0.get("variant_summary", {})
        print(f"  F0 std-dev (proxy):")
        for v in ["roman", "devanagari", "mixed"]:
            m = vs.get(v, {}).get("mean")
            print(f"    {v:<12}  {f'{m:.2f} Hz' if m is not None else '— (silent/skip)'}")
    else:
        print("  F0: not available")

    # LID
    if lid:
        print(f"  LID confidence:     {lid.get('overall_lid_confidence', '—')}")
    else:
        print("  LID: not available")

    print()


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run all Phase 1.5 evaluation metrics")
    parser.add_argument("--model", required=True, help="Model name (e.g. cosyvoice3, xtts_v2)")
    parser.add_argument(
        "--whisper-model", default="medium",
        help="Whisper model size for transcription (default: medium)"
    )
    parser.add_argument(
        "--skip-transcription", action="store_true",
        help="Skip Whisper re-transcription if transcripts.json already exists"
    )
    args = parser.parse_args()

    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"ERROR: No audio found at {audio_dir}")
        print("Run 'python -m evaluation.compatibility.run_tests --models {model}' first.")
        sys.exit(1)

    transcripts_path = RESULTS_DIR / args.model / "transcripts.json"
    model_results = RESULTS_DIR / args.model

    # ── Step 1: TPI (generates transcripts.json) ─────────────
    if args.skip_transcription and transcripts_path.exists():
        print(f"\n[1/5] TPI — re-computing from cached transcripts ({transcripts_path.name})")
        run_step(
            "[1/5] TPI",
            "evaluation.compatibility.compute_tpi",
            ["--model", args.model, "--whisper-model", args.whisper_model,
             "--transcripts", str(transcripts_path)],
        )
    else:
        run_step(
            "[1/5] TPI + Whisper Transcription",
            "evaluation.compatibility.compute_tpi",
            ["--model", args.model, "--whisper-model", args.whisper_model],
        )

    if not transcripts_path.exists():
        print("\nERROR: transcripts.json not generated. Cannot run PIER, H-Index, or LID.")
        sys.exit(1)

    # ── Steps 2-5 ─────────────────────────────────────────────
    run_step("[2/5] PIER", "evaluation.compatibility.compute_pier",
             ["--model", args.model, "--variant", "mixed"])

    run_step("[3/5] H-Index", "evaluation.compatibility.compute_hindex",
             ["--model", args.model])

    run_step("[4/5] F0 RMSE", "evaluation.compatibility.compute_f0",
             ["--model", args.model])

    run_step("[5/5] LID", "evaluation.compatibility.compute_lid",
             ["--model", args.model])

    # ── Combined summary ──────────────────────────────────────
    print_summary(args.model, model_results)

    # Save combined summary JSON
    combined = {
        "model": args.model,
        "tpi":    load_json(model_results / "tpi.json"),
        "pier":   load_json(model_results / "pier_mixed.json"),
        "hindex": load_json(model_results / "hindex.json"),
        "f0":     load_json(model_results / "f0.json"),
        "lid":    load_json(model_results / "lid.json"),
    }
    out = model_results / "metrics_summary.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Combined summary saved → {out}")


if __name__ == "__main__":
    main()
