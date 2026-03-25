# evaluation/compatibility/compute_tpi.py
"""
ASR-proxy TPI (Transliteration Performance Index) computation.

Uses Whisper to transcribe synthesized audio, then computes WER against
reference texts. TPI is defined as:

    TPI = (WER_Roman - WER_Devanagari) / WER_Devanagari * 100

A positive TPI means Roman script yields higher WER than Devanagari
(model struggles more with Roman input). Zero means parity.

Usage:
    python -m evaluation.compatibility.compute_tpi --model qwen3_tts
    python -m evaluation.compatibility.compute_tpi --model qwen3_tts --whisper-model medium
"""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
TEST_SET_PATH = HERE / "test_set.csv"
RESULTS_DIR = HERE / "results"

TEXT_COLUMNS = {
    "roman": "text_roman",
    "devanagari": "text_devanagari",
    "mixed": "text_mixed",
}


# ── Text normalisation ────────────────────────────────────────
def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।॥,\.!?;:\-\"'()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Whisper transcription ─────────────────────────────────────
def transcribe_all(audio_dir: Path, whisper_model_name: str) -> dict[str, str]:
    """Return {filename_stem: transcript} for all WAVs in audio_dir."""
    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper {whisper_model_name}...")
    model = WhisperModel(whisper_model_name, device="auto", compute_type="auto")

    transcripts = {}
    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"Transcribing {len(wav_files)} files...")

    for wav in wav_files:
        segments, _ = model.transcribe(str(wav), language=None, task="transcribe")
        text = " ".join(seg.text for seg in segments).strip()
        transcripts[wav.stem] = text
        print(f"  {wav.stem}: {text[:60]}")

    return transcripts


# ── WER ───────────────────────────────────────────────────────
def wer(reference: str, hypothesis: str) -> float:
    from jiwer import wer as _wer
    return _wer(normalise(reference), normalise(hypothesis))


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3_tts")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model size")
    parser.add_argument(
        "--transcripts",
        help="Path to existing transcripts JSON (skip re-running Whisper)",
    )
    args = parser.parse_args()

    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    # Load or generate transcripts
    transcripts_path = RESULTS_DIR / args.model / "transcripts.json"
    if args.transcripts:
        with open(args.transcripts) as f:
            transcripts = json.load(f)
    elif transcripts_path.exists():
        print(f"Loading cached transcripts from {transcripts_path}")
        with open(transcripts_path) as f:
            transcripts = json.load(f)
    else:
        transcripts = transcribe_all(audio_dir, args.whisper_model)
        with open(transcripts_path, "w") as f:
            json.dump(transcripts, f, indent=2, ensure_ascii=False)
        print(f"Saved transcripts → {transcripts_path}")

    # Load test set
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_rows = {row["test_id"]: row for row in csv.DictReader(f)}

    # Compute per-sentence WER for each variant
    results = []
    for test_id, row in test_rows.items():
        row_data = {"test_id": test_id, "category": row["category"]}
        for variant in ["roman", "devanagari", "mixed"]:
            key = f"{test_id}_{variant}"
            ref = row[TEXT_COLUMNS[variant]]
            hyp = transcripts.get(key, "")
            w = wer(ref, hyp) if hyp else 1.0
            row_data[f"wer_{variant}"] = round(w, 4)
            row_data[f"ref_{variant}"] = ref
            row_data[f"hyp_{variant}"] = hyp
        results.append(row_data)

    # Aggregate WER per variant
    avg_wer = {}
    for variant in ["roman", "devanagari", "mixed"]:
        vals = [r[f"wer_{variant}"] for r in results]
        avg_wer[variant] = round(sum(vals) / len(vals), 4)

    # TPI
    wr, wd, wm = avg_wer["roman"], avg_wer["devanagari"], avg_wer["mixed"]
    tpi_roman_vs_dev = round((wr - wd) / wd * 100, 2) if wd > 0 else None
    tpi_mixed_vs_dev = round((wm - wd) / wd * 100, 2) if wd > 0 else None

    # Print report
    print("\n" + "=" * 60)
    print(f"ASR-proxy TPI — {args.model}")
    print("=" * 60)
    print(f"\n  Avg WER  Roman:      {wr:.4f}")
    print(f"  Avg WER  Devanagari: {wd:.4f}")
    print(f"  Avg WER  Mixed:      {wm:.4f}")
    print(f"\n  TPI (Roman vs Devanagari):  {tpi_roman_vs_dev:+.1f}%")
    print(f"  TPI (Mixed  vs Devanagari): {tpi_mixed_vs_dev:+.1f}%")
    print()
    print("  Interpretation:")
    print("  > 0%  → Roman/Mixed harder for model (higher WER)")
    print("  = 0%  → script parity")
    print("  < 0%  → Roman/Mixed easier (lower WER)")

    # Per-category breakdown
    print("\n" + "-" * 60)
    print(f"  {'Test ID':<8} {'Category':<26} {'WER_R':>7} {'WER_D':>7} {'WER_M':>7}")
    print(f"  {'-'*8} {'-'*26} {'-'*7} {'-'*7} {'-'*7}")
    for r in results:
        print(
            f"  {r['test_id']:<8} {r['category']:<26} "
            f"{r['wer_roman']:>7.4f} {r['wer_devanagari']:>7.4f} {r['wer_mixed']:>7.4f}"
        )

    # Save JSON
    output = {
        "model": args.model,
        "whisper_model": args.whisper_model,
        "avg_wer": avg_wer,
        "tpi_roman_vs_devanagari": tpi_roman_vs_dev,
        "tpi_mixed_vs_devanagari": tpi_mixed_vs_dev,
        "per_sentence": results,
    }
    out_path = RESULTS_DIR / args.model / "tpi.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
