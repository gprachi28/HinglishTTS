# evaluation/compatibility/compute_eindex.py
"""
E-Index (English Phonetic Fidelity) — Phase 3.4 metric (NEW).

Mirrors H-Index but measures what fraction of English-tagged tokens are correctly
transcribed by ASR (Whisper). An English token is "correctly recognised" when the
ASR hypothesis matches the reference at that aligned position.

Definition:
  E-Index = (correctly recognised English tokens) / (total English tokens)

Weighting: weighted by English token count per sentence, consistent with H-Index methodology.

Evaluated on the `mixed` script variant where English tokens are in Roman script
(making them easy to identify and verify against ASR output).

Usage:
    python -m evaluation.compatibility.compute_eindex --model qwen3_tts
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


# ── Normalisation ─────────────────────────────────────────────
def normalise(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।॥,\.!?;:\-\"'()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    return normalise(text).split()


# ── Levenshtein alignment (same as H-Index) ─────────────────────
def align_tokens(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((ref[i - 1], hyp[j - 1]))  # substitution
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None))  # deletion
            i -= 1
        else:
            alignment.append((None, hyp[j - 1]))  # insertion
            j -= 1
    alignment.reverse()
    return alignment


# ── E-Index for one sentence ──────────────────────────────────
def compute_sentence_eindex(
    ref_text: str, hyp_text: str, tags: list[str]
) -> dict:
    """
    Returns:
        english_token_count: number of EN-tagged tokens in reference
        correct: number correctly transcribed by ASR
        per_token: list of {ref, hyp, correct} for each EN token
    """
    ref_tokens = tokenise(ref_text)
    hyp_tokens = tokenise(hyp_text)

    if not hyp_text.strip():
        return {"english_token_count": 0, "correct": 0, "per_token": [], "skipped": True}

    if len(ref_tokens) != len(tags):
        return {"english_token_count": 0, "correct": 0, "per_token": [], "skipped": True}

    english_positions = [i for i, tag in enumerate(tags) if tag == "EN"]
    if not english_positions:
        return {"english_token_count": 0, "correct": 0, "per_token": [], "skipped": False}

    alignment = align_tokens(ref_tokens, hyp_tokens)

    # Map ref index → aligned hyp token
    ref_to_hyp: dict[int, str | None] = {}
    ref_idx = 0
    for ref_tok, hyp_tok in alignment:
        if ref_tok is not None:
            ref_to_hyp[ref_idx] = hyp_tok
            ref_idx += 1

    per_token = []
    correct_count = 0
    for pos in english_positions:
        ref_tok = ref_tokens[pos]
        hyp_tok = ref_to_hyp.get(pos)
        is_correct = hyp_tok is not None and hyp_tok == ref_tok
        if is_correct:
            correct_count += 1
        per_token.append({"ref": ref_tok, "hyp": hyp_tok, "correct": is_correct})

    return {
        "english_token_count": len(english_positions),
        "correct": correct_count,
        "per_token": per_token,
        "skipped": False,
    }


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3_tts")
    args = parser.parse_args()

    transcripts_path = RESULTS_DIR / args.model / "transcripts.json"
    if not transcripts_path.exists():
        print(f"Transcripts not found: {transcripts_path}")
        print("Run compute_tpi.py first to generate transcripts.")
        return

    with open(transcripts_path) as f:
        transcripts = json.load(f)

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_rows = {row["test_id"]: row for row in csv.DictReader(f)}

    results = []
    total_english = 0
    total_correct = 0

    print(f"\nE-Index (English Phonetic Fidelity) — {args.model} (mixed variant)")
    print("-" * 75)
    print(f"  {'Test ID':<8} {'Category':<26} {'EN tokens':>10} {'Correct':>8} {'E-Index':>8}")
    print(f"  {'-'*8} {'-'*26} {'-'*10} {'-'*8} {'-'*8}")

    for test_id, row in test_rows.items():
        ref = row["text_mixed"]
        hyp = transcripts.get(f"{test_id}_mixed", "")
        tags = row["language_tags"].split()

        data = compute_sentence_eindex(ref, hyp, tags)

        en_count = data["english_token_count"]
        correct = data["correct"]
        e_idx = round(correct / en_count, 4) if en_count > 0 else None
        skipped = data["skipped"]

        # Weight by english token count when aggregating
        total_english += en_count
        total_correct += correct

        e_str = f"{e_idx:.4f}" if e_idx is not None else ("skip" if skipped else "n/a")
        print(f"  {test_id:<8} {row['category']:<26} {en_count:>10} {correct:>8} {e_str:>8}")

        results.append({
            "test_id": test_id,
            "category": row["category"],
            "ref": ref,
            "hyp": hyp,
            "english_token_count": en_count,
            "correct": correct,
            "e_index": e_idx,
            "per_token": data["per_token"],
            "skipped": skipped,
        })

    weighted_eindex = round(total_correct / total_english, 4) if total_english > 0 else None

    print(f"\n  Total English tokens:   {total_english}")
    print(f"  Correctly recognised:   {total_correct}")
    print(f"\n  E-Index (weighted):     {weighted_eindex}")
    print(f"  (1.0 = all English tokens correctly transcribed)")

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"en": 0, "correct": 0}
        categories[cat]["en"] += r["english_token_count"]
        categories[cat]["correct"] += r["correct"]

    print("\n  Per-category E-Index:")
    for cat, data in sorted(categories.items()):
        cat_e = round(data["correct"] / data["en"], 4) if data["en"] > 0 else None
        bar = ""
        if cat_e is not None:
            filled = int(cat_e * 20)
            bar = f"  [{'█' * filled}{'░' * (20 - filled)}]"
        print(f"    {cat:<28} {str(cat_e):>6}{bar}")

    output = {
        "model": args.model,
        "variant": "mixed",
        "weighted_eindex": weighted_eindex,
        "total_english_tokens": total_english,
        "total_correct": total_correct,
        "per_category": {
            cat: round(d["correct"] / d["en"], 4) if d["en"] > 0 else None
            for cat, d in categories.items()
        },
        "per_sentence": results,
    }

    out_path = RESULTS_DIR / args.model / "eindex.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
