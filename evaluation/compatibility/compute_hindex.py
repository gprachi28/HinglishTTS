# evaluation/compatibility/compute_hindex.py
"""
H-Index (Phonetic Fidelity) — Phase 3.4 metric.

Measures what fraction of Hindi-tagged tokens are correctly transcribed
by ASR (Whisper). A Hindi token is "correctly recognised" when the
ASR hypothesis matches the reference at that aligned position.

Definition (from project plan):
  H-Index = (correctly recognised Hindi tokens) / (total Hindi tokens)

Weighting: weighted by Hindi token count per sentence, so a CS-07
intraword sentence with a single Hindi token doesn't bias the aggregate
the same as a long Hindi-matrix sentence.

Evaluated on the `mixed` script variant where Hindi tokens are in
Devanagari (making them easy to identify and verify against ASR output).

Usage:
    python -m evaluation.compatibility.compute_hindex --model qwen3_tts
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


# ── Levenshtein alignment (same as PIER) ─────────────────────
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


# ── H-Index for one sentence ──────────────────────────────────
def compute_sentence_hindex(
    ref_text: str, hyp_text: str, tags: list[str]
) -> dict:
    """
    Returns:
        hindi_token_count: number of HI-tagged tokens in reference
        correct: number correctly transcribed by ASR
        per_token: list of {ref, hyp, correct} for each HI token
    """
    ref_tokens = tokenise(ref_text)
    hyp_tokens = tokenise(hyp_text)

    if not hyp_text.strip():
        return {"hindi_token_count": 0, "correct": 0, "per_token": [], "skipped": True}

    if len(ref_tokens) != len(tags):
        return {"hindi_token_count": 0, "correct": 0, "per_token": [], "skipped": True}

    hindi_positions = [i for i, tag in enumerate(tags) if tag == "HI"]
    if not hindi_positions:
        return {"hindi_token_count": 0, "correct": 0, "per_token": [], "skipped": False}

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
    for pos in hindi_positions:
        ref_tok = ref_tokens[pos]
        hyp_tok = ref_to_hyp.get(pos)
        is_correct = hyp_tok is not None and hyp_tok == ref_tok
        if is_correct:
            correct_count += 1
        per_token.append({"ref": ref_tok, "hyp": hyp_tok, "correct": is_correct})

    return {
        "hindi_token_count": len(hindi_positions),
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
    total_hindi = 0
    total_correct = 0

    print(f"\nH-Index (Phonetic Fidelity) — {args.model} (mixed variant)")
    print("-" * 75)
    print(f"  {'Test ID':<8} {'Category':<26} {'HI tokens':>10} {'Correct':>8} {'H-Index':>8}")
    print(f"  {'-'*8} {'-'*26} {'-'*10} {'-'*8} {'-'*8}")

    for test_id, row in test_rows.items():
        ref = row["text_mixed"]
        hyp = transcripts.get(f"{test_id}_mixed", "")
        tags = row["language_tags"].split()

        data = compute_sentence_hindex(ref, hyp, tags)

        hi_count = data["hindi_token_count"]
        correct = data["correct"]
        h_idx = round(correct / hi_count, 4) if hi_count > 0 else None
        skipped = data["skipped"]

        # Weight by hindi token count when aggregating
        total_hindi += hi_count
        total_correct += correct

        h_str = f"{h_idx:.4f}" if h_idx is not None else ("skip" if skipped else "n/a")
        print(f"  {test_id:<8} {row['category']:<26} {hi_count:>10} {correct:>8} {h_str:>8}")

        results.append({
            "test_id": test_id,
            "category": row["category"],
            "ref": ref,
            "hyp": hyp,
            "hindi_token_count": hi_count,
            "correct": correct,
            "h_index": h_idx,
            "per_token": data["per_token"],
            "skipped": skipped,
        })

    weighted_hindex = round(total_correct / total_hindi, 4) if total_hindi > 0 else None

    print(f"\n  Total Hindi tokens:    {total_hindi}")
    print(f"  Correctly recognised:  {total_correct}")
    print(f"\n  H-Index (weighted):    {weighted_hindex}")
    print(f"  (1.0 = all Hindi tokens correctly transcribed)")

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"hi": 0, "correct": 0}
        categories[cat]["hi"] += r["hindi_token_count"]
        categories[cat]["correct"] += r["correct"]

    print("\n  Per-category H-Index:")
    for cat, data in sorted(categories.items()):
        cat_h = round(data["correct"] / data["hi"], 4) if data["hi"] > 0 else None
        bar = ""
        if cat_h is not None:
            filled = int(cat_h * 20)
            bar = f"  [{'█' * filled}{'░' * (20 - filled)}]"
        print(f"    {cat:<28} {str(cat_h):>6}{bar}")

    output = {
        "model": args.model,
        "variant": "mixed",
        "weighted_hindex": weighted_hindex,
        "total_hindi_tokens": total_hindi,
        "total_correct": total_correct,
        "per_category": {
            cat: round(d["correct"] / d["hi"], 4) if d["hi"] > 0 else None
            for cat, d in categories.items()
        },
        "per_sentence": results,
    }

    out_path = RESULTS_DIR / args.model / "hindex.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
