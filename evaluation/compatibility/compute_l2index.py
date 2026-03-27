# evaluation/compatibility/compute_l2index.py
"""
L2-Index (Embedded Language Phonetic Fidelity) — Phase 3.4 metric (NEW).

Mirrors L1-Index but measures what fraction of L2-tagged tokens are correctly
transcribed by ASR (Whisper). An English token is "correctly recognised" when the
ASR hypothesis matches the reference at that aligned position.

Definition:
  L2-Index = (correctly recognised L2 tokens) / (total L2 tokens)

Weighting: weighted by L2 token count per sentence, consistent with L1-Index methodology.

Evaluated on the `mixed` script variant where English tokens are in Roman script
(making them easy to identify and verify against ASR output).

Usage:
    python -m evaluation.compatibility.compute_l2index --model qwen3_tts
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

HERE = Path(__file__).parent
TEST_SET_PATH = HERE / "test_set.csv"
RESULTS_DIR = HERE / "results"

# Import devanagari_map for hand-curated Roman→Devanagari conversion
sys.path.insert(0, str(HERE.parents[1] / "data"))
from devanagari_map import transliterate_hindi


# ── Text transliteration (Roman → Devanagari) ──────────────────────
def transliterate_roman_to_devanagari(text: str) -> str:
    """Convert Roman Hinglish to Devanagari using hand-curated dictionary."""
    words = text.split()
    transliterated = [transliterate_hindi(word) for word in words]
    return " ".join(transliterated)


# ── Character-level similarity ─────────────────────────────────────
def char_similarity(ref: str, hyp: str) -> float:
    """Compare tokens at character level using SequenceMatcher.

    Returns: ratio of matching characters (0.0 = completely different, 1.0 = identical)
    """
    if not ref or not hyp:
        return 1.0 if ref == hyp else 0.0
    return SequenceMatcher(None, ref, hyp).ratio()


# ── Normalisation ─────────────────────────────────────────────
def normalise(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।॥,\.!?;:\-\"'()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    return normalise(text).split()


# ── Levenshtein alignment (same as L1-Index) ─────────────────────
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


# ── L2-Index for one sentence ──────────────────────────────────
def compute_sentence_l2index(
    ref_text: str, hyp_text: str, tags: list[str]
) -> dict:
    """
    Returns:
        l2_token_count: number of EN-tagged tokens in reference
        correct: number correctly transcribed by ASR
        per_token: list of {ref, hyp, correct} for each EN token
    """
    ref_tokens = tokenise(ref_text)
    hyp_tokens = tokenise(hyp_text)

    if not hyp_text.strip():
        return {"l2_token_count": 0, "correct": 0, "per_token": [], "skipped": True}

    if len(ref_tokens) != len(tags):
        return {"l2_token_count": 0, "correct": 0, "per_token": [], "skipped": True}

    l2_positions = [i for i, tag in enumerate(tags) if tag == "EN"]
    if not l2_positions:
        return {"l2_token_count": 0, "correct": 0, "per_token": [], "skipped": False}

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
    char_sim_threshold = 0.7  # Require 70% character overlap
    for pos in l2_positions:
        ref_tok = ref_tokens[pos]
        hyp_tok = ref_to_hyp.get(pos)
        # Use character-level similarity instead of exact match
        similarity = char_similarity(ref_tok, hyp_tok) if hyp_tok is not None else 0.0
        is_correct = similarity >= char_sim_threshold
        if is_correct:
            correct_count += 1
        per_token.append({
            "ref": ref_tok,
            "hyp": hyp_tok,
            "char_similarity": round(similarity, 4),
            "correct": is_correct
        })

    return {
        "l2_token_count": len(l2_positions),
        "correct": correct_count,
        "per_token": per_token,
        "skipped": False,
    }


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3_tts")
    parser.add_argument("--variant", default="roman", choices=["roman", "mixed"])
    parser.add_argument("--limit", type=int, default=None, help="Process only first N sentences")
    args = parser.parse_args()

    transcripts_path = RESULTS_DIR / args.model / "transcripts.json"
    if not transcripts_path.exists():
        print(f"Transcripts not found: {transcripts_path}")
        print("Run compute_tpi.py first to generate transcripts.")
        return

    with open(transcripts_path) as f:
        transcripts = json.load(f)

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if args.limit:
            rows = rows[:args.limit]
        test_rows = {row["test_id"]: row for row in rows}

    results = []
    total_l2 = 0
    total_correct = 0

    print(f"\nL2-Index (Embedded Language Phonetic Fidelity) — {args.model} ({args.variant} variant)")
    print("-" * 75)
    print(f"  {'Test ID':<8} {'Category':<26} {'L2 tokens':>10} {'Correct':>8} {'E-Index':>8}")
    print(f"  {'-'*8} {'-'*26} {'-'*10} {'-'*8} {'-'*8}")

    for test_id, row in test_rows.items():
        ref = row["text_roman"]
        ref_normalized = transliterate_roman_to_devanagari(ref)
        hyp_raw = transcripts.get(f"{test_id}_{args.variant}", "")
        hyp = transliterate_roman_to_devanagari(hyp_raw)  # normalize: handles Whisper Roman↔Devanagari inconsistency
        tags = row["language_tags"].split()

        # Compare using normalized (Devanagari) reference
        data = compute_sentence_l2index(ref_normalized, hyp, tags)

        en_count = data["l2_token_count"]
        correct = data["correct"]
        e_idx = round(correct / en_count, 4) if en_count > 0 else None
        skipped = data["skipped"]

        # Weight by english token count when aggregating
        total_l2 += en_count
        total_correct += correct

        e_str = f"{e_idx:.4f}" if e_idx is not None else ("skip" if skipped else "n/a")
        print(f"  {test_id:<8} {row['category']:<26} {en_count:>10} {correct:>8} {e_str:>8}")

        results.append({
            "test_id": test_id,
            "category": row["category"],
            "ref": ref,
            "ref_normalized": ref_normalized,
            "hyp": hyp,
            "l2_token_count": en_count,
            "correct": correct,
            "l2_index": e_idx,
            "per_token": data["per_token"],
            "skipped": skipped,
        })

    weighted_l2index = round(total_correct / total_l2, 4) if total_l2 > 0 else None

    print(f"\n  Total L2 tokens:   {total_l2}")
    print(f"  Correctly recognised:   {total_correct}")
    print(f"\n  L2-Index (weighted):     {weighted_l2index}")
    print(f"  (1.0 = all L2 tokens correctly transcribed)")

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"en": 0, "correct": 0}
        categories[cat]["en"] += r["l2_token_count"]
        categories[cat]["correct"] += r["correct"]

    print("\n  Per-category L2-Index:")
    for cat, data in sorted(categories.items()):
        cat_e = round(data["correct"] / data["en"], 4) if data["en"] > 0 else None
        bar = ""
        if cat_e is not None:
            filled = int(cat_e * 20)
            bar = f"  [{'█' * filled}{'░' * (20 - filled)}]"
        print(f"    {cat:<28} {str(cat_e):>6}{bar}")

    output = {
        "model": args.model,
        "variant": args.variant,
        "weighted_l2index": weighted_l2index,
        "total_l2_tokens": total_l2,
        "total_correct": total_correct,
        "per_category": {
            cat: round(d["correct"] / d["en"], 4) if d["en"] > 0 else None
            for cat, d in categories.items()
        },
        "per_sentence": results,
    }

    out_path = RESULTS_DIR / args.model / f"l2index_{args.variant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
