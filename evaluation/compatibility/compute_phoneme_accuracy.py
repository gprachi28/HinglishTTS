# evaluation/compatibility/compute_phoneme_accuracy.py
"""
H-Phoneme-Accuracy & E-Phoneme-Accuracy — Phase 3.4 metric (Step 2).

Measures phoneme-level accuracy for Hindi and English tokens separately.
Uses character-level string matching as a proxy for phoneme accuracy:
- Hindi: Devanagari characters are phonetic; direct character comparison
- English: Normalization + character-level matching of transcribed words

Definition:
  H-Phoneme-Accuracy = (% of Hindi token phonemes correctly transcribed)
  E-Phoneme-Accuracy = (% of English token phonemes correctly transcribed)

This captures errors like "meeting" → "making" at the phoneme level,
which token-level metrics (H-Index, E-Index) miss.

Usage:
    python -m evaluation.compatibility.compute_phoneme_accuracy --model qwen3_tts
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


# ── Normalisation ─────────────────────────────────────────────
def normalise(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।॥,\.!?;:\-\"'()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    return normalise(text).split()


# ── Levenshtein alignment ─────────────────────────────────────
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
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None))
            i -= 1
        else:
            alignment.append((None, hyp[j - 1]))
            j -= 1
    alignment.reverse()
    return alignment


# ── Phoneme-level accuracy (character-level proxy) ────────────────────
def character_similarity(ref: str, hyp: str) -> float:
    """
    Compute character-level similarity as a proxy for phoneme accuracy.
    Returns ratio of matching characters to reference length.

    Example: "meeting" vs "making"
      Alignment: m-m(✓), e-a(✗), e-k(✗), t-i(✗), i-n(✗), n-g(✗), g-(✗)
      Score: 1/7 = 0.1429
    """
    if not ref:
        return 1.0 if not hyp else 0.0

    # Use SequenceMatcher to find matching characters
    matcher = SequenceMatcher(None, ref, hyp)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    return matches / max(len(ref), len(hyp))


def compute_sentence_phoneme_accuracy(
    ref_text: str, hyp_text: str, tags: list[str]
) -> dict:
    """
    Returns phoneme-level accuracy for Hindi and English tokens separately.
    """
    ref_tokens = tokenise(ref_text)
    hyp_tokens = tokenise(hyp_text)

    if not hyp_text.strip():
        return {
            "hindi_tokens": [],
            "english_tokens": [],
            "skipped": True
        }

    if len(ref_tokens) != len(tags):
        return {
            "hindi_tokens": [],
            "english_tokens": [],
            "skipped": True
        }

    alignment = align_tokens(ref_tokens, hyp_tokens)

    # Map ref index → aligned hyp token
    ref_to_hyp: dict[int, str | None] = {}
    ref_idx = 0
    for ref_tok, hyp_tok in alignment:
        if ref_tok is not None:
            ref_to_hyp[ref_idx] = hyp_tok
            ref_idx += 1

    hindi_tokens = []
    english_tokens = []

    for pos, tag in enumerate(tags):
        if pos >= len(ref_tokens):
            break
        ref_tok = ref_tokens[pos]
        hyp_tok = ref_to_hyp.get(pos)

        if hyp_tok is None:
            # Token was deleted by ASR
            similarity = 0.0
        else:
            similarity = character_similarity(ref_tok, hyp_tok)

        token_data = {
            "ref": ref_tok,
            "hyp": hyp_tok,
            "similarity": round(similarity, 4)
        }

        if tag == "HI":
            hindi_tokens.append(token_data)
        elif tag == "EN":
            english_tokens.append(token_data)

    return {
        "hindi_tokens": hindi_tokens,
        "english_tokens": english_tokens,
        "skipped": False
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
    hindi_similarities = []
    english_similarities = []

    print(f"\nPhoneme-Level Accuracy — {args.model} (roman variant)")
    print("-" * 85)
    print(f"  {'Test ID':<8} {'Category':<20} {'HI Acc':>8} {'EN Acc':>8} {'Notes':>45}")
    print(f"  {'-'*8} {'-'*20} {'-'*8} {'-'*8} {'-'*45}")

    for test_id, row in test_rows.items():
        ref = row["text"]
        ref_normalized = transliterate_roman_to_devanagari(ref)
        hyp = transcripts.get(f"{test_id}_roman", "")
        tags = row["language_tags"].split()

        # Compare using normalized (Devanagari) reference
        data = compute_sentence_phoneme_accuracy(ref_normalized, hyp, tags)

        hi_tokens = data["hindi_tokens"]
        en_tokens = data["english_tokens"]

        # Calculate averages
        hi_acc = round(sum(t["similarity"] for t in hi_tokens) / len(hi_tokens), 4) if hi_tokens else None
        en_acc = round(sum(t["similarity"] for t in en_tokens) / len(en_tokens), 4) if en_tokens else None

        if hi_tokens:
            hindi_similarities.extend([t["similarity"] for t in hi_tokens])
        if en_tokens:
            english_similarities.extend([t["similarity"] for t in en_tokens])

        # Highlight errors (< 0.5 similarity)
        notes = ""
        errors = []
        for t in hi_tokens:
            if t["similarity"] < 0.5:
                errors.append(f"HI:{t['ref']}→{t['hyp']}")
        for t in en_tokens:
            if t["similarity"] < 0.5:
                errors.append(f"EN:{t['ref']}→{t['hyp']}")
        if errors:
            notes = ", ".join(errors[:2])

        hi_str = f"{hi_acc:.3f}" if hi_acc is not None else "—"
        en_str = f"{en_acc:.3f}" if en_acc is not None else "—"

        print(f"  {test_id:<8} {row['category']:<20} {hi_str:>8} {en_str:>8} {notes:>45}")

        results.append({
            "test_id": test_id,
            "category": row["category"],
            "ref": ref,
            "ref_normalized": ref_normalized,
            "hyp": hyp,
            "hindi_phoneme_acc": hi_acc,
            "english_phoneme_acc": en_acc,
            "hindi_tokens": hi_tokens,
            "english_tokens": en_tokens,
            "skipped": data["skipped"],
        })

    # Aggregate statistics
    avg_hi_acc = round(sum(hindi_similarities) / len(hindi_similarities), 4) if hindi_similarities else None
    avg_en_acc = round(sum(english_similarities) / len(english_similarities), 4) if english_similarities else None

    print(f"\n  H-Phoneme-Accuracy (average):    {avg_hi_acc}")
    print(f"  E-Phoneme-Accuracy (average):    {avg_en_acc}")
    print(f"  (1.0 = perfect phoneme match, 0.0 = completely wrong)")
    print(f"  Note: Using character-level similarity as phoneme proxy")

    output = {
        "model": args.model,
        "variant": "roman",
        "h_phoneme_accuracy": avg_hi_acc,
        "e_phoneme_accuracy": avg_en_acc,
        "per_sentence": results,
        "note": "Character-level similarity used as phoneme-level proxy"
    }

    out_path = RESULTS_DIR / args.model / "phoneme_accuracy.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
