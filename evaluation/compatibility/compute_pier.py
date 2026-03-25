# evaluation/compatibility/compute_pier.py
"""
PIER (Point-of-Interest Error Rate) — Phase 3.1 metric.

Computes WER *only* at code-switch boundary tokens — the tokens immediately
adjacent to a HI↔EN language switch in the reference. Normalized per
switch-point token, not per sentence (so multi-switch sentences are not
weighted more heavily).

Algorithm:
  1. For each sentence, parse reference tokens + language tags.
  2. Find switch-boundary positions: indices where tag[i] != tag[i+1].
     Both tokens on each boundary (i and i+1) are switch-point tokens.
  3. Align ASR hypothesis tokens to reference tokens via Levenshtein path.
  4. Extract aligned pairs at switch-point positions.
  5. Compute token error rate over those pairs only.

Run on the `mixed` script variant — this preserves natural HI/EN boundaries.

Usage:
    python -m evaluation.compatibility.compute_pier --model qwen3_tts
    python -m evaluation.compatibility.compute_pier --model qwen3_tts --variant mixed
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
    """
    Returns a list of (ref_token, hyp_token) pairs representing the
    minimum-edit alignment. Insertions have ref=None; deletions have hyp=None.
    """
    n, m = len(ref), len(hyp)
    # DP table
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

    # Backtrack
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((ref[i - 1], hyp[j - 1]))  # substitution
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None))  # deletion
            i -= 1
        else:
            alignment.append((None, hyp[j - 1]))  # insertion
            j -= 1
    alignment.reverse()
    return alignment


def find_switch_positions(tags: list[str]) -> set[int]:
    """
    Return set of reference token indices that are at a HI↔EN boundary.
    For a boundary between position i and i+1, both i and i+1 are included.
    """
    positions = set()
    for i in range(len(tags) - 1):
        if tags[i] != tags[i + 1]:
            positions.add(i)
            positions.add(i + 1)
    return positions


# ── PIER for one sentence ─────────────────────────────────────
def compute_sentence_pier(
    ref_text: str, hyp_text: str, tags: list[str]
) -> dict:
    """
    Returns:
        switch_token_count: number of switch-point reference tokens
        errors: number of wrong/missing switch-point tokens in hypothesis
        switch_positions: set of positions that are switch points
    """
    ref_tokens = tokenise(ref_text)
    hyp_tokens = tokenise(hyp_text)

    if len(ref_tokens) != len(tags):
        # Tag count mismatch — skip PIER for this sentence
        return {"switch_token_count": 0, "errors": 0, "skipped": True}

    switch_pos = find_switch_positions(tags)
    if not switch_pos:
        return {"switch_token_count": 0, "errors": 0, "skipped": False}

    # Align full sentence
    alignment = align_tokens(ref_tokens, hyp_tokens)

    # Walk alignment to map ref index → aligned hyp token
    ref_to_hyp: dict[int, str | None] = {}
    ref_idx = 0
    for ref_tok, hyp_tok in alignment:
        if ref_tok is not None:
            ref_to_hyp[ref_idx] = hyp_tok
            ref_idx += 1
        # insertions (ref=None) are not at any ref position — skip

    errors = 0
    for pos in switch_pos:
        ref_tok = ref_tokens[pos]
        hyp_tok = ref_to_hyp.get(pos)
        if hyp_tok is None or hyp_tok != ref_tok:
            errors += 1

    return {
        "switch_token_count": len(switch_pos),
        "errors": errors,
        "skipped": False,
    }


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3_tts")
    parser.add_argument(
        "--variant",
        default="mixed",
        choices=["roman", "devanagari", "mixed"],
        help="Script variant to evaluate (default: mixed)",
    )
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

    text_col = TEXT_COLUMNS[args.variant]
    results = []
    total_switch_tokens = 0
    total_errors = 0

    print(f"\nPIER — {args.model} ({args.variant} variant)")
    print("-" * 70)
    print(f"  {'Test ID':<8} {'Category':<26} {'Switch pts':>10} {'Errors':>7} {'PIER':>8}")
    print(f"  {'-'*8} {'-'*26} {'-'*10} {'-'*7} {'-'*8}")

    for test_id, row in test_rows.items():
        ref = row[text_col]
        hyp = transcripts.get(f"{test_id}_{args.variant}", "")
        tags = row["language_tags"].split()

        pier_data = compute_sentence_pier(ref, hyp, tags)

        sw = pier_data["switch_token_count"]
        err = pier_data["errors"]
        pier = round(err / sw, 4) if sw > 0 else None
        skipped = pier_data["skipped"]

        total_switch_tokens += sw
        total_errors += err

        pier_str = f"{pier:.4f}" if pier is not None else ("skip" if skipped else "n/a")
        print(f"  {test_id:<8} {row['category']:<26} {sw:>10} {err:>7} {pier_str:>8}")

        results.append({
            "test_id": test_id,
            "category": row["category"],
            "ref": ref,
            "hyp": hyp,
            "switch_token_count": sw,
            "errors": err,
            "pier": pier,
            "skipped": skipped,
        })

    overall_pier = round(total_errors / total_switch_tokens, 4) if total_switch_tokens > 0 else None

    print(f"\n  {'Total switch tokens:':<30} {total_switch_tokens}")
    print(f"  {'Total errors at switch points:':<30} {total_errors}")
    print(f"\n  Overall PIER ({args.variant}): {overall_pier}")

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"sw": 0, "err": 0}
        categories[cat]["sw"] += r["switch_token_count"]
        categories[cat]["err"] += r["errors"]

    print("\n  Per-category PIER:")
    for cat, data in sorted(categories.items()):
        cat_pier = round(data["err"] / data["sw"], 4) if data["sw"] > 0 else None
        print(f"    {cat:<28} {cat_pier}")

    output = {
        "model": args.model,
        "variant": args.variant,
        "overall_pier": overall_pier,
        "total_switch_tokens": total_switch_tokens,
        "total_errors": total_errors,
        "per_category": {
            cat: round(d["err"] / d["sw"], 4) if d["sw"] > 0 else None
            for cat, d in categories.items()
        },
        "per_sentence": results,
    }

    out_path = RESULTS_DIR / args.model / f"pier_{args.variant}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
