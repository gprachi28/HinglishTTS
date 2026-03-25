# evaluation/compatibility/compute_lid.py
"""
Language Boundary Confidence (LID) — script-type accuracy at switch points.

For the mixed-script variant, the reference text uses Devanagari for Hindi tokens
and Latin for English tokens. This metric checks whether the ASR transcription
(Whisper) also uses the matching script at code-switch boundary positions.

Interpretation:
  High confidence (→ 1.0): ASR correctly identifies the language at each boundary —
    the model produces acoustically distinct Hindi vs English signals.
  Low confidence (→ 0.0): ASR cannot distinguish the language at boundaries —
    the model is blending languages acoustically (harder to identify but more natural).

Method: Unicode script detection (Devanagari U+0900–U+097F vs Latin A–Z) applied
to ASR output tokens at switch-point positions. This is a heuristic proxy that
works well for Hindi↔English where the scripts are visually and acoustically distinct.

Note: Whisper, when run without a forced language, will output Devanagari when it
hears clearly Hindi phonemes and Latin when it hears English. Models that produce
poor Hindi phonemes will cause Whisper to romanise the output, reducing LID.

Dependency: transcripts.json (run compute_tpi.py first)

Usage:
    python -m evaluation.compatibility.compute_lid --model cosyvoice3
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


# ── Script detection ──────────────────────────────────────────
def _has_devanagari(token: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", token))


def _has_latin(token: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", token))


def detected_script(token: str) -> str | None:
    """Return 'devanagari', 'latin', or None (ambiguous/empty)."""
    d = _has_devanagari(token)
    lat = _has_latin(token)
    if d and not lat:
        return "devanagari"
    if lat and not d:
        return "latin"
    return None


def expected_script(tag: str) -> str:
    """HI → 'devanagari', EN → 'latin'."""
    return "devanagari" if tag == "HI" else "latin"


# ── Normalisation + alignment (same conventions as PIER/H-Index) ─
def normalise(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[।॥,\.!?;:\-\"'()]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenise(text: str) -> list[str]:
    return normalise(text).split()


def align_tokens(ref: list[str], hyp: list[str]) -> list[tuple]:
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
            alignment.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref[i - 1], None)); i -= 1
        else:
            alignment.append((None, hyp[j - 1])); j -= 1
    alignment.reverse()
    return alignment


def find_switch_positions(tags: list[str]) -> set[int]:
    positions = set()
    for i in range(len(tags) - 1):
        if tags[i] != tags[i + 1]:
            positions.add(i)
            positions.add(i + 1)
    return positions


# ── LID for one sentence ──────────────────────────────────────
def compute_sentence_lid(ref_mixed: str, hyp_text: str, tags: list[str]) -> dict:
    if not hyp_text.strip():
        return {"boundary_tokens": 0, "confident": 0, "skipped": True}

    ref_tokens = tokenise(ref_mixed)
    hyp_tokens = tokenise(hyp_text)

    if len(ref_tokens) != len(tags):
        return {"boundary_tokens": 0, "confident": 0, "skipped": True}

    switch_pos = find_switch_positions(tags)
    if not switch_pos:
        return {"boundary_tokens": 0, "confident": 0, "skipped": False}

    alignment = align_tokens(ref_tokens, hyp_tokens)
    ref_to_hyp: dict[int, str | None] = {}
    ref_idx = 0
    for ref_tok, hyp_tok in alignment:
        if ref_tok is not None:
            ref_to_hyp[ref_idx] = hyp_tok
            ref_idx += 1

    confident = 0
    details = []
    for pos in sorted(switch_pos):
        tag = tags[pos]
        hyp_tok = ref_to_hyp.get(pos)
        exp = expected_script(tag)
        det = detected_script(hyp_tok) if hyp_tok else None
        match = det == exp
        if match:
            confident += 1
        details.append({"pos": pos, "tag": tag, "hyp": hyp_tok, "expected_script": exp, "detected_script": det, "match": match})

    return {
        "boundary_tokens": len(switch_pos),
        "confident": confident,
        "skipped": False,
        "details": details,
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

    total_boundary = 0
    total_confident = 0
    results = []

    print(f"\nLID (Language Boundary Confidence) — {args.model} (mixed variant)")
    print("-" * 70)
    print(f"  {'Test ID':<8} {'Category':<26} {'Boundary':>9} {'Correct':>8} {'LID':>8}")
    print(f"  {'-'*8} {'-'*26} {'-'*9} {'-'*8} {'-'*8}")

    for test_id, row in test_rows.items():
        ref = row["text_mixed"]
        hyp = transcripts.get(f"{test_id}_mixed", "")
        tags = row["language_tags"].split()

        data = compute_sentence_lid(ref, hyp, tags)
        bt     = data["boundary_tokens"]
        conf   = data["confident"]
        lid    = round(conf / bt, 4) if bt > 0 else None
        skip   = data["skipped"]

        total_boundary  += bt
        total_confident += conf

        lid_str = f"{lid:.4f}" if lid is not None else ("skip" if skip else "n/a")
        print(f"  {test_id:<8} {row['category']:<26} {bt:>9} {conf:>8} {lid_str:>8}")

        results.append({
            "test_id": test_id,
            "category": row["category"],
            "boundary_tokens": bt,
            "confident": conf,
            "lid_confidence": lid,
            "skipped": skip,
            "details": data.get("details", []),
        })

    overall_lid = round(total_confident / total_boundary, 4) if total_boundary > 0 else None

    print(f"\n  Total boundary tokens:  {total_boundary}")
    print(f"  Correctly script-typed: {total_confident}")
    print(f"\n  Overall LID confidence: {overall_lid}")
    print(f"  (1.0 = ASR sees acoustically distinct language signals at every boundary)")

    output = {
        "model": args.model,
        "variant": "mixed",
        "overall_lid_confidence": overall_lid,
        "total_boundary_tokens": total_boundary,
        "total_confident": total_confident,
        "per_sentence": results,
    }

    out_path = RESULTS_DIR / args.model / "lid.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
