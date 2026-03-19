# data/script_variants.py
"""
Generate the three script variants (Set A, B, C) from sentences.csv.

Set A (text_devanagari):
    HI-tagged tokens → Devanagari (via indic-transliteration, ITRANS scheme).
    EN-tagged tokens stay Roman.
    Machine-generated baseline; pending human verification for the golden set.
    Note: output is approximate for informal Romanized Hindi — ITRANS was
    designed for formal/classical romanization. Golden set verification corrects errors.

Set B (text_roman):
    All tokens stay Romanized. The base output from codeswitching.py — no changes.

Set C (text_mixed):
    Same algorithm as Set A (HI → Devanagari, EN → Roman).
    Always machine-generated, never human-corrected.
    Set A and Set C are identical until the golden set verification step updates Set A.

Why Set A ≠ "pure Devanagari":
    English loanwords (meeting, download, API) are NOT transliterated to Devanagari.
    Only Hindi-tagged tokens are converted. This matches how Hinglish is actually
    written in Devanagari script — English words stay in Roman even in Devanagari text.

Output: data/codeswitched/benchmark_v1.csv
Columns: sentence_id, pattern_id, cmi_bucket, text_roman, text_devanagari,
         text_mixed, language_tags
"""

import argparse
import csv
import logging
from pathlib import Path

from indic_transliteration import sanscript

from data.codeswitching import BUILDER_CS_PATTERN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cmi_bucket(cmi: float) -> str:
    if cmi < 0.3:
        return "low"
    elif cmi < 0.5:
        return "mid"
    else:
        return "high"


def roman_to_devanagari(word: str) -> str:
    """Transliterate a single Romanized Hindi token to Devanagari (ITRANS scheme).

    Lowercases before transliteration — our sentences use sentence-initial caps
    (e.g. "Aaj", "Kal") which ITRANS would misread as different phonemes (A ≠ a).

    Handles punctuation attached to words (e.g. "hai," → "है,") by stripping
    trailing punctuation, transliterating, then re-attaching.

    Known limitation: ITRANS cannot distinguish short/long 'a' in informal Hindi
    romanization ("bura" → "बुर" instead of "बुरा"). This is expected for the
    machine-generated baseline; the golden set is human-verified.
    """
    PUNCT = ".,?!;:"
    suffix = ""
    while word and word[-1] in PUNCT:
        suffix = word[-1] + suffix
        word = word[:-1]
    if not word:
        return suffix
    devanagari = sanscript.transliterate(word.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
    return devanagari + suffix


def apply_tag_based_script(tokens: list, tags: list) -> str:
    """HI-tagged tokens → Devanagari, EN-tagged tokens stay Roman."""
    parts = []
    for word, tag in zip(tokens, tags):
        if tag == "HI":
            parts.append(roman_to_devanagari(word))
        else:
            parts.append(word)
    return " ".join(parts)


def generate_benchmark_csv(input_path: Path, output_path: Path) -> None:
    rows = []
    skipped = 0

    with open(input_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            sentence = row["sentence"]
            pattern = row["pattern"]
            cmi = float(row["cmi"])
            language_tags_str = row["language_tags"]
            language_tags = language_tags_str.split()
            tokens = sentence.split()

            if len(tokens) != len(language_tags):
                logger.warning(
                    f"Row {idx}: token count ({len(tokens)}) != "
                    f"tag count ({len(language_tags)}), skipping"
                )
                skipped += 1
                continue

            text_devanagari = apply_tag_based_script(tokens, language_tags)
            text_mixed = apply_tag_based_script(tokens, language_tags)

            rows.append({
                "sentence_id": f"CS_{idx + 1:05d}",
                "pattern_id": BUILDER_CS_PATTERN.get(pattern, "unknown"),
                "cmi_bucket": get_cmi_bucket(cmi),
                "text_roman": sentence,
                "text_devanagari": text_devanagari,
                "text_mixed": text_mixed,
                "language_tags": language_tags_str,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sentence_id", "pattern_id", "cmi_bucket",
        "text_roman", "text_devanagari", "text_mixed", "language_tags",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} rows to {output_path}")
    if skipped:
        logger.warning(f"Skipped {skipped} rows due to token/tag length mismatch")


def main(input_path: str, output_path: str) -> None:
    generate_benchmark_csv(Path(input_path), Path(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Set A/B/C script variants from sentences.csv"
    )
    parser.add_argument(
        "--input_path", type=str, default="data/codeswitched/sentences.csv"
    )
    parser.add_argument(
        "--output_path", type=str, default="data/codeswitched/benchmark_v1.csv"
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path)
