# evaluation/compatibility/compute_llm_naturalness.py
"""
LLM Naturalness Score — Gemini audio-based judge metric.

Uses Gemini 3 Flash Preview to evaluate the perceptual naturalness of
synthesized Hinglish audio directly, without going through ASR transcription.
The model listens to each WAV file and scores it on a 1–5 scale.

Scale:
  1 (Unnatural):  Sounds like a bad machine translation read aloud.
                  Pronunciation or rhythm feels wrong.
  2 (Poor):       Mostly understandable but with clear synthesis artifacts
                  or awkward stress patterns.
  3 (Acceptable): Understandable but slightly "off" — forced rhythm,
                  unnatural stress, or awkward code-switches.
  4 (Good):       Sounds natural most of the time; minor roughness at
                  code-switch points.
  5 (Native):     Sounds like everyday speech from a fluent Hinglish speaker.

Output:
  results/{model}/llm_naturalness.json
    - per_file: {stem: {score, reasoning, variant, test_id}}
    - variant_summary: {roman: {mean, std, n}, mixed: {...}}

Setup:
  1. Get a free Gemini API key from https://aistudio.google.com
  2. Add to .env:  GEMINI_API_KEY=your_key_here
  3. pip install google-genai python-dotenv

Usage:
    python -m evaluation.compatibility.compute_llm_naturalness --model sarvam_tts
    python -m evaluation.compatibility.compute_llm_naturalness --model qwen3_tts
    python -m evaluation.compatibility.compute_llm_naturalness --model sarvam_tts --limit 5
    python -m evaluation.compatibility.compute_llm_naturalness --model sarvam_tts --variant roman
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TEST_SET_PATH = HERE / "test_set.csv"
RESULTS_DIR = HERE / "results"

GEMINI_MODEL = "gemini-3-flash-preview"
RETRY_DELAY = 5  # seconds between retries on rate limit
MAX_RETRIES = 3

JUDGE_PROMPT_TEMPLATE = """You are an expert linguist specialising in Indian colloquialisms and Hinglish code-switching.

Listen to the attached audio clip. It is a synthesized Text-to-Speech rendering of the following Hinglish sentence:

Reference sentence: {reference_text}

Rate the NATURALNESS of the spoken audio on a scale of 1 to 5.

Scoring guide:
  1 (Unnatural):  Sounds like a bad machine translation read aloud.
                  Pronunciation or rhythm feels clearly wrong.
  2 (Poor):       Mostly understandable but with obvious synthesis artifacts
                  or awkward stress patterns.
  3 (Acceptable): Understandable but slightly "off" — forced rhythm,
                  unnatural stress, or awkward transitions at code-switch points.
  4 (Good):       Sounds natural most of the time; only minor roughness.
  5 (Native):     Sounds like everyday speech from a fluent Hinglish speaker.

Focus specifically on:
- Pronunciation accuracy for both Hindi and English tokens
- Prosody and rhythm at code-switch boundaries
- Whether stress and intonation feel natural for Hinglish

Respond in this exact format:
Score: [integer 1–5]
Reasoning: [one or two sentences explaining the score, mentioning specific tokens or transitions if relevant]"""


def load_env_key() -> str:
    """Load GEMINI_API_KEY from .env or environment."""
    try:
        from dotenv import load_dotenv

        load_dotenv(HERE.parents[1] / ".env")
    except ImportError:
        pass  # fall through to os.environ

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found.\n"
            "Add it to your .env file:  GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com"
        )
    return key


def parse_response(text: str) -> tuple[int | None, str]:
    """
    Extract score and reasoning from Gemini response.

    Returns (score, reasoning). Score is None if parsing fails.
    """
    score_match = re.search(r"Score:\s*([1-5])", text)
    reasoning_match = re.search(r"Reasoning:\s*(.+)", text, re.DOTALL)

    score = int(score_match.group(1)) if score_match else None
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()
    return score, reasoning


def judge_audio(
    client,
    wav_path: Path,
    reference_text: str,
) -> dict | None:
    """
    Send one WAV file to Gemini and return {score, reasoning}.

    Returns None on persistent failure.
    """
    from google import genai as _genai
    from google.genai import types

    prompt = JUDGE_PROMPT_TEMPLATE.format(reference_text=reference_text)

    for attempt in range(MAX_RETRIES):
        try:
            audio_data = wav_path.read_bytes()
            audio_part = types.Part.from_bytes(data=audio_data, mime_type="audio/wav")
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=[audio_part, prompt]
            )
            score, reasoning = parse_response(response.text)
            return {"score": score, "reasoning": reasoning, "raw_response": response.text}
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    Rate limit (429) — full error: {err[:300]}")
                print(f"    Waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"    API error ({type(e).__name__}): {err[:300]}")
                return None

    print(f"    Failed after {MAX_RETRIES} retries: {wav_path.name}")
    return None


def _write_results(
    out_path: Path,
    model: str,
    per_file: dict,
    summary: dict,
) -> None:
    """Write results JSON — called after each score and at the end."""
    output = {
        "model": model,
        "metric": "llm_naturalness",
        "judge_model": GEMINI_MODEL,
        "variant_summary": summary,
        "per_file": per_file,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM naturalness judge via Gemini audio")
    parser.add_argument("--model", default="sarvam_tts", help="Model name")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N sentences")
    parser.add_argument(
        "--variant",
        choices=["roman", "mixed", "both"],
        default="both",
        help="Which script variant to evaluate (default: both)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=6.0,
        help="Seconds to wait between API calls (default: 6.0, stays under free-tier 10 RPM)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already present in the output JSON (resume a partial run)",
    )
    args = parser.parse_args()

    # --- Setup Gemini client ---
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai required: pip install google-genai python-dotenv"
        )

    api_key = load_env_key()
    client = genai.Client(api_key=api_key)

    # --- Load test set ---
    audio_dir = RESULTS_DIR / args.model / "audio"
    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    with open(TEST_SET_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    # Build lookup: test_id -> {roman: text, mixed: text}
    text_map: dict[str, dict[str, str]] = {
        row["test_id"]: {
            "roman": row["text_roman"],
            "mixed": row["text_mixed"],
        }
        for row in rows
    }
    valid_ids = set(text_map.keys())

    # --- Select WAV files ---
    wav_files = sorted(audio_dir.glob("*.wav"))

    variants_to_run = ["roman", "mixed"] if args.variant == "both" else [args.variant]
    wav_files = [
        w
        for w in wav_files
        if any(w.stem.endswith(f"_{v}") for v in variants_to_run)
        and w.stem.rsplit("_", 1)[0] in valid_ids
    ]

    print(f"\nLLM Naturalness Score — {args.model} ({GEMINI_MODEL})")
    print("  Judging audio directly: pronunciation, prosody, code-switch naturalness")
    print(f"  Files to evaluate: {len(wav_files)}")
    print("-" * 90)
    print(f"  {'File':<30} {'Score':>7}  Reasoning")
    print(f"  {'-'*30} {'-'*7}  {'-'*45}")

    per_file: dict[str, dict] = {}
    variant_scores: dict[str, list[float]] = {"roman": [], "mixed": []}

    # Load existing results if resuming
    out_path = RESULTS_DIR / args.model / "llm_naturalness.json"
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        for stem, entry in existing.get("per_file", {}).items():
            per_file[stem] = entry
            v = entry.get("variant")
            s = entry.get("score")
            if v in variant_scores and s is not None:
                variant_scores[v].append(s)
        print(f"  Resuming: loaded {len(per_file)} existing results, skipping those files.")

    for wav in wav_files:
        stem = wav.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        test_id, variant = parts[0], parts[1]

        reference_text = text_map.get(test_id, {}).get(variant, "")
        if not reference_text:
            continue

        if stem in per_file:
            score = per_file[stem]["score"]
            reasoning_short = per_file[stem]["reasoning"][:60] + (
                "..." if len(per_file[stem]["reasoning"]) > 60 else ""
            )
            print(f"  {stem:<30} {score:>7}  [cached] {reasoning_short}")
            continue

        result = judge_audio(client, wav, reference_text)  # client is the GenerativeModel
        time.sleep(args.delay)  # respect free-tier rate limits

        if result and result["score"] is not None:
            score = result["score"]
            reasoning_short = result["reasoning"][:60] + (
                "..." if len(result["reasoning"]) > 60 else ""
            )
            print(f"  {stem:<30} {score:>7}  {reasoning_short}")
            per_file[stem] = {
                "score": score,
                "reasoning": result["reasoning"],
                "variant": variant,
                "test_id": test_id,
            }
            if variant in variant_scores:
                variant_scores[variant].append(score)
            # Write after each successful score so partial runs are not lost
            _write_results(out_path, args.model, per_file=per_file, summary={})
        else:
            print(f"  {stem:<30} {'—':>7}  (API error or parse failure)")

    # --- Summary by variant ---
    print(f"\n  {'Variant':<15} {'Mean':>8} {'Median':>8} {'Std':>8} {'N':>5}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")

    summary: dict[str, dict] = {}
    for v in ["roman", "mixed"]:
        vals = variant_scores[v]
        if vals:
            arr = np.array(vals, dtype=float)
            mean_s = round(float(np.mean(arr)), 3)
            median_s = round(float(np.median(arr)), 3)
            std_s = round(float(np.std(arr)), 3)
            print(f"  {v:<15} {mean_s:>8.2f} {median_s:>8.2f} {std_s:>8.2f} {len(vals):>5}")
        else:
            mean_s = median_s = std_s = None
            print(f"  {v:<15} {'—':>8} {'—':>8} {'—':>8} {'0':>5}  (no audio)")
        summary[v] = {"mean": mean_s, "median": median_s, "std": std_s, "n": len(vals)}

    print("""
  Interpretation:
    5.0       — native-sounding Hinglish throughout
    4.0–4.9   — good; only minor roughness at switch points
    3.0–3.9   — acceptable; noticeable but not distracting issues
    2.0–2.9   — poor; frequent unnatural stress or pronunciation
    1.0–1.9   — unnatural; sounds like machine translation read aloud
""")

    # --- Save final results (with complete summary) ---
    _write_results(out_path, args.model, per_file=per_file, summary=summary)
    print(f"  Full results → {out_path}")


if __name__ == "__main__":
    main()
