# evaluation/compatibility/run_tests.py
"""
Model Compatibility Test Runner — Phase 1.5.

For each model × test sentence × script variant:
  - Synthesize audio
  - Record success/failure, latency, errors
  - Save audio to results/{model}/{test_id}_{variant}.wav

Outputs:
  - results/{model}/report.json   — per-model structured results
  - CAPABILITY_REPORT.md          — human-readable summary table

Usage:
    # Test all available models:
    python -m evaluation.compatibility.run_tests

    # Test specific models:
    python -m evaluation.compatibility.run_tests --models xtts_v2 fish_speech

    # Dry-run — check availability only, no synthesis:
    python -m evaluation.compatibility.run_tests --dry-run

Notes:
  - Requires GPU. Run on a machine with CUDA available.
  - Models not installed will show as SKIP in the report (not a failure).
  - Audio is saved at 22050 Hz WAV for downstream MOS/MCD evaluation.
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_VARIANTS = ["roman", "devanagari", "mixed"]
TEXT_COLUMNS = {
    "roman": "text_roman",
    "devanagari": "text_devanagari",
    "mixed": "text_mixed",
}

HERE = Path(__file__).parent
TEST_SET_PATH = HERE / "test_set.csv"
RESULTS_DIR = HERE / "results"
CAPABILITY_REPORT_PATH = HERE.parent.parent / "CAPABILITY_REPORT.md"


# ── Registry ─────────────────────────────────────────────────
def get_all_adapters():
    from .adapters.cosyvoice2 import CosyVoice2Adapter
    from .adapters.fish_speech import FishSpeechAdapter
    from .adapters.glow_tts import GlowTTSAdapter
    from .adapters.qwen3_tts import Qwen3TTSAdapter
    from .adapters.xtts_v2 import XTTSV2Adapter

    return {
        "glow_tts": GlowTTSAdapter,
        "qwen3_tts": Qwen3TTSAdapter,
        "cosyvoice2": CosyVoice2Adapter,
        "xtts_v2": XTTSV2Adapter,
        "fish_speech": FishSpeechAdapter,
    }


# ── Data structures ───────────────────────────────────────────
@dataclass
class TestResult:
    test_id: str
    category: str
    script_variant: str
    success: bool
    latency_s: float
    error: Optional[str]
    audio_path: Optional[str]


# ── Helpers ───────────────────────────────────────────────────
def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_audio(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    try:
        import soundfile as sf
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), audio, sample_rate)
    except Exception as e:
        logger.warning(f"Could not save audio to {path}: {e}")


def run_model(adapter_cls, test_rows: list[dict], dry_run: bool) -> list[TestResult]:
    adapter = adapter_cls()
    results = []

    if not adapter.is_available():
        logger.warning(f"[{adapter.name}] Not available — skipping")
        for row in test_rows:
            for variant in SCRIPT_VARIANTS:
                results.append(TestResult(
                    test_id=row["test_id"],
                    category=row["category"],
                    script_variant=variant,
                    success=False,
                    latency_s=0.0,
                    error="not_installed",
                    audio_path=None,
                ))
        return results

    if dry_run:
        logger.info(f"[{adapter.name}] Available ✓ (dry-run, skipping synthesis)")
        for row in test_rows:
            for variant in SCRIPT_VARIANTS:
                results.append(TestResult(
                    test_id=row["test_id"],
                    category=row["category"],
                    script_variant=variant,
                    success=True,
                    latency_s=0.0,
                    error=None,
                    audio_path=None,
                ))
        return results

    logger.info(f"[{adapter.name}] Loading model...")
    try:
        adapter.load()
    except Exception as e:
        logger.error(f"[{adapter.name}] Load failed: {e}")
        for row in test_rows:
            for variant in SCRIPT_VARIANTS:
                results.append(TestResult(
                    test_id=row["test_id"],
                    category=row["category"],
                    script_variant=variant,
                    success=False,
                    latency_s=0.0,
                    error=f"load_failed: {e}",
                    audio_path=None,
                ))
        return results

    audio_dir = RESULTS_DIR / adapter.name / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for row in test_rows:
        for variant in SCRIPT_VARIANTS:
            text = row[TEXT_COLUMNS[variant]]
            logger.info(f"  [{adapter.name}] {row['test_id']} / {variant}: {text[:50]}")

            result = adapter.synthesize(text, variant)

            audio_path = None
            if result.success and result.audio is not None:
                audio_path = audio_dir / f"{row['test_id']}_{variant}.wav"
                save_audio(result.audio, result.sample_rate, audio_path)
                audio_path = str(audio_path)

            results.append(TestResult(
                test_id=row["test_id"],
                category=row["category"],
                script_variant=variant,
                success=result.success,
                latency_s=round(result.latency_s, 3),
                error=result.error,
                audio_path=audio_path,
            ))

    adapter.unload()
    return results


# ── Report generation ─────────────────────────────────────────
def save_model_report(model_name: str, results: list[TestResult]) -> None:
    report_dir = RESULTS_DIR / model_name
    report_dir.mkdir(parents=True, exist_ok=True)

    # Per-script summary
    script_summary = {}
    for variant in SCRIPT_VARIANTS:
        variant_results = [r for r in results if r.script_variant == variant]
        passed = sum(1 for r in variant_results if r.success)
        total = len(variant_results)
        latencies = [r.latency_s for r in variant_results if r.success]
        script_summary[variant] = {
            "pass": passed,
            "total": total,
            "pass_rate": round(passed / total, 2) if total else 0,
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        }

    # Per-category summary (roman only — canonical)
    category_summary = {}
    roman_results = [r for r in results if r.script_variant == "roman"]
    for r in roman_results:
        category_summary[r.test_id] = {
            "category": r.category,
            "success": r.success,
            "latency_s": r.latency_s,
            "error": r.error,
        }

    report = {
        "model": model_name,
        "script_summary": script_summary,
        "category_summary": category_summary,
        "raw": [asdict(r) for r in results],
    }

    with open(report_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"[{model_name}] Report → {report_dir / 'report.json'}")


def generate_capability_report(model_names: list[str]) -> None:
    """Write CAPABILITY_REPORT.md from saved per-model report.json files."""
    rows = []
    for model_name in model_names:
        report_path = RESULTS_DIR / model_name / "report.json"
        if not report_path.exists():
            continue
        with open(report_path) as f:
            report = json.load(f)

        summary = report.get("script_summary", {})

        def fmt(variant):
            s = summary.get(variant, {})
            if s.get("total", 0) == 0:
                return "—"
            passed = s.get("pass", 0)
            total = s.get("total", 0)
            lat = s.get("avg_latency_s")
            mark = "✓" if passed == total else ("~" if passed > 0 else "✗")
            lat_str = f"{lat:.2f}s" if lat else "n/a"
            return f"{mark} ({passed}/{total}, {lat_str})"

        rows.append({
            "model": model_name,
            "roman": fmt("roman"),
            "devanagari": fmt("devanagari"),
            "mixed": fmt("mixed"),
        })

    lines = [
        "# CAPABILITY_REPORT.md",
        "",
        "Model compatibility test results across three script variants.",
        "Generated by `evaluation/compatibility/run_tests.py`.",
        "",
        "Legend: ✓ all passed · ~ partial · ✗ all failed · — not tested",
        "Latency: avg per sentence on the test machine.",
        "",
        "| Model | Roman | Devanagari | Mixed |",
        "|-------|-------|------------|-------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['roman']} | {row['devanagari']} | {row['mixed']} |"
        )

    lines += [
        "",
        "## Per-Pattern Results (Roman script)",
        "",
        "| Test ID | Category | " + " | ".join(model_names) + " |",
        "|---------|----------|" + "|".join(["---"] * len(model_names)) + "|",
    ]

    # Collect category results per model
    category_data: dict[str, dict[str, str]] = {}
    for model_name in model_names:
        report_path = RESULTS_DIR / model_name / "report.json"
        if not report_path.exists():
            continue
        with open(report_path) as f:
            report = json.load(f)
        for tid, data in report.get("category_summary", {}).items():
            if tid not in category_data:
                category_data[tid] = {"category": data["category"]}
            mark = "✓" if data["success"] else "✗"
            if data.get("error") == "not_installed":
                mark = "—"
            category_data[tid][model_name] = mark

    for tid in sorted(category_data.keys()):
        row = category_data[tid]
        cells = [row.get(m, "—") for m in model_names]
        lines.append(f"| {tid} | {row['category']} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Notes",
        "",
        "- Models marked `—` were not installed at test time.",
        "- Devanagari and mixed-script failures may indicate script rendering issues,",
        "  not phoneme mapping failures — check audio output manually.",
        "- Use this report to decide which script variants to synthesize per model (Phase 2).",
    ]

    CAPABILITY_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Capability report → {CAPABILITY_REPORT_PATH}")


# ── Entry point ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run TTS model compatibility tests (Phase 1.5)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(get_all_adapters().keys()),
        default=None,
        help="Models to test (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check availability only — do not synthesize",
    )
    args = parser.parse_args()

    test_rows = load_test_set()
    logger.info(f"Loaded {len(test_rows)} test sentences")

    all_adapters = get_all_adapters()
    selected = args.models or list(all_adapters.keys())

    for model_name in selected:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'='*50}")
        adapter_cls = all_adapters[model_name]
        results = run_model(adapter_cls, test_rows, dry_run=args.dry_run)
        save_model_report(model_name, results)

    generate_capability_report(selected)
    logger.info("\nDone. See CAPABILITY_REPORT.md for summary.")


if __name__ == "__main__":
    main()
