# data/golden/select.py
"""
Select the golden set from benchmark_v1.csv.

Golden set: 300 sentences stratified by CS pattern and CMI bucket,
filtered by token length, for human verification and evaluation.

Selection rules (from PROJECT_PLAN.md Phase 1.1):
    - 300 total, ~43 per CS pattern (CS-01..07)
    - Token length: 6–12 (CS-07 intraword: relaxed to 5–12)
    - Oversample mid/high CMI where available
    - Deterministic (seeded shuffle for reproducibility)

Output: data/golden/golden_set.csv (same columns as benchmark_v1.csv
        plus golden_id column)
"""

import argparse
import csv
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# How many sentences per CS pattern
TOTAL_GOLDEN = 300
PATTERNS = ["CS-01", "CS-02", "CS-03", "CS-04", "CS-05", "CS-06", "CS-07"]
PER_PATTERN = TOTAL_GOLDEN // len(PATTERNS)  # 42, remainder distributed below

# CMI target proportions — oversample mid/high
# Applied where available; shortfalls filled from other buckets
CMI_TARGETS = {"high": 0.35, "mid": 0.40, "low": 0.25}

# Token length bounds
LENGTH_RANGE = (6, 12)
LENGTH_RANGE_CS07 = (5, 12)


def length_filter(row: dict) -> bool:
    """Check if a row meets the token length requirement."""
    n_tokens = len(row["text_roman"].split())
    lo, hi = LENGTH_RANGE_CS07 if row["pattern_id"] == "CS-07" else LENGTH_RANGE
    return lo <= n_tokens <= hi


def select_from_bucket(pool: list, target: int, rng: random.Random) -> list:
    """Select up to `target` rows from a shuffled pool."""
    rng.shuffle(pool)
    return pool[:target]


def select_golden(
    input_path: Path,
    output_path: Path,
    seed: int = 42,
    total: int = TOTAL_GOLDEN,
) -> None:
    rng = random.Random(seed)

    # Load and filter by length
    with open(input_path, encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    eligible = [r for r in all_rows if length_filter(r)]
    logger.info(
        f"Loaded {len(all_rows)} rows, {len(eligible)} eligible after length filter"
    )

    # Group eligible rows by (pattern, cmi_bucket)
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for r in eligible:
        grouped[(r["pattern_id"], r["cmi_bucket"])].append(r)

    # Compute per-pattern targets (distribute remainder to largest patterns)
    per_pattern = total // len(PATTERNS)
    remainder = total - per_pattern * len(PATTERNS)
    pattern_targets = {p: per_pattern for p in PATTERNS}
    # Give extra sentences to patterns with the most eligible rows
    eligible_by_pattern = Counter(r["pattern_id"] for r in eligible)
    for p in sorted(PATTERNS, key=lambda p: -eligible_by_pattern.get(p, 0)):
        if remainder <= 0:
            break
        pattern_targets[p] += 1
        remainder -= 1

    selected = []

    for pattern in PATTERNS:
        target = pattern_targets[pattern]

        # Calculate CMI targets for this pattern
        buckets_available = {}
        for bucket in ["high", "mid", "low"]:
            pool = grouped.get((pattern, bucket), [])
            if pool:
                buckets_available[bucket] = pool

        if not buckets_available:
            logger.warning(f"{pattern}: no eligible rows at all, skipping")
            continue

        # Allocate targets proportionally, oversample mid/high
        bucket_targets = {}
        total_weight = sum(
            CMI_TARGETS[b] for b in buckets_available
        )
        for bucket in buckets_available:
            bucket_targets[bucket] = round(
                target * CMI_TARGETS[bucket] / total_weight
            )

        # Fix rounding to hit exact target
        diff = target - sum(bucket_targets.values())
        if diff != 0:
            # Adjust the bucket with the most available rows
            adjust_bucket = max(
                buckets_available, key=lambda b: len(buckets_available[b])
            )
            bucket_targets[adjust_bucket] += diff

        # Select from each bucket
        pattern_selected = []
        shortfall = 0
        for bucket in ["high", "mid", "low"]:
            if bucket not in buckets_available:
                continue
            bt = bucket_targets[bucket] + shortfall
            pool = buckets_available[bucket]
            picked = select_from_bucket(pool, bt, rng)
            pattern_selected.extend(picked)
            shortfall = bt - len(picked)

        # If still short after all buckets, we can't do more
        if len(pattern_selected) < target:
            logger.warning(
                f"{pattern}: only got {len(pattern_selected)}/{target} "
                f"(not enough eligible rows)"
            )

        selected.extend(pattern_selected)

    # Sort by original sentence_id for stable output
    selected.sort(key=lambda r: r["sentence_id"])

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "golden_id", "sentence_id", "pattern_id", "cmi_bucket",
        "text_roman", "text_devanagari", "text_mixed", "language_tags",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(selected, 1):
            row_out = {"golden_id": f"G_{idx:03d}", **row}
            writer.writerow(row_out)

    logger.info(f"Selected {len(selected)} golden sentences → {output_path}")

    # Print distribution summary
    pat_dist = Counter(r["pattern_id"] for r in selected)
    cmi_dist = Counter(r["cmi_bucket"] for r in selected)
    cross = Counter(
        (r["pattern_id"], r["cmi_bucket"]) for r in selected
    )

    logger.info("Pattern distribution:")
    for p in PATTERNS:
        logger.info(f"  {p}: {pat_dist.get(p, 0)}")
    logger.info(f"CMI distribution: {dict(sorted(cmi_dist.items()))}")
    logger.info("Cross-tab (pattern x CMI):")
    for p in PATTERNS:
        parts = {b: cross.get((p, b), 0) for b in ["low", "mid", "high"]}
        logger.info(f"  {p}: {parts}")

    # Write metadata
    lengths = [len(r["text_roman"].split()) for r in selected]
    metadata = {
        "total": len(selected),
        "seed": seed,
        "source": str(input_path),
        "pattern_counts": dict(sorted(pat_dist.items())),
        "cmi_counts": dict(sorted(cmi_dist.items())),
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
        },
        "cross_tab": {
            p: {b: cross.get((p, b), 0) for b in ["low", "mid", "high"]}
            for p in PATTERNS
        },
    }
    meta_path = output_path.parent / "golden_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata → {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select golden set from benchmark CSV"
    )
    parser.add_argument(
        "--input", type=str, default="data/codeswitched/benchmark_v1.csv"
    )
    parser.add_argument(
        "--output", type=str, default="data/golden/golden_set.csv"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total", type=int, default=300)
    args = parser.parse_args()
    select_golden(Path(args.input), Path(args.output), args.seed, args.total)
