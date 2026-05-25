"""Pilot run for the framing analyzer.

Samples N rows stratified by stimulus_set x language (and, where possible,
category / sub_bucket), runs the framing judge on them, and writes a CSV
with the prompt, response, and machine scores side-by-side. You then add
your own columns (your_stance, your_individualism, etc.) and compute
agreement against the judge.

This is the framing-rubric validation step. Until inter-rater agreement is
acceptable (Cohen's kappa >= ~0.5 for categorical fields, weighted kappa
for the 1-5 individualism scale), don't trust full-corpus content scores.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from . import config
from .evaluate_framing import judge

load_dotenv()

SEED = 2024
DEFAULT_N = 30
PILOT_CSV = config.DATA_DIR / "framing_pilot.csv"


def stratum_key(row: dict) -> tuple:
    """Group rows so the sampler can spread across conditions evenly."""
    if row["stimulus_set"] == "xsafety":
        sub = row.get("category") or ""
    else:
        sub = row.get("sub_bucket") or ""
    return (row["stimulus_set"], row["language"], sub)


def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin draw across strata, then top up to n with leftovers."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[stratum_key(r)].append(r)
    for b in buckets.values():
        rng.shuffle(b)

    keys = list(buckets.keys())
    rng.shuffle(keys)

    picked = []
    while len(picked) < n and any(buckets.values()):
        for k in keys:
            if not buckets[k]:
                continue
            picked.append(buckets[k].pop())
            if len(picked) >= n:
                break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=DEFAULT_N,
                    help=f"sample size (default {DEFAULT_N})")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic()

    with open(config.RAW_RESPONSES) as f:
        rows = [json.loads(l) for l in f]

    rng = random.Random(SEED)
    sample = stratified_sample(rows, args.n, rng)

    print(f"sampled {len(sample)} rows; strata covered:")
    cov = defaultdict(int)
    for r in sample:
        cov[stratum_key(r)] += 1
    for k, c in sorted(cov.items()):
        print(f"  {k}: {c}")

    fieldnames = [
        "model", "stimulus_set", "language", "category_or_sub_bucket",
        "prompt_id", "prompt_text", "raw_response",
        "judge_stance", "judge_individual_framing", "judge_collectivist_framing",
        "judge_recommendation", "judge_tone", "judge_refused",
        "judge_framing_notes", "judge_status",
        "your_stance", "your_individual_framing", "your_collectivist_framing",
        "your_recommendation", "your_tone", "your_refused",
        "notes",
    ]
    with open(PILOT_CSV, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for r in tqdm(sample, desc="framing-pilot"):
            scores = judge(client, r)
            sub = r.get("category") or r.get("sub_bucket") or ""
            writer.writerow({
                "model": r["model"],
                "stimulus_set": r["stimulus_set"],
                "language": r["language"],
                "category_or_sub_bucket": sub,
                "prompt_id": r["prompt_id"],
                "prompt_text": r["prompt_text"],
                "raw_response": r.get("raw_response") or "",
                "judge_stance": scores.get("stance"),
                "judge_individual_framing": scores.get("individual_framing"),
                "judge_collectivist_framing": scores.get("collectivist_framing"),
                "judge_recommendation": scores.get("recommendation"),
                "judge_tone": scores.get("tone"),
                "judge_refused": scores.get("refused"),
                "judge_framing_notes": scores.get("framing_notes"),
                "judge_status": scores.get("judge_status"),
                "your_stance": "",
                "your_individual_framing": "",
                "your_collectivist_framing": "",
                "your_recommendation": "",
                "your_tone": "",
                "your_refused": "",
                "notes": "",
            })

    print(f"\nwrote {PILOT_CSV}")
    print("\nNext: open the CSV, fill in your_* columns blind to the judge_* "
          "columns (hide them first), then compare. Aim for Cohen's kappa "
          ">= 0.5 on categorical fields before trusting full-corpus framing scores.")


if __name__ == "__main__":
    main()
