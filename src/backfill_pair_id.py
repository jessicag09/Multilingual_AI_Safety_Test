"""One-shot backfill: add pair_id to raw_responses.jsonl rows generated
before the pair_id schema was introduced.

Joins raw_responses.jsonl against prompts.csv on (stimulus_set, prompt_id,
language) and writes pair_id back in place. Idempotent — safe to re-run.

IMPORTANT: do NOT run while generate.py is still running. It rewrites
raw_responses.jsonl atomically; concurrent appends would be lost.
"""

import json
import sys

import pandas as pd

from . import config


def main():
    if not config.RAW_RESPONSES.exists():
        print(f"no file at {config.RAW_RESPONSES}", file=sys.stderr)
        sys.exit(1)
    if not config.PROMPTS.exists():
        print(f"no file at {config.PROMPTS}", file=sys.stderr)
        sys.exit(1)

    # Build the (stimulus_set, prompt_id, language) -> pair_id lookup
    prompts = pd.read_csv(config.PROMPTS, keep_default_na=False)
    if "pair_id" not in prompts.columns:
        print("prompts.csv has no pair_id column — nothing to backfill",
              file=sys.stderr)
        sys.exit(1)
    lookup = {}
    for _, r in prompts.iterrows():
        key = (r["stimulus_set"], int(r["prompt_id"]), r["language"])
        lookup[key] = str(r.get("pair_id", "") or "")

    with open(config.RAW_RESPONSES) as f:
        rows = [json.loads(line) for line in f]

    n_added = 0
    n_already_set = 0
    n_missing_lookup = 0
    for r in rows:
        existing = r.get("pair_id", "")
        if existing:
            n_already_set += 1
            continue
        key = (r["stimulus_set"], int(r["prompt_id"]), r["language"])
        pair_id = lookup.get(key)
        if pair_id is None:
            n_missing_lookup += 1
            r["pair_id"] = ""
        else:
            r["pair_id"] = pair_id
            if pair_id:
                n_added += 1

    # Atomic write: write to .tmp, then move
    tmp = config.RAW_RESPONSES.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(config.RAW_RESPONSES)

    print(f"total rows:          {len(rows)}")
    print(f"already had pair_id: {n_already_set}")
    print(f"backfilled pair_id:  {n_added}")
    print(f"no lookup match:     {n_missing_lookup}  (set to empty)")


if __name__ == "__main__":
    main()
