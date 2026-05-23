import csv
import json

import pandas as pd

from . import config
from .load_xsafety import COLUMNS


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    with open(config.CULTURAL_PROBE_SOURCE) as f:
        base = json.load(f)["prompts"]

    rows = []
    for pid, p in enumerate(base, start=1):
        for lang in config.LANGUAGES:
            text = p.get(f"text_{lang}")
            if not text or text.startswith("TODO"):
                raise ValueError(
                    f"prompt {pid} ({p.get('topic')}) is missing text_{lang}; "
                    f"fill in {config.CULTURAL_PROBE_SOURCE.name} before running"
                )
            rows.append({
                "stimulus_set": "cultural_probe",
                "prompt_id": pid,
                "language": lang,
                "prompt_text": text,
                "category": "",
                "row_index": "",
                "origin_culture": p["origin_culture"],
                "sub_bucket": p["sub_bucket"],
                "topic": p["topic"],
            })

    new = pd.DataFrame(rows, columns=COLUMNS)
    if config.PROMPTS.exists():
        existing = pd.read_csv(config.PROMPTS, keep_default_na=False)
        existing = existing[existing["stimulus_set"] != "cultural_probe"]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
        print("warning: data/prompts.csv didn't exist; cultural probe will be the only stimulus set")

    combined.to_csv(config.PROMPTS, index=False, quoting=csv.QUOTE_ALL)
    print(f"appended {len(new)} cultural probe rows -> {config.PROMPTS} (total: {len(combined)})")


if __name__ == "__main__":
    main()
