import csv
import io

import httpx
import pandas as pd

from . import config

BASE = "https://raw.githubusercontent.com/Jarviswang94/Multilingual_safety_benchmark/main"

COLUMNS = [
    "stimulus_set", "prompt_id", "language", "prompt_text",
    "category", "row_index", "origin_culture", "sub_bucket", "topic",
]


def filename(category: str, lang: str) -> str:
    if lang == "en":
        return f"{category}_n.csv" if category != "Crimes_And_Illegal_Activities" else f"{category}_en.csv"
    return f"{category}.csv"


def fetch(category: str, lang: str) -> list[str]:
    url = f"{BASE}/{lang}/{filename(category, lang)}"
    r = httpx.get(url, timeout=60)
    r.raise_for_status()
    text = r.text.lstrip("﻿")
    rows = list(csv.reader(io.StringIO(text)))
    return [(r[0] if r else "").strip() for r in rows]


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    out = []
    pid = 0
    for category in config.CATEGORIES:
        per_lang = {lang: fetch(category, lang) for lang in config.LANGUAGES}
        n_aligned = min(len(per_lang[l]) for l in config.LANGUAGES)
        take = min(config.PROMPTS_PER_CATEGORY, n_aligned)
        for row_idx in range(take):
            pid += 1
            for lang in config.LANGUAGES:
                out.append({
                    "stimulus_set": "xsafety",
                    "prompt_id": pid,
                    "language": lang,
                    "prompt_text": per_lang[lang][row_idx],
                    "category": category,
                    "row_index": row_idx,
                    "origin_culture": "",
                    "sub_bucket": "",
                    "topic": "",
                })
        print(f"{category}: took {take} / {n_aligned} aligned rows")

    df = pd.DataFrame(out, columns=COLUMNS)
    df.to_csv(config.PROMPTS, index=False, quoting=csv.QUOTE_ALL)
    print(f"wrote {len(df)} XSafety prompt-language pairs -> {config.PROMPTS}")
    print("To also include the cultural probe, now run: python -m src.load_cultural_probe")


if __name__ == "__main__":
    main()
