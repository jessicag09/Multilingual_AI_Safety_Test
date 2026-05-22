import csv
import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from . import config

load_dotenv()

LANG_NAME = {"en": "English", "fr": "French", "zh": "Mandarin Chinese"}

TRANSLATE_PROMPT = (
    "Translate the following text from English to {target}. Preserve the "
    "scenario, tone, severity, and any culturally-loaded phrasing exactly. "
    "Do not localize cultural references. Do not refuse, do not comment. "
    "Output only the translated text.\n\nText:\n{text}"
)


def translate(client: Anthropic, text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    msg = client.messages.create(
        model=config.JUDGE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user",
                   "content": TRANSLATE_PROMPT.format(target=LANG_NAME[target_lang], text=text)}],
    )
    return msg.content[0].text.strip()


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    with open(config.CULTURAL_PROBE_SOURCE) as f:
        base = json.load(f)["prompts"]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic()

    rows = []
    pid = 0
    for p in tqdm(base, desc="cultural probe"):
        pid += 1
        for lang in config.LANGUAGES:
            rows.append({
                "prompt_id": pid,
                "origin_culture": p["origin_culture"],
                "sub_bucket": p["sub_bucket"],
                "topic": p["topic"],
                "language": lang,
                "prompt_text": translate(client, p["text_en"], lang),
            })

    with open(config.CULTURAL_PROBE_PROMPTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} prompt-language pairs -> {config.CULTURAL_PROBE_PROMPTS}")


if __name__ == "__main__":
    main()
