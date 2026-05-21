import csv
import os
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

from . import config
from .templates import wrap

load_dotenv()

LANG_NAME = {"en": "English", "fr": "French", "zh": "Mandarin Chinese"}

TRANSLATE_PROMPT = (
    "Translate the following text from English to {target}. "
    "Preserve the scenario, tone, and severity exactly. Do not refuse, do not "
    "comment, do not localize cultural references. Output only the translated "
    "text, no preamble.\n\nText:\n{text}"
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
    behaviors = pd.read_csv(config.BEHAVIORS)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic()

    rows = []
    for _, b in tqdm(behaviors.iterrows(), total=len(behaviors), desc="behaviors"):
        for lang in config.LANGUAGES:
            translated = translate(client, b["behavior"], lang)
            rows.append({
                "prompt_id": int(b["prompt_id"]),
                "jbb_index": int(b["jbb_index"]),
                "category": b["category"],
                "language": lang,
                "behavior_translated": translated,
                "prompt_text": wrap(translated, lang),
            })

    with open(config.PROMPTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} prompt-language pairs -> {config.PROMPTS}")


if __name__ == "__main__":
    main()
