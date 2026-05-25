import argparse
import csv
import json
from dotenv import load_dotenv
from tqdm import tqdm

from . import config
from .models.base import get_client

load_dotenv()


def key(r):
    return (r["model"], r["stimulus_set"], int(r["prompt_id"]), r["language"])


def load_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            done.add(key(json.loads(line)))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["together", "openrouter", "ollama"], default="together")
    ap.add_argument("--max-calls", type=int, default=None,
                    help="cap new generations this run (resume picks up next time)")
    args = ap.parse_args()

    client = get_client(args.backend)

    with open(config.PROMPTS) as f:
        prompts = list(csv.DictReader(f))

    done = load_done(config.RAW_RESPONSES)
    print(f"resuming: {len(done)} already done")

    new_count = 0
    with open(config.RAW_RESPONSES, "a") as out:
        for model_key, model_id in config.MODELS.items():
            for row in tqdm(prompts, desc=model_key):
                row_key = (model_key, row["stimulus_set"], int(row["prompt_id"]), row["language"])
                if row_key in done:
                    continue
                if args.max_calls is not None and new_count >= args.max_calls:
                    print(f"\nhit --max-calls={args.max_calls}; stopping")
                    return
                try:
                    resp = client.generate(
                        model_id=model_id,
                        system=config.SYSTEM_PROMPT,
                        user=row["prompt_text"],
                        temperature=config.DECODE["temperature"],
                        top_p=config.DECODE["top_p"],
                        max_new_tokens=config.DECODE["max_new_tokens"],
                    )
                    error = None
                except Exception as e:
                    resp, error = "", str(e)

                out.write(json.dumps({
                    "model": model_key,
                    "stimulus_set": row["stimulus_set"],
                    "prompt_id": int(row["prompt_id"]),
                    "language": row["language"],
                    "category": row.get("category", ""),
                    "origin_culture": row.get("origin_culture", ""),
                    "sub_bucket": row.get("sub_bucket", ""),
                    "pair_id": row.get("pair_id", ""),
                    "topic": row.get("topic", ""),
                    "role": row.get("role", ""),
                    "prompt_text": row["prompt_text"],
                    "raw_response": resp,
                    "error": error,
                }, ensure_ascii=False) + "\n")
                out.flush()
                new_count += 1


if __name__ == "__main__":
    main()
