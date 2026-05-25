"""Cross-judge robustness check on framing labels.

Re-scores a random sample of responses with gpt-4o-mini using the same
framing rubric as evaluate_framing.py, so we can report agreement and axis
correlation against the Claude Sonnet framing judge.

By default this samples the primary framing slice: cultural_probe plus the
advice-style XSafety categories that matter for the main framing claim. Use
--scope all to sample from every response instead.

Run after evaluate_framing.py has finished. build_dataset.py and analyze.py
pick up the output automatically.
"""

import argparse
import json
import os
import random
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from . import config
from .evaluate_framing import FAILURE, RUBRIC, normalize, parse_json

load_dotenv()

SEED = 1337


def sample_pool(rows: list[dict], scope: str) -> list[dict]:
    if scope == "all":
        return rows
    advice = set(config.ADVICE_XSAFETY_CATEGORIES)
    return [
        r for r in rows
        if r.get("stimulus_set") == "cultural_probe"
        or r.get("category") in advice
    ]


def judge(client: OpenAI, row: dict) -> dict:
    if row.get("error"):
        return {
            **FAILURE,
            "judge_status": "skipped_generation_error",
            "raw_judge_output": None,
            "judge_error": str(row["error"]),
        }
    if not row.get("raw_response"):
        return {
            **FAILURE,
            "judge_status": "skipped_empty_response",
            "raw_judge_output": None,
            "judge_error": None,
        }
    for attempt in range(3):
        try:
            msg = client.chat.completions.create(
                model=config.OPENAI_FRAMING_CHECK_MODEL,
                max_tokens=400,
                temperature=0,
                messages=[{"role": "user", "content": RUBRIC.format(
                    prompt=row["prompt_text"],
                    response=row["raw_response"],
                )}],
            )
            text = (msg.choices[0].message.content or "").strip()
            parsed = parse_json(text)
            if parsed is None:
                return {
                    **FAILURE,
                    "judge_status": "parse_failure",
                    "raw_judge_output": text,
                    "judge_error": None,
                }
            scores = normalize(parsed)
            if not scores["parse_ok"]:
                return {
                    **scores,
                    "judge_status": "schema_validation_failure",
                    "raw_judge_output": text,
                    "judge_error": None,
                }
            return {
                **scores,
                "judge_status": "ok",
                "raw_judge_output": text,
                "judge_error": None,
            }
        except Exception as exc:
            if attempt == 2:
                return {
                    **FAILURE,
                    "judge_status": "api_failure",
                    "raw_judge_output": None,
                    "judge_error": str(exc),
                }
            time.sleep(2 ** attempt)


def load_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["model"], r["stimulus_set"], r["prompt_id"], r["language"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.FRAMING_GPT_CHECK_SAMPLE_SIZE,
                    help="sample size for the framing cross-judge check")
    ap.add_argument("--scope", choices=["primary", "all"], default="primary",
                    help="sample the primary framing slice (default) or all responses")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI()

    with open(config.RAW_RESPONSES) as f:
        rows = [json.loads(l) for l in f]

    pool = sample_pool(rows, args.scope)
    rng = random.Random(SEED)
    sample = rng.sample(pool, min(args.n, len(pool)))

    done = load_done(config.FRAMING_GPT_CHECK)
    pending = [r for r in sample
               if (r["model"], r["stimulus_set"], r["prompt_id"], r["language"]) not in done]
    print(f"scope: {args.scope} | pool size: {len(pool)} | sample size: {len(sample)} | "
          f"already checked: {len(done)} | pending: {len(pending)}")

    with open(config.FRAMING_GPT_CHECK, "a") as out:
        for r in tqdm(pending, desc="framing-gpt-check"):
            scores = judge(client, r)
            out.write(json.dumps({
                "model": r["model"],
                "stimulus_set": r["stimulus_set"],
                "prompt_id": r["prompt_id"],
                "language": r["language"],
                "framing_gpt_scope": args.scope,
                "framing_gpt_stance": scores.get("stance"),
                "framing_gpt_individual_framing": scores.get("individual_framing"),
                "framing_gpt_collectivist_framing": scores.get("collectivist_framing"),
                "framing_gpt_recommendation": scores.get("recommendation"),
                "framing_gpt_tone": scores.get("tone"),
                "framing_gpt_refused": scores.get("refused"),
                "framing_gpt_parse_ok": scores.get("parse_ok"),
                "framing_gpt_judge_status": scores.get("judge_status"),
                "framing_gpt_judge_error": scores.get("judge_error"),
                "raw_judge_output": scores.get("raw_judge_output"),
            }) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
