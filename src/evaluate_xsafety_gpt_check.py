"""Cross-judge robustness check on XSafety labels.

Re-scores a random N-sample of rows with gpt-4o-mini using the same
XSafety prompt, so we can report inter-rater agreement (Cohen's kappa)
against the Claude judge. Sample size is set in config; the same seed
is used so the sample is reproducible.

Run after evaluate_xsafety.py has finished. analyze.py picks up the file
and reports agreement.
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
from .evaluate_xsafety import JUDGE, scenario_label

load_dotenv()

SEED = 1337


def judge(client: OpenAI, row: dict) -> str:
    if not row.get("raw_response"):
        return "yes"
    for attempt in range(3):
        try:
            msg = client.chat.completions.create(
                model=config.OPENAI_XSAFETY_CHECK_MODEL,
                max_tokens=4,
                temperature=0,
                messages=[{"role": "user", "content": JUDGE.format(
                    prompt=row["prompt_text"],
                    response=row["raw_response"],
                    scenario=scenario_label(row),
                )}],
            )
            ans = msg.choices[0].message.content.strip().lower()
            return "yes" if ans.startswith("y") else "no"
        except Exception:
            if attempt == 2:
                return "error"
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
    ap.add_argument("--n", type=int, default=config.XSAFETY_GPT_CHECK_SAMPLE_SIZE,
                    help="sample size for the cross-judge check")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI()

    with open(config.RAW_RESPONSES) as f:
        rows = [json.loads(l) for l in f]

    # XSafety methodology cross-check: only sample XSafety rows. The XSafety
    # paper's prompt template was designed for its own benchmark categories;
    # cultural_probe rows use an adapted scenario label that's our extension.
    xsafety_rows = [r for r in rows if r["stimulus_set"] == "xsafety"]
    rng = random.Random(SEED)
    sample = rng.sample(xsafety_rows, min(args.n, len(xsafety_rows)))

    done = load_done(config.XSAFETY_GPT_CHECK)
    pending = [r for r in sample
               if (r["model"], r["stimulus_set"], r["prompt_id"], r["language"]) not in done]
    print(f"sample size: {len(sample)} | already checked: {len(done)} | pending: {len(pending)}")

    with open(config.XSAFETY_GPT_CHECK, "a") as out:
        for r in tqdm(pending, desc="xsafety-gpt-check"):
            ans = judge(client, r)
            out.write(json.dumps({
                "model": r["model"],
                "stimulus_set": r["stimulus_set"],
                "prompt_id": r["prompt_id"],
                "language": r["language"],
                "xsafety_gpt_answer": ans,
                "xsafety_gpt_unsafe": int(ans == "no") if ans in ("yes", "no") else None,
            }) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
