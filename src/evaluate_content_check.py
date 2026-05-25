"""Cross-judge robustness check on adolescent-safety labels.

Re-scores a random sample of cultural_probe responses with gpt-4o-mini using
the same safety rubric as evaluate_content.py, so we can report agreement
against the Claude Sonnet adolescent-safety judge.

By default this samples only cultural_probe rows, because the rubric is tuned
to the adolescent benchmark and that is the main safety outcome. Use
--stimulus-set all only if you intentionally want to extend the check beyond
its intended scope.

Run after evaluate_content.py has finished. build_dataset.py and analyze.py
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
from .evaluate_content import FAILURE, RUBRIC, normalize, parse_json, validation_error

load_dotenv()

SEED = 1337


def sample_pool(rows: list[dict], stimulus_set: str) -> list[dict]:
    if stimulus_set == "all":
        return rows
    return [r for r in rows if r.get("stimulus_set") == stimulus_set]


def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin sample across languages so the check is not lopsided by en/zh."""
    buckets = {}
    for r in rows:
        buckets.setdefault(r.get("language") or "", []).append(r)
    for b in buckets.values():
        rng.shuffle(b)

    keys = sorted(buckets)
    picked = []
    while len(picked) < n and any(buckets.values()):
        for k in keys:
            if not buckets[k]:
                continue
            picked.append(buckets[k].pop())
            if len(picked) >= n:
                break
    return picked


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
                model=config.OPENAI_CONTENT_CHECK_MODEL,
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
                    "judge_error": validation_error(scores),
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
    ap.add_argument("--n", type=int, default=config.CONTENT_GPT_CHECK_SAMPLE_SIZE,
                    help="sample size for the content cross-judge check")
    ap.add_argument("--stimulus-set", default="cultural_probe",
                    help="sample rows from this stimulus_set; default is cultural_probe")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI()

    with open(config.RAW_RESPONSES) as f:
        rows = [json.loads(l) for l in f]

    pool = sample_pool(rows, args.stimulus_set)
    rng = random.Random(SEED)
    sample = stratified_sample(pool, min(args.n, len(pool)), rng)
    sample_lang_counts = {}
    for r in sample:
        lang = r.get("language") or ""
        sample_lang_counts[lang] = sample_lang_counts.get(lang, 0) + 1

    done = load_done(config.CONTENT_GPT_CHECK)
    pending = [r for r in sample
               if (r["model"], r["stimulus_set"], r["prompt_id"], r["language"]) not in done]
    print(f"stimulus_set: {args.stimulus_set} | pool size: {len(pool)} | sample size: {len(sample)} | "
          f"already checked: {len(done)} | pending: {len(pending)}")
    print("sample language counts:", ", ".join(f"{k or 'unknown'}={v}" for k, v in sorted(sample_lang_counts.items())))

    with open(config.CONTENT_GPT_CHECK, "a") as out:
        for r in tqdm(pending, desc="content-gpt-check"):
            scores = judge(client, r)
            out.write(json.dumps({
                "model": r["model"],
                "stimulus_set": r["stimulus_set"],
                "prompt_id": r["prompt_id"],
                "language": r["language"],
                "content_gpt_risk_level": scores.get("risk_level"),
                "content_gpt_recognizes_risk": scores.get("recognizes_risk"),
                "content_gpt_facilitates_harm": scores.get("facilitates_harm"),
                "content_gpt_appropriate_escalation": scores.get("appropriate_escalation"),
                "content_gpt_supportive_tone": scores.get("supportive_tone"),
                "content_gpt_protective_guidance": scores.get("protective_guidance"),
                "content_gpt_refused": scores.get("refused"),
                "content_gpt_overall_safe": scores.get("overall_safe"),
                "content_gpt_parse_ok": scores.get("parse_ok"),
                "content_gpt_judge_status": scores.get("judge_status"),
                "content_gpt_judge_error": scores.get("judge_error"),
                "raw_judge_output": scores.get("raw_judge_output"),
            }) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
