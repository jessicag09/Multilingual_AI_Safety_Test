import json
import os

import jailbreakbench as jbb
from dotenv import load_dotenv
from tqdm import tqdm

from . import config

load_dotenv()


def main():
    if not os.environ.get("TOGETHER_API_KEY"):
        raise RuntimeError("TOGETHER_API_KEY not set (needed for Llama Guard via JBB)")

    with open(config.RAW_RESPONSES) as f:
        rows = [json.loads(l) for l in f]

    classifier = jbb.classifier.Classifier(api_key=os.environ["TOGETHER_API_KEY"])

    prompts = [r["behavior_translated"] for r in rows]
    responses = [r["raw_response"] or "" for r in rows]

    BATCH = 25
    labels = []
    for i in tqdm(range(0, len(rows), BATCH), desc="llamaguard"):
        labels.extend(classifier(prompts=prompts[i:i+BATCH], responses=responses[i:i+BATCH]))

    with open(config.LLAMAGUARD, "w") as out:
        for r, lab in zip(rows, labels):
            out.write(json.dumps({
                "model": r["model"],
                "prompt_id": r["prompt_id"],
                "language": r["language"],
                "llamaguard_unsafe": int(bool(lab)),
            }) + "\n")
    print(f"wrote {len(labels)} llamaguard labels -> {config.LLAMAGUARD}")


if __name__ == "__main__":
    main()
