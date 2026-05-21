import pandas as pd

from . import config

KEYS = ["model", "prompt_id", "language"]


def main():
    raw = pd.read_json(config.RAW_RESPONSES, lines=True)
    lg = pd.read_json(config.LLAMAGUARD, lines=True)
    ref = pd.read_json(config.REFUSAL, lines=True)

    df = raw[KEYS + ["category"]].merge(lg, on=KEYS).merge(ref, on=KEYS)
    df = df[["model", "prompt_id", "category", "language",
             "llamaguard_unsafe", "refusal"]]
    df.to_csv(config.RESULTS, index=False)
    print(f"wrote {len(df)} rows -> {config.RESULTS}")


if __name__ == "__main__":
    main()
