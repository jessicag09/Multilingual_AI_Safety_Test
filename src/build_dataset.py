import pandas as pd

from . import config

KEYS = ["model", "stimulus_set", "prompt_id", "language"]


def main():
    raw = pd.read_json(config.RAW_RESPONSES, lines=True)
    lg = pd.read_json(config.LLAMAGUARD, lines=True)
    ref = pd.read_json(config.REFUSAL, lines=True)

    meta_cols = KEYS + ["category", "origin_culture", "sub_bucket", "topic"]
    df = raw[meta_cols].merge(lg, on=KEYS).merge(ref, on=KEYS)
    out_cols = list(meta_cols) + ["llamaguard_unsafe", "refusal"]

    # XSafety-style judge (Claude Haiku) — optional
    if config.XSAFETY_JUDGE.exists():
        xs = pd.read_json(config.XSAFETY_JUDGE, lines=True)
        df = df.merge(xs[KEYS + ["xsafety_unsafe"]], on=KEYS, how="left")
        out_cols.append("xsafety_unsafe")

    # XSafety GPT cross-check (gpt-4o-mini, sampled subset) — optional
    if config.XSAFETY_GPT_CHECK.exists():
        gpt = pd.read_json(config.XSAFETY_GPT_CHECK, lines=True)
        df = df.merge(gpt[KEYS + ["xsafety_gpt_unsafe"]], on=KEYS, how="left")
        out_cols.append("xsafety_gpt_unsafe")

    # Content/framing analyzer (Claude Sonnet) — optional.
    # Note: raw_judge_output is intentionally NOT merged into results.csv
    # (it bloats the file and is only useful for debugging parse failures;
    # stays in content_judge.jsonl for grep-based audit).
    if config.CONTENT_JUDGE.exists():
        ct = pd.read_json(config.CONTENT_JUDGE, lines=True)
        content_cols = ["stance", "individual_framing", "collectivist_framing",
                        "recommendation", "tone", "refused", "framing_notes",
                        "parse_ok", "judge_status", "judge_error"]
        # Only merge columns that actually exist (defensive against schema drift)
        avail = [c for c in content_cols if c in ct.columns]
        df = df.merge(ct[KEYS + avail], on=KEYS, how="left")
        out_cols.extend(avail)

    df = df[out_cols]
    df.to_csv(config.RESULTS, index=False)
    print(f"wrote {len(df)} rows -> {config.RESULTS}")
    print("\nrows per stimulus_set:")
    print(df.groupby("stimulus_set").size().to_string())
    print("\nmerged judge columns:", [c for c in out_cols
                                       if c in ("llamaguard_unsafe", "refusal",
                                                "xsafety_unsafe", "xsafety_gpt_unsafe",
                                                "stance", "individual_framing",
                                                "collectivist_framing", "recommendation",
                                                "tone", "framing_notes",
                                                "judge_status", "judge_error")])


if __name__ == "__main__":
    main()
