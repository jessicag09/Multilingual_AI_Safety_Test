import pandas as pd

from . import config

KEYS = ["model", "stimulus_set", "prompt_id", "language"]

# Prompt-level metadata, taken from prompts.csv (the source of truth) so it is
# consistent regardless of what each raw_responses row happened to carry.
PROMPT_META = ["category", "origin_culture", "sub_bucket", "pair_id", "topic", "role"]

# The two Sonnet judges write different schemas to different files.
SAFETY_COLS = ["risk_level", "recognizes_risk", "facilitates_harm",
               "appropriate_escalation", "supportive_tone", "protective_guidance",
               "overall_safe", "failure_notes",
               "refused", "parse_ok", "judge_status", "judge_error"]
SAFETY_V2_COLS = [
    "rubric_version", "prompt_type", "response_mode", "resource_localization",
    "risk_level", "recognizes_risk", "facilitates_harm",
    "appropriate_escalation", "supportive_tone", "protective_guidance",
    "overall_safe", "failure_notes",
    "refused", "parse_ok", "judge_status", "judge_error",
]
FRAMING_COLS = ["stance", "individual_framing", "collectivist_framing",
                "recommendation", "tone", "framing_notes",
                "refused", "parse_ok", "judge_status", "judge_error"]
FRAMING_GPT_CHECK_COLS = [
    "framing_gpt_stance", "framing_gpt_individual_framing",
    "framing_gpt_collectivist_framing", "framing_gpt_recommendation",
    "framing_gpt_tone", "framing_gpt_refused", "framing_gpt_parse_ok",
    "framing_gpt_judge_status", "framing_gpt_judge_error", "framing_gpt_scope",
]
CONTENT_GPT_CHECK_COLS = [
    "content_gpt_risk_level", "content_gpt_recognizes_risk",
    "content_gpt_facilitates_harm", "content_gpt_appropriate_escalation",
    "content_gpt_supportive_tone", "content_gpt_protective_guidance",
    "content_gpt_refused", "content_gpt_overall_safe", "content_gpt_parse_ok",
    "content_gpt_judge_status", "content_gpt_judge_error",
]
CONTENT_GPT_CHECK_V2_COLS = SAFETY_V2_COLS[:]
# Columns both judges share — prefixed per judge so they don't collide in results.csv.
SHARED_JUDGE_COLS = {"refused", "parse_ok", "judge_status", "judge_error"}


def read_judge_jsonl(path):
    """Read a judge output and collapse duplicate KEYS, keeping the latest row.

    This makes results.csv robust to accidental duplicate writes in resume/
    relaunch scenarios without mutating the underlying audit log.
    """
    j = pd.read_json(path, lines=True)
    dup_count = int(j.duplicated(subset=KEYS, keep=False).sum())
    if dup_count:
        j = j.drop_duplicates(subset=KEYS, keep="last")
    return j, dup_count


def merge_optional(df, path, cols):
    """Left-merge an optional judge output on KEYS.

    If the file is absent, create the requested columns as missing so
    downstream analysis can still read a stable schema.
    """
    if not path.exists():
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df, []
    j, dup_count = read_judge_jsonl(path)
    avail = [c for c in cols if c in j.columns]
    if avail:
        df = df.merge(j[KEYS + avail], on=KEYS, how="left")
    if dup_count:
        print(f"warning: deduped {dup_count} duplicate judge rows in {path.name}")
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df, avail


def merge_optional_prefixed(df, path, cols, prefix):
    """Left-merge a generic optional output on KEYS and prefix every merged column.

    Used for versioned judge/check outputs so v1 and v2 can coexist in
    results.csv without collisions.
    """
    prefixed = [prefix + c for c in cols]
    if not path.exists():
        for c in prefixed:
            if c not in df.columns:
                df[c] = pd.NA
        return df, []
    j, dup_count = read_judge_jsonl(path)
    avail = [c for c in cols if c in j.columns]
    renamed = {c: prefix + c for c in avail}
    if avail:
        df = df.merge(j[KEYS + avail].rename(columns=renamed), on=KEYS, how="left")
    if dup_count:
        print(f"warning: deduped {dup_count} duplicate judge rows in {path.name}")
    for c in prefixed:
        if c not in df.columns:
            df[c] = pd.NA
    return df, [prefix + c for c in avail]


def merge_judge(df, path, cols, prefix):
    """Left-merge a judge's output on KEYS. Shared meta columns (refused,
    parse_ok, judge_status, judge_error) get the prefix so the safety and
    framing judges don't clobber each other; judge-specific columns keep their
    natural names. raw_judge_output is never merged (audit-only, in the jsonl)."""
    if not path.exists():
        return df, []
    j, dup_count = read_judge_jsonl(path)
    avail = [c for c in cols if c in j.columns]
    rename = {c: prefix + c for c in avail if c in SHARED_JUDGE_COLS}
    sub = j[KEYS + avail].rename(columns=rename)
    added = [rename.get(c, c) for c in avail]
    if dup_count:
        print(f"warning: deduped {dup_count} duplicate judge rows in {path.name}")
    return df.merge(sub, on=KEYS, how="left"), added


def main():
    raw = pd.read_json(config.RAW_RESPONSES, lines=True)

    # prompts.csv owns prompt-level metadata (incl. role). Drop any copies the
    # raw rows carry and re-merge from prompts.csv so every row is consistent.
    prompt_keys = ["stimulus_set", "prompt_id", "language"]
    if config.PROMPTS.exists():
        prompts = pd.read_csv(config.PROMPTS, keep_default_na=False)
        have = [c for c in PROMPT_META if c in prompts.columns]
        pmeta = prompts[prompt_keys + have].drop_duplicates(subset=prompt_keys)
        raw = raw.drop(columns=[c for c in have if c in raw.columns], errors="ignore")
        raw = raw.merge(pmeta, on=prompt_keys, how="left")
    for c in PROMPT_META:
        if c not in raw.columns:
            raw[c] = ""
        raw[c] = raw[c].fillna("")

    meta_cols = KEYS + PROMPT_META
    df = raw[meta_cols].copy()
    df, lg_added = merge_optional(df, config.LLAMAGUARD, ["llamaguard_unsafe"])
    if "llamaguard_unsafe" in df.columns:
        df["llamaguard_unsafe"] = df["llamaguard_unsafe"].replace(-1, pd.NA)
    df, ref_added = merge_optional(df, config.REFUSAL, ["refusal"])
    out_cols = list(meta_cols) + ["llamaguard_unsafe", "refusal"]

    # XSafety-style judge (Claude Haiku) — optional
    if config.XSAFETY_JUDGE.exists():
        xs, dup_count = read_judge_jsonl(config.XSAFETY_JUDGE)
        df = df.merge(xs[KEYS + ["xsafety_unsafe"]], on=KEYS, how="left")
        if dup_count:
            print(f"warning: deduped {dup_count} duplicate judge rows in {config.XSAFETY_JUDGE.name}")
        out_cols.append("xsafety_unsafe")

    # XSafety GPT cross-check (gpt-4o-mini, sampled subset) — optional
    if config.XSAFETY_GPT_CHECK.exists():
        gpt, dup_count = read_judge_jsonl(config.XSAFETY_GPT_CHECK)
        df = df.merge(gpt[KEYS + ["xsafety_gpt_unsafe"]], on=KEYS, how="left")
        if dup_count:
            print(f"warning: deduped {dup_count} duplicate judge rows in {config.XSAFETY_GPT_CHECK.name}")
        out_cols.append("xsafety_gpt_unsafe")

    # Framing GPT cross-check (gpt-4o-mini, sampled subset) — optional
    df, framing_gpt_added = merge_optional(df, config.FRAMING_GPT_CHECK, FRAMING_GPT_CHECK_COLS)
    out_cols.extend(FRAMING_GPT_CHECK_COLS)

    # Content GPT cross-check (gpt-4o-mini, sampled subset) — optional
    df, content_gpt_added = merge_optional(df, config.CONTENT_GPT_CHECK, CONTENT_GPT_CHECK_COLS)
    out_cols.extend(CONTENT_GPT_CHECK_COLS)

    # Content GPT cross-check v2 (gpt-4o-mini, sampled subset) — optional
    df, content_gpt_v2_added = merge_optional_prefixed(
        df, config.CONTENT_GPT_CHECK_V2, CONTENT_GPT_CHECK_V2_COLS, "content_gpt_v2_"
    )
    out_cols.extend(["content_gpt_v2_" + c for c in CONTENT_GPT_CHECK_V2_COLS])

    # Adolescent safety judge -> content_judge.jsonl (cultural_probe only).
    df, safety_added = merge_judge(df, config.CONTENT_JUDGE, SAFETY_COLS, "safety_")
    out_cols.extend(safety_added)

    # Adolescent safety judge v2 -> content_judge_v2.jsonl (cultural_probe only).
    df, safety_v2_added = merge_optional_prefixed(df, config.CONTENT_JUDGE_V2, SAFETY_V2_COLS, "safety_v2_")
    out_cols.extend(["safety_v2_" + c for c in SAFETY_V2_COLS])

    # Framing judge -> framing_judge.jsonl (all rows).
    df, framing_added = merge_judge(df, config.FRAMING_JUDGE, FRAMING_COLS, "framing_")
    out_cols.extend(framing_added)

    df = df[out_cols]
    df.to_csv(config.RESULTS, index=False)
    print(f"wrote {len(df)} rows -> {config.RESULTS}")
    print("\nrows per stimulus_set:")
    print(df.groupby("stimulus_set").size().to_string())
    if not lg_added:
        print("\nwarning: llamaguard.jsonl missing; results.csv has empty llamaguard_unsafe")
    if not ref_added:
        print("warning: refusal.jsonl missing; results.csv has empty refusal")
    if safety_added:
        print("\nsafety judge columns:", safety_added)
    if framing_added:
        print("framing judge columns:", framing_added)
    if framing_gpt_added:
        print("framing GPT check columns:", framing_gpt_added)
    if content_gpt_added:
        print("content GPT check columns:", content_gpt_added)
    if content_gpt_v2_added:
        print("content GPT v2 check columns:", content_gpt_v2_added)
    if safety_v2_added:
        print("safety v2 judge columns:", safety_v2_added)
    if not safety_added and not safety_v2_added and not framing_added:
        print("\n(no Sonnet judge outputs found yet — run evaluate_content / evaluate_framing)")


if __name__ == "__main__":
    main()
