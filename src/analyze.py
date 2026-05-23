import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

PLOTS = config.DATA_DIR / "plots"


def bar(df, x, hue, y, title, fname, ymax=1.0):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x=x, hue=hue, y=y)
    plt.title(title)
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()


def main():
    PLOTS.mkdir(exist_ok=True)
    df = pd.read_csv(config.RESULTS)
    df["origin"] = df["model"].map(config.MODEL_ORIGIN)

    has_xsafety = (df["stimulus_set"] == "xsafety").any()
    has_probe = (df["stimulus_set"] == "cultural_probe").any()
    has_content = "stance" in df.columns and df["stance"].notna().any()
    has_xs_gpt = "xsafety_gpt_unsafe" in df.columns and df["xsafety_gpt_unsafe"].notna().any()
    print(f"stimulus sets present: xsafety={has_xsafety}, cultural_probe={has_probe}")
    print(f"content analyzer present: {has_content}")
    print(f"GPT cross-check present: {has_xs_gpt}")

    print("\n== overall ==")
    print(df[["llamaguard_unsafe", "refusal"]].mean().round(3).to_string())

    if has_xsafety:
        xs = df[df["stimulus_set"] == "xsafety"]

        print("\n== [XSafety] ORIGIN x LANGUAGE (WEIRD hypothesis) ==")
        by_ol = xs.groupby(["origin", "language"])[["llamaguard_unsafe", "refusal"]].mean().round(3).reset_index()
        print(by_ol.to_string(index=False))
        by_ol.to_csv(config.DATA_DIR / "summary_xsafety_origin_x_language.csv", index=False)

        print("\n== [XSafety] MODEL x LANGUAGE (per-model language effect) ==")
        by_ml = xs.groupby(["model", "language"])[["llamaguard_unsafe", "refusal"]].mean().round(3).reset_index()
        print(by_ml.to_string(index=False))
        by_ml.to_csv(config.DATA_DIR / "summary_xsafety_model_x_language.csv", index=False)
        bar(by_ml, "model", "language", "llamaguard_unsafe",
            "XSafety: Llama-Guard unsafe rate by model x language", "xsafety_unsafe_by_model_language.png")
        bar(by_ml, "model", "language", "refusal",
            "XSafety: Refusal rate by model x language", "xsafety_refusal_by_model_language.png")

        print("\n== [XSafety] WITHIN-ORIGIN CLUSTERING ==")
        for origin in ["WEIRD", "non_WEIRD"]:
            sub = by_ml[by_ml["model"].isin([m for m, o in config.MODEL_ORIGIN.items() if o == origin])]
            print(f"\n[{origin}]")
            print(sub.to_string(index=False))

    if has_probe:
        cp = df[df["stimulus_set"] == "cultural_probe"]

        print("\n== [Cultural Probe] ORIGIN-OF-MODEL x ORIGIN-OF-PROMPT ==")
        by_oo = cp.groupby(["origin", "origin_culture"])[["llamaguard_unsafe", "refusal"]].mean().round(3).reset_index()
        print(by_oo.to_string(index=False))
        by_oo.to_csv(config.DATA_DIR / "summary_probe_modelorigin_x_promptorigin.csv", index=False)

        # Cultural probe uses cultural_norm only (political_taboo dropped per Path D)
        print("\n== [Cultural Probe] MODEL x LANGUAGE ==")
        by_ml = cp.groupby(["model", "language"])[["llamaguard_unsafe", "refusal"]].mean().round(3).reset_index()
        print(by_ml.to_string(index=False))
        bar(by_ml, "model", "language", "refusal",
            "Cultural Probe: Refusal rate by model x language", "probe_refusal_by_model_language.png")

    if has_xsafety and has_probe:
        print("\n== RQ4: NEUTRALIZED vs CULTURALLY-LOADED (XSafety vs Cultural Probe) ==")
        contrast = df.groupby(["model", "stimulus_set"])[["llamaguard_unsafe", "refusal"]].mean().round(3).reset_index()
        print(contrast.to_string(index=False))
        contrast.to_csv(config.DATA_DIR / "summary_rq4_stimulus_contrast.csv", index=False)
        bar(contrast, "model", "stimulus_set", "refusal",
            "RQ4: Refusal rate by model x stimulus set", "rq4_refusal_by_stimulus.png")
        bar(contrast, "model", "stimulus_set", "llamaguard_unsafe",
            "RQ4: Unsafe rate by model x stimulus set", "rq4_unsafe_by_stimulus.png")

        print("\n== RQ4 deltas: cultural_probe - xsafety, per model ==")
        pivot_r = contrast.pivot(index="model", columns="stimulus_set", values="refusal")
        pivot_u = contrast.pivot(index="model", columns="stimulus_set", values="llamaguard_unsafe")
        deltas = pd.DataFrame({
            "refusal_delta": (pivot_r["cultural_probe"] - pivot_r["xsafety"]).round(3),
            "unsafe_delta": (pivot_u["cultural_probe"] - pivot_u["xsafety"]).round(3),
        })
        print(deltas.to_string())
        deltas.to_csv(config.DATA_DIR / "summary_rq4_deltas.csv")

    df["disagree"] = (df["llamaguard_unsafe"] != df["refusal"]).astype(int)
    print("\n== judge disagreement (llamaguard vs refusal) ==")
    print(df.groupby(["model", "language"])["disagree"].mean().round(3).reset_index().to_string(index=False))

    if "xsafety_unsafe" in df.columns and df["xsafety_unsafe"].notna().any():
        print("\n== XSafety-paper-style judge (Claude Haiku, XSafety paper prompt) ==")
        by_m = df.groupby("model")["xsafety_unsafe"].mean().round(3)
        print(by_m.to_string())
        by_ml = df.groupby(["model", "language"])["xsafety_unsafe"].mean().round(3).reset_index()
        print(by_ml.to_string(index=False))
        by_ml.to_csv(config.DATA_DIR / "summary_xsafety_judge_model_x_language.csv", index=False)

        lg_vs_xs = (df["llamaguard_unsafe"] != df["xsafety_unsafe"]).mean()
        ref_vs_xs = (df["refusal"] != df["xsafety_unsafe"]).mean()
        print(f"\n  judge disagreement: llamaguard vs xsafety = {lg_vs_xs:.3f}, refusal vs xsafety = {ref_vs_xs:.3f}")

    if has_xs_gpt:
        print("\n== XSafety judge robustness check (Claude Haiku vs gpt-4o-mini, sampled) ==")
        sub = df[df["xsafety_gpt_unsafe"].notna() & df["xsafety_unsafe"].notna()]
        agree = (sub["xsafety_unsafe"] == sub["xsafety_gpt_unsafe"]).mean()
        print(f"  sample size: {len(sub)}")
        print(f"  raw agreement: {agree:.3f}")
        # Cohen's kappa
        po = agree
        p1_claude = sub["xsafety_unsafe"].mean()
        p1_gpt = sub["xsafety_gpt_unsafe"].mean()
        pe = p1_claude * p1_gpt + (1 - p1_claude) * (1 - p1_gpt)
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        print(f"  Cohen's kappa: {kappa:.3f}")

    if has_content:
        print("\n== CONTENT ANALYZER (Claude Sonnet) ==")
        if "judge_status" in df.columns:
            print("\n  -- content judge status counts --")
            status_counts = df["judge_status"].fillna("missing").value_counts()
            print(status_counts.to_string())
            status_counts.to_csv(config.DATA_DIR / "summary_content_judge_status.csv")

        # Filter to successful judgments only — exclude parse failures, API errors, skipped rows
        if "judge_status" in df.columns:
            valid = df[df["judge_status"] == "ok"]
        elif "parse_ok" in df.columns:
            valid = df[df["parse_ok"] == True]
        else:
            valid = df.dropna(subset=["stance"])
        n_excluded = len(df) - len(valid)
        print(f"  valid responses for content analysis: {len(valid)} / {len(df)} ({n_excluded} excluded)")

        print("\n  -- stance distribution by model x stimulus_set --")
        stance_dist = valid.groupby(["model", "stimulus_set", "stance"]).size().unstack(fill_value=0)
        print(stance_dist.to_string())
        stance_dist.to_csv(config.DATA_DIR / "summary_content_stance.csv")

        # Two independent framing axes (1-5 each). A response can be high on both,
        # low on both, or asymmetric. Net framing = individual - collectivist (range -4 to +4).
        print("\n  -- individual_framing (1-5) by model x language --")
        ind_by_ml = valid.groupby(["model", "language"])["individual_framing"].mean().round(2).reset_index()
        print(ind_by_ml.to_string(index=False))
        ind_by_ml.to_csv(config.DATA_DIR / "summary_content_individual_framing.csv", index=False)
        bar(ind_by_ml, "model", "language", "individual_framing",
            "Content judge: individual framing (1-5) by model x language",
            "content_individual_framing_by_model_language.png", ymax=5.0)

        print("\n  -- collectivist_framing (1-5) by model x language --")
        col_by_ml = valid.groupby(["model", "language"])["collectivist_framing"].mean().round(2).reset_index()
        print(col_by_ml.to_string(index=False))
        col_by_ml.to_csv(config.DATA_DIR / "summary_content_collectivist_framing.csv", index=False)
        bar(col_by_ml, "model", "language", "collectivist_framing",
            "Content judge: collectivist framing (1-5) by model x language",
            "content_collectivist_framing_by_model_language.png", ymax=5.0)

        # Net framing: positive = leans individualist, negative = leans collectivist
        valid = valid.assign(net_framing=valid["individual_framing"] - valid["collectivist_framing"])
        print("\n  -- net_framing (individual - collectivist, range -4..+4) by model x language --")
        net_by_ml = valid.groupby(["model", "language"])["net_framing"].mean().round(2).reset_index()
        print(net_by_ml.to_string(index=False))
        net_by_ml.to_csv(config.DATA_DIR / "summary_content_net_framing.csv", index=False)

        print("\n  -- recommendation distribution by model x stimulus_set --")
        rec_dist = valid.groupby(["model", "stimulus_set", "recommendation"]).size().unstack(fill_value=0)
        print(rec_dist.to_string())
        rec_dist.to_csv(config.DATA_DIR / "summary_content_recommendation.csv")

        print("\n  -- tone distribution by model x stimulus_set --")
        tone_dist = valid.groupby(["model", "stimulus_set", "tone"]).size().unstack(fill_value=0)
        print(tone_dist.to_string())
        tone_dist.to_csv(config.DATA_DIR / "summary_content_tone.csv")

        if has_probe:
            print("\n  -- [Cultural Probe] framing axes by ORIGIN-OF-MODEL x ORIGIN-OF-PROMPT --")
            cp_valid = valid[valid["stimulus_set"] == "cultural_probe"]
            cp_frame = cp_valid.groupby(["origin", "origin_culture"])[
                ["individual_framing", "collectivist_framing", "net_framing"]
            ].mean().round(2).reset_index()
            print(cp_frame.to_string(index=False))
            cp_frame.to_csv(config.DATA_DIR / "summary_content_probe_framing.csv", index=False)

    n = len(df)
    n_unsafe = int(df["llamaguard_unsafe"].sum())
    n_refused = int(df["refusal"].sum())
    n_both = int(((df["llamaguard_unsafe"] == 1) & (df["refusal"] == 1)).sum())
    n_neither = int(((df["llamaguard_unsafe"] == 0) & (df["refusal"] == 0)).sum())
    print(f"\nconfusion: unsafe={n_unsafe}, refused={n_refused}, both={n_both}, neither={n_neither}, total={n}")


if __name__ == "__main__":
    main()
