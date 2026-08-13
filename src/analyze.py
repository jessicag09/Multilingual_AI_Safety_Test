import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

PLOTS = config.DATA_DIR / "plots"
ADVICE_XSAFETY_CATEGORIES = set(config.ADVICE_XSAFETY_CATEGORIES)
DIRECT_HARM_XSAFETY_CATEGORIES = {
    "Crimes_And_Illegal_Activities",
    "Unsafe_Instruction_Topic",
}
MODEL_ORDER = list(config.MODEL_ORIGIN.keys())
ORIGIN_ORDER = ["US-grouped", "China-grouped"]
PROMPT_ORIGIN_ORDER = ["US", "CN"]
XSAFETY_SLICE_ORDER = ["advice_style", "direct_harm"]
MODEL_LABELS = {
    "llama": "Llama",
    "gpt_oss": "gpt-oss",
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
}
AXIS_LABELS = {
    "model": "model",
    "origin": "developer-country grouping",
    "origin_culture": "prompt setting",
    "xsafety_slice": "xsafety slice",
}


def relabel_axis(ax, axis_name, values):
    if values is None:
        return
    labels = list(values)
    if axis_name == "model":
        labels = [MODEL_LABELS.get(v, v) for v in values]
    if axis_name == "x":
        ax.set_xticklabels(labels)
    elif axis_name == "y":
        ax.set_yticklabels(labels, rotation=0)


def pretty_axis_label(name):
    return AXIS_LABELS.get(name, name)


def bar(df, x, hue, y, title, fname, ymax=1.0):
    plt.figure(figsize=(9, 5))
    order = None
    hue_order = None
    if x == "model":
        order = MODEL_ORDER
    elif x == "origin":
        order = ORIGIN_ORDER
    elif x == "origin_culture":
        order = PROMPT_ORIGIN_ORDER
    elif x == "xsafety_slice":
        order = XSAFETY_SLICE_ORDER

    if hue == "model":
        hue_order = MODEL_ORDER
    elif hue == "origin":
        hue_order = ORIGIN_ORDER
    elif hue == "origin_culture":
        hue_order = PROMPT_ORIGIN_ORDER
    elif hue == "xsafety_slice":
        hue_order = XSAFETY_SLICE_ORDER
    elif hue == "language":
        hue_order = ["en", "zh"]

    sns.barplot(data=df, x=x, hue=hue, y=y, order=order, hue_order=hue_order)
    ax = plt.gca()
    if x == "model":
        relabel_axis(ax, "x", order or MODEL_ORDER)
    if hue == "model" and ax.legend_ is not None:
        for text, key in zip(ax.legend_.texts, hue_order or MODEL_ORDER):
            text.set_text(MODEL_LABELS.get(key, key))
    plt.title(title)
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()


def heatmap(
    df,
    index,
    columns,
    value,
    title,
    fname,
    vmin=None,
    vmax=None,
    cmap="YlGnBu",
    fmt=".2f",
    index_order=None,
    column_order=None,
):
    pivot = df.pivot(index=index, columns=columns, values=value)
    if index_order is not None:
        pivot = pivot.reindex(index=index_order)
    if column_order is not None:
        pivot = pivot.reindex(columns=column_order)
    plt.figure(figsize=(7, 4))
    ax = sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax)
    if index == "model":
        relabel_axis(ax, "y", pivot.index.tolist())
    if columns == "model":
        relabel_axis(ax, "x", pivot.columns.tolist())
    ax.set_ylabel(pretty_axis_label(index))
    ax.set_xlabel(pretty_axis_label(columns))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()


def heatmap_panel(
    df,
    index,
    columns,
    metrics,
    titles,
    fname,
    vmins=None,
    vmaxs=None,
    cmap="YlGnBu",
    fmt=".2f",
    index_order=None,
    column_order=None,
):
    vmins = vmins or [None] * len(metrics)
    vmaxs = vmaxs or [None] * len(metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric, subtitle, vmin, vmax in zip(axes, metrics, titles, vmins, vmaxs):
        pivot = df.pivot(index=index, columns=columns, values=metric)
        if index_order is not None:
            pivot = pivot.reindex(index=index_order)
        if column_order is not None:
            pivot = pivot.reindex(columns=column_order)
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
        if index == "model":
            relabel_axis(ax, "y", pivot.index.tolist())
        if columns == "model":
            relabel_axis(ax, "x", pivot.columns.tolist())
        ax.set_ylabel(pretty_axis_label(index))
        ax.set_xlabel(pretty_axis_label(columns))
        ax.set_title(subtitle)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close(fig)


def coerce_binary(series: pd.Series) -> pd.Series:
    """Normalize bool-like judge outputs to numeric 0/1 without breaking
    values that are already stored as floats in results.csv."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map({
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
    })
    return mapped


def main():
    PLOTS.mkdir(exist_ok=True)
    df = pd.read_csv(config.RESULTS)
    df["origin"] = df["model"].map(config.MODEL_ORIGIN)
    if "llamaguard_unsafe" in df.columns:
        df["llamaguard_unsafe"] = df["llamaguard_unsafe"].replace(-1, pd.NA)

    def present(col):
        return col in df.columns and df[col].notna().any()

    has_xsafety = (df["stimulus_set"] == "xsafety").any()
    has_probe = (df["stimulus_set"] == "cultural_probe").any()
    has_framing = present("stance")
    has_safety = present("overall_safe")
    has_llamaguard = present("llamaguard_unsafe")
    has_refusal = present("refusal")
    has_xs_gpt = present("xsafety_gpt_unsafe")
    has_framing_gpt = present("framing_gpt_individual_framing")
    has_content_gpt = present("content_gpt_overall_safe")
    base_cols = [c for c in ["llamaguard_unsafe", "refusal"] if present(c)]
    print(f"stimulus sets present: xsafety={has_xsafety}, cultural_probe={has_probe}")
    print(f"safety judge present: {has_safety} | framing judge present: {has_framing}")
    print(f"GPT cross-check present: xsafety={has_xs_gpt}, framing={has_framing_gpt}, content={has_content_gpt}")

    if base_cols:
        print("\n== overall ==")
        print(df[base_cols].mean().round(3).to_string())
    else:
        print("\n== overall ==")
        print("baseline judge outputs missing; run evaluate_llamaguard and/or evaluate_refusal first")

    if has_xsafety and base_cols:
        xs = df[df["stimulus_set"] == "xsafety"]

        print("\n== [XSafety] GROUPING x LANGUAGE (US-grouped vs China-grouped) ==")
        by_ol = xs.groupby(["origin", "language"])[base_cols].mean().round(3).reset_index()
        print(by_ol.to_string(index=False))
        by_ol.to_csv(config.DATA_DIR / "summary_xsafety_origin_x_language.csv", index=False)

        print("\n== [XSafety] MODEL x LANGUAGE (per-model language effect) ==")
        by_ml = xs.groupby(["model", "language"])[base_cols].mean().round(3).reset_index()
        print(by_ml.to_string(index=False))
        by_ml.to_csv(config.DATA_DIR / "summary_xsafety_model_x_language.csv", index=False)
        if has_llamaguard:
            bar(by_ml, "model", "language", "llamaguard_unsafe",
                "XSafety: Llama-Guard unsafe rate by model x language", "xsafety_unsafe_by_model_language.png")
        if has_refusal:
            bar(by_ml, "model", "language", "refusal",
                "XSafety: Refusal rate by model x language", "xsafety_refusal_by_model_language.png")

        print("\n== [XSafety] WITHIN-ORIGIN CLUSTERING ==")
        for origin in ORIGIN_ORDER:
            sub = by_ml[by_ml["model"].isin([m for m, o in config.MODEL_ORIGIN.items() if o == origin])]
            print(f"\n[{origin}]")
            print(sub.to_string(index=False))

    if has_probe and base_cols:
        cp = df[df["stimulus_set"] == "cultural_probe"]

        print("\n== [Cultural Probe] GROUPING-OF-MODEL x PROMPT-SETTING ==")
        by_oo = cp.groupby(["origin", "origin_culture"])[base_cols].mean().round(3).reset_index()
        print(by_oo.to_string(index=False))
        by_oo.to_csv(config.DATA_DIR / "summary_probe_modelorigin_x_promptorigin.csv", index=False)

        print("\n== [Cultural Probe] MODEL x LANGUAGE ==")
        by_ml = cp.groupby(["model", "language"])[base_cols].mean().round(3).reset_index()
        print(by_ml.to_string(index=False))
        if has_refusal:
            bar(by_ml, "model", "language", "refusal",
                "Cultural Probe: Refusal rate by model x language", "probe_refusal_by_model_language.png")

    if has_xsafety and has_probe and base_cols:
        print("\n== RQ4: NEUTRALIZED vs CULTURALLY-LOADED (XSafety vs Cultural Probe) ==")
        contrast = df.groupby(["model", "stimulus_set"])[base_cols].mean().round(3).reset_index()
        print(contrast.to_string(index=False))
        contrast.to_csv(config.DATA_DIR / "summary_rq4_stimulus_contrast.csv", index=False)
        if has_refusal:
            bar(contrast, "model", "stimulus_set", "refusal",
                "RQ4: Refusal rate by model x stimulus set", "rq4_refusal_by_stimulus.png")
        if has_llamaguard:
            bar(contrast, "model", "stimulus_set", "llamaguard_unsafe",
                "RQ4: Unsafe rate by model x stimulus set", "rq4_unsafe_by_stimulus.png")

        print("\n== RQ4 deltas: cultural_probe - xsafety, per model ==")
        deltas = pd.DataFrame(index=sorted(df["model"].dropna().unique()))
        if has_refusal:
            pivot_r = contrast.pivot(index="model", columns="stimulus_set", values="refusal")
            deltas["refusal_delta"] = (pivot_r["cultural_probe"] - pivot_r["xsafety"]).round(3)
        if has_llamaguard:
            pivot_u = contrast.pivot(index="model", columns="stimulus_set", values="llamaguard_unsafe")
            deltas["unsafe_delta"] = (pivot_u["cultural_probe"] - pivot_u["xsafety"]).round(3)
        print(deltas.to_string())
        deltas.to_csv(config.DATA_DIR / "summary_rq4_deltas.csv")

    if has_llamaguard and has_refusal:
        disagree_df = df[df["llamaguard_unsafe"].notna() & df["refusal"].notna()].copy()
        disagree_df["disagree"] = (disagree_df["llamaguard_unsafe"] != disagree_df["refusal"]).astype(int)
        print("\n== judge disagreement (llamaguard vs refusal) ==")
        print(disagree_df.groupby(["model", "language"])["disagree"].mean().round(3).reset_index().to_string(index=False))

    if has_xsafety and present("xsafety_unsafe"):
        xs_eval = df[df["stimulus_set"] == "xsafety"].copy()
        print("\n== XSafety-paper-style judge (Claude Haiku, XSafety paper prompt) ==")
        by_m = xs_eval.groupby("model")["xsafety_unsafe"].mean().round(3)
        print(by_m.to_string())
        by_ml = xs_eval.groupby(["model", "language"])["xsafety_unsafe"].mean().round(3).reset_index()
        by_ml["xsafety_safe"] = (1 - by_ml["xsafety_unsafe"]).round(3)
        print(by_ml.to_string(index=False))
        by_ml.to_csv(config.DATA_DIR / "summary_xsafety_judge_model_x_language.csv", index=False)
        bar(
            by_ml,
            "model",
            "language",
            "xsafety_safe",
            "XSafety: safe rate by model x language",
            "xsafety_safe_by_model_language.png",
        )

        msg = []
        if has_llamaguard:
            lg_sub = xs_eval[xs_eval["llamaguard_unsafe"].notna() & xs_eval["xsafety_unsafe"].notna()]
            lg_vs_xs = (lg_sub["llamaguard_unsafe"] != lg_sub["xsafety_unsafe"]).mean()
            msg.append(f"llamaguard vs xsafety = {lg_vs_xs:.3f}")
        if has_refusal:
            ref_sub = xs_eval[xs_eval["refusal"].notna() & xs_eval["xsafety_unsafe"].notna()]
            ref_vs_xs = (ref_sub["refusal"] != ref_sub["xsafety_unsafe"]).mean()
            msg.append(f"refusal vs xsafety = {ref_vs_xs:.3f}")
        if msg:
            print(f"\n  judge disagreement: {', '.join(msg)}")

        if "category" in xs_eval.columns:
            xs_eval["xsafety_slice"] = pd.NA
            xs_eval.loc[
                xs_eval["category"].isin(DIRECT_HARM_XSAFETY_CATEGORIES),
                "xsafety_slice",
            ] = "direct_harm"
            xs_eval.loc[
                xs_eval["category"].isin(ADVICE_XSAFETY_CATEGORIES),
                "xsafety_slice",
            ] = "advice_style"
            sliced = xs_eval[xs_eval["xsafety_slice"].notna()].copy()
            if len(sliced):
                slice_metrics = ["xsafety_unsafe"]
                if has_refusal:
                    slice_metrics.append("refusal")
                if has_llamaguard:
                    slice_metrics.append("llamaguard_unsafe")

                print("\n== XSafety sliced control (primary metric = xsafety_unsafe) ==")
                print("  slices: direct_harm = crimes/how-to | advice_style = mental-health/opinion/discrimination")

                by_so = sliced.groupby(["xsafety_slice", "origin"])[slice_metrics].mean().round(3).reset_index()
                by_so["xsafety_safe"] = (1 - by_so["xsafety_unsafe"]).round(3)
                if has_llamaguard and "llamaguard_unsafe" in by_so.columns:
                    by_so["llamaguard_safe"] = (1 - by_so["llamaguard_unsafe"]).round(3)
                print("\n  -- by slice x origin --")
                print(by_so.to_string(index=False))
                by_so.to_csv(config.DATA_DIR / "summary_xsafety_slice_origin.csv", index=False)
                bar(
                    by_so,
                    "xsafety_slice",
                    "origin",
                    "xsafety_safe",
                    "XSafety control: safe rate by slice x developer-country grouping",
                    "xsafety_slice_origin_safe.png",
                )
                bar(
                    by_so,
                    "origin",
                    "xsafety_slice",
                    "xsafety_safe",
                    "XSafety control: safe rate by developer-country grouping x slice",
                    "xsafety_origin_slice_safe_bar.png",
                )

                by_sm = sliced.groupby(["xsafety_slice", "model"])[slice_metrics].mean().round(3).reset_index()
                by_sm["xsafety_safe"] = (1 - by_sm["xsafety_unsafe"]).round(3)
                print("\n  -- by slice x model --")
                print(by_sm.to_string(index=False))
                by_sm.to_csv(config.DATA_DIR / "summary_xsafety_slice_model.csv", index=False)
                bar(
                    by_sm,
                    "model",
                    "xsafety_slice",
                    "xsafety_safe",
                    "XSafety control: safe rate by model x slice",
                    "xsafety_slice_model_safe.png",
                )
                heatmap(
                    by_so,
                    "xsafety_slice",
                    "origin",
                    "xsafety_unsafe",
                    "XSafety control: xsafety_unsafe by slice x grouping",
                    "xsafety_slice_origin_unsafe_heatmap.png",
                    vmin=0.0,
                    vmax=max(0.2, float(by_so["xsafety_unsafe"].max())),
                    index_order=XSAFETY_SLICE_ORDER,
                    column_order=ORIGIN_ORDER,
                )
                panel_metrics = ["xsafety_safe"]
                panel_titles = ["xsafety_safe"]
                panel_vmins = [0.0]
                panel_vmaxs = [1.0]
                if has_refusal:
                    panel_metrics.append("refusal")
                    panel_titles.append("refusal")
                    panel_vmins.append(0.0)
                    panel_vmaxs.append(1.0)
                if has_llamaguard and "llamaguard_safe" in by_so.columns:
                    panel_metrics.append("llamaguard_safe")
                    panel_titles.append("llamaguard_safe")
                    panel_vmins.append(0.0)
                    panel_vmaxs.append(1.0)
                heatmap_panel(
                    by_so,
                    "origin",
                    "xsafety_slice",
                    panel_metrics,
                    panel_titles,
                    "xsafety_origin_slice_panel.png",
                    vmins=panel_vmins,
                    vmaxs=panel_vmaxs,
                    index_order=ORIGIN_ORDER,
                    column_order=XSAFETY_SLICE_ORDER,
                )
                heatmap_panel(
                    by_sm,
                    "model",
                    "xsafety_slice",
                    ["xsafety_unsafe"] + (["refusal"] if has_refusal else []),
                    ["xsafety_unsafe", "refusal"] if has_refusal else ["xsafety_unsafe"],
                    "xsafety_slice_model_panel.png",
                    vmins=[0.0] * (2 if has_refusal else 1),
                    vmaxs=[max(0.2, float(by_sm["xsafety_unsafe"].max()))] + ([1.0] if has_refusal else []),
                    index_order=MODEL_ORDER,
                    column_order=XSAFETY_SLICE_ORDER,
                )

                by_sc = sliced.groupby(["xsafety_slice", "category"])[slice_metrics].mean().round(3).reset_index()
                print("\n  -- by slice x category --")
                print(by_sc.to_string(index=False))
                by_sc.to_csv(config.DATA_DIR / "summary_xsafety_slice_category.csv", index=False)

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

    if has_safety:
        print("\n== ADOLESCENT SAFETY JUDGE (Claude Sonnet, cultural_probe only) ==")
        if "safety_judge_status" in df.columns:
            print("\n  -- safety judge status counts --")
            sc = df["safety_judge_status"].fillna("missing").value_counts()
            print(sc.to_string())
            sc.to_csv(config.DATA_DIR / "summary_safety_judge_status.csv")

        if "safety_judge_status" in df.columns:
            sv = df[df["safety_judge_status"] == "ok"].copy()
        else:
            sv = df[df["overall_safe"].notna()].copy()
        sv = sv[sv["stimulus_set"] == "cultural_probe"]
        print(f"  valid adolescent safety judgments: {len(sv)}")

        if len(sv):
            safety_metrics = ["overall_safe", "facilitates_harm", "recognizes_risk",
                              "appropriate_escalation", "supportive_tone", "protective_guidance"]
            for m in safety_metrics:
                sv[m] = coerce_binary(sv[m])

            print("\n  -- safety rates by model x language --")
            by_ml = sv.groupby(["model", "language"])[safety_metrics].mean().round(3).reset_index()
            print(by_ml.to_string(index=False))
            by_ml.to_csv(config.DATA_DIR / "summary_safety_model_x_language.csv", index=False)
            bar(by_ml, "model", "language", "overall_safe",
                "Adolescent safety: overall_safe rate by model x language",
                "safety_overall_by_model_language.png")
            bar(by_ml, "model", "language", "facilitates_harm",
                "Adolescent safety: facilitates_harm rate by model x language",
                "safety_facilitates_harm_by_model_language.png")

            print("\n  -- safety rates by ORIGIN-OF-MODEL x ORIGIN-OF-PROMPT --")
            by_oo = sv.groupby(["origin", "origin_culture"])[safety_metrics].mean().round(3).reset_index()
            print(by_oo.to_string(index=False))
            by_oo.to_csv(config.DATA_DIR / "summary_safety_modelorigin_x_promptorigin.csv", index=False)
            heatmap_panel(
                by_oo,
                "origin",
                "origin_culture",
                ["overall_safe", "recognizes_risk", "appropriate_escalation", "protective_guidance"],
                ["overall_safe", "recognizes_risk", "appropriate_escalation", "protective_guidance"],
                "probe_safety_origin_prompt_panel.png",
                vmins=[0.0, 0.0, 0.0, 0.0],
                vmaxs=[1.0, 1.0, 1.0, 1.0],
                index_order=ORIGIN_ORDER,
                column_order=PROMPT_ORIGIN_ORDER,
            )

            if "role" in sv.columns and sv["role"].astype(str).str.len().gt(0).any():
                print("\n  -- safety rates by role (core / control / supplemental) x language --")
                by_role = sv.groupby(["role", "language"])[["overall_safe", "facilitates_harm"]].mean().round(3).reset_index()
                print(by_role.to_string(index=False))
                by_role.to_csv(config.DATA_DIR / "summary_safety_by_role.csv", index=False)

            if "risk_level" in sv.columns:
                print("\n  -- risk_level distribution by model --")
                rl = sv.groupby(["model", "risk_level"]).size().unstack(fill_value=0)
                print(rl.to_string())

    if has_safety and has_content_gpt:
        print("\n== CONTENT judge robustness check (Claude Sonnet vs gpt-4o-mini, sampled) ==")
        comp = df[df["stimulus_set"] == "cultural_probe"].copy()
        if "safety_judge_status" in comp.columns:
            comp = comp[comp["safety_judge_status"] == "ok"]
        if "content_gpt_judge_status" in comp.columns:
            comp = comp[comp["content_gpt_judge_status"] == "ok"]
        if "safety_parse_ok" in comp.columns:
            comp = comp[comp["safety_parse_ok"] == True]
        if "content_gpt_parse_ok" in comp.columns:
            comp = comp[comp["content_gpt_parse_ok"] == True]

        def exact_agreement(sub, left, right):
            pair = sub[sub[left].notna() & sub[right].notna()]
            if not len(pair):
                return float("nan")
            return (pair[left] == pair[right]).mean()

        def binary_kappa(sub, left, right):
            pair = sub[sub[left].notna() & sub[right].notna()]
            if not len(pair):
                return float("nan")
            po = (pair[left] == pair[right]).mean()
            p1_left = pair[left].mean()
            p1_right = pair[right].mean()
            pe = p1_left * p1_right + (1 - p1_left) * (1 - p1_right)
            return (po - pe) / (1 - pe) if pe < 1 else float("nan")

        def true_count(sub, col):
            pair = sub[sub[col].notna()]
            if not len(pair):
                return 0
            return int(pair[col].sum())

        def risk_level_mae(sub, left, right):
            pair = sub[sub[left].notna() & sub[right].notna()].copy()
            if not len(pair):
                return float("nan")
            mapping = {"low": 0, "moderate": 1, "high": 2, "acute": 3}
            left_num = pair[left].map(mapping)
            right_num = pair[right].map(mapping)
            return (left_num - right_num).abs().mean()

        rows = []
        for lang, sub in [("all", comp)] + [(l, comp[comp["language"] == l]) for l in sorted(comp["language"].dropna().unique())]:
            if not len(sub):
                continue
            rows.append({
                "language": lang,
                "n": len(sub),
                "overall_safe_agreement": round(exact_agreement(sub, "overall_safe", "content_gpt_overall_safe"), 3),
                "overall_safe_kappa": round(binary_kappa(sub, "overall_safe", "content_gpt_overall_safe"), 3),
                "facilitates_harm_agreement": round(exact_agreement(sub, "facilitates_harm", "content_gpt_facilitates_harm"), 3),
                "facilitates_harm_kappa": round(binary_kappa(sub, "facilitates_harm", "content_gpt_facilitates_harm"), 3),
                "facilitates_harm_true_sonnet": true_count(sub, "facilitates_harm"),
                "facilitates_harm_true_gpt": true_count(sub, "content_gpt_facilitates_harm"),
                "recognizes_risk_agreement": round(exact_agreement(sub, "recognizes_risk", "content_gpt_recognizes_risk"), 3),
                "appropriate_escalation_agreement": round(exact_agreement(sub, "appropriate_escalation", "content_gpt_appropriate_escalation"), 3),
                "supportive_tone_agreement": round(exact_agreement(sub, "supportive_tone", "content_gpt_supportive_tone"), 3),
                "protective_guidance_agreement": round(exact_agreement(sub, "protective_guidance", "content_gpt_protective_guidance"), 3),
                "risk_level_agreement": round(exact_agreement(sub, "risk_level", "content_gpt_risk_level"), 3),
                "risk_level_mae": round(risk_level_mae(sub, "risk_level", "content_gpt_risk_level"), 3),
            })
        check = pd.DataFrame(rows)
        if len(check):
            print(check.to_string(index=False))
            check.to_csv(config.DATA_DIR / "summary_content_gpt_check.csv", index=False)

    if has_framing:
        print("\n== FRAMING ANALYZER (Claude Sonnet) ==")
        if "framing_judge_status" in df.columns:
            print("\n  -- framing judge status counts --")
            status_counts = df["framing_judge_status"].fillna("missing").value_counts()
            print(status_counts.to_string())
            status_counts.to_csv(config.DATA_DIR / "summary_framing_judge_status.csv")

        # Filter to successful judgments only — exclude parse failures, API errors, skipped rows
        if "framing_judge_status" in df.columns:
            valid = df[df["framing_judge_status"] == "ok"]
        elif "framing_parse_ok" in df.columns:
            valid = df[df["framing_parse_ok"] == True]
        else:
            valid = df.dropna(subset=["stance"])
        n_excluded = len(df) - len(valid)
        print(f"  valid responses for framing analysis: {len(valid)} / {len(df)} ({n_excluded} excluded)")

        advice_mask = (
            (valid["stimulus_set"] == "cultural_probe")
            | valid["category"].isin(ADVICE_XSAFETY_CATEGORIES)
        )
        advice = valid[advice_mask]
        if len(advice):
            print("\n  -- advice-style primary framing slice (cultural_probe + advice XSafety categories) --")
            primary_by_ml = advice.groupby(["model", "language"])[
                ["individual_framing", "collectivist_framing"]
            ].mean().round(2).reset_index()
            print(primary_by_ml.to_string(index=False))
            primary_by_ml.to_csv(config.DATA_DIR / "summary_framing_primary_by_model_language.csv", index=False)

        print("\n  -- stance distribution by model x stimulus_set --")
        stance_dist = valid.groupby(["model", "stimulus_set", "stance"]).size().unstack(fill_value=0)
        print(stance_dist.to_string())
        stance_dist.to_csv(config.DATA_DIR / "summary_framing_stance.csv")

        # Two independent framing axes (1-5 each). A response can be high on both,
        # low on both, or asymmetric. Net framing = individual - collectivist (range -4 to +4).
        print("\n  -- individual_framing (1-5) by model x language --")
        ind_by_ml = valid.groupby(["model", "language"])["individual_framing"].mean().round(2).reset_index()
        print(ind_by_ml.to_string(index=False))
        ind_by_ml.to_csv(config.DATA_DIR / "summary_framing_individual.csv", index=False)
        bar(ind_by_ml, "model", "language", "individual_framing",
            "Framing judge: individual framing (1-5) by model x language",
            "framing_individual_by_model_language.png", ymax=5.0)

        print("\n  -- collectivist_framing (1-5) by model x language --")
        col_by_ml = valid.groupby(["model", "language"])["collectivist_framing"].mean().round(2).reset_index()
        print(col_by_ml.to_string(index=False))
        col_by_ml.to_csv(config.DATA_DIR / "summary_framing_collectivist.csv", index=False)
        bar(col_by_ml, "model", "language", "collectivist_framing",
            "Framing judge: collectivist framing (1-5) by model x language",
            "framing_collectivist_by_model_language.png", ymax=5.0)

        # Net framing: positive = leans individualist, negative = leans collectivist
        valid = valid.assign(net_framing=valid["individual_framing"] - valid["collectivist_framing"])
        print("\n  -- net_framing (individual - collectivist, range -4..+4) by model x language --")
        net_by_ml = valid.groupby(["model", "language"])["net_framing"].mean().round(2).reset_index()
        print(net_by_ml.to_string(index=False))
        net_by_ml.to_csv(config.DATA_DIR / "summary_framing_net.csv", index=False)

        print("\n  -- recommendation distribution by model x stimulus_set --")
        rec_dist = valid.groupby(["model", "stimulus_set", "recommendation"]).size().unstack(fill_value=0)
        print(rec_dist.to_string())
        rec_dist.to_csv(config.DATA_DIR / "summary_framing_recommendation.csv")

        print("\n  -- tone distribution by model x stimulus_set --")
        tone_dist = valid.groupby(["model", "stimulus_set", "tone"]).size().unstack(fill_value=0)
        print(tone_dist.to_string())
        tone_dist.to_csv(config.DATA_DIR / "summary_framing_tone.csv")

        if has_probe:
            print("\n  -- [Cultural Probe] framing axes by ORIGIN-OF-MODEL x ORIGIN-OF-PROMPT --")
            cp_valid = valid[valid["stimulus_set"] == "cultural_probe"]
            cp_frame = cp_valid.groupby(["origin", "origin_culture"])[
                ["individual_framing", "collectivist_framing", "net_framing"]
            ].mean().round(2).reset_index()
            print(cp_frame.to_string(index=False))
            cp_frame.to_csv(config.DATA_DIR / "summary_framing_probe.csv", index=False)
            heatmap_panel(
                cp_frame,
                "origin",
                "origin_culture",
                ["individual_framing", "collectivist_framing", "net_framing"],
                ["individual_framing", "collectivist_framing", "net_framing"],
                "framing_probe_origin_prompt_panel.png",
                vmins=[1.0, 1.0, -1.0],
                vmaxs=[5.0, 5.0, 1.5],
                index_order=ORIGIN_ORDER,
                column_order=PROMPT_ORIGIN_ORDER,
            )

    if has_framing and has_framing_gpt:
        print("\n== FRAMING judge robustness check (Claude Sonnet vs gpt-4o-mini, sampled) ==")
        comp = df.copy()
        if "framing_judge_status" in comp.columns:
            comp = comp[comp["framing_judge_status"] == "ok"]
        if "framing_gpt_judge_status" in comp.columns:
            comp = comp[comp["framing_gpt_judge_status"] == "ok"]
        if "framing_parse_ok" in comp.columns:
            comp = comp[comp["framing_parse_ok"] == True]
        if "framing_gpt_parse_ok" in comp.columns:
            comp = comp[comp["framing_gpt_parse_ok"] == True]

        def exact_agreement(sub, left, right):
            pair = sub[sub[left].notna() & sub[right].notna()]
            if not len(pair):
                return float("nan")
            return (pair[left] == pair[right]).mean()

        def spearman_corr(sub, left, right):
            pair = sub[sub[left].notna() & sub[right].notna()]
            if len(pair) < 2:
                return float("nan")
            # Avoid a hard scipy dependency by correlating average ranks directly.
            left_rank = pair[left].rank(method="average")
            right_rank = pair[right].rank(method="average")
            return left_rank.corr(right_rank, method="pearson")

        def mae(sub, left, right):
            pair = sub[sub[left].notna() & sub[right].notna()]
            if not len(pair):
                return float("nan")
            return (pair[left] - pair[right]).abs().mean()

        rows = []
        for lang, sub in [("all", comp)] + [(l, comp[comp["language"] == l]) for l in sorted(comp["language"].dropna().unique())]:
            if not len(sub):
                continue
            rows.append({
                "language": lang,
                "n": len(sub),
                "stance_agreement": round(exact_agreement(sub, "stance", "framing_gpt_stance"), 3),
                "recommendation_agreement": round(exact_agreement(sub, "recommendation", "framing_gpt_recommendation"), 3),
                "tone_agreement": round(exact_agreement(sub, "tone", "framing_gpt_tone"), 3),
                "refused_agreement": round(exact_agreement(sub, "framing_refused", "framing_gpt_refused"), 3),
                "individual_spearman": round(spearman_corr(sub, "individual_framing", "framing_gpt_individual_framing"), 3),
                "individual_mae": round(mae(sub, "individual_framing", "framing_gpt_individual_framing"), 3),
                "collectivist_spearman": round(spearman_corr(sub, "collectivist_framing", "framing_gpt_collectivist_framing"), 3),
                "collectivist_mae": round(mae(sub, "collectivist_framing", "framing_gpt_collectivist_framing"), 3),
            })
        check = pd.DataFrame(rows)
        if len(check):
            print(check.to_string(index=False))
            check.to_csv(config.DATA_DIR / "summary_framing_gpt_check.csv", index=False)

    if has_llamaguard and has_refusal:
        conf = df[df["llamaguard_unsafe"].notna() & df["refusal"].notna()]
        n = len(conf)
        n_unsafe = int(conf["llamaguard_unsafe"].sum())
        n_refused = int(conf["refusal"].sum())
        n_both = int(((conf["llamaguard_unsafe"] == 1) & (conf["refusal"] == 1)).sum())
        n_neither = int(((conf["llamaguard_unsafe"] == 0) & (conf["refusal"] == 0)).sum())
        print(f"\nconfusion: unsafe={n_unsafe}, refused={n_refused}, both={n_both}, neither={n_neither}, total={n}")


if __name__ == "__main__":
    main()
