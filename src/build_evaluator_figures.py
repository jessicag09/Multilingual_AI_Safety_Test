"""Generate the composite-evaluator figures 01-07 into results/figures/.

Until now these figures had no tracked generator — they were committed as
binaries and could silently drift from the analysis code. This module
reconstructs them from data/results.csv (the same judged rows analyze.py
summarises) so they are reproducible. Re-run with:

    python -m src.build_evaluator_figures

Figure map:
  01  adolescent overall_safe by model x language        (caveat target)
  02  safety sub-scores by grouping x prompt-setting      (caveat target)
  03  XSafety control panel by grouping x slice
  04  language effect per model (en - zh, overall_safe)
  05  language effect, cross-judge (Sonnet vs gpt-4o-mini)
  06  framing axes by grouping x prompt-setting
  07  safety components by grouping                        (caveat target)

Run order: analyze.py (dumps summary CSVs) -> this module (01-07) ->
extra_analyses.py (08-13 + stamps the tone caveat onto 01/02/07).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from . import config

FIG = config.ROOT / "results" / "figures"
UNCAVEATED = FIG / "extra" / "uncaveated"
EXTRA = FIG / "extra"

# Order so the safety story reads top-to-bottom / left-to-right: the two
# China-grouped models first, then the two US-grouped models.
MODEL_ORDER = ["deepseek", "qwen", "gpt_oss", "llama"]
ORIGIN_ORDER = ["China-grouped", "US-grouped"]
PROMPT_ORIGIN_ORDER = ["US", "CN"]
MODEL_LABELS = {
    "deepseek": "DeepSeek",
    "qwen": "Qwen",
    "gpt_oss": "gpt-oss",
    "llama": "Llama",
}
ORIGIN_PALETTE = {"China-grouped": "#2563eb", "US-grouped": "#dc2626"}
LANG_ORDER = ["en", "zh"]
AXIS_LABELS = {"origin": "developer-country grouping", "origin_culture": "prompt setting"}

DIRECT_HARM_XSAFETY_CATEGORIES = {"Crimes_And_Illegal_Activities", "Unsafe_Instruction_Topic"}
ADVICE_XSAFETY_CATEGORIES = set(config.ADVICE_XSAFETY_CATEGORIES)
HARMFUL_REQUEST = {"self_harm_concealment", "disordered_eating", "online_older_stranger",
                   "controlling_parents_runaway", "counterfeit_pill_fentanyl", "substance_peer_pressure"}
SAFETY_METRICS = ["overall_safe", "facilitates_harm", "recognizes_risk",
                  "appropriate_escalation", "supportive_tone", "protective_guidance"]


def coerce_binary(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return s.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})


def save(fname, caveat=False):
    """Save the current figure as a pristine PNG. For caveat targets, also refresh
    the uncaveated backup so extra_analyses.py re-stamps from the new pristine image."""
    FIG.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG / fname, dpi=150)
    if caveat:
        UNCAVEATED.mkdir(parents=True, exist_ok=True)
        plt.savefig(UNCAVEATED / fname, dpi=150)
    plt.close()


def relabel_models(ax, order, axis="x"):
    labels = [MODEL_LABELS.get(m, m) for m in order]
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(labels, rotation=0)


def heatmap_panel(df, index, columns, metrics, fname, vmins, vmaxs,
                  index_order, column_order, caveat=False, cmap="YlGnBu"):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric, vmin, vmax in zip(axes, metrics, vmins, vmaxs):
        pivot = (df.pivot(index=index, columns=columns, values=metric)
                 .reindex(index=index_order, columns=column_order))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
        ax.set_ylabel(AXIS_LABELS.get(index, index))
        ax.set_xlabel(AXIS_LABELS.get(columns, columns))
        ax.set_title(metric)
    plt.tight_layout()
    save(fname, caveat=caveat)


def build():
    df = pd.read_csv(config.RESULTS)
    df["origin"] = df["model"].map(config.MODEL_ORIGIN)

    # Adolescent (cultural_probe) judged rows, same filter analyze.py uses.
    sv = df[df["stimulus_set"] == "cultural_probe"].copy()
    if "safety_judge_status" in sv.columns:
        sv = sv[sv["safety_judge_status"] == "ok"]
    for m in SAFETY_METRICS:
        sv[m] = coerce_binary(sv[m])
    sv_hr = sv[sv["pair_id"].isin(HARMFUL_REQUEST)].copy()
    sv_nonhr = sv[~sv["pair_id"].isin(HARMFUL_REQUEST)].copy()
    sv_nonhr_paired = sv_nonhr[sv_nonhr["role"] != "supplemental_unpaired"].copy()

    # ---- 01: adolescent overall_safe by model x language (reordered) ----
    by_ml = sv.groupby(["model", "language"])["overall_safe"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=by_ml, x="model", y="overall_safe", hue="language",
                     order=MODEL_ORDER, hue_order=LANG_ORDER)
    relabel_models(ax, MODEL_ORDER)
    plt.ylim(0, 1.0)
    plt.title("Adolescent safety: overall_safe by model x language")
    plt.tight_layout()
    save("01_adolescent_safety_by_model.png", caveat=True)

    # ---- 02: safety sub-scores by grouping x prompt-setting ----
    by_oo = sv.groupby(["origin", "origin_culture"])[SAFETY_METRICS].mean().reset_index()
    heatmap_panel(
        by_oo, "origin", "origin_culture",
        ["overall_safe", "recognizes_risk", "appropriate_escalation", "protective_guidance"],
        "02_adolescent_safety_origin_x_prompt.png",
        vmins=[0.0, 0.0, 0.0, 0.0], vmaxs=[1.0, 1.0, 1.0, 1.0],
        index_order=ORIGIN_ORDER, column_order=PROMPT_ORIGIN_ORDER, caveat=True,
    )

    # ---- 03: XSafety control panel by grouping x slice ----
    xs = df[df["stimulus_set"] == "xsafety"].copy()
    xs["xsafety_unsafe"] = coerce_binary(xs["xsafety_unsafe"]) if "xsafety_unsafe" in xs else pd.NA
    xs["refusal"] = coerce_binary(xs["refusal"])
    if "llamaguard_unsafe" in xs:
        xs["llamaguard_unsafe"] = pd.to_numeric(xs["llamaguard_unsafe"], errors="coerce").replace(-1, pd.NA)
    xs["xsafety_slice"] = pd.NA
    xs.loc[xs["category"].isin(DIRECT_HARM_XSAFETY_CATEGORIES), "xsafety_slice"] = "direct_harm"
    xs.loc[xs["category"].isin(ADVICE_XSAFETY_CATEGORIES), "xsafety_slice"] = "advice_style"
    sl = xs[xs["xsafety_slice"].notna()]
    by_so = sl.groupby(["xsafety_slice", "origin"]).agg(
        xsafety_unsafe=("xsafety_unsafe", "mean"),
        refusal=("refusal", "mean"),
        llamaguard_unsafe=("llamaguard_unsafe", "mean"),
    ).reset_index()
    by_so["xsafety_safe"] = 1 - by_so["xsafety_unsafe"]
    by_so["llamaguard_safe"] = 1 - by_so["llamaguard_unsafe"]
    heatmap_panel(
        by_so, "origin", "xsafety_slice",
        ["xsafety_safe", "refusal", "llamaguard_safe"],
        "03_xsafety_control_panel.png",
        vmins=[0.0, 0.0, 0.0], vmaxs=[1.0, 1.0, 1.0],
        index_order=ORIGIN_ORDER, column_order=["advice_style", "direct_harm"],
    )

    # ---- 04: language effect per model (en - zh, overall_safe) ----
    piv = by_ml.pivot(index="model", columns="language", values="overall_safe").reindex(MODEL_ORDER)
    delta = (piv["en"] - piv["zh"]).reset_index(name="en_minus_zh")
    plt.figure(figsize=(9, 5))
    colors = ["#2563eb" if v >= 0 else "#dc2626" for v in delta["en_minus_zh"]]
    ax = sns.barplot(data=delta, x="model", y="en_minus_zh", order=MODEL_ORDER, palette=colors)
    relabel_models(ax, MODEL_ORDER)
    ax.axhline(0, color="black", linewidth=0.8)
    hi = float(delta["en_minus_zh"].max())
    ax.set_ylim(min(0.0, float(delta["en_minus_zh"].min()) - 0.01), hi + 0.012)
    for i, v in enumerate(delta["en_minus_zh"]):
        ax.text(i, v + (0.003 if v >= 0 else -0.008), f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    plt.ylabel("overall_safe(en) - overall_safe(zh)")
    plt.title("Language effect per model (positive = safer in English)")
    plt.tight_layout()
    save("04_language_effect_per_model.png")

    # ---- 05: language effect, cross-judge (Sonnet vs gpt-4o-mini) ----
    cj = sv.copy()
    cj["Claude Sonnet"] = cj["overall_safe"]
    if "content_gpt_overall_safe" in cj:
        cj["gpt-4o-mini"] = coerce_binary(cj["content_gpt_overall_safe"])
    cj = cj[cj["gpt-4o-mini"].notna()]
    long = (cj[["language", "Claude Sonnet", "gpt-4o-mini"]]
            .melt(id_vars="language", var_name="judge", value_name="overall_safe"))
    plt.figure(figsize=(8, 5))
    sns.barplot(data=long, x="judge", y="overall_safe", hue="language",
                hue_order=LANG_ORDER, order=["Claude Sonnet", "gpt-4o-mini"])
    plt.ylim(0, 1.0)
    plt.title(f"Adolescent overall_safe by language under two judges (n={len(cj)} overlap)")
    plt.tight_layout()
    save("05_language_effect_cross_judge.png")

    # ---- 06: framing axes by grouping x prompt-setting ----
    fr = df[df["stimulus_set"] == "cultural_probe"].copy()
    if "framing_judge_status" in fr.columns:
        fr = fr[fr["framing_judge_status"] == "ok"]
    fr = fr.dropna(subset=["individual_framing", "collectivist_framing"])
    fr["net_framing"] = fr["individual_framing"] - fr["collectivist_framing"]
    by_fo = fr.groupby(["origin", "origin_culture"])[
        ["individual_framing", "collectivist_framing", "net_framing"]].mean().reset_index()
    heatmap_panel(
        by_fo, "origin", "origin_culture",
        ["individual_framing", "collectivist_framing", "net_framing"],
        "06_framing_by_origin.png",
        vmins=[1.0, 1.0, -1.0], vmaxs=[5.0, 5.0, 1.5],
        index_order=ORIGIN_ORDER, column_order=PROMPT_ORIGIN_ORDER,
    )

    # ---- 07: safety components by grouping (truncated axis, all higher = safer) ----
    comp = sv.groupby("origin")[SAFETY_METRICS].mean()
    comp["avoids_harm"] = 1 - comp["facilitates_harm"]
    safe_dir = ["overall_safe", "avoids_harm", "recognizes_risk",
                "appropriate_escalation", "supportive_tone", "protective_guidance"]
    long = comp[safe_dir].reindex(ORIGIN_ORDER).reset_index().melt(
        id_vars="origin", var_name="component", value_name="rate")
    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=long, x="component", y="rate", hue="origin",
                     order=safe_dir, hue_order=ORIGIN_ORDER, palette=ORIGIN_PALETTE)
    ax.set_ylim(0.5, 1.04)  # truncated: cluster sits in 0.67-1.00, gaps become legible
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
    ax.legend(title="grouping", loc="lower right")
    plt.ylabel("rate (higher = safer; axis truncated at 0.5)")
    plt.title("Safety components by developer-country grouping\n(avoids_harm = 1 - facilitates_harm; ~no gap. "
              "Gaps live in escalation / tone / protective guidance)", pad=12)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    save("07_safety_components_by_origin.png", caveat=True)

    # ---- extra: directional user-fit comparison (who is serving whom?) ----
    EXTRA.mkdir(parents=True, exist_ok=True)
    directional = by_oo.copy()
    directional["direction"] = directional.apply(
        lambda r: f"{r['origin']} -> {r['origin_culture']}", axis=1
    )
    direction_order = [
        "US-grouped -> CN",
        "China-grouped -> CN",
        "US-grouped -> US",
        "China-grouped -> US",
    ]
    dir_palette = {
        "US-grouped -> CN": "#dc2626",
        "China-grouped -> CN": "#2563eb",
        "US-grouped -> US": "#fca5a5",
        "China-grouped -> US": "#93c5fd",
    }
    metrics = [
        ("overall_safe", "overall_safe"),
        ("appropriate_escalation", "appropriate_escalation"),
        ("protective_guidance", "protective_guidance"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        sub = directional[["direction", metric]].copy()
        sns.barplot(
            data=sub,
            x="direction",
            y=metric,
            order=direction_order,
            palette=dir_palette,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("rate" if ax is axes[0] else "")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.suptitle("Directional user fit on the adolescent benchmark\n(compare US-grouped -> CN vs China-grouped -> CN directly)", fontsize=11)
    plt.tight_layout()
    plt.savefig(EXTRA / "08_directional_user_fit.png", dpi=150)
    plt.close()

    # ---- extra: within-group comparison (same grouping, different prompt setting) ----
    prompt_palette = {"US": "#f59e0b", "CN": "#2563eb"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        sns.barplot(
            data=by_oo,
            x="origin",
            y=metric,
            hue="origin_culture",
            order=ORIGIN_ORDER,
            hue_order=PROMPT_ORIGIN_ORDER,
            palette=prompt_palette,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("rate" if ax is axes[0] else "")
        ax.set_title(title)
        if ax is axes[0]:
            ax.legend(title="prompt setting", loc="lower left")
        else:
            ax.get_legend().remove()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.suptitle("All adolescent prompts: same grouping, different prompt setting\n(compare responses to CN-setting vs US-setting scenarios)", fontsize=11)
    plt.tight_layout()
    plt.savefig(EXTRA / "09_within_group_user_fit_by_prompt_setting.png", dpi=150)
    plt.close()

    # ---- extra: within-group comparison on harmful-request prompts only ----
    by_oo_hr = sv_hr.groupby(["origin", "origin_culture"])[SAFETY_METRICS].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        sns.barplot(
            data=by_oo_hr,
            x="origin",
            y=metric,
            hue="origin_culture",
            order=ORIGIN_ORDER,
            hue_order=PROMPT_ORIGIN_ORDER,
            palette=prompt_palette,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("rate" if ax is axes[0] else "")
        ax.set_title(title)
        if ax is axes[0]:
            ax.legend(title="prompt setting", loc="lower left")
        else:
            ax.get_legend().remove()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.suptitle("Harmful-request prompts only: same grouping, different prompt setting\n(compare responses to CN-setting vs US-setting scenarios)", fontsize=11)
    plt.tight_layout()
    plt.savefig(EXTRA / "10_harmful_within_group_user_fit_by_prompt_setting.png", dpi=150)
    plt.close()

    # ---- extra: within-group comparison on non-harmful prompts only ----
    by_oo_nonhr = sv_nonhr.groupby(["origin", "origin_culture"])[SAFETY_METRICS].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        sns.barplot(
            data=by_oo_nonhr,
            x="origin",
            y=metric,
            hue="origin_culture",
            order=ORIGIN_ORDER,
            hue_order=PROMPT_ORIGIN_ORDER,
            palette=prompt_palette,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("rate" if ax is axes[0] else "")
        ax.set_title(title)
        if ax is axes[0]:
            ax.legend(title="prompt setting", loc="lower left")
        else:
            ax.get_legend().remove()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.suptitle("Non-harmful prompts only: same grouping, different prompt setting\n(compare responses to CN-setting vs US-setting scenarios)", fontsize=11)
    plt.tight_layout()
    plt.savefig(EXTRA / "11_nonharmful_within_group_user_fit_by_prompt_setting.png", dpi=150)
    plt.close()

    # ---- extra: paired-only non-harmful comparison (conservative setting effect) ----
    by_oo_nonhr_paired = sv_nonhr_paired.groupby(["origin", "origin_culture"])[SAFETY_METRICS].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        sns.barplot(
            data=by_oo_nonhr_paired,
            x="origin",
            y=metric,
            hue="origin_culture",
            order=ORIGIN_ORDER,
            hue_order=PROMPT_ORIGIN_ORDER,
            palette=prompt_palette,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("rate" if ax is axes[0] else "")
        ax.set_title(title)
        if ax is axes[0]:
            ax.legend(title="prompt setting", loc="lower left")
        else:
            ax.get_legend().remove()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.suptitle("Paired-only non-harmful prompts: same grouping, different prompt setting\n(archetype-matched US/CN scenarios; topic-controlled comparison)", fontsize=11)
    plt.tight_layout()
    plt.savefig(EXTRA / "12_paired_nonharmful_within_group_user_fit_by_prompt_setting.png", dpi=150)
    plt.close()

    # ---- extra: paired-only non-harmful comparison by model (same slice, no grouping collapse) ----
    by_model_nonhr_paired = sv_nonhr_paired.groupby(["model", "origin_culture"])[SAFETY_METRICS].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        sns.barplot(
            data=by_model_nonhr_paired,
            x="model",
            y=metric,
            hue="origin_culture",
            order=MODEL_ORDER,
            hue_order=PROMPT_ORIGIN_ORDER,
            palette=prompt_palette,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("rate" if ax is axes[0] else "")
        ax.set_title(title)
        if ax is axes[0]:
            ax.legend(title="prompt setting", loc="lower left")
        else:
            ax.get_legend().remove()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    fig.suptitle("Paired-only non-harmful prompts: by model, different prompt setting\n(archetype-matched US/CN scenarios; topic-controlled comparison)", fontsize=11)
    plt.tight_layout()
    plt.savefig(EXTRA / "13_paired_nonharmful_by_model_prompt_setting.png", dpi=150)
    plt.close()

    print("evaluator figures 01-07 ->", FIG)
    print("refreshed uncaveated backups for 01/02/07 ->", UNCAVEATED)
    print("extra directional figure ->", EXTRA / "08_directional_user_fit.png")
    print("extra within-group figure ->", EXTRA / "09_within_group_user_fit_by_prompt_setting.png")
    print("extra harmful within-group figure ->", EXTRA / "10_harmful_within_group_user_fit_by_prompt_setting.png")
    print("extra non-harmful within-group figure ->", EXTRA / "11_nonharmful_within_group_user_fit_by_prompt_setting.png")
    print("extra paired-only non-harmful figure ->", EXTRA / "12_paired_nonharmful_within_group_user_fit_by_prompt_setting.png")
    print("extra paired-only non-harmful model figure ->", EXTRA / "13_paired_nonharmful_by_model_prompt_setting.png")


if __name__ == "__main__":
    build()
