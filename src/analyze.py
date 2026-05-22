import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

PLOTS = config.DATA_DIR / "plots"


def bar(df, x, hue, y, title, fname):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x=x, hue=hue, y=y)
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()


def main():
    PLOTS.mkdir(exist_ok=True)
    df = pd.read_csv(config.RESULTS)
    df["origin"] = df["model"].map(config.MODEL_ORIGIN)

    print("\n== overall ==")
    print(df[["llamaguard_unsafe", "refusal"]].mean().round(3).to_string())

    print("\n== ORIGIN x LANGUAGE (WEIRD hypothesis) ==")
    by_ol = df.groupby(["origin", "language"])[["llamaguard_unsafe", "refusal"]].mean().round(3).reset_index()
    print(by_ol.to_string(index=False))
    by_ol.to_csv(config.DATA_DIR / "summary_origin_x_language.csv", index=False)

    print("\n== MODEL x LANGUAGE (per-model language effect) ==")
    by_ml = df.groupby(["model", "language"])[["llamaguard_unsafe", "refusal"]].mean().round(3).reset_index()
    print(by_ml.to_string(index=False))
    by_ml.to_csv(config.DATA_DIR / "summary_model_x_language.csv", index=False)

    print("\n== WITHIN-ORIGIN CLUSTERING (do models from the same origin pattern together?) ==")
    for origin in ["WEIRD", "non_WEIRD"]:
        sub = by_ml[by_ml["model"].isin([m for m, o in config.MODEL_ORIGIN.items() if o == origin])]
        print(f"\n[{origin}]")
        print(sub.to_string(index=False))

    print("\n== per-language deltas from English baseline ==")
    pivot_u = by_ml.pivot(index="model", columns="language", values="llamaguard_unsafe")
    pivot_r = by_ml.pivot(index="model", columns="language", values="refusal")
    deltas = pd.DataFrame({
        "unsafe_fr-en": (pivot_u["fr"] - pivot_u["en"]).round(3),
        "unsafe_zh-en": (pivot_u["zh"] - pivot_u["en"]).round(3),
        "refusal_fr-en": (pivot_r["fr"] - pivot_r["en"]).round(3),
        "refusal_zh-en": (pivot_r["zh"] - pivot_r["en"]).round(3),
    })
    print(deltas.to_string())
    deltas.to_csv(config.DATA_DIR / "summary_language_deltas.csv")

    print("\n== model x language x category (n~30/cell) ==")
    by_mlc = df.groupby(["model", "language", "category"])[["llamaguard_unsafe", "refusal"]].mean().round(3).reset_index()
    by_mlc.to_csv(config.DATA_DIR / "summary_model_x_language_x_category.csv", index=False)
    print(by_mlc.to_string(index=False))

    bar(by_ml, "model", "language", "llamaguard_unsafe",
        "Llama-Guard unsafe rate by model x language", "unsafe_by_model_language.png")
    bar(by_ml, "model", "language", "refusal",
        "Claude-judged refusal rate by model x language", "refusal_by_model_language.png")

    df["disagree"] = (df["llamaguard_unsafe"] != df["refusal"]).astype(int)
    print("\n== judge disagreement rate ==")
    print(df.groupby(["model", "language"])["disagree"].mean().round(3).reset_index().to_string(index=False))

    n = len(df)
    n_unsafe = int(df["llamaguard_unsafe"].sum())
    n_refused = int(df["refusal"].sum())
    n_both = int(((df["llamaguard_unsafe"] == 1) & (df["refusal"] == 1)).sum())
    n_neither = int(((df["llamaguard_unsafe"] == 0) & (df["refusal"] == 0)).sum())
    print(f"\nconfusion: unsafe={n_unsafe}, refused={n_refused}, both={n_both}, neither={n_neither}, total={n}")


if __name__ == "__main__":
    main()
