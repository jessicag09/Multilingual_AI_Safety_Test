import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

PLOTS = config.DATA_DIR / "plots"


def bar(df, x, hue, y, title, fname):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x=x, hue=hue, y=y)
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()


def main():
    PLOTS.mkdir(exist_ok=True)
    df = pd.read_csv(config.RESULTS)
    df["harmful"] = df["llamaguard_unsafe"]

    print("\n== overall ==")
    print(df[["llamaguard_unsafe", "refusal"]].mean().to_string())

    summary = {
        "by_model": df.groupby("model")[["llamaguard_unsafe", "refusal"]].mean().reset_index(),
        "by_model_language": df.groupby(["model", "language"])[["llamaguard_unsafe", "refusal"]].mean().reset_index(),
        "by_model_category": df.groupby(["model", "category"])[["llamaguard_unsafe", "refusal"]].mean().reset_index(),
        "by_language_category": df.groupby(["language", "category"])[["llamaguard_unsafe", "refusal"]].mean().reset_index(),
    }
    for name, t in summary.items():
        print(f"\n== {name} ==")
        print(t.to_string(index=False))
        t.to_csv(config.DATA_DIR / f"summary_{name}.csv", index=False)

    bar(summary["by_model_language"], "model", "language", "llamaguard_unsafe",
        "Llama-Guard unsafe rate by model x language", "unsafe_by_model_language.png")
    bar(summary["by_model_language"], "model", "language", "refusal",
        "Claude-judged refusal rate by model x language", "refusal_by_model_language.png")

    df["agree"] = (df["llamaguard_unsafe"] != df["refusal"]).astype(int)
    print("\n== judge disagreement (unsafe vs refusal) ==")
    print("note: a high disagreement rate means the model produced harmful content without an explicit refusal, or refused without producing anything Llama-Guard flagged.")
    print(df.groupby(["model", "language"])["agree"].mean().reset_index().to_string(index=False))

    n_unsafe = int(df["llamaguard_unsafe"].sum())
    n_refused = int(df["refusal"].sum())
    n_both = int(((df["llamaguard_unsafe"] == 1) & (df["refusal"] == 1)).sum())
    n_neither = int(((df["llamaguard_unsafe"] == 0) & (df["refusal"] == 0)).sum())
    print(f"\nconfusion: unsafe={n_unsafe}, refused={n_refused}, both={n_both}, neither={n_neither}, total={len(df)}")


if __name__ == "__main__":
    main()
