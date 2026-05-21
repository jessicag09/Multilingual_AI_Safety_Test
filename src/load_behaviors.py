import csv

import jailbreakbench as jbb

from . import config


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    dataset = jbb.read_dataset()
    harmful = dataset.as_dataframe()
    harmful = harmful[harmful["Behavior"].notna()].reset_index(drop=True)
    harmful = harmful[["Index", "Goal", "Target", "Behavior", "Category", "Source"]]
    harmful.columns = ["jbb_index", "goal", "target", "behavior", "category", "source"]
    harmful["prompt_id"] = range(1, len(harmful) + 1)

    cols = ["prompt_id", "jbb_index", "category", "behavior", "goal", "target", "source"]
    harmful[cols].to_csv(config.BEHAVIORS, index=False, quoting=csv.QUOTE_ALL)
    print(f"wrote {len(harmful)} JBB harmful behaviors -> {config.BEHAVIORS}")


if __name__ == "__main__":
    main()
