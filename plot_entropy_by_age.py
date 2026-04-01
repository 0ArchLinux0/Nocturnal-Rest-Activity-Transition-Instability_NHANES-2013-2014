#!/usr/bin/env python3
"""
Shannon Entropy by Age (Aging)
==============================
Uses existing processed data (Entropy + age). No synthetic data.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
DPI = 300

# Try data files in order
DATA_PATHS = [
    OUTPUT_DIR / "processed_data_physics_deep.csv",
    OUTPUT_DIR / "processed_data_physics_ultimate.csv",
    OUTPUT_DIR / "processed_data_physics_final.csv",
]


def main():
    df = None
    for p in DATA_PATHS:
        if p.exists():
            df = pd.read_csv(p)
            print(f"Loaded: {p.name} ({len(df)} rows)")
            break
    if df is None:
        print("ERROR: No processed data found. Run nhanes_physica_physics.py or nhanes_physica_ultimate.py first.")
        return

    if "Entropy" not in df.columns or "age" not in df.columns:
        print("ERROR: Data missing 'Entropy' or 'age' column.")
        print("Available columns:", list(df.columns))
        return

    df_plot = df.dropna(subset=["Entropy", "age"])
    print(f"Analytic sample: {len(df_plot)} subjects")

    q1, q2, q3 = df_plot["age"].quantile([0.33, 0.66, 1.0]).values

    def age_stratum(a):
        if a <= q1:
            return "Young (≤33rd)"
        if a <= q2:
            return "Middle (33rd–66th)"
        return "Older (>66th)"

    df_plot = df_plot.copy()
    df_plot["age_stratum"] = df_plot["age"].apply(age_stratum)
    order = ["Young (≤33rd)", "Middle (33rd–66th)", "Older (>66th)"]
    strata = [s for s in order if s in df_plot["age_stratum"].unique()]
    data = [df_plot[df_plot["age_stratum"] == s]["Entropy"] for s in strata]

    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, tick_labels=strata, patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("#4A90A4")
    ax.set_xlabel("Age stratum", fontsize=11)
    ax.set_ylabel("Shannon Entropy (bits)", fontsize=11)
    ax.set_title("Shannon Entropy by Age (Aging)", fontsize=12)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig5b_entropy_by_age.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
