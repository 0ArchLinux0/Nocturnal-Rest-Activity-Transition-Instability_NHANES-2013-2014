#!/usr/bin/env python3
"""
fig5c: Age vs Entropy - Boxplot by decade + Scatter with regression/LOWESS
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
DPI = 300

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
            break
    if df is None:
        print("ERROR: No processed data found.")
        return

    df = df.dropna(subset=["Entropy", "age"])
    df = df[(df["age"] >= 16) & (df["age"] <= 90)]

    # 10-year age groups (20s ~ 80s)
    def age_group(a):
        if a < 20:
            return "<20"
        if a < 30:
            return "20-29"
        if a < 40:
            return "30-39"
        if a < 50:
            return "40-49"
        if a < 60:
            return "50-59"
        if a < 70:
            return "60-69"
        if a < 80:
            return "70-79"
        return "80+"

    df = df.copy()
    df["Age_Group"] = df["age"].apply(age_group)
    order = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    df["Age_Group"] = pd.Categorical(df["Age_Group"], categories=[g for g in order if g in df["Age_Group"].unique()], ordered=True)

    # Pearson r and p-value
    r, p = stats.pearsonr(df["age"], df["Entropy"])

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.3)
    FS_LABEL = 14   # axis labels (1.2~1.5x larger)
    FS_TICK = 12    # tick numbers
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=False)

    # --- Panel A: Boxplot by 10-year groups ---
    ax1 = axes[0]
    df_box = df[df["Age_Group"].notna()].copy()
    df_box = df_box.sort_values("Age_Group")
    sns.boxplot(data=df_box, x="Age_Group", y="Entropy", color="#4A90A4", ax=ax1)
    ax1.set_xlabel("Age group (years)", fontsize=FS_LABEL)
    ax1.set_ylabel("Shannon Entropy (bits)", fontsize=FS_LABEL)
    ax1.set_title(r"$(A)$ Entropy distribution by age decade", fontsize=FS_LABEL + 1)
    ax1.tick_params(axis="both", labelsize=FS_TICK)
    ax1.tick_params(axis="x", rotation=45)

    # --- Panel B: Scatter + regression ---
    ax2 = axes[1]
    ax2.scatter(df["age"], df["Entropy"], alpha=0.12, s=6, c="#2E86AB", edgecolors="none")

    # Linear regression line
    x_line = np.linspace(df["age"].min(), df["age"].max(), 100)
    slope, intercept, _, _, _ = stats.linregress(df["age"], df["Entropy"])
    ax2.plot(x_line, slope * x_line + intercept, color="#E94F37", lw=2.5, label="Linear fit")

    # LOWESS (if statsmodels available)
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lo = lowess(df["Entropy"], df["age"], frac=0.2)
        ax2.plot(lo[:, 0], lo[:, 1], color="#333333", lw=2, ls="--", label="LOWESS")
    except ImportError:
        pass

    ax2.set_xlabel("Age (years)", fontsize=FS_LABEL)
    ax2.set_ylabel("Shannon Entropy (bits)", fontsize=FS_LABEL)
    ax2.set_title(r"$(B)$ Age vs. Entropy with regression", fontsize=FS_LABEL + 1)
    ax2.tick_params(axis="both", labelsize=FS_TICK)
    ax2.legend(loc="upper right", fontsize=FS_TICK)
    stat_text = r"$r = " + f"{r:.3f}" + r"$, $p < 0.001$" if p < 0.001 else r"$r = " + f"{r:.3f}" + r"$, $p = " + f"{p:.3f}" + r"$"
    ax2.text(0.95, 0.08, stat_text, transform=ax2.transAxes,
             fontsize=FS_LABEL, ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    fig.suptitle("Shannon Entropy and Aging", fontsize=16, y=1.02)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "fig5c_entropy_aging_detailed.png"
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Pearson r = {r:.4f}, p = {p:.6f}")


if __name__ == "__main__":
    main()
