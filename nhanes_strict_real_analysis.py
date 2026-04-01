#!/usr/bin/env python3
"""
STRICT Real Data Logistic Regression & Empirical Potential (Kramers' Well)
=========================================================================
RULE: NO synthetic/fallback data. FileNotFoundError if BMX_H or DPQ_H missing.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "outputs" / "processed_data_physics_ultimate.csv"
PAXHR_PATH = None
BMX_PATH = None
DPQ_PATH = None
OUTPUT_DIR = BASE_DIR / "outputs"
DPI = 300
ACTIVITY_COLS = ["PAXMTSH", "PAXINTEN"]


def resolve_xpt(name: str) -> Path:
    """Case-insensitive lookup for .xpt or .XPT"""
    for ext in [".xpt", ".XPT"]:
        p = BASE_DIR / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"REQUIRED FILE NOT FOUND: {name}.xpt or {name}.XPT\n"
        f"Download from NHANES 2013-2014 and place in: {BASE_DIR}"
    )


def load_xpt(fp: Path) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xport(str(fp))
        return df
    except ImportError:
        return pd.read_sas(str(fp), format="xport")


def run_real_logistic(df: pd.DataFrame):
    """Task 1: Real multivariate logistic regression."""
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm

    cols_std = ["Age", "BMXBMI", "PHQ9_Score", "Night_P_01"]
    scaler = StandardScaler()
    df[cols_std] = scaler.fit_transform(df[cols_std].astype(float))
    df["Gender"] = df["Gender"].astype(int)

    X = df[["Age", "Gender", "BMXBMI", "PHQ9_Score", "Night_P_01"]]
    X = sm.add_constant(X)
    y = df["sleep_problem_reported"]

    model = sm.Logit(y, X).fit(disp=0)
    print("\n" + "=" * 60)
    print("REAL MULTIVARIATE LOGISTIC REGRESSION SUMMARY")
    print("=" * 60)
    print("Formula: sleep_problem_reported ~ Age + Gender + BMXBMI + PHQ9_Score + Night_P_01")
    print(model.summary())

    or_vals = np.exp(model.params)
    ci = np.exp(model.conf_int())
    ci.columns = ["CI_lower", "CI_upper"]
    results = pd.DataFrame({
        "variable": model.params.index,
        "OR": or_vals.values,
        "CI_lower": ci["CI_lower"].values,
        "CI_upper": ci["CI_upper"].values,
        "pvalue": model.pvalues.values,
    })
    results = results[results["variable"] != "const"].reset_index(drop=True)

    print("\n" + "=" * 60)
    print("REAL ODDS RATIOS (95% CI) & P-VALUES")
    print("=" * 60)
    for _, r in results.iterrows():
        print(f"  {r['variable']:12s} OR={r['OR']:.3f} [95%CI {r['CI_lower']:.3f}-{r['CI_upper']:.3f}] p={r['pvalue']:.6f}")

    night_row = results[results["variable"] == "Night_P_01"]
    survives = len(night_row) > 0 and night_row["pvalue"].iloc[0] < 0.05
    print("\n--- Night_P_01 vs BMI + PHQ-9 ---")
    if survives:
        print("  Night_P_01 SURVIVES (p < 0.05) as independent predictor.")
    else:
        pv = night_row["pvalue"].iloc[0] if len(night_row) > 0 else np.nan
        print(f"  Night_P_01 does NOT survive (p = {pv:.4f}) after controlling for BMI and PHQ-9.")

    return results


def compute_empirical_potential(paxhr: pd.DataFrame, cohort: pd.DataFrame, activity_col: str):
    """
    Task 2: U(x) = -ln(P(x)), Kramers' potential well.
    P(x) from KDE of activity. Rest well = min U at low activity.
    """
    from scipy.stats import gaussian_kde

    paxhr = paxhr.merge(cohort[["SEQN", "sleep_problem_reported"]], on="SEQN", how="inner")
    act0 = paxhr[paxhr["sleep_problem_reported"] == 0][activity_col].values
    act1 = paxhr[paxhr["sleep_problem_reported"] == 1][activity_col].values
    act0 = act0[act0 > 0]
    act1 = act1[act1 > 0]
    eps = 1e-10
    x_upper = min(np.percentile(act0, 90), np.percentile(act1, 90)) * 1.05
    x_grid = np.linspace(eps, x_upper, 500)

    def u_from_group(act):
        kde = gaussian_kde(act, bw_method="scott")
        p = np.maximum(kde(x_grid), eps)
        return -np.log(p)

    u0 = u_from_group(act0)
    u1 = u_from_group(act1)

    # Rest well: min U in low-activity region (bottom ~50% of x range)
    low_idx = slice(0, len(x_grid) // 2)
    well_min_0 = np.min(u0[low_idx])
    well_min_1 = np.min(u1[low_idx])
    x_well_0 = x_grid[low_idx][np.argmin(u0[low_idx])]
    x_well_1 = x_grid[low_idx][np.argmin(u1[low_idx])]
    diff_depth = well_min_1 - well_min_0
    # Transition threshold: local max in middle region (between well and high-x)
    mid_start = min(np.argmin(u0[low_idx]) + 10, len(x_grid) - 50)
    mid_end = int(len(x_grid) * 0.85)
    peak_0 = mid_start + np.argmax(u0[mid_start:mid_end])
    peak_1 = mid_start + np.argmax(u1[mid_start:mid_end])
    x_peak_0 = x_grid[peak_0]
    x_peak_1 = x_grid[peak_1]
    u_peak_0 = u0[peak_0]
    u_peak_1 = u1[peak_1]
    return x_grid, u0, u1, {
        "U_min_Group0": well_min_0,
        "U_min_Group1": well_min_1,
        "x_well_0": x_well_0,
        "x_well_1": x_well_1,
        "diff_well_depth": diff_depth,
        "x_peak_0": x_peak_0,
        "x_peak_1": x_peak_1,
        "u_peak_0": u_peak_0,
        "u_peak_1": u_peak_1,
    }


def main():
    BMX_PATH = resolve_xpt("BMX_H")
    DPQ_PATH = resolve_xpt("DPQ_H")
    PAXHR_PATH = resolve_xpt("PAXHR_H")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("STRICT REAL DATA ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"age": "Age", "gender": "Gender", "P_01_Night": "Night_P_01"})
    if "Age" not in df.columns:
        df["Age"] = df.get("age", np.nan)
    if "Gender" not in df.columns:
        df["Gender"] = df.get("gender", np.nan)

    bmx = load_xpt(BMX_PATH)[["SEQN", "BMXBMI"]]
    df = df.merge(bmx, on="SEQN", how="inner")
    dpq = load_xpt(DPQ_PATH)
    dpq_cols = [f"DPQ{i:03d}" for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]]
    dpq_cols = [c for c in dpq_cols if c in dpq.columns]
    dpq = dpq[["SEQN"] + dpq_cols].copy()
    for c in dpq_cols:
        dpq[c] = pd.to_numeric(dpq[c], errors="coerce")
        dpq.loc[dpq[c].isin([7, 9]), c] = np.nan
    dpq["PHQ9_Score"] = dpq[dpq_cols].sum(axis=1)
    dpq = dpq[["SEQN", "PHQ9_Score"]]
    df = df.merge(dpq, on="SEQN", how="inner")
    df = df.dropna(subset=["Age", "Gender", "BMXBMI", "PHQ9_Score", "Night_P_01"])
    print(f"Analytic sample: {len(df)} (REAL data, complete cases)")

    results = run_real_logistic(df.copy())

    paxhr = load_xpt(PAXHR_PATH)
    activity_col = next((c for c in ACTIVITY_COLS if c in paxhr.columns), None)
    if activity_col is None:
        raise KeyError("No activity column in PAXHR_H")
    seqn_set = set(df["SEQN"].dropna().astype(int))
    paxhr = paxhr[paxhr["SEQN"].isin(seqn_set)].copy()
    paxhr = paxhr[paxhr[activity_col].notna() & (paxhr[activity_col] > 0)]

    print("\n" + "=" * 60)
    print("EMPIRICAL POTENTIAL (Kramers' Well)")
    print("=" * 60)
    x_grid, u0, u1, well = compute_empirical_potential(paxhr, df, activity_col)
    print(f"  Group 0 (No sleep problem) - Rest well minimum U(x): {well['U_min_Group0']:.4f} at x≈{well['x_well_0']:.2f}")
    print(f"  Group 1 (Sleep problem)     - Rest well minimum U(x): {well['U_min_Group1']:.4f} at x≈{well['x_well_1']:.2f}")
    print(f"  Difference in well depth (Group1 - Group0): {well['diff_well_depth']:.4f}")

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(9, 6))
    results_plot = results.sort_values("OR", ascending=True)
    y_pos = np.arange(len(results_plot))
    var_to_label = {"Age": "Age", "Gender": "Gender", "BMXBMI": "BMI", "PHQ9_Score": "PHQ-9", "Night_P_01": "Night P₀₁"}
    def fmt_p(pv):
        if pv < 0.001:
            return "(p < 0.001)"
        return f"(p = {pv:.3f})"
    labels = [f"{var_to_label.get(v, v)} {fmt_p(r['pvalue'])}" for v, r in zip(results_plot["variable"], results_plot.to_dict("records"))]
    colors = ["#E94F37" if v == "Night_P_01" else "#333333" for v in results_plot["variable"]]
    ax.barh(y_pos, results_plot["OR"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.errorbar(
        results_plot["OR"], y_pos,
        xerr=[
            results_plot["OR"] - results_plot["CI_lower"],
            results_plot["CI_upper"] - results_plot["OR"],
        ],
        fmt="none", ecolor="black", capsize=4,
    )
    ax.axvline(1, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Odds Ratio (95% CI)", fontsize=13)
    ax.set_title("REAL Data Forest Plot", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(0.3, min(4.0, results_plot["CI_upper"].max() * 1.15))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig13_real_forest_plot.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / 'fig13_real_forest_plot.png'}")

    act_all = paxhr[activity_col].values[paxhr[activity_col] > 0]
    x_max = np.percentile(act_all, 85) * 1.05 if len(act_all) > 0 else 100
    mask = x_grid <= x_max
    FS_LABEL = 14
    FS_TICK = 12
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_grid[mask], u0[mask], "b-", lw=2, label="No sleep problem (Group 0)")
    ax.plot(x_grid[mask], u1[mask], "r-", lw=2, label="Sleep problem (Group 1)")
    ax.set_xlabel(f"Activity x ({activity_col})", fontsize=FS_LABEL)
    ax.set_ylabel("Effective Potential U(x) = -ln(P(x))", fontsize=FS_LABEL)
    ax.set_title("Empirical Potential: Kramers' Rest Well", fontsize=FS_LABEL + 1)
    ax.legend(loc="upper left", fontsize=FS_TICK)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_xlim(0, x_max)
    # Annotations: Rest State, Transition Threshold, Delta U arrow
    xw0, xw1 = well["x_well_0"], well["x_well_1"]
    uw0, uw1 = well["U_min_Group0"], well["U_min_Group1"]
    dU = well["diff_well_depth"]
    xp0, xp1 = well["x_peak_0"], well["x_peak_1"]
    up0, up1 = well["u_peak_0"], well["u_peak_1"]
    bbox_style = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")
    ax.annotate("Rest State", xy=((xw0 + xw1) / 2, (uw0 + uw1) / 2), xytext=(xw0 + 25, uw0 - 0.25), fontsize=9, ha="center")
    # Transition Threshold: local max of U(x) between well and high-x (Kramers barrier)
    xp_mid = (xp0 + xp1) / 2
    ax.axvline(xp_mid, color="gray", ls="--", alpha=0.7, lw=1.2)
    ax.plot(xp0, up0, "bo", markersize=6)
    ax.plot(xp1, up1, "ro", markersize=6)
    ax.annotate(r"Transition Threshold" + "\n" + r"(local max $U(x)$)", xy=(xp_mid, (up0 + up1) / 2),
                xytext=(xp_mid - 100, 6.3), fontsize=9, ha="center", bbox=bbox_style,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    # Delta U: arrow + larger text, moved further right to clear curves
    ax.annotate("", xy=(xw1, uw1), xytext=(xw1, uw0), arrowprops=dict(arrowstyle="<->", color="black", lw=1.8))
    ax.text(xw1 + 60, (uw0 + uw1) / 2, r"$\Delta U$ = " + f"{dU:.3f}", fontsize=13, bbox=bbox_style)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig14_empirical_potential.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig14_empirical_potential.png'}")

    df.to_csv(OUTPUT_DIR / "processed_data_final_master.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'processed_data_final_master.csv'}")


if __name__ == "__main__":
    main()
