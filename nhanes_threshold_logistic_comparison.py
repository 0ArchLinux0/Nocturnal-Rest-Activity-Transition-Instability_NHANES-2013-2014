#!/usr/bin/env python3
"""
Compare logistic regression (OR, p) across Median / Q1 / Q3 thresholds.
Answers: Situation A (OR flips), B (p changes meaningfully), or C (consistent).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "outputs" / "processed_data_physics_ultimate.csv"
PAXHR_PATH = BASE_DIR / "PAXHR_H.xpt"
BMX_PATH = BASE_DIR / "BMX_H.xpt"
DPQ_PATH = BASE_DIR / "DPQ_H.xpt"
ACTIVITY_COLS = ["PAXMTSH", "PAXINTEN"]
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))


def load_xpt(fp: Path) -> pd.DataFrame:
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xport(str(fp))
        return df
    except ImportError:
        return pd.read_sas(str(fp), format="xport")


def p01_from_sequence(act: np.ndarray, thresh: float) -> float:
    state = (act > thresh).astype(int)
    if len(state) < 2:
        return np.nan
    n_01 = np.sum((state[:-1] == 0) & (state[1:] == 1))
    n_0 = np.sum(state[:-1] == 0)
    return n_01 / n_0 if n_0 > 0 else np.nan


def run_logistic(df: pd.DataFrame, night_col: str) -> dict:
    from sklearn.preprocessing import StandardScaler
    import statsmodels.api as sm

    cols_std = ["Age", "BMXBMI", "PHQ9_Score", night_col]
    df = df.dropna(subset=cols_std + ["Gender", "sleep_problem_reported"]).copy()
    scaler = StandardScaler()
    df[cols_std] = scaler.fit_transform(df[cols_std].astype(float))
    df["Gender"] = df["Gender"].astype(int)

    X = df[["Age", "Gender", "BMXBMI", "PHQ9_Score", night_col]]
    X = sm.add_constant(X)
    y = df["sleep_problem_reported"]
    model = sm.Logit(y, X).fit(disp=0)

    night_idx = list(model.params.index).index(night_col)
    or_val = np.exp(model.params[night_col])
    pval = model.pvalues[night_col]
    ci = np.exp(model.conf_int().loc[night_col])
    return {"OR": or_val, "p": pval, "CI_lower": ci[0], "CI_upper": ci[1]}


def main():
    print("=" * 60)
    print("THRESHOLD COMPARISON: Median vs Q1 vs Q3")
    print("(Multivariate logistic: Age, Gender, BMI, PHQ-9, Night P_01)")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"age": "Age", "gender": "Gender", "P_01_Night": "Night_P_01_Median"})
    if "Age" not in df.columns:
        df["Age"] = df.get("age", np.nan)
    if "Gender" not in df.columns:
        df["Gender"] = df.get("gender", np.nan)
    if "Night_P_01_Median" not in df.columns and "P_01_Night" in df.columns:
        df["Night_P_01_Median"] = df["P_01_Night"]

    seqn_set = set(df["SEQN"].dropna().astype(int))
    paxhr = load_xpt(PAXHR_PATH)
    activity_col = next((c for c in ACTIVITY_COLS if c in paxhr.columns), None)
    paxhr = paxhr[paxhr["SEQN"].isin(seqn_set)].copy()
    paxhr = paxhr[paxhr[activity_col].notna() & (paxhr[activity_col] >= 0)]
    paxhr = paxhr.sort_values(["SEQN", "PAXDAYH", "PAXSSNHP"])
    paxhr["_hour_idx"] = paxhr.groupby(["SEQN", "PAXDAYH"]).cumcount()

    rows = []
    for seqn, grp in paxhr.groupby("SEQN"):
        act_all = grp[activity_col].values
        if len(act_all) < 24:
            continue
        q1 = np.nanpercentile(act_all, 25)
        q3 = np.nanpercentile(act_all, 75)
        night_mask = grp["_hour_idx"].isin(NIGHT_HOURS)
        act_night = grp.loc[night_mask, activity_col].values
        if len(act_night) < 4:
            continue
        rows.append({
            "SEQN": seqn,
            "Night_P_01_Q1": p01_from_sequence(act_night, q1),
            "Night_P_01_Q3": p01_from_sequence(act_night, q3),
        })

    thresh_df = pd.DataFrame(rows)
    df = df.merge(thresh_df, on="SEQN", how="inner")

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
    df = df.merge(dpq[["SEQN", "PHQ9_Score"]], on="SEQN", how="inner")
    df = df.dropna(subset=["Age", "Gender", "BMXBMI", "PHQ9_Score"])
    print(f"Analytic N: {len(df)}")

    res_med = run_logistic(df, "Night_P_01_Median")
    res_q1 = run_logistic(df, "Night_P_01_Q1")
    res_q3 = run_logistic(df, "Night_P_01_Q3")

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: Night P_01 (OR, p-value)")
    print("=" * 60)
    print(f"  Median: OR = {res_med['OR']:.4f} [95% CI {res_med['CI_lower']:.4f}-{res_med['CI_upper']:.4f}]  p = {res_med['p']:.6f}")
    print(f"  Q1:     OR = {res_q1['OR']:.4f} [95% CI {res_q1['CI_lower']:.4f}-{res_q1['CI_upper']:.4f}]  p = {res_q1['p']:.6f}")
    print(f"  Q3:     OR = {res_q3['OR']:.4f} [95% CI {res_q3['CI_lower']:.4f}-{res_q3['CI_upper']:.4f}]  p = {res_q3['p']:.6f}")

    print("\n" + "=" * 60)
    print("SITUATION CHECK (GMM = Median-based multivariate model)")
    print("=" * 60)
    or_med, or_q1, or_q3 = res_med["OR"], res_q1["OR"], res_q3["OR"]
    p_med, p_q1, p_q3 = res_med["p"], res_q1["p"], res_q3["p"]

    if (or_med > 1 and or_q1 < 1) or (or_med < 1 and or_q1 > 1):
        print("  >> SITUATION A: OR flipped (Median OR>1 vs Q1 OR<1 or vice versa).")
        print("     --> 'Inconsistent' - consider removing claim.")
    elif abs(p_med - p_q1) > 0.1 or (p_med > 0.05 and p_q1 < 0.05) or (p_med < 0.05 and p_q1 > 0.05):
        print("  >> SITUATION B: p-value meaningfully different (Median vs Q1 cross significance).")
        print("     --> 'Results differ' - report honestly.")
    else:
        print("  >> SITUATION C: Results consistent across thresholds.")
        print("     --> Median, Q1, Q3 all similar p ~ 0.15-0.20. Current sentence OK.")
    print()


if __name__ == "__main__":
    main()
