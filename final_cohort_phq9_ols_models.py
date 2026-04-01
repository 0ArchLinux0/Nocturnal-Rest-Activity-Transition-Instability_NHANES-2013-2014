#!/usr/bin/env python3
"""
OLS for continuous PHQ-9: final_full_cohort.csv + per-SEQN P01 (10m) and WASO proxy.

Model 1: PHQ9_Score ~ P01 + Age + Sex + BMI
Model 2: PHQ9_Score ~ P01 + Age + Sex + BMI + PIR + TST + WASO

Prints to stdout and writes outputs/phq9_ols_p01_comparison.txt.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("NHANES_RAW_DIR", str(BASE_DIR / "nhanes_2013_2014_raw")))
COHORT_CSV = BASE_DIR / "final_full_cohort.csv"
PAM_CSV = BASE_DIR / "outputs" / "final_cohort_pam_metrics.csv"
OUT_TXT = BASE_DIR / "outputs" / "phq9_ols_p01_comparison.txt"

ALPHA = 0.05
# Z-score within each model's complete-case sample; P01 stays on native [0,1]-scale probability units.
Z_CONTINUOUS_M1 = ("Age", "BMI")
Z_CONTINUOUS_M2 = ("Age", "BMI", "PIR", "TST", "WASO")


def zscore_in_sample(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu = float(out[c].mean())
        sig = float(out[c].std(ddof=0))
        out[f"{c}_z"] = (out[c] - mu) / sig if sig >= 1e-12 else 0.0
    return out


def load_merged() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_CSV)
    cohort["SEQN"] = cohort["SEQN"].astype(int)
    pam = pd.read_csv(PAM_CSV)
    pam["SEQN"] = pam["SEQN"].astype(int)
    df = cohort.merge(pam[["SEQN", "P01_10m", "WASO_proxy_min"]], on="SEQN", how="inner")
    df = df.rename(columns={"P01_10m": "P01", "WASO_proxy_min": "WASO"})

    demo = pd.read_sas(str(RAW_DIR / "DEMO_H.xpt"), format="xport")
    demo["SEQN"] = pd.to_numeric(demo["SEQN"], errors="coerce").astype("Int64")
    demo = demo[["SEQN", "INDFMPIR"]].copy()
    demo["PIR"] = pd.to_numeric(demo["INDFMPIR"], errors="coerce")

    bmx = pd.read_sas(str(RAW_DIR / "BMX_H.xpt"), format="xport")
    bmx["SEQN"] = pd.to_numeric(bmx["SEQN"], errors="coerce").astype("Int64")
    bmx = bmx[["SEQN", "BMXBMI"]].copy()
    bmx["BMI"] = pd.to_numeric(bmx["BMXBMI"], errors="coerce")

    slq = pd.read_sas(str(RAW_DIR / "SLQ_H.xpt"), format="xport")
    slq["SEQN"] = pd.to_numeric(slq["SEQN"], errors="coerce").astype("Int64")
    slq["TST"] = pd.to_numeric(slq["SLD010H"], errors="coerce")
    slq.loc[slq["TST"].isin([77, 99]), "TST"] = pd.NA
    slq = slq[["SEQN", "TST"]]

    df = df.merge(demo[["SEQN", "PIR"]], on="SEQN", how="left")
    df = df.merge(bmx[["SEQN", "BMI"]], on="SEQN", how="left")
    df = df.merge(slq, on="SEQN", how="left")

    df["Age"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df["Sex"] = pd.to_numeric(df["RIAGENDR"], errors="coerce")
    df["PHQ9_Score"] = pd.to_numeric(df["PHQ9_Score"], errors="coerce")
    # cohort PHQ9 may already be clean; clip numerical noise
    df.loc[df["PHQ9_Score"].notna() & (df["PHQ9_Score"] < 0.5), "PHQ9_Score"] = 0.0
    df["PHQ9_Score"] = df["PHQ9_Score"].clip(0, 27)

    return df


def fit_ols(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    return sm.OLS(y.astype(float), Xc).fit()


def main() -> None:
    df = load_merged()

    raw1 = ["P01", "Age", "Sex", "BMI"]
    raw2 = raw1 + ["PIR", "TST", "WASO"]

    d1 = df.dropna(subset=["PHQ9_Score"] + raw1).copy().reset_index(drop=True)
    d2 = df.dropna(subset=["PHQ9_Score"] + raw2).copy().reset_index(drop=True)

    d1 = zscore_in_sample(d1, Z_CONTINUOUS_M1)
    d2 = zscore_in_sample(d2, Z_CONTINUOUS_M2)

    X1_cols = ["P01", "Age_z", "Sex", "BMI_z"]
    X2_cols = ["P01", "Age_z", "Sex", "BMI_z", "PIR_z", "TST_z", "WASO_z"]

    m1 = fit_ols(d1["PHQ9_Score"], d1[X1_cols])
    m2 = fit_ols(d2["PHQ9_Score"], d2[X2_cols])

    def p01_row(m, name: str) -> dict:
        return {
            "Model": name,
            "P01_coef": m.params["P01"],
            "P01_SE": m.bse["P01"],
            "P01_pvalue": m.pvalues["P01"],
            "R2": m.rsquared,
            "Adj_R2": m.rsquared_adj,
            "N": int(m.nobs),
        }

    r1 = p01_row(m1, "1 (base): +Age, Sex, BMI")
    r2 = p01_row(m2, "2 (extended): +PIR, TST, WASO")

    lines: list[str] = []
    lines.append("PHQ-9 OLS with P01 (10m night instability), NHANES 2013–2014 final cohort")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Model 1 complete-case N = {r1['N']}")
    lines.append(f"Model 2 complete-case N = {r2['N']}")
    lines.append("")
    lines.append("P01 = nocturnal P01 from 10-min block MIMS (same definition as P01_10m in PAM cache).")
    lines.append("TST = SLD010H (usual weekday sleep hours). WASO = actigraphy proxy from PAXPREDM (minute-level).")
    lines.append(
        "Scaling: P01 is left in raw units (transition probability scale). "
        f"Model 1 z-scores (within complete-case sample): {', '.join(Z_CONTINUOUS_M1)}. "
        f"Model 2 z-scores: {', '.join(Z_CONTINUOUS_M2)}. "
        "Sex (NHANES RIAGENDR) is unscaled."
    )
    lines.append("")
    # Markdown table
    hdr = "| Model | P01 coef | SE | p-value | R² | Adj. R² | N |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines.append(hdr)
    lines.append(sep)
    for r in (r1, r2):
        sig = "*" if r["P01_pvalue"] < ALPHA else ""
        lines.append(
            f"| {r['Model']} | {r['P01_coef']:.6f} | {r['P01_SE']:.6f} | {r['P01_pvalue']:.4g}{sig} | "
            f"{r['R2']:.4f} | {r['Adj_R2']:.4f} | {r['N']} |"
        )
    lines.append("")
    lines.append("* p < 0.05")
    lines.append("")
    lines.append("--- Full summaries (statsmodels) ---")
    lines.append("")
    lines.append(str(m1.summary()))
    lines.append("")
    lines.append(str(m2.summary()))
    lines.append("")
    lines.append("--- Interpretation ---")
    p1, p2 = r1["P01_pvalue"], r2["P01_pvalue"]
    if p2 < ALPHA:
        lines.append(
            f"After controlling for PIR, TST, and WASO, P01 remains statistically significant "
            f"(p = {p2:.4g} < {ALPHA})."
        )
    else:
        lines.append(
            f"After controlling for PIR, TST, and WASO, P01 is not significant at α={ALPHA} "
            f"(p = {p2:.4g})."
        )
    if p1 < ALPHA and p2 < ALPHA:
        lines.append("P01 was also significant in Model 1; the effect persists in Model 2.")
    elif p1 < ALPHA and p2 >= ALPHA:
        lines.append("P01 was significant in Model 1 but not after adding sleep/socioeconomic controls (possible mediation/collinearity).")
    elif p1 >= ALPHA and p2 < ALPHA:
        lines.append("P01 became significant only with extended controls (unusual; check sample change N1 vs N2).")

    out = "\n".join(lines)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
