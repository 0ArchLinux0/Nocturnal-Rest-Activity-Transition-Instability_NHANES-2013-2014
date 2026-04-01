#!/usr/bin/env python3
"""
Three ROC curves for depression_suspected:
  A: P01 only
  B: Age + Sex + BMI
  C: P01 + Age + Sex + BMI
Writes outputs/roc_comparison.png.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import auc, roc_curve

from roc_plot_style import ROC_COLORS, apply_roc_rcparams

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("NHANES_RAW_DIR", str(BASE_DIR / "nhanes_2013_2014_raw")))
COHORT_CSV = BASE_DIR / "final_full_cohort.csv"
PAM_CSV = BASE_DIR / "outputs" / "final_cohort_pam_metrics.csv"
OUT_PNG = BASE_DIR / "outputs" / "roc_comparison.png"


def main() -> None:
    apply_roc_rcparams()
    cohort = pd.read_csv(COHORT_CSV)
    cohort["SEQN"] = cohort["SEQN"].astype(int)
    pam = pd.read_csv(PAM_CSV)
    pam["SEQN"] = pam["SEQN"].astype(int)
    df = cohort.merge(pam[["SEQN", "P01_10m"]], on="SEQN", how="inner")

    bmx = pd.read_sas(str(RAW_DIR / "BMX_H.xpt"), format="xport")
    bmx["SEQN"] = pd.to_numeric(bmx["SEQN"], errors="coerce").astype("Int64")
    bmx = bmx[["SEQN", "BMXBMI"]].copy()
    bmx["BMI"] = pd.to_numeric(bmx["BMXBMI"], errors="coerce")
    df = df.merge(bmx[["SEQN", "BMI"]], on="SEQN", how="left")

    df["Age"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df["Sex"] = pd.to_numeric(df["RIAGENDR"], errors="coerce")
    df["y"] = pd.to_numeric(df["depression_suspected"], errors="coerce").astype("Int64")
    df["P01"] = pd.to_numeric(df["P01_10m"], errors="coerce")

    d = df.dropna(subset=["y", "P01", "Age", "Sex", "BMI"]).copy()
    d = d.reset_index(drop=True)
    y = np.asarray(d["y"], dtype=int)

    # Design matrices (float)
    p01 = d[["P01"]].to_numpy(dtype=float)
    demo = d[["Age", "Sex", "BMI"]].to_numpy(dtype=float)
    Xa = sm.add_constant(p01, has_constant="add")
    Xb = sm.add_constant(demo, has_constant="add")
    Xc = sm.add_constant(np.hstack([p01, demo]), has_constant="add")

    ma = sm.Logit(y, Xa).fit(disp=False)
    mb = sm.Logit(y, Xb).fit(disp=False)
    mc = sm.Logit(y, Xc).fit(disp=False)

    pa = np.asarray(ma.predict(Xa))
    pb = np.asarray(mb.predict(Xb))
    pc = np.asarray(mc.predict(Xc))

    curves = [
        ("A: P01 only", pa, ROC_COLORS["standalone"]),
        ("B: Age + Sex + BMI", pb, ROC_COLORS["baseline"]),
        ("C: P01 + Age + Sex + BMI", pc, ROC_COLORS["p01_model"]),
    ]

    plt.figure(figsize=(8, 8))
    plt.plot(
        [0, 1],
        [0, 1],
        color=ROC_COLORS["chance"],
        ls="--",
        lw=1.25,
        alpha=0.85,
        label="Chance",
    )
    for label, p, color in curves:
        fpr, tpr, _ = roc_curve(y, p)
        a = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.25, label=f"{label} (AUC = {a:.3f})")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC: depression suspected (PHQ-9 ≥ 10)")
    plt.legend(loc="lower right")
    plt.tight_layout(pad=1.2)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"N = {len(d)} (complete cases)")
    for label, p, _c in curves:
        fpr, tpr, _ = roc_curve(y, p)
        print(f"  {label}: AUC = {auc(fpr, tpr):.4f}")
    print(f"Wrote {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()
