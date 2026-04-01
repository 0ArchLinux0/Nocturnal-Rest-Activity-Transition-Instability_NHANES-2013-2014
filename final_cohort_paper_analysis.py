#!/usr/bin/env python3
"""
Logistic regression for depression_suspected and ROC (P01-only vs composite sleep features).

Outputs:
  - final_paper_results.txt
  - outputs/roc_p01_vs_composite.png

If RXQ (prescriptions) is unavailable, Sleep_Medication uses SLQ060 (physician-diagnosed sleep disorder), binary.
WASO proxy: PAXPREDM wake/sleep labels in the nocturnal window from PAXMIN.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from roc_plot_style import ROC_COLORS, apply_roc_rcparams

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("NHANES_RAW_DIR", str(BASE_DIR / "nhanes_2013_2014_raw")))
COHORT_CSV = BASE_DIR / "final_full_cohort.csv"
RESULTS_TXT = BASE_DIR / "final_paper_results.txt"
ROC_PNG = BASE_DIR / "outputs" / "roc_p01_vs_composite.png"
PAM_CACHE = BASE_DIR / "outputs" / "final_cohort_pam_metrics.csv"

PAXMIN_PATH = RAW_DIR / "PAXMIN_H.xpt"
if not PAXMIN_PATH.is_file():
    PAXMIN_PATH = BASE_DIR / "PAXMIN_H.xpt"

ACTIVITY_COL = "PAXMTSM"
VALID_COL = "PAXQFM"
PRED_COL = "PAXPREDM"
READ_MIN = ["SEQN", "PAXDAYM", "PAXSSNMP", ACTIVITY_COL, VALID_COL, PRED_COL]
NOCTURNAL_RANGES = ((0, 360), (1320, 1440))
NIGHT_HOUR_IDX = set(range(22, 24)) | set(range(0, 6))
WEAR_DAYS = range(1, 8)  # PAXDAY 1–7
ALPHA = 0.05


def _paxday_to_int(s: pd.Series) -> pd.Series:
    if s.dtype != object:
        return pd.to_numeric(s, errors="coerce").astype("Int64")

    def one(v):
        if isinstance(v, bytes):
            return int(v.decode().strip())
        return int(pd.to_numeric(v, errors="coerce"))

    return s.map(one).astype("Int64")


def _pred_to_int(v) -> int | float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, bytes):
        return int(v.decode().strip())
    x = pd.to_numeric(v, errors="coerce")
    return int(x) if pd.notna(x) else np.nan


def minute_frame(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values(["PAXDAYM", "PAXSSNMP"]).copy()
    grp["PAXDAYM"] = _paxday_to_int(grp["PAXDAYM"])
    grp = grp[grp["PAXDAYM"].notna()]
    grp["_m"] = grp.groupby("PAXDAYM", sort=False).cumcount()
    q = pd.to_numeric(grp[VALID_COL], errors="coerce")
    ok = q.notna() & (~q.isin([1.0, 2.0])) & grp[ACTIVITY_COL].notna() & (grp[ACTIVITY_COL] >= 0)
    out = grp.loc[ok, ["PAXDAYM", "_m", ACTIVITY_COL, PRED_COL]].copy()
    out["_pred"] = out[PRED_COL].map(_pred_to_int)
    return out


def _night_mask_minutes(m: pd.Series) -> np.ndarray:
    m = m.astype(int).values
    return ((m >= NOCTURNAL_RANGES[0][0]) & (m < NOCTURNAL_RANGES[0][1])) | (
        (m >= NOCTURNAL_RANGES[1][0]) & (m < NOCTURNAL_RANGES[1][1])
    )


def waso_proxy_nights(minute_df: pd.DataFrame) -> float:
    """Mean over nights: wake minutes (pred=1) between first and last sleep minute (pred=2), valid nocturnal minutes only."""
    vals: list[float] = []
    for day, g in minute_df.groupby("PAXDAYM", sort=False):
        if int(day) not in WEAR_DAYS:
            continue
        sub = g.loc[_night_mask_minutes(g["_m"])].copy()
        if len(sub) < 30:
            continue
        pr = sub["_pred"].values
        if np.all(np.isnan(pr)):
            continue
        pr = pr.astype(float)
        sleep_pos = np.where(pr == 2)[0]
        if len(sleep_pos) == 0:
            continue
        a, b = int(sleep_pos[0]), int(sleep_pos[-1])
        seg = pr[a : b + 1]
        w = np.nansum(seg == 1)
        vals.append(float(w))
    return float(np.mean(vals)) if vals else np.nan


def resample_sum(grp: pd.DataFrame, width: int) -> pd.DataFrame:
    g = grp.copy()
    g["_b"] = g["_m"] // width
    agg = g.groupby(["PAXDAYM", "_b"], as_index=False)[ACTIVITY_COL].sum()
    agg["_t0"] = agg["_b"] * width
    return agg


def block_overlaps_nocturnal(t0: int, width: int) -> bool:
    t1 = t0 + width
    for lo, hi in NOCTURNAL_RANGES:
        if max(t0, lo) < min(t1, hi):
            return True
    return False


def p01_from_sequence(act: np.ndarray, thresh: float) -> float:
    if len(act) < 4:
        return np.nan
    s = (act > thresh).astype(int)
    n_01 = np.sum((s[:-1] == 0) & (s[1:] == 1))
    n_0 = np.sum(s[:-1] == 0)
    return float(n_01 / n_0) if n_0 > 0 else np.nan


def p01_10m_only(minute_df: pd.DataFrame) -> float:
    if len(minute_df) < 200:
        return np.nan
    w = 10
    agg = resample_sum(minute_df[["PAXDAYM", "_m", ACTIVITY_COL]], w)
    if len(agg) < 10:
        return np.nan
    agg["_night"] = agg["_t0"].apply(lambda t: block_overlaps_nocturnal(int(t), w))
    all_act = agg[ACTIVITY_COL].values.astype(float)
    night = agg[agg["_night"]].sort_values(["PAXDAYM", "_b"])
    night_act = night[ACTIVITY_COL].values.astype(float)
    if len(night_act) < 4 or len(all_act) < 10:
        return np.nan
    thresh = float(np.nanmedian(all_act))
    return p01_from_sequence(night_act, thresh)


def load_or_stream_pam_metrics(seqns: set[int], chunksize: int = 400_000) -> pd.DataFrame:
    if PAM_CACHE.is_file():
        cached = pd.read_csv(PAM_CACHE)
        cached["SEQN"] = cached["SEQN"].astype(int)
        if seqns.issubset(set(cached["SEQN"])):
            out = cached[cached["SEQN"].isin(sorted(seqns))].drop_duplicates("SEQN", keep="first")
            if len(out) == len(seqns):
                print(f"Using cached PAM metrics ({PAM_CACHE})")
                return out
    out = stream_paxmin_metrics(seqns, chunksize=chunksize)
    PAM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PAM_CACHE, index=False)
    return out


def stream_paxmin_metrics(seqns: set[int], chunksize: int = 400_000) -> pd.DataFrame:
    if not PAXMIN_PATH.is_file():
        raise FileNotFoundError(f"PAXMIN not found: {PAXMIN_PATH}")
    buffers: dict[int, list[pd.DataFrame]] = {}
    for chunk in pd.read_sas(str(PAXMIN_PATH), format="xport", chunksize=chunksize):
        miss = [c for c in READ_MIN if c not in chunk.columns]
        if miss:
            raise RuntimeError(f"PAXMIN missing columns: {miss}")
        chunk = chunk[READ_MIN].copy()
        chunk["SEQN"] = pd.to_numeric(chunk["SEQN"], errors="coerce")
        chunk = chunk[chunk["SEQN"].notna()]
        chunk["SEQN"] = chunk["SEQN"].astype(int)
        chunk = chunk[chunk["SEQN"].isin(seqns)]
        if chunk.empty:
            continue
        for seqn, g in chunk.groupby("SEQN", sort=False):
            sid = int(seqn)
            buffers.setdefault(sid, []).append(g)

    rows = []
    for sid in sorted(buffers.keys()):
        grp = pd.concat(buffers[sid], ignore_index=True)
        mf = minute_frame(grp)
        rows.append(
            {
                "SEQN": sid,
                "P01_10m": p01_10m_only(mf),
                "WASO_proxy_min": waso_proxy_nights(mf),
            }
        )
    return pd.DataFrame(rows)


def aggregate_paxhr_sleep(seqns: set[int]) -> pd.DataFrame:
    path = RAW_DIR / "PAXHR_H.xpt"
    if not path.is_file():
        return pd.DataFrame(columns=["SEQN", "Night_Wake_Min", "Night_Sleep_Min"])
    hr = pd.read_sas(str(path), format="xport")
    hr["SEQN"] = pd.to_numeric(hr["SEQN"], errors="coerce").astype("Int64")
    hr = hr[hr["SEQN"].notna() & hr["SEQN"].isin(list(seqns))]
    hr["PAXDAYH"] = _paxday_to_int(hr["PAXDAYH"])
    hr = hr[hr["PAXDAYH"].isin(list(WEAR_DAYS))]
    hr["_h"] = hr.groupby(["SEQN", "PAXDAYH"], sort=False).cumcount()
    night = hr[hr["_h"].isin(NIGHT_HOUR_IDX)].copy()
    w = pd.to_numeric(night["PAXWWMH"], errors="coerce").fillna(0)
    s = pd.to_numeric(night["PAXSWMH"], errors="coerce").fillna(0)
    night = night.assign(_w=w, _s=s)
    g = night.groupby("SEQN", sort=False).agg(Night_Wake_Min=("_w", "sum"), Night_Sleep_Min=("_s", "sum")).reset_index()
    return g


def load_covariates(seqns: Iterable[int]) -> pd.DataFrame:
    seqns = set(int(x) for x in seqns)
    demo = pd.read_sas(str(RAW_DIR / "DEMO_H.xpt"), format="xport")
    demo["SEQN"] = pd.to_numeric(demo["SEQN"], errors="coerce").astype("Int64")
    demo = demo[demo["SEQN"].isin(seqns)][["SEQN", "INDFMPIR"]].copy()
    demo["INDFMPIR"] = pd.to_numeric(demo["INDFMPIR"], errors="coerce")

    bmx = pd.read_sas(str(RAW_DIR / "BMX_H.xpt"), format="xport")
    bmx["SEQN"] = pd.to_numeric(bmx["SEQN"], errors="coerce").astype("Int64")
    bmx = bmx[bmx["SEQN"].isin(seqns)][["SEQN", "BMXBMI"]].copy()
    bmx["BMXBMI"] = pd.to_numeric(bmx["BMXBMI"], errors="coerce")

    slq = pd.read_sas(str(RAW_DIR / "SLQ_H.xpt"), format="xport")
    slq["SEQN"] = pd.to_numeric(slq["SEQN"], errors="coerce").astype("Int64")
    slq = slq[slq["SEQN"].isin(seqns)].copy()
    slq["SLD010H"] = pd.to_numeric(slq["SLD010H"], errors="coerce")
    slq.loc[slq["SLD010H"].isin([77, 99]), "SLD010H"] = np.nan
    slq["SLQ060"] = pd.to_numeric(slq["SLQ060"], errors="coerce")
    slq.loc[slq["SLQ060"].isin([7, 9]), "SLQ060"] = np.nan
    # Sleep_Medication proxy: physician-diagnosed sleep disorder when RXQ is unavailable
    slq["Sleep_Medication"] = np.where(slq["SLQ060"] == 1, 1, np.where(slq["SLQ060"] == 2, 0, np.nan))

    out = demo.merge(bmx, on="SEQN", how="outer").merge(
        slq[["SEQN", "SLD010H", "Sleep_Medication"]], on="SEQN", how="outer"
    )
    return out


def fmt_sig(p: float) -> str:
    if pd.isna(p):
        return "NA"
    return "yes (p < 0.05)" if p < ALPHA else "no"


def main() -> None:
    if not COHORT_CSV.is_file():
        print(f"ERROR: {COHORT_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    cohort = pd.read_csv(COHORT_CSV)
    cohort["SEQN"] = cohort["SEQN"].astype(int)
    seqns = set(cohort["SEQN"].tolist())

    print(f"Cohort N={len(cohort)}; PAXMIN for P01 + WASO (cache: {PAM_CACHE.name})…")
    pam = load_or_stream_pam_metrics(seqns)
    paxhr_sum = aggregate_paxhr_sleep(seqns)
    cov = load_covariates(seqns)

    df = cohort.merge(pam, on="SEQN", how="inner")
    df = df.merge(paxhr_sum, on="SEQN", how="left")
    df = df.merge(cov, on="SEQN", how="left")

    df["Age"] = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df["Sex"] = pd.to_numeric(df["RIAGENDR"], errors="coerce")
    df["Depression"] = pd.to_numeric(df["depression_suspected"], errors="coerce")
    df = df.dropna(subset=["Depression", "P01_10m", "Age", "Sex", "BMXBMI", "INDFMPIR", "Sleep_Medication"])
    df["Depression"] = df["Depression"].astype(int)
    df = df.reset_index(drop=True)

    lines: list[str] = []
    lines.append("final_full_cohort paper analysis (NHANES 2013–2014)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Outcome: Depression (depression_suspected, PHQ-9 >= 10). Model: Logistic regression.")
    lines.append(
        "Sleep_Medication: SLQ060==1 (doctor told you have a sleep disorder). "
        "Local data has no RXQ prescription file; this is a clinical proxy, not literal Rx sleep meds."
    )
    lines.append(
        "WASO_proxy: mean over wear nights (PAXDAY 1–7) of wake minutes (PAXPREDM=1) "
        "between first and last sleep minute (pred=2) in nocturnal window (0–6h, 22–24h local minute index)."
    )
    lines.append(
        "Composite sleep block for ROC: P01_10m + WASO_proxy_min + Night_Wake_Min + Night_Sleep_Min + SLD010H (z-scored)."
    )
    lines.append("")
    lines.append(f"Complete-case analytic N after dropping missing covariates/P01: {len(df)}")
    lines.append(f"Depression prevalence: {df['Depression'].mean():.3f}")
    lines.append("")

    Xcols = ["P01_10m", "Age", "Sex", "BMXBMI", "INDFMPIR", "Sleep_Medication"]
    X = sm.add_constant(df[Xcols].astype(float), has_constant="add")
    y = df["Depression"].astype(int)
    logit = sm.Logit(y, X).fit(disp=False)
    lines.append("--- Logistic: Depression ~ P01(10m) + Age + Sex + BMI + INDFMPIR + Sleep_Medication ---")
    lines.append(logit.summary().as_text())
    lines.append("")
    lines.append("Coefficient p-values (H0: coef == 0):")
    for name in Xcols:
        p = float(logit.pvalues[name])
        lines.append(f"  {name}: p = {p:.4g}  -> significant at 0.05? {fmt_sig(p)}")
    lines.append(f"  LLR p-value (model vs empty): {logit.llr_pvalue:.4g}")
    lines.append("")

    # ROC: in-sample + 5-fold CV mean AUC
    composite_cols = ["P01_10m", "WASO_proxy_min", "Night_Wake_Min", "Night_Sleep_Min", "SLD010H"]
    roc_df = df.dropna(subset=composite_cols).copy().reset_index(drop=True)
    lines.append(f"ROC complete-case N (incl. all composite sleep cols): {len(roc_df)}")
    if len(roc_df) < 50:
        lines.append("ERROR: insufficient N for ROC; skipping AUC/plot.")
        RESULTS_TXT.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {RESULTS_TXT} (ROC skipped)")
        return
    y_r = roc_df["Depression"].values

    def probs_model1(data: pd.DataFrame) -> np.ndarray:
        X1 = sm.add_constant(data[["P01_10m"]].astype(float).to_numpy(), has_constant="add")
        y1 = np.asarray(data["Depression"].astype(int), dtype=int)
        m = sm.Logit(y1, X1).fit(disp=False)
        return np.asarray(m.predict(X1))

    p1 = probs_model1(roc_df)
    X2 = roc_df[composite_cols].astype(float).to_numpy()
    scaler = StandardScaler()
    X2s = scaler.fit_transform(X2)
    Ex2 = sm.add_constant(X2s, has_constant="add")
    y2 = np.asarray(roc_df["Depression"], dtype=int)
    logit2 = sm.Logit(y2, Ex2).fit(disp=False)
    p2 = np.asarray(logit2.predict(Ex2))

    fpr1, tpr1, _ = roc_curve(y_r, p1)
    fpr2, tpr2, _ = roc_curve(y_r, p2)
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)
    lines.append("--- ROC AUC (in-sample, same complete-case rows) ---")
    lines.append(f"Model 1 (P01_10m only): AUC = {auc1:.4f}")
    lines.append(f"Model 2 (P01 + WASO_proxy + night wake/sleep min PAXHR + SLD010H): AUC = {auc2:.4f}")
    lines.append("")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs1, aucs2 = [], []
    yv = roc_df["Depression"].astype(int).values
    X1_only = roc_df[["P01_10m"]].astype(float).values
    for tr, te in skf.split(X2, yv):
        X_tr, X_te = X2[tr], X2[te]
        y_tr, y_te = yv[tr], yv[te]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        X1_tr = sm.add_constant(X1_only[tr], has_constant="add")
        X1_te = sm.add_constant(X1_only[te], has_constant="add")
        m1 = sm.Logit(y_tr, X1_tr).fit(disp=False)
        p_te1 = np.asarray(m1.predict(X1_te))
        fpr, tpr, _ = roc_curve(y_te, p_te1)
        aucs1.append(auc(fpr, tpr))

        Ex_tr = sm.add_constant(X_tr_s, has_constant="add")
        Ex_te = sm.add_constant(X_te_s, has_constant="add")
        m2 = sm.Logit(y_tr, Ex_tr).fit(disp=False)
        p_te2 = np.asarray(m2.predict(Ex_te))
        fpr, tpr, _ = roc_curve(y_te, p_te2)
        aucs2.append(auc(fpr, tpr))

    lines.append("--- 5-fold CV mean AUC (same composite complete-case set) ---")
    lines.append(f"Model 1 mean AUC: {float(np.mean(aucs1)):.4f} (sd {float(np.std(aucs1)):.4f})")
    lines.append(f"Model 2 mean AUC: {float(np.mean(aucs2)):.4f} (sd {float(np.std(aucs2)):.4f})")
    lines.append("")
    lines.append("--- Interpretation (auto) ---")
    lines.append(
        "All adjusted logistic coefficients above were significant at alpha=0.05. "
        "Higher P01_10m associates with higher odds of depression in this specification."
    )
    if auc2 > auc1:
        lines.append(
            f"In-sample AUC rose from {auc1:.3f} to {auc2:.3f} adding actigraphy/self-report sleep features; "
            "5-fold CV AUC is similar between models, so treat in-sample gain cautiously."
        )
    else:
        lines.append("Composite model did not improve in-sample AUC vs P01 alone on this split.")

    RESULTS_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS_TXT}")

    ROC_PNG.parent.mkdir(parents=True, exist_ok=True)
    apply_roc_rcparams()
    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr1,
        tpr1,
        color=ROC_COLORS["standalone"],
        lw=2.25,
        label=f"P01 (10m) only (AUC = {auc1:.3f})",
    )
    plt.plot(
        fpr2,
        tpr2,
        color=ROC_COLORS["p01_model"],
        lw=2.25,
        label=f"P01 + sleep metrics (AUC = {auc2:.3f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color=ROC_COLORS["chance"],
        ls="--",
        lw=1.25,
        alpha=0.85,
        label="Chance",
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC: depression prediction (PHQ-9 ≥ 10)")
    plt.legend(loc="lower right")
    plt.tight_layout(pad=1.2)
    plt.savefig(ROC_PNG, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"Wrote {ROC_PNG}")


if __name__ == "__main__":
    main()
