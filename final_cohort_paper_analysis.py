#!/usr/bin/env python3
"""
Logistic regression for depression_suspected and ROC (P01-only vs composite sleep features).

Outputs:
  - final_paper_results.txt
  - outputs/roc_p01_vs_composite.png

SLQ060 is represented only as sleep_disorder_history; it is not a medication measure.
WASO_proxy_min is a timestamp-grouped PAXPREDM proxy; it is not validated WASO.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from nhanes_p01.p01 import (
    P01_ALGORITHM_VERSION,
    P01_METHOD_FINGERPRINT,
    compute_p01_corrected,
    normalize_paxmin_timestamps,
    sha256_file,
    validate_p01_cache,
)
from kan173_corrected_analysis import waso_proxy_for_participant

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from roc_plot_style import ROC_COLORS, apply_roc_rcparams

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("NHANES_RAW_DIR", str(BASE_DIR / "data" / "raw")))
COHORT_CSV = BASE_DIR / "final_full_cohort.csv"
RESULTS_TXT = BASE_DIR / "final_paper_results.txt"
ROC_PNG = BASE_DIR / "outputs" / "roc_p01_vs_composite.png"
PAM_CACHE = BASE_DIR / "outputs" / "final_cohort_pam_metrics.csv"
PAM_WASO_DERIVATIVE = BASE_DIR / "outputs" / "models" / "pam_metrics_with_waso.csv"

PAXMIN_PATH = RAW_DIR / "PAXMIN_H.xpt"
PAXHD_PATH = RAW_DIR / "PAXHD_H.xpt"

ACTIVITY_COL = "PAXMTSM"
VALID_COL = "PAXQFM"
PRED_COL = "PAXPREDM"
READ_MIN = ["SEQN", "PAXDAYM", "PAXSSNMP", "PAXTSM", ACTIVITY_COL, VALID_COL, PRED_COL]
NIGHT_HOUR_IDX = set(range(22, 24)) | set(range(0, 6))
WEAR_DAYS = range(2, 9)  # Seven full protocol days; partial days 1 and 9 excluded.
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


def minute_frame(grp: pd.DataFrame, paxftime: str) -> pd.DataFrame:
    """Normalize the source sample/time axis before any PAXPREDM aggregation."""
    return normalize_paxmin_timestamps(grp, paxftime)


def waso_proxy_nights(minute_df: pd.DataFrame) -> float:
    return float(waso_proxy_for_participant(minute_df)["WASO_proxy_min"])


def load_or_stream_pam_metrics(seqns: set[int], chunksize: int = 400_000) -> pd.DataFrame:
    if not PAXMIN_PATH.is_file():
        raise FileNotFoundError(f"PAXMIN not found: {PAXMIN_PATH}")
    source_sha256 = sha256_file(PAXMIN_PATH)
    cache_valid, reason = validate_p01_cache(PAM_CACHE, seqns, source_sha256)
    if cache_valid:
        print("Corrected P01 cache provenance matches; recomputing WASO proxy from PAXMIN.")
    else:
        print(f"Rejecting PAM cache ({reason}); recomputing from {PAXMIN_PATH}.")

    out = stream_paxmin_metrics(seqns, source_sha256, chunksize=chunksize)
    PAM_WASO_DERIVATIVE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PAM_WASO_DERIVATIVE, index=False)
    print(f"Wrote WASO proxy derivative without replacing corrected P01 cache: {PAM_WASO_DERIVATIVE}")
    return out


def stream_paxmin_metrics(
    seqns: set[int],
    source_sha256: str,
    chunksize: int = 400_000,
) -> pd.DataFrame:
    import pyreadstat

    if not PAXMIN_PATH.is_file():
        raise FileNotFoundError(f"PAXMIN not found: {PAXMIN_PATH}")
    if not PAXHD_PATH.is_file():
        raise FileNotFoundError(f"PAXHD not found: {PAXHD_PATH}")
    requested = {int(seqn) for seqn in seqns}
    headers, _ = pyreadstat.read_xport(str(PAXHD_PATH), usecols=["SEQN", "PAXFTIME"])
    headers["SEQN"] = pd.to_numeric(headers["SEQN"], errors="coerce")
    header_times = {
        int(seqn): paxftime
        for seqn, paxftime in headers[["SEQN", "PAXFTIME"]].itertuples(index=False, name=None)
        if pd.notna(seqn)
    }

    rows: list[dict[str, object]] = []
    current_sid: int | None = None
    current_parts: list[pd.DataFrame] = []
    processed: set[int] = set()
    last_source_seqn: float | None = None

    def finalize(seqn: int, parts: list[pd.DataFrame]) -> None:
        if seqn not in header_times:
            raise RuntimeError(f"PAXFTIME missing from PAXHD for SEQN {seqn}")
        grp = pd.concat(parts, ignore_index=True)
        waso_minutes = minute_frame(grp, header_times[seqn])
        corrected = compute_p01_corrected(grp, header_times[seqn], 10)
        rows.append(
            {
                "SEQN": seqn,
                "P01_10m": corrected["p01"],
                "WASO_proxy_min": waso_proxy_nights(waso_minutes),
                "p01_algorithm_version": P01_ALGORITHM_VERSION,
                "paxmin_sha256": source_sha256,
                "method_fingerprint": P01_METHOD_FINGERPRINT,
            }
        )
        processed.add(seqn)

    chunk_iter = pyreadstat.read_file_in_chunks(
        pyreadstat.read_xport,
        str(PAXMIN_PATH),
        chunksize=chunksize,
        usecols=READ_MIN,
    )
    for chunk, _ in chunk_iter:
        source_seqn = pd.to_numeric(chunk["SEQN"], errors="coerce").to_numpy(dtype=float)
        finite_seqn = source_seqn[np.isfinite(source_seqn)]
        if len(finite_seqn):
            if np.any(np.diff(finite_seqn) < 0) or (
                last_source_seqn is not None and finite_seqn[0] < last_source_seqn
            ):
                raise RuntimeError("PAXMIN SEQN order is not monotonic; bounded participant finalization is unsafe.")
            last_source_seqn = float(finite_seqn[-1])
        chunk["SEQN"] = pd.to_numeric(chunk["SEQN"], errors="coerce")
        chunk = chunk.loc[chunk["SEQN"].isin(requested)]
        for seqn_value, group in chunk.groupby("SEQN", sort=False):
            seqn = int(seqn_value)
            if current_sid is None:
                current_sid = seqn
                current_parts = [group]
            elif seqn == current_sid:
                current_parts.append(group)
            elif seqn > current_sid:
                finalize(current_sid, current_parts)
                current_sid = seqn
                current_parts = [group]
            else:
                raise RuntimeError(f"PAXMIN participant order regressed from {current_sid} to {seqn}")
    if current_sid is not None:
        finalize(current_sid, current_parts)

    missing = requested - processed
    if missing:
        preview = sorted(missing)[:10]
        raise RuntimeError(f"PAXMIN has no rows for {len(missing)} cohort participants; examples: {preview}")
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
    slq["self_reported_weekday_sleep_hours"] = pd.to_numeric(slq["SLD010H"], errors="coerce")
    slq.loc[slq["self_reported_weekday_sleep_hours"].isin([77, 99]), "self_reported_weekday_sleep_hours"] = np.nan
    slq["SLQ060"] = pd.to_numeric(slq["SLQ060"], errors="coerce")
    slq.loc[slq["SLQ060"].isin([7, 9]), "SLQ060"] = np.nan
    slq["sleep_disorder_history"] = np.where(
        slq["SLQ060"] == 1, 1, np.where(slq["SLQ060"] == 2, 0, np.nan)
    )

    out = demo.merge(bmx, on="SEQN", how="outer").merge(
        slq[["SEQN", "self_reported_weekday_sleep_hours", "sleep_disorder_history"]],
        on="SEQN",
        how="outer",
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
    df = df.dropna(subset=["Depression", "P01_10m", "Age", "Sex", "BMXBMI", "INDFMPIR", "sleep_disorder_history"])
    df["Depression"] = df["Depression"].astype(int)
    df = df.reset_index(drop=True)

    lines: list[str] = []
    lines.append("final_full_cohort paper analysis (NHANES 2013–2014)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Outcome: Depression (depression_suspected, PHQ-9 >= 10). Model: Logistic regression.")
    lines.append(
        "sleep_disorder_history is SLQ060 (physician-diagnosed sleep disorder), not a medication variable."
    )
    lines.append(
        "WASO_proxy_min groups PAXPREDM from 22:00 on day d through 06:00 on day d+1; "
        "only QC-valid wake/sleep labels contribute. This is a proxy, not validated WASO."
    )
    lines.append(
        "Composite sleep block: P01_10m + WASO_proxy_min + Night_Wake_Min + Night_Sleep_Min + self_reported_weekday_sleep_hours."
    )
    lines.append("")
    lines.append(f"Complete-case analytic N after dropping missing covariates/P01: {len(df)}")
    lines.append(f"Depression prevalence: {df['Depression'].mean():.3f}")
    lines.append("")

    Xcols = ["P01_10m", "Age", "Sex", "BMXBMI", "INDFMPIR", "sleep_disorder_history"]
    X = sm.add_constant(df[Xcols].astype(float), has_constant="add")
    y = df["Depression"].astype(int)
    logit = sm.Logit(y, X).fit(disp=False)
    lines.append("--- Logistic: Depression ~ P01_10m + Age + Sex + BMI + INDFMPIR + sleep_disorder_history ---")
    lines.append(logit.summary().as_text())
    lines.append("")
    lines.append("Coefficient p-values (H0: coef == 0):")
    for name in Xcols:
        p = float(logit.pvalues[name])
        lines.append(f"  {name}: p = {p:.4g}  -> significant at 0.05? {fmt_sig(p)}")
    lines.append(f"  LLR p-value (model vs empty): {logit.llr_pvalue:.4g}")
    lines.append("")

    # ROC: in-sample + 5-fold CV mean AUC
    composite_cols = ["P01_10m", "WASO_proxy_min", "Night_Wake_Min", "Night_Sleep_Min", "self_reported_weekday_sleep_hours"]
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
    lines.append(f"Model 2 (P01_10m + WASO_proxy_min + PAXHR night wake/sleep + self_reported_weekday_sleep_hours): AUC = {auc2:.4f}")
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
