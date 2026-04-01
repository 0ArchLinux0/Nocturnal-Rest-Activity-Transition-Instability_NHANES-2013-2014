#!/usr/bin/env python3
"""
NHANES DEMO + DPQ + accelerometer → merged cohort with PHQ-9 and wear rules
==========================================================================
- Loads DEMO_H (demographics), DPQ_H (depression), and accelerometer-related
  files, merges on SEQN.
- PHQ-9 = sum of DPQ010–DPQ090 (0–3 each); 7/9 → missing per item.
- depression_suspected = 1 if PHQ9_Score >= 10 else 0 (NaN PHQ → NaN label).
- Wear rule (protocol days 1–7): ≥4 days with adequate recording AND ≥1 of
  those days on a weekend (PAXDAYWD Sunday=1 or Saturday=7).
  Adequate day: PAXVMD >= MIN_VALID_MINUTES and no day-level QC flags (PAXQFD).

NHANES 2013–2014 note: there is no public single-file PAXRAW_H.xpt; raw 80 Hz
data is PAX80_H on CDC FTP (per-participant archives). This script defaults to
PAXDAY_H + PAXHD_H (same PAM device/pipeline). If you place a legacy
PAXRAW_*.xpt (e.g. another cycle) at --paxraw-path, SEQN must appear there to
pass the accelerometer gate (wear counts still come from PAXDAY_H).

Usage:
  ./venv/bin/python build_final_full_cohort.py
  NHANES_RAW_DIR=/path/to/xpt ./venv/bin/python build_final_full_cohort.py --out final_full_cohort.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW = BASE_DIR / "nhanes_2013_2014_raw"

PHQ9_ITEMS = [f"DPQ{i:03d}" for i in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
DEPRESSION_CUTOFF = 10
# Days 1–7 = first week of wear in PAXDAYD codebook
WEAR_DAY_MIN = 1
WEAR_DAY_MAX = 7
MIN_VALID_DAYS = 4
MIN_VALID_MINUTES = 600  # common wear-time threshold (minutes with valid data)
# PAXDAYWD: 1=Sunday, 7=Saturday
WEEKEND_CODES = (1, 7)


def _sas_int_series(s: pd.Series) -> pd.Series:
    """Decode bytes/object from SAS xport to int; NaN if invalid."""
    if s.dtype != object:
        return pd.to_numeric(s, errors="coerce")

    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, bytes):
            return int(v.decode().strip())
        return int(pd.to_numeric(v, errors="coerce"))

    return s.map(one)


def _qflag_ok_day(qfd: float) -> bool:
    """PAXQFD: 0 = no QC flags; SAS may map 0 to tiny float."""
    if qfd is None or (isinstance(qfd, float) and np.isnan(qfd)):
        return False
    return float(qfd) < 0.5


def load_demo(path: Path) -> pd.DataFrame:
    df = pd.read_sas(str(path), format="xport")
    keep = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "WTMEC2YR"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out["SEQN"] = pd.to_numeric(out["SEQN"], errors="coerce").astype("Int64")
    return out.dropna(subset=["SEQN"])


def load_dpq_phq9(path: Path) -> pd.DataFrame:
    df = pd.read_sas(str(path), format="xport").copy()
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["SEQN"])
    for c in PHQ9_ITEMS:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c} in DPQ file")
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c].isin([7, 9]), c] = np.nan
        # SAS xport often maps 0 to ~5e-79
        df.loc[df[c].notna() & (df[c] < 0.5), c] = 0.0
    # PHQ-9 total only if all 9 items answered
    phq = df[PHQ9_ITEMS].sum(axis=1, min_count=len(PHQ9_ITEMS)).clip(0, 27)
    out = df[["SEQN"]].copy()
    out["PHQ9_Score"] = phq
    out["depression_suspected"] = np.where(
        out["PHQ9_Score"].notna(),
        (out["PHQ9_Score"] >= DEPRESSION_CUTOFF).astype(int),
        np.nan,
    )
    return out


def load_paxhd(path: Path) -> pd.DataFrame:
    df = pd.read_sas(str(path), format="xport").copy()
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["SEQN"])
    cols = ["SEQN", "PAXSTS", "PAXSENID", "PAXFDAY", "PAXLDAY"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def load_paxday_wear_summary(path: Path) -> pd.DataFrame:
    """One row per SEQN: valid-day counts in days 1–7, weekend coverage."""
    df = pd.read_sas(str(path), format="xport").copy()
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["SEQN"])
    df["PAXDAYD"] = _sas_int_series(df["PAXDAYD"])
    df["PAXDAYWD"] = _sas_int_series(df["PAXDAYWD"])
    df["PAXVMD"] = pd.to_numeric(df["PAXVMD"], errors="coerce")
    df["PAXQFD"] = pd.to_numeric(df["PAXQFD"], errors="coerce")

    win = df[
        (df["PAXDAYD"] >= WEAR_DAY_MIN)
        & (df["PAXDAYD"] <= WEAR_DAY_MAX)
    ].copy()
    win["day_adequate"] = win["PAXQFD"].map(_qflag_ok_day) & (win["PAXVMD"] >= MIN_VALID_MINUTES)
    win["weekend_day"] = win["PAXDAYWD"].isin(WEEKEND_CODES)

    adeq = win[win["day_adequate"]]
    g = adeq.groupby("SEQN", sort=False)
    summary = g.agg(
        n_valid_accel_days=("PAXDAYD", "nunique"),
        n_valid_weekend_days=("weekend_day", lambda s: int(s.sum())),
    ).reset_index()
    summary["has_weekend_valid_day"] = (summary["n_valid_weekend_days"] >= 1).astype(int)
    return summary


def load_paxraw_seqns(path: Path, chunksize: int = 500_000) -> set[int]:
    """Unique SEQN present in a legacy PAXRAW xpt (chunked)."""
    seqns: set[int] = set()
    reader = pd.read_sas(str(path), format="xport", chunksize=chunksize)
    for chunk in reader:
        if "SEQN" not in chunk.columns:
            raise RuntimeError(f"PAXRAW file missing SEQN: {path}")
        s = pd.to_numeric(chunk["SEQN"], errors="coerce").dropna().astype(int).unique()
        seqns.update(int(x) for x in s)
    return seqns


def main() -> None:
    ap = argparse.ArgumentParser(description="Build NHANES final_full_cohort.csv")
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(os.environ.get("NHANES_RAW_DIR", str(DEFAULT_RAW))),
        help="Directory containing DEMO_H, DPQ, PAXDAY, PAXHD xpt files",
    )
    ap.add_argument(
        "--paxraw-path",
        type=Path,
        default=None,
        help="Optional PAXRAW_*.xpt: if set and file exists, restrict to these SEQN",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "final_full_cohort.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()
    raw = args.raw_dir.resolve()

    demo_p = raw / "DEMO_H.xpt"
    dpq_p = raw / "DPQ_H.xpt"
    paxday_p = raw / "PAXDAY_H.xpt"
    paxhd_p = raw / "PAXHD_H.xpt"
    for p in (demo_p, dpq_p, paxday_p, paxhd_p):
        if not p.is_file():
            print(f"ERROR: missing required file: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Raw directory: {raw}")
    demo = load_demo(demo_p)
    dpq = load_dpq_phq9(dpq_p)
    paxhd = load_paxhd(paxhd_p)
    wear = load_paxday_wear_summary(paxday_p)

    # Participants with monitor summary data (PAXSTS==1 when present)
    if "PAXSTS" in paxhd.columns:
        paxhd = paxhd[pd.to_numeric(paxhd["PAXSTS"], errors="coerce") == 1].copy()

    merged = demo.merge(dpq, on="SEQN", how="inner")
    merged = merged.merge(paxhd, on="SEQN", how="inner")
    merged = merged.merge(wear, on="SEQN", how="inner")

    paxraw_path = args.paxraw_path
    if paxraw_path is None:
        cand = raw / "PAXRAW_H.xpt"
        paxraw_path = cand if cand.is_file() else None
    elif paxraw_path is not None and not paxraw_path.is_file():
        print(f"Warning: --paxraw-path not found, ignoring: {paxraw_path}")
        paxraw_path = None

    if paxraw_path is not None:
        print(f"Restricting to SEQN in PAXRAW: {paxraw_path}")
        raw_seqns = load_paxraw_seqns(paxraw_path)
        merged["in_paxraw_xpt"] = merged["SEQN"].isin(raw_seqns).astype(int)
        merged = merged[merged["SEQN"].isin(raw_seqns)].copy()
    else:
        print(
            "Note: PAXRAW_*.xpt not used (not in raw dir). "
            "NHANES 2013–2014 raw 80 Hz is PAX80_H on FTP; wear rules use PAXDAY_H."
        )

    keep_mask = (
        (merged["n_valid_accel_days"] >= MIN_VALID_DAYS)
        & (merged["has_weekend_valid_day"] == 1)
    )
    cohort = merged.loc[keep_mask].copy()
    cohort = cohort.sort_values("SEQN").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.out, index=False)

    print(
        f"Merged DEMO∩DPQ∩PAXHD∩PAXDAY (before wear filter): {len(merged)} "
        f"→ final cohort: {len(cohort)}"
    )
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
