#!/usr/bin/env python3
"""
NHANES PAXMIN: multi-resolution nocturnal instability (P_01) vs 60-min reference
===============================================================================
Resamples minute-level MIMS (PAXMTSM) to 5, 10, 30, 60 min; computes Night P_01
(Markov Rest→Active, median threshold on full-day blocks at that resolution).

Outputs:
  - outputs/paxmin_resolution_metrics.csv
  - outputs/paxmin_resolution_correlations.csv
  - outputs/paxmin_resolution_scatter_vs_60m.png
  - outputs/paxmin_resolution_phq9_regression.txt

Data: data/raw/PAXMIN_H.xpt and data/raw/PAXHD_H.xpt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from nhanes_p01.p01 import compute_p01_corrected_resolutions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from roc_plot_style import apply_roc_rcparams

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("NHANES_RAW_DIR", str(BASE_DIR / "data" / "raw")))
OUTPUT_DIR = BASE_DIR / "outputs"
COHORT_CSV = BASE_DIR / "final_full_cohort.csv"

PAXMIN_PATH = RAW_DIR / "PAXMIN_H.xpt"
PAXHD_PATH = RAW_DIR / "PAXHD_H.xpt"

RESOLUTIONS_MIN = (5, 10, 30, 60)
PHQ9_ITEMS = [f"DPQ{i:03d}" for i in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
ACTIVITY_COL = "PAXMTSM"
VALID_COL = "PAXQFM"
READ_COLS = ["SEQN", "PAXDAYM", "PAXSSNMP", "PAXTSM", ACTIVITY_COL, VALID_COL]


def metrics_all_resolutions(grp: pd.DataFrame, paxftime: str) -> dict[str, float]:
    metrics = compute_p01_corrected_resolutions(grp, paxftime)
    return {f"P01_{resolution}m": metrics[resolution]["p01"] for resolution in RESOLUTIONS_MIN}


def load_cohort_seqns(limit: int | None = None) -> list[int]:
    if not COHORT_CSV.is_file():
        raise FileNotFoundError(f"Cohort CSV not found: {COHORT_CSV}")
    df = pd.read_csv(COHORT_CSV, usecols=["SEQN"])
    s = df["SEQN"].dropna().astype(int).unique().tolist()
    s = sorted(set(s))
    if limit:
        s = s[:limit]
    return s


def resolve_target_seqns(sample: int | None) -> set[int]:
    seqns = load_cohort_seqns(limit=sample)
    print(f"Using canonical cohort SEQN list ({len(seqns)} ids)")
    return set(seqns)

def load_phq9_bmi_age() -> pd.DataFrame:
    """Merge PHQ-9, BMI, age from raw xpt (same folder as PAXMIN)."""
    dpq_path = RAW_DIR / "DPQ_H.xpt"
    bmx_path = RAW_DIR / "BMX_H.xpt"
    demo_path = RAW_DIR / "DEMO_H.xpt"
    for path in (dpq_path, bmx_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required raw input not found: {path}")
    import pyreadstat
    dpq, _ = pyreadstat.read_xport(str(dpq_path))
    bmx, _ = pyreadstat.read_xport(str(bmx_path))
    missing = [column for column in PHQ9_ITEMS if column not in dpq.columns]
    if missing:
        raise ValueError(f"DPQ file is missing canonical PHQ-9 items: {missing}")
    for c in PHQ9_ITEMS:
        dpq[c] = pd.to_numeric(dpq[c], errors="coerce")
        dpq.loc[dpq[c].isin([7, 9]), c] = np.nan
        dpq.loc[dpq[c].notna() & (dpq[c].abs() < 1e-10), c] = 0.0
        dpq.loc[~dpq[c].isin([0, 1, 2, 3]), c] = np.nan
    dpq["PHQ9_Score"] = dpq[PHQ9_ITEMS].sum(axis=1, min_count=9).clip(0, 27)
    out = dpq[["SEQN", "PHQ9_Score"]].merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="inner")
    if demo_path.exists():
        demo, _ = pyreadstat.read_xport(str(demo_path))
        age_col = "RIDAGEYR" if "RIDAGEYR" in demo.columns else None
        if age_col:
            out = out.merge(demo[["SEQN", age_col]], on="SEQN", how="left")
            out = out.rename(columns={age_col: "Age"})
    return out


def main() -> None:
    sample = None
    if "--sample" in sys.argv:
        i = sys.argv.index("--sample")
        sample = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 300
    chunksize = 800_000
    if "--chunksize" in sys.argv:
        i = sys.argv.index("--chunksize")
        chunksize = int(sys.argv[i + 1])

    target = resolve_target_seqns(sample)
    if not PAXMIN_PATH.is_file():
        raise FileNotFoundError(f"PAXMIN not found: {PAXMIN_PATH}")
    if not PAXHD_PATH.is_file():
        raise FileNotFoundError(f"PAXHD not found: {PAXHD_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"PAXMIN: {PAXMIN_PATH}")
    print(f"Target SEQN count: {len(target)}")

    import pyreadstat

    headers, _ = pyreadstat.read_xport(str(PAXHD_PATH), usecols=["SEQN", "PAXFTIME"])
    headers["SEQN"] = pd.to_numeric(headers["SEQN"], errors="coerce")
    header_times = {
        int(seqn): paxftime
        for seqn, paxftime in headers[["SEQN", "PAXFTIME"]].itertuples(index=False, name=None)
        if pd.notna(seqn)
    }
    rows: list[dict[str, float | int]] = []
    current_sid: int | None = None
    current_parts: list[pd.DataFrame] = []
    processed: set[int] = set()
    last_source_seqn: float | None = None

    def finalize(seqn: int, parts: list[pd.DataFrame]) -> None:
        if seqn not in header_times:
            raise RuntimeError(f"PAXFTIME missing from PAXHD for SEQN {seqn}")
        group = pd.concat(parts, ignore_index=True)
        metrics = metrics_all_resolutions(group, header_times[seqn])
        metrics["SEQN"] = seqn
        rows.append(metrics)
        processed.add(seqn)

    chunk_iter = pyreadstat.read_file_in_chunks(
        pyreadstat.read_xport,
        str(PAXMIN_PATH),
        chunksize=chunksize,
        usecols=READ_COLS,
    )
    for chunk, _ in chunk_iter:
        source_seqn = pd.to_numeric(chunk["SEQN"], errors="coerce").to_numpy(dtype=float)
        finite_seqn = source_seqn[np.isfinite(source_seqn)]
        if len(finite_seqn):
            if np.any(np.diff(finite_seqn) < 0) or (
                last_source_seqn is not None and finite_seqn[0] < last_source_seqn
            ):
                raise RuntimeError("PAXMIN SEQN order is not monotonic.")
            last_source_seqn = float(finite_seqn[-1])
        chunk["SEQN"] = pd.to_numeric(chunk["SEQN"], errors="coerce")
        chunk = chunk.loc[chunk["SEQN"].isin(target)]
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

    missing = target - processed
    if missing:
        raise RuntimeError(f"PAXMIN has no rows for {len(missing)} cohort participants.")
    df = pd.DataFrame(rows)
    if df.empty:
        print("No participants processed.")
        sys.exit(1)

    df.to_csv(OUTPUT_DIR / "paxmin_resolution_metrics.csv", index=False)
    print(f"Saved metrics: {len(df)} participants")

    # Pearson: 60m vs others
    ref = "P01_60m"
    corr_rows = []
    for col in ["P01_5m", "P01_10m", "P01_30m"]:
        sub = df[[ref, col]].dropna()
        if len(sub) < 30:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pearsonr(sub[ref], sub[col])
        corr_rows.append({"pair": f"{ref}_vs_{col}", "n": len(sub), "pearson_r": r, "p_value": p})
    pd.DataFrame(corr_rows).to_csv(OUTPUT_DIR / "paxmin_resolution_correlations.csv", index=False)

    # Scatter plots vs 60m (2x2 grid: three panels + blank for A4-friendly aspect)
    apply_roc_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panel_axes = (axes[0, 0], axes[0, 1], axes[1, 0])
    for ax, col in zip(panel_axes, ["P01_5m", "P01_10m", "P01_30m"]):
        sub = df[[ref, col]].dropna()
        ax.scatter(sub[col], sub[ref], alpha=0.25, s=14, c="#2E86AB", edgecolors="none")
        if len(sub) >= 30:
            r, p = stats.pearsonr(sub[ref], sub[col])
            z = np.polyfit(sub[col], sub[ref], 1)
            xs = np.linspace(sub[col].min(), sub[col].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color="#E94F37", lw=2.5)
            ax.set_title(f"60m vs {col.replace('P01_', '').replace('m', '')} min\nr = {r:.3f}, p = {p:.2e}")
        else:
            ax.set_title(col)
        ax.set_xlabel(col.replace("P01_", "P₀₁ ") + " (resampled)")
        ax.set_ylabel("P₀₁ 60 min")
    axes[1, 1].axis("off")
    fig.suptitle("Nocturnal instability (P₀₁): 60-min vs coarser resampling", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.2)
    fig.savefig(
        OUTPUT_DIR / "paxmin_resolution_scatter_vs_60m.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)
    print(f"Saved {OUTPUT_DIR / 'paxmin_resolution_scatter_vs_60m.png'}")

    # PHQ-9 regression per resolution
    try:
        cov = load_phq9_bmi_age()
        merged = df.merge(cov, on="SEQN", how="inner")
        merged = merged.dropna(subset=["PHQ9_Score"])
    except Exception as e:
        print(f"Warning: could not load covariates for PHQ-9 analysis: {e}")
        merged = None

    lines = []
    if merged is not None and len(merged) > 50:
        import statsmodels.api as sm
        from sklearn.preprocessing import StandardScaler
        for col in ["P01_5m", "P01_10m", "P01_30m", "P01_60m"]:
            base_cols = ["PHQ9_Score", col, "BMXBMI"]
            if "Age" in merged.columns:
                base_cols.append("Age")
            use = merged[[c for c in base_cols if c in merged.columns]].dropna()
            if len(use) < 80:
                lines.append(f"{col}: insufficient n after dropna ({len(use)})\n")
                continue
            y = use["PHQ9_Score"].astype(float)
            Xcols = [c for c in [col, "Age", "BMXBMI"] if c in use.columns]
            X = use[Xcols].astype(float)
            if X[col].std() < 1e-12:
                lines.append(f"{col}: zero variance, skip\n")
                continue
            Xs = pd.DataFrame(
                StandardScaler().fit_transform(X),
                columns=X.columns,
                index=X.index,
            )
            Xs = sm.add_constant(Xs)
            model = sm.OLS(y, Xs).fit()
            lines.append(f"\n--- OLS: PHQ-9 ~ {col} (z-scored) + " + ", ".join(c for c in Xcols if c != col) + " ---\n")
            lines.append(model.summary().as_text())
            lines.append("\n")
        Path(OUTPUT_DIR / "paxmin_resolution_phq9_regression.txt").write_text("".join(lines))
        print(f"Saved {OUTPUT_DIR / 'paxmin_resolution_phq9_regression.txt'}")
    else:
        Path(OUTPUT_DIR / "paxmin_resolution_phq9_regression.txt").write_text(
            "Skipped: merge PHQ-9 failed or n too small.\n"
        )

    print("Done.")


if __name__ == "__main__":
    main()
