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

Data: nhanes_2013_2014_raw/PAXMIN_H.xpt (or PAXMIN_H.xpt in project root)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from roc_plot_style import apply_roc_rcparams

BASE_DIR = Path(__file__).parent
RAW_DIR = Path(os.environ.get("NHANES_RAW_DIR", str(BASE_DIR / "nhanes_2013_2014_raw")))
OUTPUT_DIR = BASE_DIR / "outputs"
COHORT_CSV = OUTPUT_DIR / "processed_data_physics_ultimate.csv"

PAXMIN_PATH = RAW_DIR / "PAXMIN_H.xpt"
if not PAXMIN_PATH.exists():
    PAXMIN_PATH = BASE_DIR / "PAXMIN_H.xpt"

RESOLUTIONS_MIN = (5, 10, 30, 60)
ACTIVITY_COL = "PAXMTSM"
VALID_COL = "PAXQFM"  # 0 = reliable minute
READ_COLS = ["SEQN", "PAXDAYM", "PAXSSNMP", ACTIVITY_COL, VALID_COL]

# Nocturnal clock minutes within each calendar day (0 = midnight)
NOCTURNAL_RANGES = ((0, 360), (1320, 1440))


def block_overlaps_nocturnal(t0: int, width: int) -> bool:
    """True if [t0, t0+width) overlaps 00:00–06:00 or 22:00–24:00."""
    t1 = t0 + width
    for lo, hi in NOCTURNAL_RANGES:
        if max(t0, lo) < min(t1, hi):
            return True
    return False


def p01_from_sequence(act: np.ndarray, thresh: float) -> float:
    """P(0→1) with Rest=activity<=thresh, Active>thresh."""
    if len(act) < 4:
        return np.nan
    s = (act > thresh).astype(int)
    n_01 = np.sum((s[:-1] == 0) & (s[1:] == 1))
    n_0 = np.sum(s[:-1] == 0)
    return float(n_01 / n_0) if n_0 > 0 else np.nan


def _paxdaym_to_int(s: pd.Series) -> pd.Series:
    """SAS xport may load PAXDAYM as bytes (e.g. b'1')."""
    if s.dtype != object:
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    def one(v):
        if isinstance(v, bytes):
            return int(v.decode().strip())
        return int(pd.to_numeric(v, errors="coerce"))

    return s.map(one).astype("Int64")


def minute_frame_one_participant(grp: pd.DataFrame) -> pd.DataFrame:
    """Valid minutes only, with minute-of-day index _m per PAXDAYM."""
    grp = grp.sort_values(["PAXDAYM", "PAXSSNMP"]).copy()
    grp["PAXDAYM"] = _paxdaym_to_int(grp["PAXDAYM"])
    grp = grp[grp["PAXDAYM"].notna()]
    grp["_m"] = grp.groupby("PAXDAYM", sort=False).cumcount()
    # NHANES: PAXQFM 0 = reliable; 1/2 = not. read_sas often maps 0 to ~5e-79.
    q = pd.to_numeric(grp[VALID_COL], errors="coerce")
    ok = q.notna() & (~q.isin([1.0, 2.0])) & grp[ACTIVITY_COL].notna() & (grp[ACTIVITY_COL] >= 0)
    return grp.loc[ok, ["PAXDAYM", "_m", ACTIVITY_COL]]


def resample_sum(grp: pd.DataFrame, width: int) -> pd.DataFrame:
    """Sum activity in non-overlapping blocks of `width` minutes within each day."""
    g = grp.copy()
    g["_b"] = g["_m"] // width
    agg = g.groupby(["PAXDAYM", "_b"], as_index=False)[ACTIVITY_COL].sum()
    agg["_t0"] = agg["_b"] * width
    return agg


def metrics_all_resolutions(grp: pd.DataFrame) -> dict[str, float]:
    """
    For each resolution: median threshold from ALL blocks that day; P_01 on
    nocturnal blocks only, ordered by time (PAXDAYM, _b).
    """
    minute_df = minute_frame_one_participant(grp)
    if len(minute_df) < 200:
        return {f"P01_{w}m": np.nan for w in RESOLUTIONS_MIN}

    out: dict[str, float] = {}
    for w in RESOLUTIONS_MIN:
        agg = resample_sum(minute_df, w)
        if len(agg) < 10:
            out[f"P01_{w}m"] = np.nan
            continue
        agg["_night"] = agg["_t0"].apply(lambda t: block_overlaps_nocturnal(int(t), w))
        all_act = agg[ACTIVITY_COL].values.astype(float)
        night = agg[agg["_night"]].sort_values(["PAXDAYM", "_b"])
        night_act = night[ACTIVITY_COL].values.astype(float)
        if len(night_act) < 4 or len(all_act) < 10:
            out[f"P01_{w}m"] = np.nan
            continue
        thresh = float(np.nanmedian(all_act))
        out[f"P01_{w}m"] = p01_from_sequence(night_act, thresh)
    return out


def load_cohort_seqns(limit: int | None = None) -> list[int]:
    if not COHORT_CSV.is_file():
        raise FileNotFoundError(f"Cohort CSV not found: {COHORT_CSV}")
    df = pd.read_csv(COHORT_CSV, usecols=["SEQN"])
    s = df["SEQN"].dropna().astype(int).unique().tolist()
    s = sorted(set(s))
    if limit:
        s = s[:limit]
    return s


def discover_seqns_first_chunk(paxmin_path: Path, chunksize: int, limit: int) -> set[int]:
    """First-chunk SEQN discovery when cohort CSV is missing (PAXMIN usually SEQN-sorted)."""
    ch0 = next(pd.read_sas(str(paxmin_path), format="xport", chunksize=chunksize))
    u = pd.unique(ch0["SEQN"].dropna().astype(int))
    u = sorted(set(int(x) for x in u))[:limit]
    return set(u)


def resolve_target_seqns(paxmin_path: Path, chunksize: int, sample: int | None) -> set[int]:
    lim = sample if sample is not None else None
    try:
        s = load_cohort_seqns(limit=lim)
        print(f"Using cohort SEQN list ({len(s)} ids)")
        return set(s)
    except FileNotFoundError:
        n = lim if lim is not None else 200
        t = discover_seqns_first_chunk(paxmin_path, chunksize, n)
        print(f"No cohort CSV — first-chunk SEQN discovery: {len(t)} ids (cap {n})")
        return t


def load_phq9_bmi_age() -> pd.DataFrame:
    """Merge PHQ-9, BMI, age from raw xpt (same folder as PAXMIN)."""
    dpq_path = RAW_DIR / "DPQ_H.xpt"
    bmx_path = RAW_DIR / "BMX_H.xpt"
    demo_path = RAW_DIR / "DEMO_H.xpt"
    for p in (dpq_path, bmx_path):
        if not p.exists():
            p2 = BASE_DIR / p.name
            if p2.exists():
                if p is dpq_path:
                    dpq_path = p2
                else:
                    bmx_path = p2
    import pyreadstat
    dpq, _ = pyreadstat.read_xport(str(dpq_path))
    bmx, _ = pyreadstat.read_xport(str(bmx_path))
    dpq_cols = [c for c in dpq.columns if c.startswith("DPQ") and len(c) >= 6 and c[3:6].isdigit()]
    for c in dpq_cols:
        dpq[c] = pd.to_numeric(dpq[c], errors="coerce")
        dpq.loc[dpq[c].isin([7, 9]), c] = np.nan
    dpq["PHQ9_Score"] = dpq[dpq_cols].sum(axis=1)
    out = dpq[["SEQN", "PHQ9_Score"]].merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="inner")
    if demo_path.exists():
        demo, _ = pyreadstat.read_xport(str(demo_path))
        age_col = "RIDAGEYR" if "RIDAGEYR" in demo.columns else None
        if age_col:
            out = out.merge(demo[["SEQN", age_col]], on="SEQN", how="left")
            out = out.rename(columns={age_col: "Age"})
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = None
    if "--sample" in sys.argv:
        i = sys.argv.index("--sample")
        sample = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 300
    chunksize = 800_000
    if "--chunksize" in sys.argv:
        i = sys.argv.index("--chunksize")
        chunksize = int(sys.argv[i + 1])

    if not PAXMIN_PATH.exists():
        print(f"ERROR: PAXMIN not found at {PAXMIN_PATH}")
        sys.exit(1)

    print(f"PAXMIN: {PAXMIN_PATH}")
    target = resolve_target_seqns(PAXMIN_PATH, chunksize, sample)
    print(f"Target SEQN count: {len(target)}")

    buffers: dict[int, list[pd.DataFrame]] = {}
    chunk_iter = pd.read_sas(str(PAXMIN_PATH), format="xport", chunksize=chunksize)
    for chunk in chunk_iter:
        miss = [c for c in READ_COLS if c not in chunk.columns]
        if miss:
            raise RuntimeError(f"Missing columns: {miss}")
        chunk = chunk[READ_COLS].copy()
        chunk["SEQN"] = pd.to_numeric(chunk["SEQN"], errors="coerce")
        chunk = chunk[chunk["SEQN"].notna()]
        chunk["SEQN"] = chunk["SEQN"].astype(int)
        chunk = chunk[chunk["SEQN"].isin(target)]
        if chunk.empty:
            continue
        for seqn, g in chunk.groupby("SEQN", sort=False):
            sid = int(seqn)
            buffers.setdefault(sid, []).append(g)

    rows = []
    for seqn in sorted(buffers.keys()):
        parts = buffers.get(seqn, [])
        if not parts:
            continue
        grp = pd.concat(parts, ignore_index=True)
        m = metrics_all_resolutions(grp)
        m["SEQN"] = seqn
        rows.append(m)

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
