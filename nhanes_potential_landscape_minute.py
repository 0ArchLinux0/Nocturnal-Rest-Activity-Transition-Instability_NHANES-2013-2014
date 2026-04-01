#!/usr/bin/env python3
"""
NHANES 1-Minute Potential Landscape Analysis (Sequential Baseline)
==================================================================
Processes minute-level PAXMIN data: nocturnal window (22:00–06:00), 15-state Markov,
stationary distribution, potential U = -ln(P_st), ΔU, Transition Entropy.
Data integrity first: iterate one SEQN at a time, save every 100 participants.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg

BASE_DIR = Path(__file__).parent
# Default: project root; else symlink or set NHANES_RAW_DIR to download_nhanes_2013_2014 output
_RAW = Path(os.environ.get("NHANES_RAW_DIR", str(BASE_DIR / "nhanes_2013_2014_raw")))
PAXMIN_PATH = BASE_DIR / "PAXMIN_H.xpt"
if not PAXMIN_PATH.exists() and (_RAW / "PAXMIN_H.xpt").exists():
    PAXMIN_PATH = _RAW / "PAXMIN_H.xpt"
OUTPUT_DIR = BASE_DIR / "outputs"
INTERIM_PATH = OUTPUT_DIR / "interim_results.csv"
ERROR_LOG = OUTPUT_DIR / "error_log.txt"
N_BINS = 15
EPS = 1e-9
VALID_FRAC = 0.80  # Skip if <80% of nocturnal window has valid data
NOCTURNAL_MINUTES = 8 * 60  # 22:00–06:00 = 480 minutes

# Nocturnal minute indices: 00:00–05:59 (0–359) and 22:00–23:59 (1320–1439)
NOCTURNAL_IDX = set(range(360)) | set(range(1320, 1440))


def get_seqn_list(paxmin: pd.DataFrame) -> list:
    """Get unique SEQN in order."""
    return paxmin["SEQN"].dropna().unique().tolist()


def filter_nocturnal_valid(grp: pd.DataFrame, activity_col: str, valid_col: str) -> np.ndarray:
    """
    Filter for nocturnal window (22:00–06:00) and paxstat/valid data only.
    Uses minute index from row order within each (SEQN, PAXDAYM).
    Returns 1D array of activity counts for valid nocturnal minutes.
    """
    grp = grp.sort_values(["PAXDAYM", "PAXSSNMP"])
    grp = grp.copy()
    grp["_minute_idx"] = grp.groupby("PAXDAYM").cumcount()
    valid_mask = (grp[valid_col] == 0) & grp[activity_col].notna() & (grp[activity_col] >= 0)
    grp = grp[valid_mask]
    nocturnal_mask = grp["_minute_idx"].isin(NOCTURNAL_IDX)
    act = grp.loc[nocturnal_mask, activity_col].values.astype(float)
    return act


def discretize_equal_width(act: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width binning. Returns bin indices 0..n_bins-1."""
    act = np.asarray(act, dtype=float)
    act = act[~np.isnan(act) & (act >= 0)]
    if len(act) < 10:
        return np.array([], dtype=int)
    lo, hi = np.percentile(act, [0.5, 99.5])  # Robust range
    if hi <= lo:
        hi = lo + 1
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.digitize(act, edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


def build_transition_matrix(bins: np.ndarray, n_bins: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Count matrix N_ij and row-normalized M_ij. Add eps to avoid zero-division.
    Returns (M_row_normalized, N_counts) for stationary and entropy.
    """
    n = len(bins)
    if n < 2:
        M = np.full((n_bins, n_bins), 1.0 / n_bins)
        return M, np.ones((n_bins, n_bins))
    N = np.zeros((n_bins, n_bins))
    for i in range(n - 1):
        a, b = int(bins[i]), int(bins[i + 1])
        if 0 <= a < n_bins and 0 <= b < n_bins:
            N[a, b] += 1
    M = N + eps
    M = M / M.sum(axis=1, keepdims=True)
    return M, N


def stationary_distribution(M: np.ndarray) -> np.ndarray:
    """Left eigenvector with eigenvalue 1 (stationary distribution)."""
    eigvals, eigvecs = linalg.eig(M.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.maximum(pi, 0)
    pi = pi / pi.sum()
    return pi


def potential_from_stationary(pi: np.ndarray) -> np.ndarray:
    """U_i = -ln(pi_i)."""
    pi = np.maximum(pi, EPS)
    return -np.log(pi)


def delta_u_barrier(U: np.ndarray) -> float:
    """Barrier height: max(U) - min(U) as dynamic range of potential."""
    U = U[np.isfinite(U)]
    if len(U) < 2:
        return 0.0
    return float(np.max(U) - np.min(U))


def transition_entropy_from_counts(N: np.ndarray, eps: float) -> float:
    """
    Joint transition entropy H = -sum_{i,j} p_ij * ln(p_ij).
    p_ij = N_ij / N_total (empirical joint), consistent with 2-state formulation.
    """
    p_joint = N + eps
    p_joint = p_joint / p_joint.sum()
    H = 0.0
    for p in p_joint.ravel():
        if p > 0:
            H -= p * np.log(p)
    return H


def process_one_participant(
    grp: pd.DataFrame,
    seqn,
    activity_col: str,
    valid_col: str,
    n_bins: int,
    eps: float,
    valid_frac: float,
) -> dict | None:
    """
    Process one SEQN. Returns dict with Delta_U, Transition_Entropy, Stationary_Dist (summary),
    or None if skipped.
    """
    act = filter_nocturnal_valid(grp, activity_col, valid_col)
    n_days = grp["PAXDAYM"].nunique()
    window_expected = int(480 * n_days * valid_frac)  # 80% of nocturnal window across days
    if len(act) < max(400, window_expected):
        return None
    bins = discretize_equal_width(act, n_bins)
    if len(bins) < 50:
        return None
    M, N = build_transition_matrix(bins, n_bins, eps)
    pi = stationary_distribution(M)
    U = potential_from_stationary(pi)
    delta_u = delta_u_barrier(U)
    H = transition_entropy_from_counts(N, eps)
    return {
        "SEQN": seqn,
        "Delta_U": delta_u,
        "Transition_Entropy": H,
        "N_nocturnal_valid": len(act),
        "pi_max": float(np.max(pi)),
        "pi_min": float(np.min(pi[pi > eps])),
    }


def main():
    import sys
    sample_mode = "--sample" in sys.argv or "-n" in sys.argv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("NHANES 1-Minute Potential Landscape (Sequential)")
    print("=" * 60)

    # Load PAXMIN (expect xpt; fallback logic for parquet/csv if added)
    if not PAXMIN_PATH.exists():
        print(f"ERROR: {PAXMIN_PATH} not found. Download from CDC NHANES.")
        return
    import pyreadstat
    paxmin, _ = pyreadstat.read_xport(str(PAXMIN_PATH))
    print(f"Loaded {len(paxmin):,} minute records")
    activity_col = "PAXMTSM"
    valid_col = "PAXQFM"  # 0 = valid (paxstat-like: reliable data only), >0 = QC flagged
    if activity_col not in paxmin.columns:
        print(f"ERROR: {activity_col} not in PAXMIN")
        return

    seqn_list = get_seqn_list(paxmin)
    if sample_mode:
        seqn_list = seqn_list[:10]
        print(f"Sample mode: first {len(seqn_list)} participants only")
    else:
        print(f"Participants: {len(seqn_list)}")

    results = []
    with open(ERROR_LOG, "w") as errf:
        for k, seqn in enumerate(seqn_list):
            try:
                grp = paxmin[paxmin["SEQN"] == seqn]
                row = process_one_participant(
                    grp, seqn, activity_col, valid_col,
                    n_bins=N_BINS, eps=EPS, valid_frac=VALID_FRAC,
                )
                if row is not None:
                    results.append(row)
            except Exception as e:
                errf.write(f"SEQN={seqn}: {e}\n")
                errf.flush()
                print(f"  [SKIP] SEQN={seqn}: {e}")
                continue
            if (k + 1) % 100 == 0:
                pd.DataFrame(results).to_csv(INTERIM_PATH, index=False)
                print(f"  Checkpoint: {len(results)} participants saved")

    df_out = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "potential_landscape_minute_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Total participants with valid metrics: {len(df_out)}")
    if len(df_out) > 0:
        print(f"  Delta_U: median {df_out['Delta_U'].median():.3f}")
        print(f"  Transition_Entropy: median {df_out['Transition_Entropy'].median():.3f}")


if __name__ == "__main__":
    main()
