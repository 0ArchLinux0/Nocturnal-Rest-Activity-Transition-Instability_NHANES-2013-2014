#!/usr/bin/env python3
"""
NHANES 1-minute potential landscape — parallel pipeline (5,124+ participants)
==============================================================================
Memory-safe: streams PAXMIN in chunks → per-SEQN Parquet staging, then
ProcessPoolExecutor (cpu_count()-1 workers). tqdm + failed_log.txt + final Parquet.

Usage:
  ./venv/bin/python nhanes_potential_landscape_minute_parallel.py
  ./venv/bin/python nhanes_potential_landscape_minute_parallel.py --skip-stage1   # reuse staging
  ./venv/bin/python nhanes_potential_landscape_minute_parallel.py --sample 50     # first 50 SEQN only
"""

from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from tqdm import tqdm

# Reuse scientific core from sequential module
from nhanes_potential_landscape_minute import (
    BASE_DIR,
    EPS,
    N_BINS,
    NOCTURNAL_IDX,
    PAXMIN_PATH,
    OUTPUT_DIR,
    VALID_FRAC,
    build_transition_matrix,
    delta_u_barrier,
    discretize_equal_width,
    filter_nocturnal_valid,
    potential_from_stationary,
    transition_entropy_from_counts,
)

STAGING_DIR = OUTPUT_DIR / "paxmin_staging_parquet"
STAGING_PARTS = OUTPUT_DIR / "paxmin_staging_parts"
RESULT_PARQUET = OUTPUT_DIR / "potential_landscape_minute_parallel.parquet"
FAILED_LOG = OUTPUT_DIR / "failed_log.txt"
COHORT_CSV = OUTPUT_DIR / "processed_data_physics_ultimate.csv"

ACTIVITY_COL = "PAXMTSM"
VALID_COL = "PAXQFM"  # 0 = reliable (NHANES QC); analogous to paxstat valid
READ_COLS = ["SEQN", "PAXDAYM", "PAXSSNMP", ACTIVITY_COL, VALID_COL]


def stationary_distribution(M: np.ndarray) -> np.ndarray:
    """
    Stationary distribution π with π M = π (left eigenvector, eigenvalue 1).
    Falls back to power iteration if eig numerically unstable.
    """
    n = M.shape[0]
    eigvals, eigvecs = linalg.eig(M.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi = np.real(eigvecs[:, idx])
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= 0 or not np.isfinite(s):
        return stationary_distribution_power(M)
    pi = pi / s
    # Sanity check: π M ≈ π
    if np.max(np.abs(pi @ M - pi)) > 1e-5:
        pi = stationary_distribution_power(M)
    return pi


def stationary_distribution_power(M: np.ndarray, tol: float = 1e-12, max_iter: int = 50_000) -> np.ndarray:
    """Power iteration: π_{k+1} = π_k M, normalized."""
    n = M.shape[0]
    pi = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        pi_new = pi @ M
        pi_new = pi_new / (pi_new.sum() + 1e-300)
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            return pi_new
        pi = pi_new
    return pi


def compute_metrics_from_df(grp: pd.DataFrame, seqn: int) -> dict | None:
    """Core metrics for one participant DataFrame (full minute history for that SEQN)."""
    act = filter_nocturnal_valid(grp, ACTIVITY_COL, VALID_COL)
    n_days = grp["PAXDAYM"].nunique()
    window_expected = int(480 * n_days * VALID_FRAC)
    if len(act) < max(400, window_expected):
        return None
    bins = discretize_equal_width(act, N_BINS)
    if len(bins) < 50:
        return None
    M, N = build_transition_matrix(bins, N_BINS, EPS)
    pi = stationary_distribution(M)
    U = potential_from_stationary(pi)
    delta_u = delta_u_barrier(U)
    H = transition_entropy_from_counts(N, EPS)
    return {
        "SEQN": seqn,
        "Delta_U": delta_u,
        "Transition_Entropy": H,
        "N_nocturnal_valid": len(act),
        "pi_max": float(np.max(pi)),
        "pi_min": float(np.min(pi[pi > EPS])),
    }


def process_participant(seqn: int, staging_root: str) -> dict:
    """
    Worker: load only this SEQN's parquet from staging; compute metrics or return error record.
    Must be top-level for pickling on spawn.
    """
    root = Path(staging_root)
    path = root / f"{int(seqn)}.parquet"
    try:
        if not path.is_file():
            return {"SEQN": seqn, "status": "error", "error": "missing_staging_parquet"}
        df = pd.read_parquet(path, columns=READ_COLS)
        if df.empty:
            return {"SEQN": seqn, "status": "error", "error": "empty_parquet"}
        row = compute_metrics_from_df(df, int(seqn))
        if row is None:
            return {"SEQN": seqn, "status": "skipped", "error": "quality_threshold"}
        row["status"] = "ok"
        return row
    except Exception as e:
        return {
            "SEQN": seqn,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }


def load_target_seqns(limit: int | None) -> list[int]:
    """Cohort SEQNs (e.g. 5,124); fallback: scan staging folder."""
    if COHORT_CSV.is_file():
        df = pd.read_csv(COHORT_CSV, usecols=["SEQN"])
        seqns = df["SEQN"].dropna().astype(int).unique().tolist()
    else:
        seqns = []
    if not seqns and STAGING_DIR.is_dir():
        seqns = [int(p.stem) for p in STAGING_DIR.glob("*.parquet") if p.stem.isdigit()]
    seqns = sorted(set(seqns))
    if limit is not None:
        seqns = seqns[:limit]
    return seqns


def stage_paxmin_chunked(
    xpt_path: Path,
    target_seqns: set[int],
    staging_dir: Path,
    parts_root: Path,
    chunksize: int = 800_000,
) -> None:
    """
    Stream-read PAXMIN .xpt in chunks (only READ_COLS + target SEQN).
    Writes part-{chunk_id}.parquet under parts_root/{SEQN}/ (bounded memory: one chunk).
    Then merges to staging_dir/{SEQN}.parquet.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    parts_root.mkdir(parents=True, exist_ok=True)

    print(f"Staging PAXMIN → {staging_dir} (chunk size {chunksize:,})")
    chunk_iter = pd.read_sas(str(xpt_path), format="xport", chunksize=chunksize)
    chunk_id = 0
    for chunk in tqdm(chunk_iter, desc="Chunk read (SAS)", unit="chunk"):
        missing = [c for c in READ_COLS if c not in chunk.columns]
        if missing:
            raise RuntimeError(f"PAXMIN missing columns: {missing}")
        chunk = chunk[READ_COLS].copy()
        chunk["SEQN"] = pd.to_numeric(chunk["SEQN"], errors="coerce")
        chunk = chunk[chunk["SEQN"].notna()]
        chunk["SEQN"] = chunk["SEQN"].astype(int)
        chunk = chunk[chunk["SEQN"].isin(target_seqns)]
        if chunk.empty:
            chunk_id += 1
            continue
        for seqn, g in chunk.groupby("SEQN", sort=False):
            sid = int(seqn)
            d = parts_root / str(sid)
            d.mkdir(parents=True, exist_ok=True)
            g.to_parquet(d / f"part_{chunk_id:06d}.parquet", index=False, compression="snappy")
        chunk_id += 1

    print(f"Merging part files → {staging_dir} …")
    seqn_dirs = [p for p in parts_root.iterdir() if p.is_dir() and p.name.isdigit()]
    for d in tqdm(sorted(seqn_dirs, key=lambda x: int(x.name)), desc="Merge SEQN", unit="id"):
        sid = int(d.name)
        parts = sorted(d.glob("part_*.parquet"))
        if not parts:
            continue
        dfs = [pd.read_parquet(p) for p in parts]
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(staging_dir / f"{sid}.parquet", index=False, compression="snappy")
    print(f"Staging done. One parquet per SEQN: {len(list(staging_dir.glob('*.parquet')))}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stage1", action="store_true", help="Skip SAS→Parquet staging (reuse folder)")
    ap.add_argument("--sample", type=int, default=None, metavar="N", help="Process only first N cohort SEQNs")
    ap.add_argument("--chunksize", type=int, default=800_000, help="SAS chunk size for staging")
    args = ap.parse_args()

    if not PAXMIN_PATH.exists():
        raise FileNotFoundError(f"PAXMIN not found: {PAXMIN_PATH}")

    target_list = load_target_seqns(args.sample)
    if not target_list:
        raise RuntimeError(f"No SEQNs: add {COHORT_CSV} or run staging first.")
    target_set = set(target_list)
    print(f"Target participants: {len(target_list)}")

    if not args.skip_stage1:
        import shutil
        if STAGING_PARTS.exists():
            shutil.rmtree(STAGING_PARTS)
        if STAGING_DIR.exists():
            for p in STAGING_DIR.glob("*.parquet"):
                p.unlink()
        stage_paxmin_chunked(
            PAXMIN_PATH, target_set, STAGING_DIR, STAGING_PARTS, chunksize=args.chunksize
        )
        shutil.rmtree(STAGING_PARTS, ignore_errors=True)
    else:
        print(f"Skipping staging; using {STAGING_DIR}")

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    staging_str = str(STAGING_DIR.resolve())

    results: list[dict] = []
    failed: list[str] = []

    print(f"Parallel workers: {n_workers}")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(process_participant, int(s), staging_str): s for s in target_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Participants", unit="id"):
            rec = fut.result()
            results.append(rec)
            if rec.get("status") == "error":
                failed.append(f"SEQN={rec.get('SEQN')}: {rec.get('error', '')}")

    # Failed log
    with open(FAILED_LOG, "w") as f:
        for line in failed:
            f.write(line + "\n")
        if not failed:
            f.write("(no errors)\n")
    print(f"Wrote {FAILED_LOG} ({len(failed)} error lines)")

    ok_rows = [r for r in results if r.get("status") == "ok"]
    df_out = pd.DataFrame(ok_rows)
    if not df_out.empty:
        df_out = df_out.drop(columns=["status"], errors="ignore")
    df_out.to_parquet(RESULT_PARQUET, index=False, compression="snappy")
    print(f"Saved {RESULT_PARQUET} (n_ok={len(ok_rows)}, n_total_results={len(results)})")
    if len(ok_rows) > 0:
        print(f"  Delta_U median: {df_out['Delta_U'].median():.4f}")
        print(f"  Transition_Entropy median: {df_out['Transition_Entropy'].median():.4f}")


if __name__ == "__main__":
    main()
