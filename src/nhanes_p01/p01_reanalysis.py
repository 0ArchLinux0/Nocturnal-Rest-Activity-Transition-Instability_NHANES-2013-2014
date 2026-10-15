from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .p01 import (
    P01_ALGORITHM_VERSION,
    P01_METHOD_FINGERPRINT,
    P01_RESOLUTIONS_MIN,
    compute_p01_comparison,
    sha256_file,
    validate_p01_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTICIPANT_COLUMNS = [
    "SEQN",
    "resolution_min",
    "p01_old",
    "p01_corrected",
    "threshold_old",
    "threshold_corrected",
    "old_nocturnal_blocks",
    "corrected_complete_blocks",
    "corrected_valid_nights",
    "corrected_valid_edges",
    "corrected_rest_edges",
    "corrected_0_to_1_edges",
    "excluded_qc_minutes",
    "excluded_qc_missing_minutes",
    "excluded_uncomputed_minutes",
    "excluded_incomplete_minutes",
    "excluded_missing_timestamp_minutes",
    "sample_counter_non_4800_steps",
    "incomplete_blocks",
    "gap_breaks",
    "false_daylight_edges_removed",
    "legacy_200_minute_gate",
    "status",
]
SUMMARY_COLUMNS = [
    "resolution_min",
    "N_old",
    "N_corrected",
    "N_paired",
    "N_old_missing",
    "N_corrected_missing",
    "old_mean",
    "old_median",
    "corrected_mean",
    "corrected_median",
    "mean_delta",
    "median_delta",
    "mean_abs_delta",
    "pearson_r",
    "spearman_r",
]


def cohort_seqns(path: Path) -> set[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical cohort CSV not found: {path}")
    cohort = pd.read_csv(path, usecols=["SEQN"])
    values = pd.to_numeric(cohort["SEQN"], errors="coerce").dropna()
    seqns = {int(value) for value in values}
    if not seqns:
        raise ValueError(f"Canonical cohort has no SEQN values: {path}")
    return seqns


def cohort_fingerprint(seqns: set[int]) -> str:
    digest = hashlib.sha256()
    for seqn in sorted(seqns):
        digest.update(f"{seqn}\n".encode("ascii"))
    return digest.hexdigest()


def load_paxftime(path: Path, target_seqns: set[int]) -> dict[int, Any]:
    import pyreadstat

    if not path.is_file():
        raise FileNotFoundError(f"PAXHD not found: {path}")
    header, _ = pyreadstat.read_xport(str(path), usecols=["SEQN", "PAXFTIME"])
    header["SEQN"] = pd.to_numeric(header["SEQN"], errors="coerce")
    return {
        int(seqn): paxftime
        for seqn, paxftime in header[["SEQN", "PAXFTIME"]].itertuples(index=False, name=None)
        if pd.notna(seqn) and int(seqn) in target_seqns
    }


def summary_by_resolution(participants: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for resolution in P01_RESOLUTIONS_MIN:
        subset = participants.loc[participants["resolution_min"] == resolution]
        old = pd.to_numeric(subset["p01_old"], errors="coerce").to_numpy(dtype=float)
        corrected = pd.to_numeric(subset["p01_corrected"], errors="coerce").to_numpy(dtype=float)
        old_finite = np.isfinite(old)
        corrected_finite = np.isfinite(corrected)
        paired = old_finite & corrected_finite
        deltas = corrected[paired] - old[paired]
        paired_frame = pd.DataFrame({"old": old[paired], "corrected": corrected[paired]})
        pearson_r = paired_frame.corr(method="pearson").iloc[0, 1] if paired.sum() >= 2 else np.nan
        spearman_r = paired_frame.corr(method="spearman").iloc[0, 1] if paired.sum() >= 2 else np.nan
        summaries.append(
            {
                "resolution_min": resolution,
                "N_old": int(old_finite.sum()),
                "N_corrected": int(corrected_finite.sum()),
                "N_paired": int(paired.sum()),
                "N_old_missing": int((~old_finite).sum()),
                "N_corrected_missing": int((~corrected_finite).sum()),
                "old_mean": float(np.mean(old[old_finite])) if old_finite.any() else np.nan,
                "old_median": float(np.median(old[old_finite])) if old_finite.any() else np.nan,
                "corrected_mean": float(np.mean(corrected[corrected_finite])) if corrected_finite.any() else np.nan,
                "corrected_median": float(np.median(corrected[corrected_finite])) if corrected_finite.any() else np.nan,
                "mean_delta": float(np.mean(deltas)) if len(deltas) else np.nan,
                "median_delta": float(np.median(deltas)) if len(deltas) else np.nan,
                "mean_abs_delta": float(np.mean(np.abs(deltas))) if len(deltas) else np.nan,
                "pearson_r": float(pearson_r) if pd.notna(pearson_r) else np.nan,
                "spearman_r": float(spearman_r) if pd.notna(spearman_r) else np.nan,
            }
        )
    return pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)


def _checkpoint_payload(
    source_sha256: str,
    cohort_sha256: str,
    resume_row_offset: int,
) -> dict[str, Any]:
    return {
        "paxmin_sha256": source_sha256,
        "cohort_sha256": cohort_sha256,
        "p01_algorithm_version": P01_ALGORITHM_VERSION,
        "method_fingerprint": P01_METHOD_FINGERPRINT,
        "resume_row_offset": int(resume_row_offset),
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finite_sum(frame: pd.DataFrame, column: str) -> int:
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int(values.sum())


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_reanalysis(
    paxmin_path: Path,
    paxhd_path: Path,
    cohort_path: Path,
    audit_dir: Path,
    cache_path: Path,
    chunksize: int = 400_000,
    resume: bool = False,
) -> dict[str, Any]:
    import pyreadstat

    if not paxmin_path.is_file():
        raise FileNotFoundError(f"PAXMIN not found: {paxmin_path}")
    target = cohort_seqns(cohort_path)
    source_sha256 = sha256_file(paxmin_path)
    cohort_sha256 = cohort_fingerprint(target)
    paxftime = load_paxftime(paxhd_path, target)
    audit_dir.mkdir(parents=True, exist_ok=True)

    partial_path = audit_dir / "p01_old_vs_corrected_participant.partial.csv"
    checkpoint_path = audit_dir / "p01_reanalysis_checkpoint.json"
    if resume:
        if not partial_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("--resume requires both the participant partial CSV and checkpoint JSON")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = _checkpoint_payload(source_sha256, cohort_sha256, checkpoint.get("resume_row_offset", -1))
        for key in ("paxmin_sha256", "cohort_sha256", "p01_algorithm_version", "method_fingerprint"):
            if checkpoint.get(key) != expected[key]:
                raise ValueError(f"Checkpoint {key} does not match current source, cohort, or method")
        start_offset = int(checkpoint["resume_row_offset"])
        prior = pd.read_csv(partial_path)
        if not set(PARTICIPANT_COLUMNS).issubset(prior.columns):
            raise ValueError("Checkpoint participant CSV has an incompatible schema")
        if prior.duplicated(["SEQN", "resolution_min"]).any():
            raise ValueError("Checkpoint participant CSV has duplicate participant/resolution rows")
        if not prior.empty:
            resolution_counts = prior.groupby("SEQN")["resolution_min"].nunique()
            incomplete_seqns = set(
                resolution_counts[
                    resolution_counts != len(P01_RESOLUTIONS_MIN)
                ].index.astype(int)
            )
            if incomplete_seqns:
                prior = prior.loc[~prior["SEQN"].isin(incomplete_seqns)].copy()
                prior.to_csv(partial_path, index=False)
        processed = set(pd.to_numeric(prior["SEQN"], errors="raise").astype(int))
        if not processed.issubset(target):
            raise ValueError("Checkpoint contains participants outside the current cohort")
    else:
        if partial_path.exists() or checkpoint_path.exists():
            raise FileExistsError(
                f"Incomplete P01 checkpoint exists under {audit_dir}; use --resume or move it before starting over"
            )
        start_offset = 0
        processed: set[int] = set()
        pd.DataFrame(columns=PARTICIPANT_COLUMNS).to_csv(partial_path, index=False)
        _write_checkpoint(
            checkpoint_path,
            _checkpoint_payload(source_sha256, cohort_sha256, start_offset),
        )

    def append_participant(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        pd.DataFrame(rows, columns=PARTICIPANT_COLUMNS).to_csv(
            partial_path,
            mode="a",
            header=not partial_path.exists(),
            index=False,
        )

    rows_seen = start_offset
    last_source_seqn: float | None = None
    current_sid: int | None = None
    current_parts: list[pd.DataFrame] = []

    def finalize(seqn: int, parts: list[pd.DataFrame]) -> None:
        if seqn in processed:
            return
        if seqn not in paxftime:
            raise RuntimeError(f"PAXFTIME missing from PAXHD for cohort participant {seqn}")
        participant = pd.concat(parts, ignore_index=True)
        records = compute_p01_comparison(participant, paxftime[seqn], 1, 9)
        if len(records) != len(P01_RESOLUTIONS_MIN):
            raise RuntimeError(f"P01 engine returned incomplete resolutions for SEQN {seqn}")
        append_participant(records)
        processed.add(seqn)

    chunks = pyreadstat.read_file_in_chunks(
        pyreadstat.read_xport,
        str(paxmin_path),
        chunksize=chunksize,
        offset=start_offset,
        usecols=["SEQN", "PAXDAYM", "PAXSSNMP", "PAXTSM", "PAXQFM", "PAXMTSM"],
    )
    for chunk, _ in chunks:
        chunk_start = rows_seen
        source_seqn = pd.to_numeric(chunk["SEQN"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(source_seqn).all():
            raise RuntimeError("PAXMIN contains missing/non-numeric SEQN values")
        if np.any(np.diff(source_seqn) < 0) or (
            last_source_seqn is not None and source_seqn[0] < last_source_seqn
        ):
            raise RuntimeError("PAXMIN SEQN order is not monotonic; participant finalization is unsafe")
        last_source_seqn = float(source_seqn[-1])
        chunk["SEQN"] = source_seqn.astype(np.int64)
        chunk["_source_row"] = np.arange(chunk_start, chunk_start + len(chunk), dtype=np.int64)

        for seqn_value, group in chunk.groupby("SEQN", sort=False):
            seqn = int(seqn_value)
            group_start = int(group["_source_row"].iloc[0])
            participant_rows = group.drop(columns="_source_row")
            if current_sid is not None and seqn != current_sid:
                finalize(current_sid, current_parts)
                _write_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(source_sha256, cohort_sha256, group_start),
                )
                current_sid = None
                current_parts = []
            if seqn in target:
                if current_sid is None:
                    current_sid = seqn
                    current_parts = [participant_rows]
                elif seqn == current_sid:
                    current_parts.append(participant_rows)
                else:
                    raise RuntimeError(f"PAXMIN participant order regressed from {current_sid} to {seqn}")

        rows_seen += len(chunk)
        if current_sid is None:
            _write_checkpoint(
                checkpoint_path,
                _checkpoint_payload(source_sha256, cohort_sha256, rows_seen),
            )

    if current_sid is not None:
        finalize(current_sid, current_parts)
    _write_checkpoint(
        checkpoint_path,
        _checkpoint_payload(source_sha256, cohort_sha256, rows_seen),
    )
    if processed != target:
        missing = sorted(target - processed)
        raise RuntimeError(
            f"PAXMIN processing did not cover {len(missing)} cohort participants; examples: {missing[:10]}"
        )

    participants = pd.read_csv(partial_path)
    participants["SEQN"] = pd.to_numeric(participants["SEQN"], errors="raise").astype(int)
    participants["resolution_min"] = pd.to_numeric(
        participants["resolution_min"], errors="raise"
    ).astype(int)
    participants = participants.sort_values(["SEQN", "resolution_min"]).reset_index(drop=True)
    if participants.duplicated(["SEQN", "resolution_min"]).any():
        raise RuntimeError("Participant comparison contains duplicate participant/resolution rows")
    if len(participants) != len(target) * len(P01_RESOLUTIONS_MIN):
        raise RuntimeError("Participant comparison row count does not match cohort × resolution count")

    participant_path = audit_dir / "p01_old_vs_corrected_participant.csv"
    summary_path = audit_dir / "p01_old_vs_corrected_summary.csv"
    manifest_path = audit_dir / "p01_correction_run_manifest.json"
    participants.to_csv(participant_path, index=False)
    summary = summary_by_resolution(participants)
    summary.to_csv(summary_path, index=False)

    corrected = participants.pivot(
        index="SEQN", columns="resolution_min", values="p01_corrected"
    ).reindex(sorted(target))
    cache = pd.DataFrame({"SEQN": sorted(target)})
    for resolution in P01_RESOLUTIONS_MIN:
        cache[f"P01_{resolution}m"] = corrected[resolution].to_numpy()
    cache["p01_algorithm_version"] = P01_ALGORITHM_VERSION
    cache["paxmin_sha256"] = source_sha256
    cache["method_fingerprint"] = P01_METHOD_FINGERPRINT
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(cache_path, index=False)
    cache_valid, cache_reason = validate_p01_cache(cache_path, target, source_sha256)
    if not cache_valid:
        raise RuntimeError(f"Rebuilt P01 cache failed provenance validation: {cache_reason}")

    rows_10m = participants.loc[participants["resolution_min"] == 10]
    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_day_window_paxdaym": [1, 9],
        "participants_processed": len(target),
        "participant_resolution_rows": len(participants),
        "paxmin_path": str(paxmin_path),
        "paxmin_sha256": source_sha256,
        "paxhd_path": str(paxhd_path),
        "cohort_path": str(cohort_path),
        "cohort_sha256": cohort_sha256,
        "p01_algorithm_version": P01_ALGORITHM_VERSION,
        "method_fingerprint": P01_METHOD_FINGERPRINT,
        "time_axis": {
            "anchor": "PAXFTIME",
            "sample_counter": "PAXSSNMP",
            "sample_rate_hz": 80,
            "formula": "local_timestamp_ticks = PAXFTIME_seconds*80 + (PAXSSNMP - first_participant_sample)",
        },
        "stream_validation": {
            "seqn_monotonic": True,
            "paxssnmp_non_decreasing_within_target_participants": True,
            "paxdaym_matches_derived_day_for_timed_rows": True,
        },
        "cache_path": str(cache_path),
        "cache_columns": list(cache.columns),
        "cache_validation": cache_reason,
        "outputs": {
            "participant_comparison": str(participant_path),
            "resolution_summary": str(summary_path),
            "run_manifest": str(manifest_path),
        },
        "descriptive_impact_counts": {
            "false_daylight_edges_removed_across_resolutions": _finite_sum(
                participants, "false_daylight_edges_removed"
            ),
            "gap_edges_removed_across_resolutions": _finite_sum(participants, "gap_breaks"),
            "incomplete_blocks_removed_across_resolutions": _finite_sum(
                participants, "incomplete_blocks"
            ),
            "qc_invalid_minutes_removed_unique_participant_rows": _finite_sum(
                rows_10m, "excluded_qc_minutes"
            ),
            "qc_missing_minutes_unique_participant_rows": _finite_sum(
                rows_10m, "excluded_qc_missing_minutes"
            ),
            "uncomputed_minutes_removed_unique_participant_rows": _finite_sum(
                rows_10m, "excluded_uncomputed_minutes"
            ),
            "incomplete_minutes_removed_unique_participant_rows": _finite_sum(
                rows_10m, "excluded_incomplete_minutes"
            ),
            "missing_timestamp_minutes_unique_participant_rows": _finite_sum(
                rows_10m, "excluded_missing_timestamp_minutes"
            ),
            "sample_counter_non_4800_steps_unique_participant_rows": _finite_sum(
                rows_10m, "sample_counter_non_4800_steps"
            ),
        },
        "resolution_summary": [
            {key: _json_value(value) for key, value in record.items()}
            for record in summary.to_dict(orient="records")
        ],
        "downstream_analyses_run": {"ols": False, "logistic": False, "roc": False},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial_path.unlink()
    checkpoint_path.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded KAN-172 P01 comparison.")
    parser.add_argument("--paxmin", type=Path, default=PROJECT_ROOT / "data" / "raw" / "PAXMIN_H.xpt")
    parser.add_argument("--paxhd", type=Path, default=PROJECT_ROOT / "data" / "raw" / "PAXHD_H.xpt")
    parser.add_argument("--cohort", type=Path, default=PROJECT_ROOT / "final_full_cohort.csv")
    parser.add_argument("--audit-dir", type=Path, default=PROJECT_ROOT / "outputs" / "audit")
    parser.add_argument(
        "--cache",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_cohort_pam_metrics.csv",
    )
    parser.add_argument("--chunksize", type=int, default=400_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = run_reanalysis(
        args.paxmin,
        args.paxhd,
        args.cohort,
        args.audit_dir,
        args.cache,
        chunksize=args.chunksize,
        resume=args.resume,
    )
    print(
        f"P01 reanalysis complete: {manifest['participants_processed']} participants; "
        f"cache={manifest['cache_path']}"
    )


if __name__ == "__main__":
    main()
