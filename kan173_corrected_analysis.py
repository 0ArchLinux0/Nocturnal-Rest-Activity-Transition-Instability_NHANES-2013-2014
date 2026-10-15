#!/usr/bin/env python3
"""Run the corrected, auditable KAN-173 NHANES outcome analysis."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_final_full_cohort import PHQ9_ITEMS
from nhanes_p01.p01 import (
    P01_ALGORITHM_VERSION,
    P01_METHOD_FINGERPRINT,
    P01_RESOLUTIONS_MIN,
    compute_p01_corrected_resolutions,
    normalize_paxmin_timestamps,
    sha256_file,
    validate_p01_cache,
)

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_COHORT = PROJECT_ROOT / "final_full_cohort.csv"
DEFAULT_CACHE = PROJECT_ROOT / "outputs" / "final_cohort_pam_metrics.csv"
DEFAULT_KAN172_MANIFEST = PROJECT_ROOT / "outputs" / "audit" / "p01_correction_run_manifest.json"
PAXMIN_COLUMNS = ["SEQN", "PAXDAYM", "PAXSSNMP", "PAXTSM", "PAXQFM", "PAXMTSM", "PAXPREDM"]
P01_COLUMNS = [f"P01_{resolution}m" for resolution in P01_RESOLUTIONS_MIN]
HISTORICAL_REFERENCES = {
    "Model 1": {"beta": 10.707, "se": 1.277, "p": "<1e-16", "p_significant": True},
    "Model 2": {"beta": 9.105, "se": 1.350, "p": 1.76e-11, "p_significant": True},
}


def _numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.map(
            lambda value: value.decode().strip()
            if isinstance(value, bytes)
            else value.strip()
            if isinstance(value, str)
            else value
        )
    return pd.to_numeric(series, errors="coerce")


def _with_integer_seqn(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    out["SEQN"] = pd.to_numeric(out["SEQN"], errors="coerce")
    out = out.dropna(subset=["SEQN"])
    out["SEQN"] = out["SEQN"].astype("int64")
    if out["SEQN"].duplicated().any():
        raise ValueError(f"{source} contains duplicate SEQN values")
    return out


def score_phq9(dpq: pd.DataFrame) -> pd.DataFrame:
    """Score only the nine canonical DPQ items; require all nine responses."""
    missing = [item for item in PHQ9_ITEMS if item not in dpq.columns]
    if missing:
        raise ValueError(f"DPQ is missing canonical PHQ-9 items: {missing}")
    frame = dpq[["SEQN", *PHQ9_ITEMS]].copy()
    frame["SEQN"] = pd.to_numeric(frame["SEQN"], errors="coerce")
    frame = frame.dropna(subset=["SEQN"])
    frame["SEQN"] = frame["SEQN"].astype("int64")
    for item in PHQ9_ITEMS:
        values = _numeric(frame[item])
        values = values.mask(values.isin([7, 9]))
        values = values.mask(values.notna() & (values.abs() < 1e-10), 0.0)
        frame[item] = values.where(values.isin([0, 1, 2, 3]))
    frame["PHQ9_Score"] = frame[PHQ9_ITEMS].sum(axis=1, min_count=9).clip(0, 27)
    result = frame[["SEQN", "PHQ9_Score"]].copy()
    result["PHQ9_ge_10"] = np.where(
        result["PHQ9_Score"].notna(), (result["PHQ9_Score"] >= 10).astype(float), np.nan
    )
    return result


def _load_xport(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required NHANES source missing: {path}")
    frame, _ = pyreadstat.read_xport(str(path), usecols=columns)
    return _with_integer_seqn(frame, str(path))


def waso_proxy_for_participant(normalized: pd.DataFrame) -> dict[str, float | int]:
    """Average wake-coded minutes between first/last sleep-coded minute per full night.

    Nights are assigned from the timestamp axis: 22:00–24:00 on day d and
    00:00–06:00 on day d+1 share night_id d. Only full nights d=2..7 fit wholly
    within full protocol days 2..8. Prediction codes other than 1 (wake) and 2
    (sleep), including non-wear/unknown codes, are excluded rather than treated
    as either state. Minute QC, duration, and computed-activity checks also apply.
    """
    frame = normalized
    seconds = pd.to_numeric(frame["seconds_since_midnight"], errors="coerce").to_numpy(float)
    day = pd.to_numeric(frame["PAXDAYM_INT"], errors="coerce").to_numpy(float)
    pred = _numeric(frame["PAXPREDM"]).to_numpy(dtype=float)
    q = pd.to_numeric(frame["PAXQFM_NUM"], errors="coerce").to_numpy(float)
    activity = pd.to_numeric(frame["PAXMTSM_NUM"], errors="coerce").to_numpy(float)
    duration = pd.to_numeric(frame["PAXTSM_NUM"], errors="coerce").to_numpy(float)
    ticks = pd.to_numeric(frame["local_timestamp_ticks"], errors="coerce").to_numpy(float)
    evening = seconds >= 22 * 3600
    morning = seconds < 6 * 3600
    nocturnal = evening | morning
    night_id = np.where(evening, day, day - 1)
    minute_ticks = 80 * 60
    qc_valid = (
        (q == 0)
        & np.isfinite(activity)
        & (activity >= 0)
        & (duration == 60)
        & np.isfinite(ticks)
        & (ticks % minute_ticks == 0)
    )
    known_prediction = np.isin(pred, [1, 2])
    select = (
        nocturnal
        & (night_id >= 2)
        & (night_id <= 7)
        & qc_valid
        & known_prediction
    )
    excluded_predictions = int(np.count_nonzero(nocturnal & ~known_prediction))
    work = pd.DataFrame(
        {
            "night_id": night_id[select].astype(int),
            "timestamp_ticks": ticks[select],
            "prediction": pred[select].astype(int),
        }
    )
    wake_counts: list[int] = []
    for _, night in work.groupby("night_id", sort=True):
        night = night.sort_values("timestamp_ticks")
        if len(night) < 30:
            continue
        sleep_ticks = night.loc[night["prediction"] == 2, "timestamp_ticks"]
        if sleep_ticks.empty:
            continue
        first_sleep = float(sleep_ticks.iloc[0])
        last_sleep = float(sleep_ticks.iloc[-1])
        sleep_interval = night.loc[night["timestamp_ticks"].between(first_sleep, last_sleep)]
        wake_counts.append(int((sleep_interval["prediction"] == 1).sum()))
    return {
        "WASO_proxy_min": float(np.mean(wake_counts)) if wake_counts else np.nan,
        "WASO_valid_nights": len(wake_counts),
        "WASO_nonwake_nights_excluded_prediction_minutes": excluded_predictions,
    }


def _read_paxftime(paxhd_path: Path, target: set[int]) -> tuple[dict[int, Any], pd.DataFrame]:
    headers, _ = pyreadstat.read_xport(str(paxhd_path), usecols=["SEQN", "PAXFTIME", "PAXSTS"])
    headers["SEQN"] = pd.to_numeric(headers["SEQN"], errors="coerce")
    headers = headers.dropna(subset=["SEQN"])
    headers["SEQN"] = headers["SEQN"].astype(int)
    if headers["SEQN"].duplicated().any():
        raise ValueError("PAXHD contains duplicate SEQN values")
    times = {
        int(seqn): paxftime
        for seqn, paxftime in headers[["SEQN", "PAXFTIME"]].itertuples(index=False, name=None)
        if int(seqn) in target
    }
    missing = target - set(times)
    if missing:
        raise ValueError(f"PAXHD lacks PAXFTIME for {len(missing)} cohort participants")
    return times, headers[["SEQN", "PAXSTS"]].copy()


def _extend_corrected_cache_and_stream_waso(
    target: set[int],
    paxmin_path: Path,
    paxhd_path: Path,
    cache_path: Path,
    kan172_manifest_path: Path,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str, int]:
    """Validate the KAN-172 cache, fill only newly eligible SEQNs, and stream WASO."""
    if not cache_path.is_file():
        raise FileNotFoundError(f"Corrected KAN-172 P01 cache is missing: {cache_path}")
    if not kan172_manifest_path.is_file():
        raise FileNotFoundError(f"KAN-172 provenance manifest is missing: {kan172_manifest_path}")
    source_sha256 = sha256_file(paxmin_path)
    old_cache = pd.read_csv(cache_path)
    if "SEQN" not in old_cache.columns or old_cache["SEQN"].isna().any():
        raise ValueError("Corrected P01 cache has missing SEQN values")
    old_cache["SEQN"] = pd.to_numeric(old_cache["SEQN"], errors="raise")
    if np.any(old_cache["SEQN"].to_numpy(float) != np.floor(old_cache["SEQN"].to_numpy(float))):
        raise ValueError("Corrected P01 cache has non-integer SEQN values")
    old_cache["SEQN"] = old_cache["SEQN"].astype(int)
    if old_cache["SEQN"].duplicated().any():
        raise ValueError("Corrected P01 cache has duplicate SEQN values")
    missing_metrics = [column for column in P01_COLUMNS if column not in old_cache.columns]
    if missing_metrics:
        raise ValueError(f"Corrected KAN-172 cache lacks resolution metrics: {missing_metrics}")
    old_ids = set(old_cache["SEQN"])
    valid, reason = validate_p01_cache(cache_path, old_ids, source_sha256)
    if not valid:
        raise ValueError(f"Rejected corrected P01 cache: {reason}")
    manifest = json.loads(kan172_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("paxmin_sha256") != source_sha256
        or manifest.get("p01_algorithm_version") != P01_ALGORITHM_VERSION
        or manifest.get("method_fingerprint") != P01_METHOD_FINGERPRINT
        or manifest.get("cache_validation") != "matched"
    ):
        raise ValueError("KAN-172 manifest does not match the corrected cache/source provenance")
    historical_cohort_n = int(manifest["participants_processed"])
    if historical_cohort_n > len(old_ids):
        raise ValueError("KAN-172 original participant count exceeds the corrected cache coverage")

    paxftime, paxhd = _read_paxftime(paxhd_path, target)
    missing_ids = target - old_ids
    print(
        f"Validated KAN-172 cache ({len(old_ids)} participants); "
        f"streaming {len(target)} target participants for WASO and {len(missing_ids)} missing P01 rows.",
        flush=True,
    )
    waso_rows: list[dict[str, Any]] = []
    new_p01_rows: list[dict[str, Any]] = []
    processed: set[int] = set()
    current_seqn: int | None = None
    current_parts: list[pd.DataFrame] = []
    last_source_seqn: float | None = None
    chunk_number = 0

    def finalize(seqn: int, pieces: list[pd.DataFrame]) -> None:
        if seqn not in target:
            return
        participant = pd.concat(pieces, ignore_index=True)
        normalized = normalize_paxmin_timestamps(participant, paxftime[seqn])
        waso = waso_proxy_for_participant(normalized)
        waso_rows.append({"SEQN": seqn, **waso})
        if seqn in missing_ids:
            metrics = compute_p01_corrected_resolutions(participant, paxftime[seqn])
            row: dict[str, Any] = {"SEQN": seqn}
            row.update({f"P01_{resolution}m": metrics[resolution]["p01"] for resolution in P01_RESOLUTIONS_MIN})
            row.update(
                {
                    "p01_algorithm_version": P01_ALGORITHM_VERSION,
                    "paxmin_sha256": source_sha256,
                    "method_fingerprint": P01_METHOD_FINGERPRINT,
                }
            )
            new_p01_rows.append(row)
        processed.add(seqn)

    chunks = pyreadstat.read_file_in_chunks(
        pyreadstat.read_xport,
        str(paxmin_path),
        chunksize=chunksize,
        usecols=PAXMIN_COLUMNS,
    )
    for chunk, _ in chunks:
        chunk_number += 1
        source_seqn = pd.to_numeric(chunk["SEQN"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(source_seqn).all():
            raise ValueError("PAXMIN contains missing or nonnumeric SEQN values")
        if np.any(np.diff(source_seqn) < 0) or (
            last_source_seqn is not None and source_seqn[0] < last_source_seqn
        ):
            raise ValueError("PAXMIN SEQN order is not monotonic")
        last_source_seqn = float(source_seqn[-1])
        chunk["SEQN"] = source_seqn.astype(np.int64)
        for seqn_value, group in chunk.groupby("SEQN", sort=False):
            seqn = int(seqn_value)
            if current_seqn is not None and seqn != current_seqn:
                finalize(current_seqn, current_parts)
                current_seqn = None
                current_parts = []
            if seqn in target:
                if current_seqn is None:
                    current_seqn = seqn
                    current_parts = [group]
                elif seqn == current_seqn:
                    current_parts.append(group)
                else:
                    raise ValueError(f"PAXMIN participant order regressed at SEQN {seqn}")
        if chunk_number % 10 == 0:
            print(f"PAXMIN streaming chunks processed: {chunk_number}", flush=True)
    if current_seqn is not None:
        finalize(current_seqn, current_parts)
    if processed != target:
        absent = sorted(target - processed)
        raise ValueError(f"PAXMIN did not contain {len(absent)} target SEQNs; examples={absent[:10]}")
    if set(pd.DataFrame(waso_rows)["SEQN"].astype(int)) != target:
        raise ValueError("WASO stream did not cover every target participant")

    extension = pd.DataFrame(new_p01_rows)
    combined = pd.concat(
        [old_cache.loc[old_cache["SEQN"].isin(target)].copy(), extension],
        ignore_index=True,
    )
    combined["SEQN"] = pd.to_numeric(combined["SEQN"], errors="raise").astype(int)
    if combined["SEQN"].duplicated().any() or set(combined["SEQN"]) != target:
        raise ValueError("Extended corrected P01 cache has duplicate or missing cohort SEQNs")
    combined = combined.sort_values("SEQN").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(cache_path)
    valid, reason = validate_p01_cache(cache_path, target, source_sha256)
    if not valid:
        raise ValueError(f"Extended corrected P01 cache failed provenance validation: {reason}")
    waso_frame = pd.DataFrame(waso_rows).sort_values("SEQN").reset_index(drop=True)
    return combined, waso_frame, source_sha256, historical_cohort_n


def _read_analysis_sources(raw_dir: Path, cohort_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(cohort_path)
    if "SEQN" not in cohort.columns:
        raise ValueError("Corrected full cohort has no SEQN column")
    cohort["SEQN"] = pd.to_numeric(cohort["SEQN"], errors="raise").astype(int)
    if cohort["SEQN"].duplicated().any():
        raise ValueError("Corrected full cohort contains duplicate SEQN values")

    demo = _load_xport(
        raw_dir / "DEMO_H.xpt",
        ["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"],
    )
    dpq_raw, _ = pyreadstat.read_xport(str(raw_dir / "DPQ_H.xpt"))
    dpq = score_phq9(dpq_raw)
    if dpq["SEQN"].duplicated().any():
        raise ValueError("DPQ_H.xpt contains duplicate SEQN values")
    bmx = _load_xport(raw_dir / "BMX_H.xpt", ["SEQN", "BMXBMI"])
    slq = _load_xport(raw_dir / "SLQ_H.xpt", ["SEQN", "SLD010H", "SLQ060"])
    slq["self_reported_weekday_sleep_hours"] = _numeric(slq["SLD010H"])
    slq.loc[slq["self_reported_weekday_sleep_hours"].isin([77, 99]), "self_reported_weekday_sleep_hours"] = np.nan
    sleep_history = _numeric(slq["SLQ060"])
    slq["sleep_disorder_history"] = np.select(
        [sleep_history.eq(1), sleep_history.eq(2)], [1.0, 0.0], default=np.nan
    )

    data = cohort[["SEQN"]].merge(demo, on="SEQN", how="left")
    data = data.merge(dpq, on="SEQN", how="left")
    data = data.merge(bmx, on="SEQN", how="left")
    data = data.merge(
        slq[["SEQN", "self_reported_weekday_sleep_hours", "sleep_disorder_history"]],
        on="SEQN",
        how="left",
    )
    data["Age"] = _numeric(data["RIDAGEYR"])
    data["Sex"] = _numeric(data["RIAGENDR"])
    data.loc[~data["Sex"].isin([1, 2]), "Sex"] = np.nan
    data["BMI"] = _numeric(data["BMXBMI"])
    data["PIR"] = _numeric(data["INDFMPIR"])
    data["WTMEC2YR"] = _numeric(data["WTMEC2YR"])
    data["SDMVSTRA"] = _numeric(data["SDMVSTRA"])
    data["SDMVPSU"] = _numeric(data["SDMVPSU"])
    data["PHQ9_Score"] = _numeric(data["PHQ9_Score"]).clip(0, 27)
    data["PHQ9_ge_10"] = np.where(
        data["PHQ9_Score"].notna(), (data["PHQ9_Score"] >= 10).astype(float), np.nan
    )
    return cohort, data, (demo, dpq, bmx, slq)


def _zscore_complete_cases(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        values = pd.to_numeric(out[column], errors="coerce")
        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        out[f"{column}_z"] = (values - mean) / sd if sd >= 1e-12 else 0.0
    return out


def _fit_primary(frame: pd.DataFrame, model_name: str) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if model_name == "Model 1":
        required = ["PHQ9_Score", "P01_10m", "Age", "Sex", "BMI"]
        continuous = ("Age", "BMI")
        formula = "PHQ9_Score ~ P01_10m + Age_z + C(Sex) + BMI_z"
    elif model_name == "Model 2":
        required = [
            "PHQ9_Score", "P01_10m", "Age", "Sex", "BMI", "PIR",
            "self_reported_weekday_sleep_hours", "WASO_proxy_min",
        ]
        continuous = (
            "Age", "BMI", "PIR", "self_reported_weekday_sleep_hours", "WASO_proxy_min"
        )
        formula = (
            "PHQ9_Score ~ P01_10m + Age_z + C(Sex) + BMI_z + PIR_z + "
            "self_reported_weekday_sleep_hours_z + WASO_proxy_min_z"
        )
    else:
        raise ValueError(f"Unknown primary model: {model_name}")
    complete = frame.dropna(subset=required).copy()
    numeric_required = [column for column in required if column != "Sex"]
    finite = np.isfinite(complete[numeric_required].to_numpy(dtype=float)).all(axis=1)
    complete = complete.loc[finite].copy()
    complete = _zscore_complete_cases(complete, continuous)
    if len(complete) <= 8:
        raise ValueError(f"Insufficient complete cases for {model_name}: {len(complete)}")
    fit = smf.ols(formula, data=complete).fit()
    robust = fit.get_robustcov_results(cov_type="HC3", use_t=True)
    p01_index = fit.model.exog_names.index("P01_10m")
    robust_ci = np.asarray(robust.conf_int())[p01_index]
    beta = float(fit.params["P01_10m"])
    ordinary = {
        "model": model_name,
        "formula": formula,
        "N": int(fit.nobs),
        "beta_P01": beta,
        "SE_P01": float(fit.bse["P01_10m"]),
        "CI95_low": float(fit.conf_int().loc["P01_10m", 0]),
        "CI95_high": float(fit.conf_int().loc["P01_10m", 1]),
        "p_P01": float(fit.pvalues["P01_10m"]),
        "R2": float(fit.rsquared),
        "adjusted_R2": float(fit.rsquared_adj),
        "P01_effect_per_0_01": beta * 0.01,
    }
    hc3 = {
        "model": model_name,
        "formula": formula,
        "N": int(fit.nobs),
        "beta_P01": beta,
        "SE_P01": float(np.asarray(robust.bse)[p01_index]),
        "CI95_low": float(robust_ci[0]),
        "CI95_high": float(robust_ci[1]),
        "p_P01": float(np.asarray(robust.pvalues)[p01_index]),
        "R2": float(fit.rsquared),
        "adjusted_R2": float(fit.rsquared_adj),
        "P01_effect_per_0_01": beta * 0.01,
    }
    return ordinary, hc3, complete


def _historical_comparison(ordinary: list[dict[str, Any]], hc3: list[dict[str, Any]]) -> pd.DataFrame:
    ordinary_by_model = {row["model"]: row for row in ordinary}
    hc3_by_model = {row["model"]: row for row in hc3}
    rows: list[dict[str, Any]] = []
    for model, historic in HISTORICAL_REFERENCES.items():
        current = ordinary_by_model.get(model)
        robust = hc3_by_model.get(model)
        if (
            current is None
            or robust is None
            or current.get("status") == "PENDING_WASO_PROXY"
            or robust.get("status") == "PENDING_WASO_PROXY"
        ):
            rows.append(
                {
                    "model": model,
                    "historical_beta": historic["beta"],
                    "corrected_beta": np.nan,
                    "historical_se": historic["se"],
                    "corrected_se": np.nan,
                    "historical_p": historic["p"],
                    "corrected_p": np.nan,
                    "historical_N": np.nan,
                    "corrected_N": np.nan,
                    "direction_preserved": np.nan,
                    "significance_preserved": np.nan,
                    "corrected_se_hc3": np.nan,
                    "corrected_p_hc3": np.nan,
                    "historical_N_note": "not reported in supplied historical reference",
                    "status": "PENDING_WASO_PROXY",
                }
            )
            continue
        rows.append(
            {
                "model": model,
                "historical_beta": historic["beta"],
                "corrected_beta": current["beta_P01"],
                "historical_se": historic["se"],
                "corrected_se": current["SE_P01"],
                "historical_p": historic["p"],
                "corrected_p": current["p_P01"],
                "historical_N": np.nan,
                "corrected_N": current["N"],
                "direction_preserved": bool(np.sign(historic["beta"]) == np.sign(current["beta_P01"])),
                "significance_preserved": bool(
                    historic["p_significant"] and robust["p_P01"] < 0.05
                ),
                "corrected_se_hc3": robust["SE_P01"],
                "corrected_p_hc3": robust["p_P01"],
                "historical_N_note": "not reported in supplied historical reference",
                "status": "complete",
            }
        )
    return pd.DataFrame(rows)


def _print_primary_checkpoint(ordinary: list[dict[str, Any]], hc3: list[dict[str, Any]]) -> None:
    robust = {
        row["model"]: row
        for row in hc3
        if row.get("status") != "PENDING_WASO_PROXY"
    }
    evaluated = [name for name in HISTORICAL_REFERENCES if name in robust]
    direction = bool(evaluated) and all(
        np.sign(HISTORICAL_REFERENCES[name]["beta"]) == np.sign(robust[name]["beta_P01"])
        for name in evaluated
    )
    significance = {
        name: bool(HISTORICAL_REFERENCES[name]["p_significant"] and robust[name]["p_P01"] < 0.05)
        for name in evaluated
    }
    print("CORRECTED_PRIMARY_CHECKPOINT", flush=True)
    for model, prefix in (("Model 1", "MODEL1"), ("Model 2", "MODEL2")):
        result = robust.get(model)
        if result is None:
            print(f"{prefix}_STATUS=PENDING_WASO_PROXY", flush=True)
            continue
        print(f"{prefix}_N={result['N']}", flush=True)
        print(f"{prefix}_P01_BETA={result['beta_P01']:.10g}", flush=True)
        print(f"{prefix}_P01_CI=({result['CI95_low']:.10g},{result['CI95_high']:.10g})", flush=True)
        print(f"{prefix}_P01_P={result['p_P01']:.10g}", flush=True)
    print(f"DIRECTION_PRESERVED={'YES' if direction else 'NO'}", flush=True)
    for model, prefix in (("Model 1", "MODEL1"), ("Model 2", "MODEL2")):
        value = "PENDING" if model not in significance else ("YES" if significance[model] else "NO")
        print(f"{prefix}_SIGNIFICANCE_PRESERVED={value}", flush=True)





def _fit_resolution_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for resolution in P01_RESOLUTIONS_MIN:
        p01_column = f"P01_{resolution}m"
        complete = data.dropna(subset=["PHQ9_Score", p01_column, "Age", "Sex", "BMI"]).copy()
        numeric = complete[["PHQ9_Score", p01_column, "Age", "BMI"]].to_numpy(dtype=float)
        complete = complete.loc[np.isfinite(numeric).all(axis=1)].copy()
        complete = _zscore_complete_cases(complete, ("Age", "BMI"))
        fit = smf.ols(
            f"PHQ9_Score ~ {p01_column} + Age_z + C(Sex) + BMI_z", data=complete
        ).fit()
        robust = fit.get_robustcov_results(cov_type="HC3", use_t=True)
        index = fit.model.exog_names.index(p01_column)
        ci = np.asarray(robust.conf_int())[index]
        values = pd.to_numeric(complete[p01_column], errors="coerce")
        rows.append(
            {
                "resolution_min": resolution,
                "N": int(fit.nobs),
                "beta": float(fit.params[p01_column]),
                "SE_HC3": float(np.asarray(robust.bse)[index]),
                "CI95_low": float(ci[0]),
                "CI95_high": float(ci[1]),
                "p": float(np.asarray(robust.pvalues)[index]),
                "P01_mean": float(values.mean()),
                "P01_SD": float(values.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _roc_preprocessor(numeric: list[str], include_sex: bool) -> ColumnTransformer:
    transforms: list[tuple[str, Any, list[str]]] = [
        ("continuous", StandardScaler(), numeric)
    ]
    if include_sex:
        transforms.append(("sex", OneHotEncoder(handle_unknown="ignore"), ["Sex"]))
    return ColumnTransformer(transforms, remainder="drop")


def _fit_oof_roc(data: pd.DataFrame, output_dir: Path, bootstrap_replicates: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["SEQN", "PHQ9_ge_10", "Age", "Sex", "BMI", "P01_10m"]
    frame = data.dropna(subset=required).copy().reset_index(drop=True)
    finite = np.isfinite(frame[["PHQ9_ge_10", "Age", "BMI", "P01_10m"]].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)
    y = frame["PHQ9_ge_10"].astype(int).to_numpy()
    if len(np.unique(y)) != 2 or int(np.bincount(y).min()) < 5:
        raise ValueError("ROC analysis requires both outcome classes with at least five participants each")
    specifications = {
        "demographics_baseline": ["Age", "BMI"],
        "P01_only": ["P01_10m"],
        "demographics_plus_P01": ["Age", "BMI", "P01_10m"],
    }
    oof = {name: np.full(len(frame), np.nan, dtype=float) for name in specifications}
    fold_ids = np.zeros(len(frame), dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=173)
    for fold, (train_index, test_index) in enumerate(splitter.split(frame, y), start=1):
        fold_ids[test_index] = fold
        for name, numeric in specifications.items():
            model = make_pipeline(
                _roc_preprocessor(numeric, "Age" in numeric or "BMI" in numeric),
                LogisticRegression(max_iter=2000, solver="lbfgs"),
            )
            model.fit(frame.iloc[train_index], y[train_index])
            oof[name][test_index] = model.predict_proba(frame.iloc[test_index])[:, 1]
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError("ROC out-of-fold predictions are incomplete")

    aucs = {name: float(roc_auc_score(y, probability)) for name, probability in oof.items()}
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(173)
    bootstrap_auc = {name: np.empty(bootstrap_replicates) for name in specifications}
    bootstrap_delta = {
        name: np.empty(bootstrap_replicates)
        for name in ("P01_only", "demographics_plus_P01")
    }
    for replicate in range(bootstrap_replicates):
        sampled = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        sampled_y = y[sampled]
        for name, probability in oof.items():
            bootstrap_auc[name][replicate] = roc_auc_score(sampled_y, probability[sampled])
        baseline_auc = bootstrap_auc["demographics_baseline"][replicate]
        for name in bootstrap_delta:
            bootstrap_delta[name][replicate] = bootstrap_auc[name][replicate] - baseline_auc
    rows: list[dict[str, Any]] = []
    for name in specifications:
        ci = np.quantile(bootstrap_auc[name], [0.025, 0.975])
        if name == "demographics_baseline":
            delta, delta_ci = 0.0, (0.0, 0.0)
        else:
            delta = aucs[name] - aucs["demographics_baseline"]
            delta_ci = np.quantile(bootstrap_delta[name], [0.025, 0.975])
        rows.append(
            {
                "model": name,
                "N": len(frame),
                "AUC_OOF": aucs[name],
                "AUC_CI95_low": float(ci[0]),
                "AUC_CI95_high": float(ci[1]),
                "delta_AUC_vs_baseline": float(delta),
                "delta_CI95_low": float(delta_ci[0]),
                "delta_CI95_high": float(delta_ci[1]),
                "bootstrap_replicates": bootstrap_replicates,
                "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=173)",
            }
        )
    summary = pd.DataFrame(rows)
    predictions = pd.DataFrame(
        {
            "SEQN": frame["SEQN"].astype(int),
            "PHQ9_ge_10": y,
            "fold": fold_ids,
            "baseline_oof_probability": oof["demographics_baseline"],
            "p01_only_oof_probability": oof["P01_only"],
            "combined_oof_probability": oof["demographics_plus_P01"],
        }
    )
    summary.to_csv(output_dir / "roc_corrected.csv", index=False)
    predictions.to_csv(output_dir / "roc_corrected_oof_predictions.csv", index=False)
    return summary, predictions


def _survey_available() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    probe = subprocess.run(
        [rscript, "--vanilla", "-e", 'quit(status=if (requireNamespace("survey", quietly=TRUE)) 0 else 1)'],
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _run_survey_models(
    model1: pd.DataFrame,
    model2: pd.DataFrame | None,
    output_dir: Path,
    r_script: Path,
) -> tuple[str, pd.DataFrame | None]:
    rscript = shutil.which("Rscript")
    if rscript is None or not _survey_available():
        return "DEFERRED_TOOLING", None
    with tempfile.TemporaryDirectory(prefix="kan173-survey-") as temporary:
        input_path = Path(temporary) / "survey_input.csv"
        output_path = Path(temporary) / "survey_results.csv"
        first = model1.copy()
        first["model"] = "Model 1"
        frames = [first]
        if model2 is not None:
            second = model2.copy()
            second["model"] = "Model 2"
            frames.append(second)
        pd.concat(frames, ignore_index=True, sort=False).to_csv(input_path, index=False)
        try:
            subprocess.run(
                [rscript, "--vanilla", str(r_script), str(input_path), str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            survey = pd.read_csv(output_path)
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() if exc.stderr else str(exc)
            print(f"SURVEY_MODEL_BLOCKED={detail}", flush=True)
            return "BLOCKED", None
        except (OSError, pd.errors.ParserError) as exc:
            print(f"SURVEY_MODEL_BLOCKED={exc}", flush=True)
            return "BLOCKED", None
    survey.to_csv(output_dir / "survey_weighted_models.csv", index=False)
    return "PASS", survey


def _cohort_flow(
    cohort: pd.DataFrame,
    data: pd.DataFrame,
    demo: pd.DataFrame,
    dpq: pd.DataFrame,
    paxhd: pd.DataFrame,
    model1: pd.DataFrame,
    model2: pd.DataFrame | None,
    roc: pd.DataFrame,
    historical_cohort_n: int,
) -> pd.DataFrame:
    demo_ids = set(demo["SEQN"].astype(int))
    dpq_ids = set(dpq["SEQN"].astype(int))
    complete_phq_ids = set(dpq.loc[dpq["PHQ9_Score"].notna(), "SEQN"].astype(int))
    monitor_status = _numeric(paxhd["PAXSTS"])
    monitor_ids = set(paxhd.loc[monitor_status == 1, "SEQN"].astype(int))
    records = [
        ("historical_cohort_N", historical_cohort_n, "KAN-172 complete cohort"),
        ("corrected_cohort_N", len(cohort), "PAXDAYD 2–8; >=4 adequate days, >=600 min/day, >=1 weekend day"),
        ("DEMO_intersect_DPQ", len(demo_ids & dpq_ids), "unique SEQN intersection"),
        ("monitor_participants", len(monitor_ids), "PAXHD PAXSTS==1"),
        ("PHQ9_complete", len(demo_ids & complete_phq_ids), "all nine DPQ items answered"),
        ("wear_qualified", len(cohort), "corrected final cohort"),
        ("corrected_P01_available", int(np.isfinite(_numeric(data["P01_10m"]).to_numpy(dtype=float)).sum()), "finite corrected KAN-172 P01_10m"),
        ("Model_1_complete_cases", len(model1), "PHQ9, corrected P01_10m, Age, Sex, BMI"),
        (
            "Model_2_complete_cases",
            len(model2) if model2 is not None else np.nan,
            "Model 1 plus PIR, weekday sleep hours, WASO_proxy_min"
            if model2 is not None
            else "pending: WASO_proxy_min unavailable",
        ),
        ("binary_ROC_complete_cases", int(roc["N"].iloc[0]), "PHQ9_ge_10, Age, Sex, BMI, corrected P01_10m"),
    ]
    return pd.DataFrame(records, columns=["metric", "count", "definition"])


def _load_corrected_cache_for_target(
    cache_path: Path,
    manifest_path: Path,
    paxmin_path: Path,
    target: set[int],
) -> tuple[pd.DataFrame, str, int, int]:
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Corrected KAN-172 cache or provenance manifest is missing")
    source_sha256 = sha256_file(paxmin_path)
    cache = pd.read_csv(cache_path)
    if "SEQN" not in cache.columns or cache["SEQN"].isna().any():
        raise ValueError("Corrected P01 cache has missing SEQN values")
    cache["SEQN"] = pd.to_numeric(cache["SEQN"], errors="raise")
    if np.any(cache["SEQN"].to_numpy(float) != np.floor(cache["SEQN"].to_numpy(float))):
        raise ValueError("Corrected P01 cache has non-integer SEQN values")
    cache["SEQN"] = cache["SEQN"].astype(int)
    if cache["SEQN"].duplicated().any():
        raise ValueError("Corrected P01 cache has duplicate SEQN values")
    missing_metrics = [column for column in P01_COLUMNS if column not in cache.columns]
    if missing_metrics:
        raise ValueError(f"Corrected KAN-172 cache lacks resolution metrics: {missing_metrics}")
    cache_ids = set(cache["SEQN"])
    valid, reason = validate_p01_cache(cache_path, cache_ids, source_sha256)
    if not valid:
        raise ValueError(f"Rejected corrected P01 cache: {reason}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("paxmin_sha256") != source_sha256
        or manifest.get("p01_algorithm_version") != P01_ALGORITHM_VERSION
        or manifest.get("method_fingerprint") != P01_METHOD_FINGERPRINT
        or manifest.get("cache_validation") != "matched"
    ):
        raise ValueError("KAN-172 manifest does not match the corrected cache/source provenance")
    historical_cohort_n = int(manifest["participants_processed"])
    if historical_cohort_n > len(cache_ids):
        raise ValueError("KAN-172 original participant count exceeds the corrected cache coverage")
    cache = cache.loc[cache["SEQN"].isin(target)].copy()
    return cache, source_sha256, historical_cohort_n, len(target - cache_ids)


def _pending_model_row(model_name: str, status_reason: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "formula": (
            "PHQ9_Score ~ P01_10m + Age_z + C(Sex) + BMI_z + PIR_z + "
            "self_reported_weekday_sleep_hours_z + WASO_proxy_min_z"
        ),
        "N": np.nan,
        "beta_P01": np.nan,
        "SE_P01": np.nan,
        "CI95_low": np.nan,
        "CI95_high": np.nan,
        "p_P01": np.nan,
        "R2": np.nan,
        "adjusted_R2": np.nan,
        "P01_effect_per_0_01": np.nan,
        "status": "PENDING_WASO_PROXY",
        "status_reason": status_reason,
    }


def run_cache_only_primary(
    raw_dir: Path,
    cohort_path: Path,
    cache_path: Path,
    kan172_manifest_path: Path,
    paxmin_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort, data, _ = _read_analysis_sources(raw_dir, cohort_path)
    target = set(cohort["SEQN"].astype(int))
    corrected_cache, source_sha256, historical_cohort_n, cache_missing_n = (
        _load_corrected_cache_for_target(
            cache_path, kan172_manifest_path, paxmin_path, target
        )
    )
    data = data.merge(
        corrected_cache[["SEQN", *P01_COLUMNS]],
        on="SEQN",
        how="left",
        validate="one_to_one",
    )
    data["WASO_proxy_min"] = np.nan
    data["WASO_valid_nights"] = np.nan
    finite_p01 = data[P01_COLUMNS].to_numpy(dtype=float)
    finite_p01 = finite_p01[np.isfinite(finite_p01)]
    if finite_p01.size and ((finite_p01 < 0).any() or (finite_p01 > 1).any()):
        raise ValueError("Corrected KAN-172 P01 values fall outside [0,1]")
    primary_input_path = output_dir / "kan173_primary_input.csv"
    data.to_csv(primary_input_path, index=False)

    ordinary, hc3, model1 = _fit_primary(data, "Model 1")
    ordinary["status"] = "complete"
    hc3["status"] = "complete"
    reason = (
        "The PAXMIN stream exceeded the 3600-second run limit after 180 chunks; "
        "127 corrected-cohort SEQNs lack cached P01 and WASO_proxy_min remains unavailable."
    )
    ordinary_rows = [ordinary, _pending_model_row("Model 2", reason)]
    hc3_rows = [hc3, _pending_model_row("Model 2", reason)]
    ordinary_table = pd.DataFrame(ordinary_rows)
    hc3_table = pd.DataFrame(hc3_rows)
    ordinary_table.to_csv(output_dir / "primary_models.csv", index=False)
    hc3_table.to_csv(output_dir / "primary_models_hc3.csv", index=False)
    historical = _historical_comparison(ordinary_rows, hc3_rows)
    historical.to_csv(output_dir / "p01_primary_old_vs_corrected.csv", index=False)
    _print_primary_checkpoint(ordinary_rows, hc3_rows)
    available_n = int(np.isfinite(_numeric(data["P01_10m"]).to_numpy(float)).sum())
    primary_state = {
        "status": "primary_complete",
        "completion": "partial",
        "paxmin_sha256": source_sha256,
        "p01_algorithm_version": P01_ALGORITHM_VERSION,
        "method_fingerprint": P01_METHOD_FINGERPRINT,
        "historical_cohort_N": historical_cohort_n,
        "corrected_cohort_N": len(cohort),
        "p01_available_N": available_n,
        "p01_missing_from_cache_N": cache_missing_n,
        "model1_N": len(model1),
        "model2_N": None,
        "model2_status": "PENDING_WASO_PROXY",
        "model2_pending_reason": reason,
    }
    (output_dir / "kan173_primary_state.json").write_text(
        json.dumps(primary_state, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "manifest": primary_state,
        "ordinary": ordinary_table,
        "hc3": hc3_table,
        "historical": historical,
        "survey_status": "NOT_RUN",
        "survey": None,
        "data": data,
    }


def run_analysis(
    raw_dir: Path = DEFAULT_RAW_DIR,
    cohort_path: Path = DEFAULT_COHORT,
    cache_path: Path = DEFAULT_CACHE,
    paxmin_path: Path | None = None,
    paxhd_path: Path | None = None,
    kan172_manifest_path: Path = DEFAULT_KAN172_MANIFEST,
    output_dir: Path = PROJECT_ROOT / "outputs" / "models",
    chunksize: int = 400_000,
    bootstrap_replicates: int = 1000,
    primary_only: bool = False,
) -> dict[str, Any]:
    paxmin_path = paxmin_path or (raw_dir / "PAXMIN_H.xpt")
    paxhd_path = paxhd_path or (raw_dir / "PAXHD_H.xpt")
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort, data, (demo, dpq, bmx, slq) = _read_analysis_sources(raw_dir, cohort_path)
    target = set(cohort["SEQN"].astype(int))
    corrected_cache, waso, source_sha256, historical_cohort_n = _extend_corrected_cache_and_stream_waso(
        target, paxmin_path, paxhd_path, cache_path, kan172_manifest_path, chunksize
    )
    pam_columns = ["SEQN", *P01_COLUMNS]
    data = data.merge(corrected_cache[pam_columns], on="SEQN", how="left", validate="one_to_one")
    data = data.merge(waso[["SEQN", "WASO_proxy_min", "WASO_valid_nights"]], on="SEQN", how="left", validate="one_to_one")
    p01_values = data[P01_COLUMNS].to_numpy(dtype=float)
    finite_p01 = p01_values[np.isfinite(p01_values)]
    if finite_p01.size and ((finite_p01 < 0).any() or (finite_p01 > 1).any()):
        raise ValueError("Corrected KAN-172 P01 values fall outside [0,1]")
    primary_input_path = output_dir / "kan173_primary_input.csv"
    data.to_csv(primary_input_path, index=False)

    ordinary_rows: list[dict[str, Any]] = []
    hc3_rows: list[dict[str, Any]] = []
    model_frames: dict[str, pd.DataFrame] = {}
    for model_name in ("Model 1", "Model 2"):
        ordinary, hc3, complete = _fit_primary(data, model_name)
        ordinary_rows.append(ordinary)
        hc3_rows.append(hc3)
        model_frames[model_name] = complete
    ordinary_table = pd.DataFrame(ordinary_rows)
    hc3_table = pd.DataFrame(hc3_rows)
    ordinary_table.to_csv(output_dir / "primary_models.csv", index=False)
    hc3_table.to_csv(output_dir / "primary_models_hc3.csv", index=False)
    historical = _historical_comparison(ordinary_rows, hc3_rows)
    historical.to_csv(output_dir / "p01_primary_old_vs_corrected.csv", index=False)

    _print_primary_checkpoint(ordinary_rows, hc3_rows)
    if primary_only:
        primary_state = {
            "status": "primary_complete",
            "paxmin_sha256": source_sha256,
            "p01_algorithm_version": P01_ALGORITHM_VERSION,
            "method_fingerprint": P01_METHOD_FINGERPRINT,
            "historical_cohort_N": historical_cohort_n,
            "corrected_cohort_N": len(cohort),
            "model1_N": len(model_frames["Model 1"]),
            "model2_N": len(model_frames["Model 2"]),
        }
        (output_dir / "kan173_primary_state.json").write_text(
            json.dumps(primary_state, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return {
            "manifest": primary_state,
            "ordinary": ordinary_table,
            "hc3": hc3_table,
            "historical": historical,
            "survey_status": "NOT_RUN",
            "survey": None,
            "data": data,
        }

    survey_status, survey = _run_survey_models(
        model_frames["Model 1"],
        model_frames["Model 2"],
        output_dir,
        PROJECT_ROOT / "kan173_survey_models.R",
    )
    resolution = _fit_resolution_sensitivity(data)
    resolution.to_csv(output_dir / "resolution_sensitivity.csv", index=False)
    roc_summary, _ = _fit_oof_roc(data, output_dir, bootstrap_replicates)
    flow = _cohort_flow(
        cohort,
        data,
        demo,
        dpq,
        _load_xport(paxhd_path, ["SEQN", "PAXSTS"]),
        model_frames["Model 1"],
        model_frames["Model 2"],
        roc_summary,
        historical_cohort_n,
    )
    flow.to_csv(output_dir / "cohort_flow.csv", index=False)

    manifest = {
        "status": "complete",
        "paxmin_sha256": source_sha256,
        "p01_algorithm_version": P01_ALGORITHM_VERSION,
        "method_fingerprint": P01_METHOD_FINGERPRINT,
        "paxdayd_wear_window": [2, 8],
        "historical_cohort_N": historical_cohort_n,
        "corrected_cohort_N": len(cohort),
        "model1_N": len(model_frames["Model 1"]),
        "model2_N": len(model_frames["Model 2"]),
        "survey_model": survey_status,
        "downstream_analyses": {"ordinary_ols": True, "hc3": True, "survey": survey_status == "PASS", "roc": True},
        "outputs": [
            "primary_models.csv",
            "primary_models_hc3.csv",
            "p01_primary_old_vs_corrected.csv",
            "resolution_sensitivity.csv",
            "roc_corrected.csv",
            "roc_corrected_oof_predictions.csv",
            "cohort_flow.csv",
            "survey_weighted_models.csv" if survey_status == "PASS" else None,
        ],
    }
    (output_dir / "kan173_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "ordinary": ordinary_table,
        "hc3": hc3_table,
        "historical": historical,
        "resolution": resolution,
        "roc": roc_summary,
        "flow": flow,
        "survey_status": survey_status,
        "survey": survey,
        "data": data,
    }


def run_secondary_analysis(
    raw_dir: Path,
    cohort_path: Path,
    cache_path: Path,
    paxmin_path: Path,
    paxhd_path: Path,
    kan172_manifest_path: Path,
    output_dir: Path,
    bootstrap_replicates: int,
    chunksize: int,
) -> dict[str, Any]:
    primary_state_path = output_dir / "kan173_primary_state.json"
    primary_input_path = output_dir / "kan173_primary_input.csv"
    if not primary_state_path.is_file() or not primary_input_path.is_file():
        raise FileNotFoundError("Run the primary stage before --post-primary")
    primary_state = json.loads(primary_state_path.read_text(encoding="utf-8"))
    if primary_state.get("status") != "primary_complete":
        raise ValueError("KAN-173 primary-stage state is incomplete")
    if (
        primary_state.get("p01_algorithm_version") != P01_ALGORITHM_VERSION
        or primary_state.get("method_fingerprint") != P01_METHOD_FINGERPRINT
    ):
        raise ValueError("Saved primary-stage state uses a different corrected P01 method")
    cohort, _, (demo, dpq, _, _) = _read_analysis_sources(raw_dir, cohort_path)
    data = pd.read_csv(primary_input_path)
    target = set(cohort["SEQN"].astype(int))
    if set(pd.to_numeric(data["SEQN"], errors="raise").astype(int)) != target:
        raise ValueError("Saved primary-stage data does not match the corrected cohort")
    source_sha256 = sha256_file(paxmin_path)
    if source_sha256 != primary_state.get("paxmin_sha256"):
        raise ValueError("PAXMIN source changed after the corrected primary checkpoint")
    if primary_state.get("completion") == "partial":
        corrected_cache, waso, _, historical_cohort_n = _extend_corrected_cache_and_stream_waso(
            target,
            paxmin_path,
            paxhd_path,
            cache_path,
            kan172_manifest_path,
            chunksize,
        )
        data = data.drop(columns=[*P01_COLUMNS, "WASO_proxy_min", "WASO_valid_nights"], errors="ignore")
        data = data.merge(
            corrected_cache[["SEQN", *P01_COLUMNS]],
            on="SEQN",
            how="left",
            validate="one_to_one",
        )
        data = data.merge(
            waso[["SEQN", "WASO_proxy_min", "WASO_valid_nights"]],
            on="SEQN",
            how="left",
            validate="one_to_one",
        )
        data.to_csv(primary_input_path, index=False)
        primary_state.update(
            {
                "completion": "waso_complete",
                "historical_cohort_N": historical_cohort_n,
                "p01_available_N": int(np.isfinite(_numeric(data["P01_10m"]).to_numpy(float)).sum()),
                "p01_missing_from_cache_N": 0,
                "model2_status": "READY",
            }
        )
        primary_state_path.write_text(
            json.dumps(primary_state, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    cache = pd.read_csv(cache_path)
    valid, reason = validate_p01_cache(cache_path, target, source_sha256)
    if not valid:
        raise ValueError(f"Corrected P01 cache rejected before secondary analyses: {reason}")
    cache["SEQN"] = pd.to_numeric(cache["SEQN"], errors="raise").astype(int)
    checked = data[["SEQN", *P01_COLUMNS]].merge(
        cache[["SEQN", *P01_COLUMNS]],
        on="SEQN",
        how="left",
        validate="one_to_one",
        suffixes=("_saved", "_cache"),
    )
    for column in P01_COLUMNS:
        if not np.allclose(
            checked[f"{column}_saved"].to_numpy(dtype=float),
            checked[f"{column}_cache"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"Saved primary-stage {column} differs from corrected cache")
    ordinary_rows: list[dict[str, Any]] = []
    hc3_rows: list[dict[str, Any]] = []
    model_frames: dict[str, pd.DataFrame] = {}
    for model_name in ("Model 1", "Model 2"):
        ordinary, hc3, complete = _fit_primary(data, model_name)
        ordinary_rows.append(ordinary)
        hc3_rows.append(hc3)
        model_frames[model_name] = complete
    primary_state.update(
        {
            "completion": "complete",
            "model1_N": len(model_frames["Model 1"]),
            "model2_N": len(model_frames["Model 2"]),
            "model2_status": "complete",
            "model2_pending_reason": None,
        }
    )
    primary_state_path.write_text(
        json.dumps(primary_state, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ordinary_table = pd.DataFrame(ordinary_rows)
    hc3_table = pd.DataFrame(hc3_rows)
    ordinary_table.to_csv(output_dir / "primary_models.csv", index=False)
    hc3_table.to_csv(output_dir / "primary_models_hc3.csv", index=False)
    historical = _historical_comparison(ordinary_rows, hc3_rows)
    historical.to_csv(output_dir / "p01_primary_old_vs_corrected.csv", index=False)
    _print_primary_checkpoint(ordinary_rows, hc3_rows)
    survey_status, survey = _run_survey_models(
        model_frames["Model 1"],
        model_frames["Model 2"],
        output_dir,
        PROJECT_ROOT / "kan173_survey_models.R",
    )
    resolution = _fit_resolution_sensitivity(data)
    resolution.to_csv(output_dir / "resolution_sensitivity.csv", index=False)
    roc_summary, _ = _fit_oof_roc(data, output_dir, bootstrap_replicates)
    paxhd = _load_xport(raw_dir / "PAXHD_H.xpt", ["SEQN", "PAXSTS"])
    flow = _cohort_flow(
        cohort,
        data,
        demo,
        dpq,
        paxhd,
        model_frames["Model 1"],
        model_frames["Model 2"],
        roc_summary,
        int(primary_state["historical_cohort_N"]),
    )
    flow.to_csv(output_dir / "cohort_flow.csv", index=False)
    manifest = {
        **primary_state,
        "status": "complete",
        "paxdayd_wear_window": [2, 8],
        "survey_model": survey_status,
        "downstream_analyses": {
            "ordinary_ols": True,
            "hc3": True,
            "survey": survey_status == "PASS",
            "roc": True,
        },
        "outputs": [
            "primary_models.csv",
            "primary_models_hc3.csv",
            "p01_primary_old_vs_corrected.csv",
            "resolution_sensitivity.csv",
            "roc_corrected.csv",
            "roc_corrected_oof_predictions.csv",
            "cohort_flow.csv",
            "survey_weighted_models.csv" if survey_status == "PASS" else None,
        ],
    }
    (output_dir / "kan173_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    primary_input_path.unlink(missing_ok=True)
    return {
        "manifest": manifest,
        "ordinary": ordinary_table,
        "hc3": hc3_table,
        "historical": historical,
        "resolution": resolution,
        "roc": roc_summary,
        "flow": flow,
        "survey_status": survey_status,
        "survey": survey,
        "data": data,
    }
def main() -> None:
    parser = argparse.ArgumentParser(description="Run KAN-173 corrected NHANES P01 models.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--paxmin", type=Path, default=None)
    parser.add_argument("--paxhd", type=Path, default=None)
    parser.add_argument("--kan172-manifest", type=Path, default=DEFAULT_KAN172_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "models")
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--post-primary", action="store_true")
    args = parser.parse_args()
    if args.primary_only and args.post_primary:
        parser.error("--primary-only and --post-primary cannot be combined")
    if args.post_primary:
        result = run_secondary_analysis(
            args.raw_dir,
            args.cohort,
            args.cache,
            args.paxmin or (args.raw_dir / "PAXMIN_H.xpt"),
            args.paxhd or (args.raw_dir / "PAXHD_H.xpt"),
            args.kan172_manifest,
            args.output_dir,
            args.bootstrap_replicates,
            args.chunksize,
        )
    elif args.primary_only:
        result = run_cache_only_primary(
            args.raw_dir,
            args.cohort,
            args.cache,
            args.kan172_manifest,
            args.paxmin or (args.raw_dir / "PAXMIN_H.xpt"),
            args.output_dir,
        )
    else:
        result = run_analysis(
            raw_dir=args.raw_dir,
            cohort_path=args.cohort,
            cache_path=args.cache,
            paxmin_path=args.paxmin,
            paxhd_path=args.paxhd,
            kan172_manifest_path=args.kan172_manifest,
            output_dir=args.output_dir,
            chunksize=args.chunksize,
            bootstrap_replicates=args.bootstrap_replicates,
            primary_only=False,
        )
    print(
        f"KAN-173 {'primary checkpoint complete' if args.primary_only else 'analysis complete'}: "
        f"N={result['manifest']['corrected_cohort_N']}; survey={result['survey_status']}"
    )
    if result["survey"] is not None:
        print(result["survey"].to_string(index=False))
    print(result["ordinary"].to_string(index=False))
    print(result["hc3"].to_string(index=False))
    if not args.primary_only:
        print(result["resolution"].to_string(index=False))
        print(result["roc"].to_string(index=False))


if __name__ == "__main__":
    main()
