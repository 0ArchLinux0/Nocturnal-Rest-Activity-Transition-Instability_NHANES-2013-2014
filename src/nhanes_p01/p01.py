from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

P01_RESOLUTIONS_MIN = (5, 10, 30, 60)
P01_ALGORITHM_VERSION = "kan172-p01-clock-contiguity-v1"
P01_SAMPLE_RATE_HZ = 80
P01_METHOD_SPEC = (
    "paxftime-anchor;paxssnmp-sample-time-80hz;sample-counter-monotonic;"
    "paxdaym-matches-derived-local-day;"
    "paxqfm-equals-zero;paxmtsm-finite-nonnegative;paxtsm-equals-60;"
    "complete-clock-aligned-blocks-5-10-30-60;"
    "participant-median-all-complete-blocks-per-resolution;"
    "night-id-evening-22-24-next-morning-00-06;"
    "edge-exact-block-start-adjacency;s0-to-s1-over-all-eligible-s0-edges"
)
P01_METHOD_FINGERPRINT = hashlib.sha256(P01_METHOD_SPEC.encode("utf-8")).hexdigest()
PAXMIN_REQUIRED_COLUMNS = ("SEQN", "PAXDAYM", "PAXSSNMP", "PAXTSM", "PAXQFM", "PAXMTSM")
P01_CACHE_REQUIRED_COLUMNS = (
    "SEQN",
    "P01_10m",
    "p01_algorithm_version",
    "paxmin_sha256",
    "method_fingerprint",
)

def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

_TICKS_PER_SECOND = P01_SAMPLE_RATE_HZ
_TICKS_PER_MINUTE = 60 * _TICKS_PER_SECOND
_TICKS_PER_DAY = 24 * 60 * _TICKS_PER_MINUTE
_NIGHT_START_TICKS = 22 * 60 * _TICKS_PER_MINUTE
_MORNING_END_TICKS = 6 * 60 * _TICKS_PER_MINUTE


class P01TimeAxisError(ValueError):
    """Raised when a participant's source time fields cannot be reconciled."""


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


def parse_paxftime(value: Any) -> int:
    """Return PAXFTIME as whole seconds after local midnight."""
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    if not isinstance(value, str):
        raise P01TimeAxisError(f"PAXFTIME must be an HH:MM:SS string, got {value!r}")
    value = value.strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", value)
    if match is None:
        raise P01TimeAxisError(f"Invalid PAXFTIME value: {value!r}")
    hour, minute, second = (int(part) for part in match.groups())
    if hour > 23 or minute > 59 or second > 59:
        raise P01TimeAxisError(f"Out-of-range PAXFTIME value: {value!r}")
    return hour * 3600 + minute * 60 + second


def normalize_paxmin_timestamps(minutes: pd.DataFrame, paxftime: Any) -> pd.DataFrame:
    """Attach local-device time derived from PAXFTIME and PAXSSNMP, never row order."""
    missing = [column for column in PAXMIN_REQUIRED_COLUMNS if column not in minutes.columns]
    if missing:
        raise ValueError(f"PAXMIN is missing required columns: {missing}")

    out = minutes.copy()
    paxftime_seconds = parse_paxftime(paxftime)
    sample = _numeric(out["PAXSSNMP"])
    sample_values = sample.to_numpy(dtype=float)
    finite_sample = np.isfinite(sample_values)
    integer_sample = finite_sample & (sample_values == np.floor(sample_values))
    if np.any(finite_sample & ~integer_sample):
        raise P01TimeAxisError("PAXSSNMP contains non-integer sample numbers")

    if len(sample_values) == 0 or not integer_sample[0]:
        raise P01TimeAxisError("first PAXSSNMP value is missing; cannot anchor PAXFTIME")
    observed_samples = sample_values[integer_sample]
    if len(observed_samples) > 1 and np.any(np.diff(observed_samples) < 0):
        raise P01TimeAxisError("PAXSSNMP decreases within a participant")
    first_sample = int(sample_values[0])
    sample_ticks = np.full(len(out), np.nan, dtype=float)
    sample_ticks[integer_sample] = sample_values[integer_sample] - first_sample
    timestamp_ticks = sample_ticks + paxftime_seconds * _TICKS_PER_SECOND

    day = _numeric(out["PAXDAYM"])
    day_values = day.to_numpy(dtype=float)
    timestamp_day = np.floor(timestamp_ticks / _TICKS_PER_DAY) + 1
    day_check = np.isfinite(day_values) & np.isfinite(timestamp_day)
    mismatch = day_check & (day_values != timestamp_day)
    if mismatch.any():
        bad = np.flatnonzero(mismatch)[:5]
        examples = [
            {
                "PAXDAYM": day_values[index],
                "derived_day": int(timestamp_day[index]),
                "PAXSSNMP": sample_values[index],
            }
            for index in bad
        ]
        raise P01TimeAxisError(f"PAXDAYM disagrees with PAXFTIME/PAXSSNMP: {examples}")

    out["PAXDAYM_INT"] = day
    out["PAXSSNMP_NUM"] = sample
    out["PAXTSM_NUM"] = _numeric(out["PAXTSM"])
    out["PAXQFM_NUM"] = _numeric(out["PAXQFM"])
    out["PAXMTSM_NUM"] = _numeric(out["PAXMTSM"])
    out["local_timestamp_ticks"] = timestamp_ticks
    out["local_timestamp_seconds"] = timestamp_ticks / _TICKS_PER_SECOND
    out["local_day_index"] = np.floor(timestamp_ticks / _TICKS_PER_DAY)
    out["seconds_since_midnight"] = (
        timestamp_ticks % _TICKS_PER_DAY
    ) / _TICKS_PER_SECOND
    return out



def _p01_from_counts(rest_edges: int, transitions_0_to_1: int) -> float:
    if rest_edges == 0:
        return float("nan")
    return float(transitions_0_to_1 / rest_edges)


def _source_exclusion_counts(rows: pd.DataFrame) -> dict[str, int]:
    q = rows["PAXQFM_NUM"].to_numpy(dtype=float)
    activity = rows["PAXMTSM_NUM"].to_numpy(dtype=float)
    duration = rows["PAXTSM_NUM"].to_numpy(dtype=float)
    timestamp = rows["local_timestamp_ticks"].to_numpy(dtype=float)
    return {
        "excluded_qc_minutes": int(np.count_nonzero(~np.isfinite(q) | (q != 0))),
        "excluded_qc_missing_minutes": int(np.count_nonzero(~np.isfinite(q))),
        "excluded_uncomputed_minutes": int(np.count_nonzero(~np.isfinite(activity) | (activity < 0))),
        "excluded_incomplete_minutes": int(np.count_nonzero(~np.isfinite(duration) | (duration != 60))),
        "excluded_missing_timestamp_minutes": int(np.count_nonzero(~np.isfinite(timestamp))),
    }


def _compute_p01_corrected_normalized(
    normalized: pd.DataFrame,
    resolution_min: int,
    analysis_day_min: int,
    analysis_day_max: int,
) -> dict[str, Any]:
    if resolution_min not in P01_RESOLUTIONS_MIN:
        raise ValueError(f"Unsupported P01 resolution: {resolution_min}")
    day_values = normalized["PAXDAYM_INT"].to_numpy(dtype=float)
    in_window = (
        np.isfinite(day_values)
        & (day_values >= analysis_day_min)
        & (day_values <= analysis_day_max)
    )
    rows = normalized.loc[in_window].copy()
    exclusions = _source_exclusion_counts(rows)
    if rows.empty:
        return {
            "p01": float("nan"),
            "threshold": float("nan"),
            "complete_blocks": 0,
            "nocturnal_blocks": 0,
            "valid_nights": 0,
            "valid_edges": 0,
            "rest_edges": 0,
            "transitions_0_to_1": 0,
            "gap_breaks": 0,
            "incomplete_blocks": 0,
            **exclusions,
        }

    resolution_ticks = resolution_min * _TICKS_PER_MINUTE
    rows["block_start_ticks"] = (
        np.floor(rows["local_timestamp_ticks"] / resolution_ticks) * resolution_ticks
    )
    rows["valid_minute"] = (
        (rows["PAXQFM_NUM"] == 0)
        & np.isfinite(rows["PAXMTSM_NUM"])
        & (rows["PAXMTSM_NUM"] >= 0)
        & (rows["PAXTSM_NUM"] == 60)
        & np.isfinite(rows["local_timestamp_ticks"])
        & (rows["local_timestamp_ticks"] % _TICKS_PER_MINUTE == 0)
    )
    candidate_count = int(rows["block_start_ticks"].nunique(dropna=True))
    valid = rows.loc[rows["valid_minute"]]
    block_stats = valid.groupby("block_start_ticks", sort=True).agg(
        valid_count=("local_timestamp_ticks", "size"),
        unique_count=("local_timestamp_ticks", "nunique"),
        first_tick=("local_timestamp_ticks", "min"),
        last_tick=("local_timestamp_ticks", "max"),
        activity=("PAXMTSM_NUM", "sum"),
    )
    if block_stats.empty:
        complete = pd.DataFrame(columns=["block_start_ticks", "activity"])
    else:
        # Minute alignment and floor assignment bound rows to the block; count, uniqueness, and endpoints prove exact coverage.
        starts = block_stats.index.to_numpy(dtype=np.int64)
        is_complete = (
            (block_stats["valid_count"].to_numpy() == resolution_min)
            & (block_stats["unique_count"].to_numpy() == resolution_min)
            & (block_stats["first_tick"].to_numpy() == starts)
            & (
                block_stats["last_tick"].to_numpy()
                == starts + (resolution_min - 1) * _TICKS_PER_MINUTE
            )
        )
        complete = (
            block_stats.loc[is_complete, ["activity"]]
            .rename_axis("block_start_ticks")
            .reset_index()
        )

    incomplete_blocks = candidate_count - len(complete)
    if complete.empty:
        threshold = float("nan")
        nocturnal = complete.assign(night_id=pd.Series(dtype=float), state=pd.Series(dtype=np.int8))
    else:
        threshold = float(np.median(complete["activity"].to_numpy(dtype=float)))
        starts = complete["block_start_ticks"].to_numpy(dtype=np.int64)
        clock_ticks = starts % _TICKS_PER_DAY
        day_index = starts // _TICKS_PER_DAY
        night_id = np.full(len(complete), np.nan, dtype=float)
        evening = clock_ticks >= _NIGHT_START_TICKS
        morning = clock_ticks < _MORNING_END_TICKS
        night_id[evening] = day_index[evening]
        night_id[morning] = day_index[morning] - 1
        complete["night_id"] = night_id
        complete["state"] = (complete["activity"] > threshold).astype(np.int8)
        nocturnal = complete.loc[complete["night_id"].notna()].copy()

    valid_edges = 0
    rest_edges = 0
    transitions = 0
    gap_breaks = 0
    if not nocturnal.empty:
        nocturnal = nocturnal.sort_values(["night_id", "block_start_ticks"])
        grouped = nocturnal.groupby("night_id", sort=False)
        nocturnal["next_start"] = grouped["block_start_ticks"].shift(-1)
        nocturnal["next_state"] = grouped["state"].shift(-1)
        has_next = nocturnal["next_start"].notna()
        is_edge = has_next & (
            nocturnal["next_start"] - nocturnal["block_start_ticks"] == resolution_ticks
        )
        is_rest_edge = is_edge & (nocturnal["state"] == 0)
        valid_edges = int(is_edge.sum())
        rest_edges = int(is_rest_edge.sum())
        transitions = int((is_rest_edge & (nocturnal["next_state"] == 1)).sum())
        gap_breaks = int((has_next & ~is_edge).sum())

    return {
        "p01": _p01_from_counts(rest_edges, transitions),
        "threshold": threshold,
        "complete_blocks": len(complete),
        "nocturnal_blocks": len(nocturnal),
        "valid_nights": int(nocturnal["night_id"].nunique()) if not nocturnal.empty else 0,
        "valid_edges": valid_edges,
        "rest_edges": rest_edges,
        "transitions_0_to_1": transitions,
        "gap_breaks": gap_breaks,
        "incomplete_blocks": incomplete_blocks,
        **exclusions,
    }


def compute_p01_corrected(
    minutes: pd.DataFrame,
    paxftime: Any,
    resolution_min: int,
    analysis_day_min: int = 1,
    analysis_day_max: int = 9,
) -> dict[str, Any]:
    """Compute corrected participant P01 from complete, contiguous, clock-time blocks."""
    normalized = normalize_paxmin_timestamps(minutes, paxftime)
    return _compute_p01_corrected_normalized(
        normalized, resolution_min, analysis_day_min, analysis_day_max
    )

def compute_p01_corrected_resolutions(
    minutes: pd.DataFrame,
    paxftime: Any,
    analysis_day_min: int = 1,
    analysis_day_max: int = 9,
) -> dict[int, dict[str, Any]]:
    """Compute all required corrected resolutions from one normalized time axis."""
    normalized = normalize_paxmin_timestamps(minutes, paxftime)
    return {
        resolution: _compute_p01_corrected_normalized(
            normalized, resolution, analysis_day_min, analysis_day_max
        )
        for resolution in P01_RESOLUTIONS_MIN
    }


def _prepare_p01_legacy(
    minutes: pd.DataFrame,
    analysis_day_min: int,
    analysis_day_max: int,
) -> pd.DataFrame:
    missing = [
        column
        for column in ("PAXDAYM", "PAXSSNMP", "PAXQFM", "PAXMTSM")
        if column not in minutes.columns
    ]
    if missing:
        raise ValueError(f"Legacy P01 input is missing columns: {missing}")
    frame = minutes.copy()
    frame["PAXDAYM_INT"] = _numeric(frame["PAXDAYM"])
    frame["PAXSSNMP_NUM"] = _numeric(frame["PAXSSNMP"])
    frame["PAXQFM_NUM"] = _numeric(frame["PAXQFM"])
    frame["PAXMTSM_NUM"] = _numeric(frame["PAXMTSM"])
    frame = frame.loc[
        frame["PAXDAYM_INT"].between(analysis_day_min, analysis_day_max, inclusive="both")
    ].sort_values(["PAXDAYM_INT", "PAXSSNMP_NUM"])
    frame["_m"] = frame.groupby("PAXDAYM_INT", sort=False).cumcount()
    return frame.loc[
        frame["PAXQFM_NUM"].notna()
        & ~frame["PAXQFM_NUM"].isin([1.0, 2.0])
        & frame["PAXMTSM_NUM"].notna()
        & (frame["PAXMTSM_NUM"] >= 0)
    ].copy()


def _compute_p01_legacy_prepared(valid: pd.DataFrame, resolution_min: int) -> dict[str, Any]:
    if resolution_min not in P01_RESOLUTIONS_MIN:
        raise ValueError(f"Unsupported P01 resolution: {resolution_min}")
    valid = valid.copy()
    valid["_b"] = valid["_m"] // resolution_min
    aggregated = (
        valid.groupby(["PAXDAYM_INT", "_b"], as_index=False)["PAXMTSM_NUM"]
        .sum()
        .rename(columns={"PAXMTSM_NUM": "activity"})
    )
    aggregated["_t0"] = aggregated["_b"] * resolution_min
    block_end = aggregated["_t0"] + resolution_min
    aggregated["_night"] = (
        (np.maximum(aggregated["_t0"], 0) < np.minimum(block_end, 360))
        | (np.maximum(aggregated["_t0"], 1320) < np.minimum(block_end, 1440))
    )
    all_activity = aggregated["activity"].to_numpy(dtype=float)
    night = aggregated.loc[aggregated["_night"]].sort_values(["PAXDAYM_INT", "_b"])
    night_activity = night["activity"].to_numpy(dtype=float)
    threshold = float(np.median(all_activity)) if len(all_activity) else float("nan")
    passed_200_minute_gate = len(valid) >= 200

    value = float("nan")
    if (
        passed_200_minute_gate
        and len(aggregated) >= 10
        and len(night_activity) >= 4
        and len(all_activity) >= 10
    ):
        states = (night_activity > threshold).astype(np.int8)
        denominator = int(np.count_nonzero(states[:-1] == 0))
        numerator = int(np.count_nonzero((states[:-1] == 0) & (states[1:] == 1)))
        value = _p01_from_counts(denominator, numerator)

    daylight_edges = 0
    if len(night) > 1:
        ordered = night[["PAXDAYM_INT", "_t0"]].to_numpy()
        for previous, current in zip(ordered[:-1], ordered[1:]):
            if previous[0] == current[0] and previous[1] < 360 and current[1] >= 1320:
                daylight_edges += 1

    return {
        "p01": value,
        "threshold": threshold,
        "nocturnal_blocks": len(night),
        "valid_minutes": len(valid),
        "passed_200_minute_gate": bool(passed_200_minute_gate),
        "false_daylight_edges": daylight_edges,
    }


def compute_p01_legacy(
    minutes: pd.DataFrame,
    resolution_min: int,
    analysis_day_min: int = 1,
    analysis_day_max: int = 9,
) -> dict[str, Any]:
    """Reproduce the audited row-ordinal/partial-block P01 algorithm for comparison only."""
    valid = _prepare_p01_legacy(minutes, analysis_day_min, analysis_day_max)
    return _compute_p01_legacy_prepared(valid, resolution_min)


def compute_p01_comparison(
    minutes: pd.DataFrame,
    paxftime: Any,
    analysis_day_min: int = 1,
    analysis_day_max: int = 9,
) -> list[dict[str, Any]]:
    """Return paired legacy/corrected metrics for all required resolutions."""
    normalized = normalize_paxmin_timestamps(minutes, paxftime)
    sample_steps = np.diff(normalized["PAXSSNMP_NUM"].to_numpy(dtype=float))
    sample_counter_non_4800_steps = int(
        np.count_nonzero(~np.isfinite(sample_steps) | (sample_steps != _TICKS_PER_MINUTE))
    )
    legacy_valid = _prepare_p01_legacy(minutes, analysis_day_min, analysis_day_max)
    seqn_values = _numeric(minutes["SEQN"]) if "SEQN" in minutes.columns else pd.Series(dtype=float)
    seqn = int(seqn_values.dropna().iloc[0]) if not seqn_values.dropna().empty else np.nan
    rows: list[dict[str, Any]] = []
    for resolution in P01_RESOLUTIONS_MIN:
        old = _compute_p01_legacy_prepared(legacy_valid, resolution)
        corrected = _compute_p01_corrected_normalized(
            normalized, resolution, analysis_day_min, analysis_day_max
        )
        old_value = old["p01"]
        corrected_value = corrected["p01"]
        if np.isfinite(old_value) and np.isfinite(corrected_value):
            status = "paired_finite"
        elif np.isfinite(old_value):
            status = "legacy_only_finite"
        elif np.isfinite(corrected_value):
            status = "corrected_only_finite"
        else:
            status = "both_nonfinite"
        rows.append(
            {
                "SEQN": seqn,
                "resolution_min": resolution,
                "p01_old": old_value,
                "p01_corrected": corrected_value,
                "threshold_old": old["threshold"],
                "threshold_corrected": corrected["threshold"],
                "old_nocturnal_blocks": old["nocturnal_blocks"],
                "corrected_complete_blocks": corrected["complete_blocks"],
                "corrected_valid_nights": corrected["valid_nights"],
                "corrected_valid_edges": corrected["valid_edges"],
                "corrected_rest_edges": corrected["rest_edges"],
                "corrected_0_to_1_edges": corrected["transitions_0_to_1"],
                "excluded_qc_minutes": corrected["excluded_qc_minutes"],
                "sample_counter_non_4800_steps": sample_counter_non_4800_steps,
                "excluded_qc_missing_minutes": corrected["excluded_qc_missing_minutes"],
                "excluded_missing_timestamp_minutes": corrected["excluded_missing_timestamp_minutes"],
                "excluded_uncomputed_minutes": corrected["excluded_uncomputed_minutes"],
                "excluded_incomplete_minutes": corrected["excluded_incomplete_minutes"],
                "incomplete_blocks": corrected["incomplete_blocks"],
                "gap_breaks": corrected["gap_breaks"],
                "false_daylight_edges_removed": old["false_daylight_edges"],
                "legacy_200_minute_gate": old["passed_200_minute_gate"],
                "status": status,
            }
        )
    return rows


def validate_p01_cache(
    cache_path: str | Path,
    expected_seqns: Iterable[int],
    paxmin_sha256: str,
) -> tuple[bool, str]:
    """Reject caches without exact algorithm, source, method, and cohort provenance."""
    path = Path(cache_path)
    if not path.is_file():
        return False, f"cache is missing: {path}"
    try:
        cache = pd.read_csv(path)
    except Exception as exc:
        return False, f"cache could not be read: {exc}"
    missing = [column for column in P01_CACHE_REQUIRED_COLUMNS if column not in cache.columns]
    if missing:
        return False, f"cache is missing provenance columns: {missing}"
    if cache["SEQN"].isna().any():
        return False, "cache SEQN values are missing"
    try:
        numeric_seqns = pd.to_numeric(cache["SEQN"], errors="raise")
        numeric_values = numeric_seqns.to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all() or np.any(numeric_values != np.floor(numeric_values)):
            return False, "cache SEQN values are not finite integers"
        if numeric_seqns.astype(int).duplicated().any():
            return False, "cache SEQN values are duplicated"
        cache_seqns = set(numeric_seqns.astype(int))
        required_seqns = {int(value) for value in expected_seqns}
    except (TypeError, ValueError) as exc:
        return False, f"invalid SEQN values: {exc}"
    if not required_seqns.issubset(cache_seqns):
        return False, "cache does not cover the requested cohort"
    provenance = (
        ("p01_algorithm_version", P01_ALGORITHM_VERSION, "algorithm version does not match"),
        ("paxmin_sha256", paxmin_sha256, "PAXMIN SHA-256 does not match"),
        ("method_fingerprint", P01_METHOD_FINGERPRINT, "method fingerprint does not match"),
    )
    for column, expected_value, mismatch_reason in provenance:
        if cache[column].isna().any() or set(cache[column].astype(str)) != {expected_value}:
            return False, mismatch_reason
    return True, "matched"
