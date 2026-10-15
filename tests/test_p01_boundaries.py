import numpy as np
import pandas as pd
import pytest

from nhanes_p01.p01 import (
    P01_RESOLUTIONS_MIN,
    P01TimeAxisError,
    compute_p01_corrected,
    compute_p01_comparison,
)


SAMPLES_PER_MINUTE = 80 * 60


def make_contiguous_minutes(count, *, start_time, seqn=20001, activity=None):
    if activity is None:
        activity = np.ones(count, dtype=float)
    return pd.DataFrame(
        {
            "SEQN": seqn,
            "PAXDAYM": [1] * count,
            "PAXSSNMP": np.arange(count, dtype=np.int64) * SAMPLES_PER_MINUTE,
            "PAXTSM": [60] * count,
            "PAXQFM": [0] * count,
            "PAXMTSM": activity,
        }
    ), start_time


def test_threshold_is_participant_median_of_complete_blocks_per_resolution():
    minutes, paxftime = make_contiguous_minutes(120, start_time="12:00:00")

    thresholds = {
        resolution: compute_p01_corrected(minutes, paxftime, resolution)["threshold"]
        for resolution in P01_RESOLUTIONS_MIN
    }

    assert thresholds == {5: 5.0, 10: 10.0, 30: 30.0, 60: 60.0}


def test_zero_rest_edge_denominator_returns_nan():
    offsets = np.asarray(list(range(300)) + list(range(600, 720)), dtype=np.int64)
    minute_count = len(offsets)
    activity = np.concatenate([np.zeros(300), np.full(120, 10.0)])
    minutes = pd.DataFrame(
        {
            "SEQN": 20002,
            "PAXDAYM": [1] * minute_count,
            "PAXSSNMP": offsets * SAMPLES_PER_MINUTE,
            "PAXTSM": [60] * minute_count,
            "PAXQFM": [0] * minute_count,
            "PAXMTSM": activity,
        }
    )

    result = compute_p01_corrected(minutes, "12:00:00", resolution_min=60)

    assert result["valid_edges"] == 1
    assert result["rest_edges"] == 0
    assert result["transitions_0_to_1"] == 0
    assert np.isnan(result["p01"])


def test_paxdaym_clock_disagreement_fails_explicitly():
    minutes, paxftime = make_contiguous_minutes(10, start_time="22:00:00")
    minutes.loc[0, "PAXDAYM"] = 2

    with pytest.raises(P01TimeAxisError, match="PAXDAYM disagrees"):
        compute_p01_corrected(minutes, paxftime, resolution_min=10)


def test_sample_number_regression_fails_explicitly():
    minutes, paxftime = make_contiguous_minutes(10, start_time="22:00:00")
    minutes.loc[5, "PAXSSNMP"] = 3 * SAMPLES_PER_MINUTE

    with pytest.raises(P01TimeAxisError, match="PAXSSNMP decreases"):
        compute_p01_corrected(minutes, paxftime, resolution_min=10)


def test_missing_first_sample_cannot_anchor_paxftime():
    minutes, paxftime = make_contiguous_minutes(10, start_time="22:00:00")
    minutes.loc[0, "PAXSSNMP"] = np.nan

    with pytest.raises(P01TimeAxisError, match="first PAXSSNMP value is missing"):
        compute_p01_corrected(minutes, paxftime, resolution_min=10)


def test_sample_counter_gaps_are_counted_and_blocks_are_incomplete():
    minutes, paxftime = make_contiguous_minutes(15, start_time="22:00:00")
    minutes.loc[5:, "PAXSSNMP"] += SAMPLES_PER_MINUTE

    result = compute_p01_comparison(minutes, paxftime)

    assert [row["sample_counter_non_4800_steps"] for row in result] == [1] * 4
    assert any(row["incomplete_blocks"] > 0 for row in result)


def test_comparison_returns_same_participant_and_resolution_rows():
    minutes, paxftime = make_contiguous_minutes(120, start_time="12:00:00")

    result = compute_p01_comparison(minutes, paxftime)

    assert [row["resolution_min"] for row in result] == [5, 10, 30, 60]
    assert {row["SEQN"] for row in result} == {20001}
    assert all("p01_old" in row and "p01_corrected" in row for row in result)
