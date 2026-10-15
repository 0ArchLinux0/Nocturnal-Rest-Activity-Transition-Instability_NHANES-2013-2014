import numpy as np
import pandas as pd

from nhanes_p01.p01 import compute_p01_corrected


SAMPLES_PER_MINUTE = 80 * 60


def make_minutes(sample_offsets, days, activity, *, q=None, duration=None, seqn=10001):
    count = len(sample_offsets)
    return pd.DataFrame(
        {
            "SEQN": seqn,
            "PAXDAYM": days,
            "PAXSSNMP": np.asarray(sample_offsets, dtype=np.int64) * SAMPLES_PER_MINUTE,
            "PAXTSM": 60 if duration is None else duration,
            "PAXQFM": 0 if q is None else q,
            "PAXMTSM": activity,
        }
    )


def test_exact_2350_to_midnight_block_edge_is_kept():
    offsets = list(range(20))
    days = [1] * 10 + [2] * 10
    activity = [0.0] * 10 + [1.0] * 10
    result = compute_p01_corrected(
        make_minutes(offsets, days, activity), "23:50:00", resolution_min=10
    )

    assert result["complete_blocks"] == 2
    assert result["valid_nights"] == 1
    assert result["valid_edges"] == 1
    assert result["rest_edges"] == 1
    assert result["transitions_0_to_1"] == 1
    assert result["p01"] == 1.0


def test_0550_to_2200_daylight_gap_is_not_an_edge():
    offsets = list(range(10)) + list(range(970, 980))
    activity = [0.0] * 10 + [1.0] * 10
    result = compute_p01_corrected(
        make_minutes(offsets, [1] * 20, activity), "05:50:00", resolution_min=10
    )

    assert result["complete_blocks"] == 2
    assert result["valid_edges"] == 0
    assert result["rest_edges"] == 0
    assert np.isnan(result["p01"])


def test_missing_minute_breaks_edges_and_removes_incomplete_block():
    offsets = list(range(30))
    offsets.remove(14)
    days = [1 if offset < 20 else 2 for offset in offsets]
    activity = [0.0 if offset < 20 else 1.0 for offset in offsets]
    result = compute_p01_corrected(
        make_minutes(offsets, days, activity), "23:40:00", resolution_min=10
    )

    assert result["complete_blocks"] == 2
    assert result["incomplete_blocks"] == 1
    assert result["valid_edges"] == 0
    assert result["gap_breaks"] == 1


def test_nonzero_qc_flag_excludes_minute_and_block():
    q = [0] * 20
    q[4] = 3
    result = compute_p01_corrected(
        make_minutes(
            range(20), [1] * 20, [0.0] * 10 + [1.0] * 10, q=q
        ),
        "22:00:00",
        resolution_min=10,
    )

    assert result["excluded_qc_minutes"] == 1
    assert result["incomplete_blocks"] == 1
    assert result["complete_blocks"] == 1
    assert result["valid_edges"] == 0


def test_negative_uncomputed_mims_excludes_minute_and_block():
    activity = [0.0] * 20
    activity[7] = -0.01
    result = compute_p01_corrected(
        make_minutes(range(20), [1] * 20, activity), "22:00:00", resolution_min=10
    )

    assert result["excluded_uncomputed_minutes"] == 1
    assert result["incomplete_blocks"] == 1
    assert result["complete_blocks"] == 1


def test_incomplete_pax_tsm_excludes_minute_and_block():
    duration = [60] * 20
    duration[2] = 59
    result = compute_p01_corrected(
        make_minutes(
            range(20), [1] * 20, [0.0] * 10 + [1.0] * 10, duration=duration
        ),
        "22:00:00",
        resolution_min=10,
    )

    assert result["excluded_incomplete_minutes"] == 1
    assert result["incomplete_blocks"] == 1
    assert result["complete_blocks"] == 1
