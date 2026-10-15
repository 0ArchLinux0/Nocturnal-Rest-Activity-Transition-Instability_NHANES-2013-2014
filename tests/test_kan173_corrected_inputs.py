import numpy as np
import pandas as pd

from build_final_full_cohort import PHQ9_ITEMS
from kan173_corrected_analysis import score_phq9, waso_proxy_for_participant
from nhanes_p01.p01 import normalize_paxmin_timestamps


def test_phq9_requires_nine_canonical_items_and_ignores_dpq100():
    row = {"SEQN": 1, **{item: 1 for item in PHQ9_ITEMS}, "DPQ100": 3}
    invalid_7 = {"SEQN": 2, **{item: 1 for item in PHQ9_ITEMS}, "DPQ100": 0}
    invalid_7["DPQ090"] = 7
    invalid_9 = {"SEQN": 3, **{item: 1 for item in PHQ9_ITEMS}, "DPQ100": 0}
    invalid_9["DPQ080"] = 9
    invalid_response = {"SEQN": 4, **{item: 1 for item in PHQ9_ITEMS}, "DPQ100": 0}
    invalid_response["DPQ010"] = -1
    zero = {"SEQN": 5, **{item: 0 for item in PHQ9_ITEMS}, "DPQ100": 3}
    maximum = {"SEQN": 6, **{item: 3 for item in PHQ9_ITEMS}, "DPQ100": 0}
    threshold = {"SEQN": 7, **{item: 1 for item in PHQ9_ITEMS}, "DPQ100": 0}
    threshold["DPQ090"] = 2

    scored = score_phq9(
        pd.DataFrame([row, invalid_7, invalid_9, invalid_response, zero, maximum, threshold])
    ).set_index("SEQN")

    assert scored.loc[1, "PHQ9_Score"] == 9
    assert scored.loc[1, "PHQ9_ge_10"] == 0
    assert np.isnan(scored.loc[2, "PHQ9_Score"])
    assert np.isnan(scored.loc[2, "PHQ9_ge_10"])
    assert np.isnan(scored.loc[3, "PHQ9_Score"])
    assert np.isnan(scored.loc[4, "PHQ9_Score"])
    assert scored.loc[5, "PHQ9_Score"] == 0
    assert scored.loc[7, "PHQ9_Score"] == 10
    assert scored.loc[7, "PHQ9_ge_10"] == 1
    assert scored.loc[6, "PHQ9_Score"] == 27

def test_waso_groups_across_midnight_and_excludes_unknown_predictions():
    minutes = 1920
    seconds = 22 * 60 * 60 + np.arange(minutes) * 60
    sample0 = 100_000
    predictions = np.full(minutes, 2, dtype=int)
    predictions[1600:1610] = 1
    predictions[1640] = 3
    predictions[1641] = 4
    paxday = np.floor(seconds / (24 * 60 * 60)).astype(int) + 1
    raw = pd.DataFrame(
        {
            "SEQN": 12345,
            "PAXDAYM": paxday,
            "PAXSSNMP": sample0 + np.arange(minutes) * 4800,
            "PAXTSM": 60,
            "PAXQFM": 0,
            "PAXMTSM": 1.0,
            "PAXPREDM": predictions,
        }
    )

    normalized = normalize_paxmin_timestamps(raw, "22:00:00")
    result = waso_proxy_for_participant(normalized)

    assert result["WASO_valid_nights"] == 1
    assert result["WASO_proxy_min"] == 10
    assert result["WASO_nonwake_nights_excluded_prediction_minutes"] == 2
