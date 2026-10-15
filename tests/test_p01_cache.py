import pandas as pd

from nhanes_p01.p01 import (
    P01_ALGORITHM_VERSION,
    P01_METHOD_FINGERPRINT,
    validate_p01_cache,
)


def test_cache_requires_exact_algorithm_source_method_and_cohort(tmp_path):
    path = tmp_path / "cache.csv"
    cache = pd.DataFrame(
        {
            "SEQN": [10, 20],
            "P01_10m": [0.1, 0.2],
            "p01_algorithm_version": [P01_ALGORITHM_VERSION] * 2,
            "paxmin_sha256": ["a" * 64] * 2,
            "method_fingerprint": [P01_METHOD_FINGERPRINT] * 2,
        }
    )
    cache.to_csv(path, index=False)

    assert validate_p01_cache(path, {10, 20}, "a" * 64) == (True, "matched")
    assert validate_p01_cache(path, {10, 30}, "a" * 64)[0] is False
    assert validate_p01_cache(path, {10, 20}, "b" * 64)[0] is False

    cache.loc[0, "method_fingerprint"] = "stale"
    cache.to_csv(path, index=False)
    valid, reason = validate_p01_cache(path, {10, 20}, "a" * 64)
    assert not valid
    assert reason == "method fingerprint does not match"

    cache.loc[0, "method_fingerprint"] = P01_METHOD_FINGERPRINT
    cache.loc[0, "paxmin_sha256"] = None
    cache.to_csv(path, index=False)
    valid, reason = validate_p01_cache(path, {10, 20}, "a" * 64)
    assert not valid
    assert reason == "PAXMIN SHA-256 does not match"

    cache.loc[0, "paxmin_sha256"] = "a" * 64
    cache["SEQN"] = cache["SEQN"].astype(float)
    cache.loc[0, "SEQN"] = 10.5
    cache.to_csv(path, index=False)
    valid, reason = validate_p01_cache(path, {10, 20}, "a" * 64)
    assert not valid
    assert reason == "cache SEQN values are not finite integers"
