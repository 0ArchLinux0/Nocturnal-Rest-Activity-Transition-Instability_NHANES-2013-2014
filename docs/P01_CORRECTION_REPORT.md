# KAN-172 P01 correction and full-data comparison

## Status and scope

- KAN-172 full PAXMIN run: **PASS**. The existing `final_full_cohort.csv` cohort was processed with the unchanged inclusion rules and PAXDAYM 1–9 analysis window.
- Processed 4,785 cohort participants and 19,140 participant-resolution rows using a bounded reader with 400,000-row chunks. Participant records were carried across chunk boundaries; the full XPT was not loaded into memory.
- This report compares the audited legacy P01 implementation with the corrected P01 implementation on the same participant rows and day window. It does not fit outcome models or assess statistical significance.
- No OLS, logistic, or ROC analysis was run. No publication figure or manuscript was generated or changed. WASO, PAXHR, cohort criteria, survey weighting, outcome resolution, labels, and downstream inference remain outside KAN-172.

## Corrected temporal definition

For each participant, local time is anchored to PAXFTIME and elapsed time is derived from PAXSSNMP at 80 Hz:

```text
local_timestamp_ticks = PAXFTIME_seconds * 80 + (PAXSSNMP - first_participant_sample)
```

The stream verifies nondecreasing SEQN order. The P01 engine rejects a decreasing or non-integer PAXSSNMP axis, requires the first sample number to anchor PAXFTIME, and checks PAXDAYM against the derived local day for timed rows. The full run found no PAXDAYM mismatch and zero non-4,800-sample steps across the cohort participants.

A valid minute requires `PAXQFM == 0`, finite `PAXMTSM >= 0`, and `PAXTSM == 60`. Clock-aligned, non-overlapping 5-, 10-, 30-, and 60-minute blocks are retained only when all expected one-minute observations are present and valid. Partial blocks are excluded without imputation. Each participant's threshold is the median activity sum over all complete blocks in PAXDAYM 1–9, separately for each resolution and across all clock times.

The nocturnal window assigns 22:00–24:00 on day *d* and 00:00–06:00 on day *d+1* to one `night_id`. A valid edge requires exact block-start adjacency at the selected resolution. The numerator counts eligible state 0→1 edges; the denominator counts all eligible edges whose current block is state 0. Counts are pooled across nights. A zero denominator yields NaN.

The comparison-only legacy implementation retains the audited row-ordinal, partial-block, historical QC, and same-day night ordering behavior, including its 200-minute gate. Both calculations use the same participants and PAXDAYM 1–9 source rows.

## Legacy-versus-corrected summary

All participants had finite values under both definitions at every resolution (`N_old = N_corrected = N_paired = 4,785`; missing counts were zero). Deltas are corrected minus legacy.

| Resolution | Legacy mean | Legacy median | Corrected mean | Corrected median | Mean delta | Median delta | Mean absolute delta | Pearson r | Spearman r |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 0.080882 | 0.069374 | 0.072061 | 0.060065 | −0.008821 | −0.008535 | 0.009148 | 0.991763 | 0.985910 |
| 10 min | 0.086684 | 0.074627 | 0.073191 | 0.060606 | −0.013493 | −0.013093 | 0.013800 | 0.984837 | 0.971604 |
| 30 min | 0.104369 | 0.090000 | 0.072943 | 0.057692 | −0.031426 | −0.029121 | 0.032053 | 0.937603 | 0.881271 |
| 60 min | 0.119927 | 0.108108 | 0.065854 | 0.044444 | −0.054073 | −0.048929 | 0.055823 | 0.823030 | 0.730582 |

For the primary 10-minute comparison: `N_old=4,785`, `N_corrected=4,785`, `N_paired=4,785`; legacy mean `0.086684`; corrected mean `0.073191`; mean delta `−0.013493`; median delta `−0.013093`; mean absolute delta `0.013800`; Pearson r `0.984837`.

## Exclusion and edge diagnostics

Manifest totals are across resolutions except minute counts, which are unique participant-minute rows from the 10-minute results:

- False same-day daylight edges identified in the legacy comparator and removed by the corrected night grouping: **133,532** across all resolutions.
- Non-adjacent nocturnal block breaks recorded by corrected processing: **1,762** across all resolutions.
- Incomplete resolution blocks excluded: **48,001** across all resolutions.
- QC-invalid minutes (`PAXQFM` missing or nonzero): **30,715**; missing QC flags: **0**.
- Uncomputed/non-finite or negative MIMS minutes: **430**.
- Minutes with `PAXTSM != 60`: **4,544**.
- Missing-timestamp minutes: **0**; non-4,800-sample steps: **0**.

The exclusion categories can overlap. They are descriptive counts, not disjoint participant totals.

## Provenance and outputs

- PAXMIN: `data/raw/PAXMIN_H.xpt`; SHA-256 `4b1aec371ebfca456ff2d386bc21444ddcb9ac06fdc8f13c3ed8589413d053bd`.
- PAXHD: `data/raw/PAXHD_H.xpt`; participant PAXFTIME values were used as the clock anchor.
- Cohort: `final_full_cohort.csv`; sorted-SEQN cohort fingerprint `93385b62e138ab32c8b0d60a9f9904beb473b2ea5641ccfe9fdb48e16cca01ae`.
- Algorithm version: `kan172-p01-clock-contiguity-v1`.
- Method fingerprint: `7c88634eb2f616c565d57bc1e2669c4a3552ae20d1792f10fca30b6992cabe64`.
- Participant comparison: `outputs/audit/p01_old_vs_corrected_participant.csv`.
- Resolution summary: `outputs/audit/p01_old_vs_corrected_summary.csv`.
- Run manifest: `outputs/audit/p01_correction_run_manifest.json` (`status=complete`; stream validations passed; downstream model flags are false).
- Corrected production cache: `outputs/final_cohort_pam_metrics.csv`. It contains `SEQN`, corrected P01 at all four resolutions, algorithm version, PAXMIN SHA-256, and method fingerprint. Cache provenance validation returned `matched`.

## Disposition and next work

Corrected P01 means are lower at all four resolutions, with the largest mean change at 60 minutes. The changes are sufficiently material that prior P01-dependent results are **potentially invalidated** until the scoped KAN-173 downstream reanalysis is prespecified and run. This comparison does not estimate model impact or significance. KAN-173 is the next Jira item; it must address the remaining audited methods and rerun only the analyses approved there.

Synthetic tests passed: **13 passed** across `tests/test_p01_contiguity.py` and `tests/test_p01_boundaries.py`. Full suite: **14 passed** (`pytest -q`).
