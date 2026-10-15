# P01 time-axis verification

## Result

**Time-coordinate validation: PASS.** The bounded source sample agreed with the KAN-171 handoff before the production run. The full streaming run processed all 4,785 cohort participants and confirmed monotonic SEQN ordering, nondecreasing PAXSSNMP, PAXDAYM agreement for timed rows, and zero non-4,800-sample steps.

## Bounded evidence

- Read only the first 30,000 rows of `data/raw/PAXMIN_H.xpt` with `pyreadstat.read_xport(..., row_limit=30000)` and the six requested fields: `SEQN`, `PAXDAYM`, `PAXSSNMP`, `PAXTSM`, `PAXQFM`, `PAXMTSM`. The sample contained three sequential `SEQN`s and was nondecreasing by `SEQN`.
- Read all 8,913 small `PAXHD_H.xpt` header rows for `SEQN,PAXFTIME,PAXSTS`: 7,776 rows had `PAXSTS==1`; all had populated, minute-aligned PAXFTIME values. Eleven used a leading-space, one-digit hour form (for example, `' 9:12:00'`), which is accepted after trimming.
- First sampled participant: `SEQN=73557`, 11,529 minute rows; first `PAXSSNMP=0`, last `PAXSSNMP=55,334,400`, `PAXFTIME='16:30:00'`.
- All 11,528 adjacent PAXSSNMP differences for this participant were 4,800 samples, or exactly 60 seconds at 80 Hz. This is the counter increment, not an inference from row count.
- Normalizing the first PAXMIN row to the header time produced zero `PAXDAYM` mismatches across all 11,529 rows. The observed day counts were day 1: 450 minutes; days 2–8: 1,440 minutes each; day 9: 999 minutes. Day 1→2 occurred at elapsed 27,000 seconds, when the normalized local clock crossed midnight; subsequent day changes aligned likewise.
- The only `PAXTSM != 60` row in this participant was the final row on `PAXDAYM=9`, with `PAXTSM=59`. This confirms the required duration predicate excludes the sampled partial final minute.

## Verified conversion

For each `SEQN`, let `t0` be the clock-of-day seconds represented by that participant's `PAXFTIME`, and let `s0` be `PAXSSNMP` on the first PAXMIN record corresponding to that header anchor. Then:

```text
elapsed_seconds_i = (PAXSSNMP_i - s0) / 80
local_timestamp_seconds_i = t0 + elapsed_seconds_i
local_day_index_i = floor(local_timestamp_seconds_i / 86400)
seconds_since_local_midnight_i = local_timestamp_seconds_i mod 86400
```

`PAXFTIME` has no calendar date; this is a participant-relative local/device-clock timeline, not a cross-participant UTC timestamp. `PAXDAYM` is checked against `1 + local_day_index` for the source's wear-day numbering. The relative local day index and time-of-day are sufficient to align clock blocks and assign `night_id=(SEQN, evening_local_day_index)`.

The bounded sample confirms 4,800-sample increments (60 seconds), including across observed midnight boundaries, despite the codebook note about 1/80 second of raw data missing per minute. Do not adjust the timestamp by counting retained samples; use the starting sample-number coordinate. The production stream later verified monotonic SEQN order and nondecreasing within-participant PAXSSNMP for all 4,785 cohort participants; it observed zero non-4,800-sample steps and no PAXDAYM mismatch for timed rows. Incomplete blocks and timestamp gaps are excluded rather than imputed. Full-run evidence: `outputs/audit/p01_correction_run_manifest.json`; PAXMIN SHA-256: `4b1aec371ebfca456ff2d386bc21444ddcb9ac06fdc8f13c3ed8589413d053bd`.

## Source references

- [NHANES 2013–2014 PAXMIN_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXMIN_H.htm): PAXSSNMP definition, 80-Hz sample-number example, minute duration, and missing-sample note.
- [NHANES 2013–2014 PAXHD_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXHD_H.htm): PAXFTIME definition and first-data-point time format.
