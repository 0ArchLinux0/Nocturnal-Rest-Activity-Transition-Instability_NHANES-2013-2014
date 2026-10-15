# KAN-171 forensic audit — active NHANES P01 pipeline

## Audit boundary and evidence

- Preflight passed in the bound repository on `main` at `932acf3653255be1d4d10058f99910ae49460b1f`; the starting worktree was clean.
- Reviewed the five active scripts named in KAN-171, `docs/SCRIPT_AUTHORITY_MAP.md`, `docs/DATA_PROVENANCE.md`, `config/variables.yaml`, and `sources/manifest.yaml`.
- Inspected all required XPT schemas through metadata-only reads. `PAXMIN_H.xpt` was inspected with `pyreadstat.read_xport(..., metadataonly=True)`; no PAXMIN record values were materialized, and no P01 calculation was run.
- Verified official CDC/NCHS codebook descriptions for PAXMIN, PAXDAY, PAXHD, DEMO, DPQ, and SLQ. The codebook evidence and required source fields are recorded in `outputs/audit/source_variable_audit.csv`.
- No cohort regeneration, full PAXMIN analysis, final models, ROC calculation, publication figure, manuscript edit, or scientific result change occurred. No numerical model/AUC result is asserted here.
- The named root `final_full_cohort.csv`, `outputs/final_cohort_pam_metrics.csv`, historical resolution input `outputs/processed_data_physics_ultimate.csv`, and root `PAXMIN_H.xpt` were absent at audit time; the canonical `data/raw/PAXMIN_H.xpt` exists.

## Current pipeline behavior

### Cohort and PHQ-9

`build_final_full_cohort.py` joins `DEMO_H`, `DPQ_H`, `PAXHD_H`, and `PAXDAY_H` on `SEQN`, retains participants with `PAXSTS == 1` when that field exists, and applies its wear-day gate. It computes `PHQ9_Score` from exactly `DPQ010`–`DPQ090`: 7 and 9 become missing, all nine items are required (`min_count=9`), the result is clipped to 0–27, and `depression_suspected` is 1 at `PHQ9_Score >= 10` when the score is present. This primary definition matches the official nine-item PHQ-9; `DPQ100` is a separate functional-impairment follow-up.

The cohort wear gate is 4 adequate rows among `PAXDAYD` 1–7, with at least one adequate weekend row; an adequate row has `PAXVMD >= 600` and `PAXQFD < 0.5`. The source protocol reports nine wear-day indices, with the first and last data-collection days partial and seven full days between them.

### P01 from PAXMIN to `P01_10m`

1. `minute_frame` sorts by `PAXDAYM, PAXSSNMP`, converts `PAXDAYM`, then sets `_m = groupby(PAXDAYM).cumcount()`. `_m` is therefore a row ordinal reset to zero for each wear-day, not a timestamp derived from `PAXSSNMP`. The CDC codebook defines `PAXSSNMP` as the starting 80-Hz sample number; e.g. 9600 means 120 seconds since PAM measurement began. The header also contains `PAXFTIME`, the first-data-point clock time. Neither timestamp source is used to derive `_m` or test continuity.
2. Validity filtering occurs after `_m` is assigned. The code accepts nonmissing `PAXQFM` except values 1 and 2, and accepts nonmissing `PAXMTSM >= 0`. The official PAXMIN codebook says `PAXQFM == 0` means no QC flags and every value `> 0` means the minute is invalid; its range is 0–14. Thus values 3–14 are wrongly retained. `PAXMTSM == -0.01` is correctly excluded by the nonnegative check, but `PAXTSM` (seconds with data, 3–60) is not read or checked.
3. `resample_sum` assigns `_b = _m // width` and sums the available `PAXMTSM` rows per `(PAXDAYM, _b)`. It does not require 10 valid minutes for a 10-minute block (or `width` valid minutes at other resolutions), and it includes partially observed block sums in both the activity threshold and the transition sequence.
4. `p01_10m_only` has participant-wide guards (`>=200` retained minute rows; `>=10` nonempty blocks; `>=4` selected nocturnal blocks). Its threshold is the median of `all_act`: all nonempty full-day-time blocks across the participant's recorded days, not a day-specific or nocturnal median. Because partial blocks are retained and days 1–9 are not restricted, it is not strictly a median of complete full-day blocks.
5. Nocturnal block selection uses the day-clock ranges `[00:00, 06:00)` and `[22:00, 24:00)`. The selected blocks are sorted by `(PAXDAYM, _b)` and flattened into one array. Within each wear-day this orders 00:00–06:00 before 22:00–24:00. Consequently, 05:50→22:00 within the same day is treated as a neighboring pair despite the 16-hour gap. 23:50 on day `d`→00:00 on day `d+1` may be physically contiguous, but the code counts it without checking the sample/time difference. Missing blocks, nonconsecutive wear days, and missing source rows can likewise be bridged or shift the row ordinal.
6. `p01_from_sequence` sets state 0 for activity `<= threshold` and state 1 above it. It counts 0→1 changes between every consecutive array element and divides by the number of 0 states that have an array successor. There is no explicit condition equivalent to `next_timestamp - current_timestamp == epoch_width`.
7. `nhanes_paxmin_resolution_instability.py` applies the same row-cumcount, block-sum, median, and flattened nocturnal-sequence design at 5, 10, 30, and 60 minutes. It does not establish temporal adjacency either.

Transition classes in the current sequence:

- Potentially valid: adjacent blocks within one continuous nocturnal segment, if both source epochs are complete and actually contiguous.
- Artificial by construction: 05:50→22:00 within a wear-day.
- Conditionally valid but unverified: 23:50→next-day 00:00; the code assumes adjacency from sorted order rather than checking it.
- Artificial when a gap exists: neighboring array entries that straddle an omitted block, absent source row, invalid-data gap, or missing wear-day. Partially valid blocks are also treated as full-width states.

### WASO and hourly night summaries

`waso_proxy_nights` groups PAXMIN rows by the same `PAXDAYM` and only processes wear-days 1–7. Within each group it selects both 00:00–06:00 and 22:00–24:00 from that same calendar/wear-day. It requires 30 selected rows and at least one `PAXPREDM == 2`; it uses the first and last predicted-sleep positions and counts `PAXPREDM == 1` between them, then averages the resulting counts across processed wear-days. Official PAXPREDM codes are 1 wake wear, 2 sleep wear, 3 non-wear, and 4 unknown. Codes 3/4 are not counted as wake, but they do not split the interval. This is an algorithm-derived actigraphy proxy, not a PSG-measured or participant-validated WASO value.

`aggregate_paxhr_sleep` independently groups PAXHR rows by `PAXDAYH`, uses row `cumcount` rather than `PAXSSNHP`, selects hour indices 00–05 and 22–23 from each same-day group, fills missing wake/sleep counts with zero, and sums those minutes across wear-days 1–7 into `Night_Wake_Min` and `Night_Sleep_Min`. These are not coherent overnight summaries; they are seven-day sums of two disjoint same-day windows. Both variables enter the paper-analysis composite ROC.

### OLS, ROC, survey design, and paths

- OLS outcome: continuous `PHQ9_Score`, clipped to 0–27. Model 1 predictors: raw P01, z-scored Age and BMI, raw RIAGENDR code. Model 2 adds z-scored PIR, TST, and WASO. Each model has its own complete-case sample: M1 requires outcome/P01/Age/Sex/BMI; M2 additionally requires PIR/TST/WASO. Z-scores use each model's own complete-case rows and population SD (`ddof=0`). P01 is not z-scored: its coefficient is PHQ-9 points per +1.00 probability unit; a +0.01 P01 difference corresponds to `0.01 × beta_P01` PHQ-9 points. This is a unit conversion, not an observed effect estimate. `sm.OLS(...).fit()` uses statsmodels' default ordinary nonrobust covariance; it is not HC3. No weights, strata, or PSU are passed. R² and adjusted R² are `m.rsquared` and `m.rsquared_adj`.
- `final_cohort_paper_analysis.py` logistic outcome is `depression_suspected`; its base complete-case model requires Depression, P01, Age, Sex, BMI, INDFMPIR, and `Sleep_Medication`. The ROC sample further requires P01, WASO proxy, hourly Night_Wake/Night_Sleep, and SLD010H. It reports in-sample AUCs and plots in-sample ROC curves for P01-only versus the five-feature composite. It also genuinely executes stratified 5-fold CV with fold-specific fitting and training-only composite scaling, then reports mean/SD AUC; this CV does not produce the plotted ROC. No AUC CI or model-difference test is computed.
- `plot_roc_depression_three_models.py` outcome is `depression_suspected`. One shared complete-case sample requires y, P01, Age, Sex, and BMI. The three models are P01 only; Age+Sex+BMI; and P01+Age+Sex+BMI. Every model is fitted and predicted on that same sample, so all three AUCs are in-sample. There is no CV, CI, or inferential AUC/model comparison.
- DEMO schema contains `WTINT2YR`, `WTMEC2YR`, `SDMVSTRA`, and `SDMVPSU`. The official PAXMIN analytic note directs users to use NHANES examined-sample weights for PAM analyses; DEMO labels WTMEC2YR as the full-sample 2-year MEC examination weight. Thus WTMEC2YR is the supported candidate weight, with SDMVSTRA/SDMVPSU for design-based variance. No separate PAX-specific weight is named in the retrieved PAXMIN codebook. The official source does not resolve whether an additional nonresponse/valid-wear adjustment is needed for this post-MEC PAX analytic subset; send that exact question to KAN-173. Active models use none of these design variables.
- None of the five active scripts defaults to `data/raw/`. `build_final_full_cohort.py` defaults to `nhanes_2013_2014_raw/` (or `NHANES_RAW_DIR`) for DEMO/DPQ/PAXDAY/PAXHD. `final_cohort_paper_analysis.py` uses that legacy directory for PAXHR/DEMO/BMX/SLQ and falls back from PAXMIN there to a project-root `PAXMIN_H.xpt`. OLS uses the legacy directory for DEMO/BMX/SLQ. The three-model ROC uses it for BMX. The resolution script uses the legacy directory for PAXMIN/DPQ/BMX/DEMO, falls back to project-root PAXMIN and to project-root DPQ/BMX, and defaults its cohort input to the historical `outputs/processed_data_physics_ultimate.csv`. If that cohort file is absent, it silently selects at most the first 200 SEQNs from the first PAXMIN chunk for its resolution sensitivity run. The canonical `data/raw/PAXMIN_H.xpt` is present, while the checked legacy/root paths and historical CSV were absent.

## Findings

### CRITICAL

#### K171-F01 — P01 time reconstruction and adjacency are not valid

- **Severity:** CRITICAL
- **File/function:** `final_cohort_paper_analysis.py:72–80, 114–119, 139–153`; `nhanes_paxmin_resolution_instability.py:79–97, 100–124`.
- **Actual current behavior:** `PAXSSNMP` is used only as a sort key; each `PAXDAYM` group starts `_m` at zero by `cumcount`. The code selects 00–06 and 22–24, sorts by wear-day/block, and flattens these windows. It has no source timestamp and no exact-epoch adjacency test. Days 1–9 are included despite the first/last days being partial.
- **Why it matters:** A 05:50→22:00 gap is counted as a transition; absent rows can compress the clock; missing blocks and nonconsecutive days can be bridged. A 23:50→next-day 00:00 pair is accepted without proving continuity. The requested `next_timestamp - current_timestamp == epoch_width` condition does not exist.
- **Reported result that may be affected:** `P01_10m`, all 5/10/30/60-minute P01 sensitivity metrics, and downstream P01 coefficients, logistic associations, and ROC AUCs. No numerical result was recomputed or observed in this audit.
- **Required correction:** KAN-172 must derive time from the official 80-Hz sample coordinate plus header start-time anchor, assign an overnight `night_id`, and count only adjacent complete epochs at the exact requested width. Do not treat sorted array neighbors as proof of temporal adjacency.
- **Target Jira:** KAN-172.

#### K171-F02 — Invalid and incomplete P01 epochs enter thresholding and transitions

- **Severity:** CRITICAL
- **File/function:** `final_cohort_paper_analysis.py:77–79, 114–119, 139–153`; `nhanes_paxmin_resolution_instability.py:85–97, 109–123`.
- **Actual current behavior:** The valid-minute predicate excludes only `PAXQFM` values 1 and 2, although official PAXMIN documentation says every value above zero is invalid and the field ranges to 14. Block sums include any available rows; there is no `width`-minute completeness check, and `PAXTSM` is not loaded despite partial-minute durations. Entirely absent blocks disappear before the sequence is evaluated.
- **Why it matters:** QC-invalid minutes with flags 3–14 can affect activity sums and medians. Partial blocks are treated as full-width states. Empty blocks are then bridged by the flattened sequence. Thresholds, states, and transitions no longer represent comparable fixed-duration epochs.
- **Reported result that may be affected:** P01 at 10 minutes and every 5/10/30/60-minute sensitivity result; any model or ROC using those values.
- **Required correction:** Use `PAXQFM == 0`; reject MIMS-uncomputed values; require full one-minute records and exactly `width` valid contiguous minute epochs per block; exclude incomplete blocks and never bridge their gaps. Report exclusions and completeness by participant/resolution.
- **Target Jira:** KAN-172.

### MAJOR

#### K171-F03 — WASO proxy combines morning and evening of the same day

- **Severity:** MAJOR
- **File/function:** `final_cohort_paper_analysis.py:84–111`.
- **Actual current behavior:** `waso_proxy_nights` groups by `PAXDAYM` 1–7 and selects 00–06 plus 22–24 from that same day. The first and last `PAXPREDM == 2` positions bound the count of `PAXPREDM == 1`; codes 3 (non-wear) and 4 (unknown) are neither wake nor segment breaks. The official variable is machine-learning predicted wear status, not PSG scoring.
- **Why it matters:** A calendar-day vector is not one overnight interval: it orders early morning before late evening and does not join 22:00 on day d to 00:00–06:00 on day d+1. The reported variable is a derived proxy, not validated WASO.
- **Reported result that may be affected:** `WASO_proxy_min` in OLS Model 2 and the paper-analysis composite ROC.
- **Required correction:** Define night IDs from the anchored clock, attach 00:00–06:00 to the preceding evening, keep the proxy label, and explicitly handle non-wear/unknown/missing segments. Do not claim PSG-validated WASO without validation evidence.
- **Target Jira:** KAN-173.

#### K171-F04 — PAXHR `Night_*` variables are same-day multi-day sums

- **Severity:** MAJOR
- **File/function:** `final_cohort_paper_analysis.py:204–219`.
- **Actual current behavior:** The hourly path makes `_h` by row `cumcount` within `PAXDAYH`, not `PAXSSNHP`; selects row indices 00–05 and 22–23 on the same wear-day; fills missing hour values with zero; then sums those rows across days 1–7 into `Night_Wake_Min` and `Night_Sleep_Min`.
- **Why it matters:** These are not coherent overnight or per-night measures. The first partial day can also be shifted because the first observed hour is treated as hour zero. Both variables are ROC composite predictors.
- **Reported result that may be affected:** The composite ROC model and its AUC comparison in `final_cohort_paper_analysis.py`.
- **Required correction:** Reconstruct hourly timestamps from the source sample coordinate, form overnight IDs across midnight, preserve missingness/coverage, and define whether outputs are per-night or participant-level summaries.
- **Target Jira:** KAN-173.

#### K171-F05 — Resolution-regression PHQ score includes DPQ100 and partial sums

- **Severity:** MAJOR
- **File/function:** `nhanes_paxmin_resolution_instability.py:159–180, 285–322`.
- **Actual current behavior:** Generic discovery includes every `DPQ` variable with a three-digit suffix, including present `DPQ100`. The code removes 7/9 and calls `.sum(axis=1)` without `min_count=9`, so the outcome includes a non-PHQ-9 impairment follow-up, partial item sums, and zero when all detected items are missing.
- **Why it matters:** Official DPQ documentation defines the 9-item score as 0–27 and identifies DPQ100 as a separate functional-impairment follow-up. The regression outcome is not PHQ-9.
- **Reported result that may be affected:** Only the optional resolution sensitivity regressions written to `outputs/paxmin_resolution_phq9_regression.txt` for P01_5m, P01_10m, P01_30m, and P01_60m. The primary cohort PHQ-9 definition is not affected.
- **Required correction:** Use the explicit DPQ010–DPQ090 list, map 7/9 to missing, require all nine answered, and retain 0–27 bounds.
- **Target Jira:** KAN-173.

#### K171-F06 — Cohort wear window is not the seven full protocol days

- **Severity:** MAJOR
- **File/function:** `build_final_full_cohort.py:39–45, 111–135, 215–218`.
- **Actual current behavior:** The builder applies its four-day/600-minute/weekend gate to `PAXDAYD` 1–7. Official PAM protocol documentation says the first and last data-collection days are partial and describes seven full days between them (wear-day indices 2–8).
- **Why it matters:** The current rule can count a sufficiently long partial day 1 while omitting full day 8, changing who enters the final cohort and every downstream analysis. This is a mismatch requiring an explicit analytic-window decision, not an observed sample-size difference.
- **Reported result that may be affected:** Cohort membership, PHQ-9 prevalence, P01 models, and both ROC analyses.
- **Required correction:** Prespecify the valid-day window. If the intended rule is seven full protocol days, use days 2–8 with the documented thresholds; if days 1–7 are intentional, state and justify that distinct rule. Also document whether any positive `PAXQFD` invalidates an entire day when `PAXVMD` already counts unflagged valid minutes.
- **Target Jira:** KAN-173.

#### K171-F07 — Self-reported weekday sleep hours are aliased as TST

- **Severity:** MAJOR
- **File/function:** `final_cohort_phq9_ols_models.py:58–62, 122–128`; `final_cohort_paper_analysis.py:234–245, 317`.
- **Actual current behavior:** `SLD010H` is renamed `TST` in OLS. The official SLQ codebook says the question is how much sleep the respondent usually gets at night on weekdays/workdays; it is self-reported hours, not actigraphy-derived total sleep time. The paper ROC uses SLD010H directly.
- **Why it matters:** The `TST` label can be read as objectively measured total sleep time, which is a different construct. The OLS text does disclose the self-report wording, but the variable/formula label remains ambiguous.
- **Reported result that may be affected:** OLS Model 2 sleep covariate interpretation and the composite ROC sleep predictor.
- **Required correction:** Rename/document it as self-reported usual weekday sleep duration (hours); do not call it actigraphy-derived TST.
- **Target Jira:** KAN-173.

#### K171-F08 — Sleep-disorder history is named Sleep_Medication

- **Severity:** MAJOR
- **File/function:** `final_cohort_paper_analysis.py:239–245, 277–305`.
- **Actual current behavior:** `Sleep_Medication` is 1/0 from SLQ060, “Ever told by doctor have sleep disorder?”; codes 7/9 become missing. The script has no RXQ medication input. Its prose calls this a proxy, but the model variable and formula still call it medication.
- **Why it matters:** Diagnosis/history of a sleep disorder is not medication use. The covariate and complete-case filter can be interpreted as medication adjustment when no medication variable was measured.
- **Reported result that may be affected:** Adjusted logistic coefficients and the complete-case sample used for the paper-analysis ROC.
- **Required correction:** Rename and describe the covariate as physician-reported sleep-disorder history; do not state or imply prescription/medication status.
- **Target Jira:** KAN-173.

#### K171-F09 — OLS uses classical nonrobust covariance

- **Severity:** MAJOR
- **File/function:** `final_cohort_phq9_ols_models.py:78–80, 101–110`.
- **Actual current behavior:** `sm.OLS(...).fit()` is called without `cov_type`; statsmodels therefore uses its ordinary nonrobust covariance, not HC3. The report emits its standard errors and p-values.
- **Why it matters:** OLS inferential quantities rely on the classical covariance assumptions and are not heteroskedasticity-robust or survey-design-corrected. This audit does not establish whether those assumptions hold or quantify any change.
- **Reported result that may be affected:** OLS standard errors, p-values, and significance statements for P01 and covariates.
- **Required correction:** KAN-173 must prespecify an inference method appropriate to the final NHANES design and report it explicitly; do not label the current default robust.
- **Target Jira:** KAN-173.

#### K171-F10 — Active models ignore NHANES weights and design variables

- **Severity:** MAJOR
- **File/function:** `final_cohort_phq9_ols_models.py:78–99`; `final_cohort_paper_analysis.py:302–305, 327–340, 351–373`; `plot_roc_depression_three_models.py:53–66`.
- **Actual current behavior:** All active fitted OLS/Logit models are unweighted and pass no strata or PSU. DEMO contains `WTINT2YR`, `WTMEC2YR`, `SDMVSTRA`, and `SDMVPSU`; the official PAXMIN analytic note directs use of NHANES examined-sample weights for PAM analyses.
- **Why it matters:** Point estimates and uncertainty are not adjusted to the NHANES sample design; the current results cannot be described as design-based population inference.
- **Reported result that may be affected:** All OLS coefficients/p-values, adjusted logistic estimates, and ROC training/validation summaries if interpreted for the U.S. population.
- **Required correction:** KAN-173 should use the supported MEC candidate `WTMEC2YR` with `SDMVSTRA` and `SDMVPSU` for design-based inference. Resolve before fitting whether PAX participation/valid-wear selection needs additional nonresponse adjustment. Explicit KAN-173 question: does official 2013–2014 PAM guidance prescribe an adjustment beyond WTMEC2YR for respondents with available PAM data who pass the analytic wear/PHQ-9 filters? The PAXMIN codebook names no separate PAM weight; do not infer that no adjustment is needed.
- **Target Jira:** KAN-173.

#### K171-F11 — Paper ROC plot is in-sample despite actual CV calculations

- **Severity:** MAJOR
- **File/function:** `final_cohort_paper_analysis.py:316–377, 395–428`.
- **Actual current behavior:** The ROC sample is the base logistic complete-case set plus nonmissing P01, WASO proxy, hourly Night_Wake/Night_Sleep, and SLD010H. The script reports in-sample AUC for P01-only and the five-feature composite and plots those in-sample curves. It also genuinely runs stratified 5-fold CV and reports fold mean/SD AUC, with training-only composite scaling; the CV results do not feed the plotted curves.
- **Why it matters:** The figure's AUCs are training-set performance. No AUC confidence interval or model-difference test is calculated; comparison is descriptive.
- **Reported result that may be affected:** `outputs/roc_p01_vs_composite.png` and its displayed AUC comparison; the text report's in-sample AUC comparison.
- **Required correction:** Label the plot in-sample or plot held-out predictions; report uncertainty and a prespecified model-comparison method before claiming an AUC difference. Keep the actual CV path distinct from in-sample output.
- **Target Jira:** KAN-173.

#### K171-F12 — Three-model ROC has no held-out prediction path

- **Severity:** MAJOR
- **File/function:** `plot_roc_depression_three_models.py:49–102`.
- **Actual current behavior:** A common complete-case sample requires outcome, P01, Age, Sex, and BMI. P01-only, Age+Sex+BMI, and combined models are fit and predicted on those same rows. AUCs are in-sample; there is no cross-validation, confidence interval, or model-difference test.
- **Why it matters:** The ROC curves estimate training performance and do not provide validated predictive discrimination or inferential evidence that one model is better.
- **Reported result that may be affected:** `outputs/roc_comparison.png` and its three displayed AUCs.
- **Required correction:** Use held-out or cross-validated predictions and report uncertainty/model comparison; otherwise label AUCs explicitly as apparent in-sample performance.
- **Target Jira:** KAN-173.

#### K171-F13 — Paper-analysis narrative asserts significance unconditionally

- **Severity:** MAJOR
- **File/function:** `final_cohort_paper_analysis.py:379–390`.
- **Actual current behavior:** The report always appends “All adjusted logistic coefficients above were significant” and “Higher P01_10m associates with higher odds” without checking the fitted p-values or coefficient sign. The following AUC comparison is conditional, but these two sentences are not.
- **Why it matters:** The generated report can state unsupported significance and direction regardless of the actual fitted model.
- **Reported result that may be affected:** The automatic interpretation in `final_paper_results.txt`.
- **Required correction:** Generate each inference sentence only from the corresponding coefficient, direction, and prespecified inferential result; otherwise omit it.
- **Target Jira:** KAN-173.

#### K171-F14 — Active raw-data paths do not use the canonical `data/raw/`

- **Severity:** MAJOR
- **File/function:** Path constants/loaders in all five active scripts.
- **Actual current behavior:** None defaults to `data/raw/`. The cohort builder defaults to `nhanes_2013_2014_raw/` or `NHANES_RAW_DIR`; paper analysis defaults there and has a project-root PAXMIN fallback; OLS uses the legacy directory for DEMO/BMX/SLQ; three-model ROC uses it for BMX; resolution sensitivity uses the legacy directory, project-root fallbacks for PAXMIN/DPQ/BMX, and the historical `outputs/processed_data_physics_ultimate.csv` input. If that historical CSV is absent, the resolution script falls back to the first 200 SEQNs in the first PAXMIN chunk. The current canonical `data/raw/` files therefore are not consumed by defaults.
- **Why it matters:** A rerun from this repository is not reproducible from the acquired canonical source tree without environment overrides or old root-level/historical files; resolution sensitivity may silently analyze a first-chunk sample rather than the cohort.
- **Reported result that may be affected:** Cohort membership, P01 cache, resolution sensitivity outputs, models, and ROC results.
- **Required correction:** Migrate every active source loader to `data/raw/`; replace the historical resolution cohort dependency with the canonical cohort artifact and make any sample cap explicit. Keep the optional `PAXRAW_*.xpt` restriction disabled unless its cycle/source is verified.
- **Target Jira:** KAN-172.

#### K171-F15 — PAM cache can bypass corrected P01 code

- **Severity:** MAJOR
- **File/function:** `final_cohort_paper_analysis.py:156–168`.
- **Actual current behavior:** If `outputs/final_cohort_pam_metrics.csv` contains every requested cohort SEQN, `load_or_stream_pam_metrics` returns cached values without checking source hash, algorithm version, or cache provenance. The cache was absent at this audit, so this conditional path was not used here.
- **Why it matters:** After KAN-172 changes P01, an old cache can silently preserve the defective values and feed downstream models/ROCs.
- **Reported result that may be affected:** `P01_10m`, `WASO_proxy_min`, OLS, logistic, and ROC outputs on future reruns with an existing cache.
- **Required correction:** Invalidate the old cache during KAN-172 cutover or key it to source checksum and algorithm version; never accept a cache solely because SEQN coverage matches.
- **Target Jira:** KAN-172.

### MINOR

No additional minor-severity finding. All recorded findings are classified CRITICAL or MAJOR.

## Official source references

- [NHANES 2013–2014 PAXMIN_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXMIN_H.htm): PAXSSNMP 80-Hz sample-number meaning; PAXPREDM codes; PAXQFM invalidity; minute data duration; PAM analytic note on examined-sample weights; protocol and partial wear-days.
- [NHANES 2013–2014 PAXHD_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXHD_H.htm): PAXFTIME, PAXFDAY, and partial-day protocol metadata.
- [NHANES 2013–2014 PAXDAY_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXDAY_H.htm): PAXDAYD, PAXDAYWD, PAXVMD, and PAXQFD.
- [NHANES 2013–2014 PAXHR_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXHR_H.htm): PAXSSNHP and PAXWWMH/PAXSWMH hour summary definitions.
- [NHANES 2013–2014 DEMO_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.htm): WTMEC2YR, WTINT2YR, SDMVSTRA, and SDMVPSU.
- [NHANES 2013–2014 DPQ_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DPQ_H.htm): nine-item score range and DPQ100 impairment follow-up.
- [NHANES 2013–2014 SLQ_H codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SLQ_H.htm): usual weekday/workday sleep question and SLQ060 sleep-disorder question.
