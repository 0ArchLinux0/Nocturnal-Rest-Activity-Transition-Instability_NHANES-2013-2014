# KAN-173 corrected NHANES P01 results

## Status and provenance

KAN-173 completed the corrected downstream rerun. The canonical full cohort contains 4,882 participants, using PAXDAYD 2–8, at least four adequate days, at least 600 valid minutes per day, and at least one adequate weekend day. The prior KAN-172 cache covered 4,785 SEQNs; the cohort-targeted extension added P01 metrics for the 127 newly eligible SEQNs absent from that cache. All 4,882 corrected-cohort participants have finite P01_10m.

PAXMIN_H SHA-256: `4b1aec371ebfca456ff2d386bc21444ddcb9ac06fdc8f13c3ed8589413d053bd`. P01 algorithm: `kan172-p01-clock-contiguity-v1`; method fingerprint: `7c88634eb2f616c565d57bc1e2669c4a3552ae20d1792f10fca30b6992cabe64`. The KAN-172 cache was validated against its source and manifest before extension; the extended cache was validated for the 4,882-person cohort.

The cohort-flow counts are not a strict attrition funnel: source intersections are reported with their explicit definitions.

| Flow measure | N | Definition |
|---|---:|---|
| Historical KAN-172 cohort | 4,785 | Prior complete cohort |
| DEMO ∩ DPQ | 5,924 | Unique SEQN intersection |
| Monitor participants | 7,776 | PAXHD PAXSTS == 1 |
| PHQ-9 complete | 5,372 | All nine DPQ items answered |
| Corrected wear-qualified cohort | 4,882 | PAXDAYD 2–8 and wear rules above |
| Corrected P01_10m available | 4,882 | Finite corrected KAN-172 metric |
| Model 1 complete cases | 4,508 | PHQ9_Score, P01_10m, Age, Sex, BMI |
| Model 2 complete cases | 4,048 | Model 1 plus PIR, weekday sleep hours, WASO_proxy_min |
| Binary ROC complete cases | 4,508 | PHQ9_ge_10, Age, Sex, BMI, P01_10m |

## Outcome and sleep-variable definitions

PHQ9_Score uses exactly DPQ010, DPQ020, DPQ030, DPQ040, DPQ050, DPQ060, DPQ070, DPQ080, and DPQ090. Responses 7/9 and other invalid values are missing; all nine item responses are required. The score is bounded from 0 to 27. `PHQ9_ge_10` is the secondary binary outcome. DPQ100 is excluded.

SLD010H is named `self_reported_weekday_sleep_hours` (77/99 missing), not total sleep time. SLQ060 is named `sleep_disorder_history`; it is not medication use. `WASO_proxy_min` is a PAXPREDM-derived proxy, not validated WASO. Nights combine timestamps from 22:00 on protocol day d through 06:00 on day d+1; nights d=2–7 are retained. Only valid, minute-aligned records with PAXQFM==0, finite nonnegative PAXMTSM, and PAXTSM==60 are eligible. PAXPREDM codes other than wake (1) and sleep (2), including non-wear/unknown, are excluded. For nights with at least 30 known-prediction minutes, the proxy counts wake-coded minutes between the first and last sleep-coded minute; participant values average those night counts.

## Corrected primary models

P01 remains on its raw 0–1 scale. Other continuous covariates are z-scored on each model's complete-case sample; Sex is categorical. Ordinary OLS is shown for replication; HC3 is the preferred non-survey inference.

| Model | N | P01 beta | OLS SE | OLS 95% CI | OLS p | HC3 SE | HC3 95% CI | HC3 p | R² | Adjusted R² | Effect per +0.01 P01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model 1: PHQ9_Score ~ P01_10m + Age + Sex + BMI | 4,508 | 10.12663 | 1.27970 | [7.61779, 12.63547] | 3.13e-15 | 1.41342 | [7.35564, 12.89762] | 9.07e-13 | 0.05412 | 0.05328 | 0.10127 |
| Model 2: Model 1 + PIR + weekday sleep hours + WASO_proxy_min | 4,048 | 5.72278 | 1.52835 | [2.72638, 8.71918] | 1.83e-4 | 1.65435 | [2.47934, 8.96622] | 5.47e-4 | 0.09185 | 0.09027 | 0.05723 |

The immediate `CORRECTED_PRIMARY_CHECKPOINT` was emitted before survey, resolution, and ROC work. Both coefficients remain positive and statistically significant under HC3 relative to the supplied historical references.

## Historical comparison

| Model | Historical beta | Historical SE | Historical p | Corrected beta | Corrected HC3 SE | Corrected HC3 p | Direction preserved | Significance preserved |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Model 1 | 10.707 | 1.277 | <1e-16 | 10.12663 | 1.41342 | 9.07e-13 | Yes | Yes |
| Model 2 | 9.105 | 1.350 | 1.76e-11 | 5.72278 | 1.65435 | 5.47e-4 | Yes | Yes |

Historical sample sizes were not supplied; `historical_N` is therefore unavailable rather than inferred. Corrected results are canonical. Model 2's P01 estimate is about 37.1% lower than the historical reference, but remains positive with a HC3 confidence interval excluding zero.

## Survey sensitivity

R and the `survey` package were available. Both `svydesign`/`svyglm` models ran using WTMEC2YR, SDMVSTRA, and SDMVPSU; the design had 15 degrees of freedom.

| Model | N | P01 beta | SE | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| Model 1 | 4,508 | 10.19673 | 1.91190 | [6.12162, 14.27184] | 2.40e-4 |
| Model 2 | 4,048 | 5.86690 | 2.02549 | [1.54967, 10.18413] | 0.0200 |

These are survey-weighted sensitivity estimates on the wear-qualified analytic sample; they do not make the unweighted HC3 results nationally representative.

## Resolution sensitivity

| Resolution | N | P01 beta | HC3 SE | HC3 95% CI | HC3 p | P01 mean | P01 SD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 min | 4,508 | 11.23806 | 1.49219 | [8.31263, 14.16348] | 6.04e-14 | 0.071515 | 0.047253 |
| 10 min | 4,508 | 10.12663 | 1.41342 | [7.35564, 12.89762] | 9.07e-13 | 0.072512 | 0.050104 |
| 30 min | 4,508 | 7.24560 | 1.15228 | [4.98656, 9.50464] | 3.52e-10 | 0.072020 | 0.061110 |
| 60 min | 4,508 | 4.41545 | 0.97381 | [2.50630, 6.32461] | 5.93e-6 | 0.064619 | 0.070567 |

## Binary-outcome ROC sensitivity

AUCs use out-of-fold predictions from stratified five-fold CV (`random_state=173`). Confidence intervals and paired delta intervals use 1,000 stratified bootstrap resamples of the out-of-fold predictions.

| Model | N | OOF AUC | AUC 95% CI | ΔAUC vs baseline | Paired ΔAUC 95% CI |
|---|---:|---:|---:|---:|---:|
| Age + Sex + BMI | 4,508 | 0.61328 | [0.58394, 0.64143] | 0 | [0, 0] |
| P01_10m only | 4,508 | 0.59089 | [0.56396, 0.62124] | −0.02239 | [−0.05880, 0.01915] |
| Age + Sex + BMI + P01_10m | 4,508 | 0.63152 | [0.60277, 0.65895] | 0.01824 | [0.00823, 0.03001] |

Adding P01 to demographics increased the cross-validated AUC by 0.01824; the paired bootstrap interval excludes zero. P01 alone did not outperform the demographic baseline.

## Scientific impact and outputs

**Classification: ATTENUATED_BUT_SUPPORTED.** Both corrected primary HC3 estimates are positive and statistically significant, with the more adjusted Model 2 materially attenuated relative to its historical reference. The conclusion is supported in direction and significance, but the historical Model 2 effect size is not reproduced.

Machine-readable results are in `outputs/models/primary_models.csv`, `primary_models_hc3.csv`, `p01_primary_old_vs_corrected.csv`, `cohort_flow.csv`, `survey_weighted_models.csv`, `resolution_sensitivity.csv`, `roc_corrected.csv`, and `roc_corrected_oof_predictions.csv`. Run provenance is in `kan173_primary_state.json` and `kan173_run_manifest.json`. No manuscript text or figures were changed.
