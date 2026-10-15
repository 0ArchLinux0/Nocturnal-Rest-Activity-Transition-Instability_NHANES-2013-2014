# Script authority map — KAN-173 execution overlay and KAN-169 baseline

## Current KAN-173 authority

`kan173_corrected_analysis.py` is the canonical KAN-173 runner for corrected cohort construction dependencies, exact nine-item PHQ-9, corrected KAN-172 P01, primary OLS/HC3, survey sensitivity, resolution sensitivity, and out-of-fold ROC. `final_cohort_phq9_ols_models.py` is only its CLI entry point. `build_final_full_cohort.py` constructs the corrected days 2–8 cohort. `final_cohort_paper_analysis.py` and `plot_roc_depression_three_models.py` remain legacy scripts and were not run for KAN-173. The runner writes scoped deliverables under `outputs/models/`; KAN-173 does not regenerate manuscript text or figures.

## KAN-169 historical baseline

The detailed script-by-script inventory below records the repository snapshot at commit `90e92706bf07384df43c73db835a42585ad3682b`. At that snapshot, raw data and generated outputs were absent and P01 authority/current manuscript linkage was unknown. Those baseline statements do not describe the KAN-173 workspace or results; current scoped authority is listed above and in the responsibility map below.

Historical path/dependency notes below are retained for migration context. Current data roots and KAN-173 behavior are recorded in the overlay.

## Analysis responsibilities

| Responsibility | Apparent script authority in tracked code | Qualification |
|---|---|---|
| Cohort construction | `build_final_full_cohort.py` | Joins DEMO_H, DPQ_H, PAXHD_H, and PAXDAY_H; KAN-173 selects PAXDAYD 2–8 and requires ≥4 adequate days (≥600 valid minutes/day), including ≥1 weekend day. |
| PHQ-9 construction | `kan173_corrected_analysis.py`; `build_final_full_cohort.py` | KAN-173 uses exactly DPQ010–DPQ090; 7/9 and invalid values are missing, all nine items are required, and PHQ9_ge_10 is secondary. |
| P01 construction | `src/nhanes_p01/p01.py` | KAN-173 consumes provenance-validated corrected KAN-172 P01 at 5/10/30/60 minutes; legacy P01 appears only in the historical comparison. |
| Temporal-resolution comparison | `kan173_corrected_analysis.py`; `nhanes_paxmin_resolution_instability.py` | KAN-173 fits the same compact corrected-P01 PHQ-9 model at 5/10/30/60 minutes; the prior resolution script uses the same canonical nine DPQ items. |
| Sleep-variable construction | `kan173_corrected_analysis.py` | Defines `self_reported_weekday_sleep_hours`, timestamp-grouped `WASO_proxy_min` (not validated WASO), and `sleep_disorder_history`; non-wear/unknown PAXPREDM labels are excluded. |
| OLS models | `kan173_corrected_analysis.py`; `final_cohort_phq9_ols_models.py` | The latter is an entry point to the KAN-173 runner. Primary outputs include ordinary OLS and HC3 inference; `nhanes_physica_physics.py` remains a separate legacy analysis. |
| ROC analyses | `kan173_corrected_analysis.py`; `final_cohort_paper_analysis.py`; `plot_roc_depression_three_models.py` | KAN-173 reports stratified 5-fold out-of-fold AUCs and paired bootstrap uncertainty. The other scripts remain legacy and were not run for KAN-173. |
| Potential landscape | `nhanes_potential_landscape_minute.py`; `nhanes_potential_landscape_minute_parallel.py` | Sequential and parallel PAXMIN paths; parallel script directly imports calculation helpers from the sequential script. `nhanes_strict_real_analysis.py` and `publication_figures.py` contain separate hourly empirical-potential paths. |
| Figure generation | `nhanes_paxmin_resolution_instability.py`, `nhanes_physica_physics.py`, `nhanes_physica_ultimate.py`, `nhanes_logistic_validation.py`, `nhanes_strict_real_analysis.py`, `nhanes_ultimate_logistic.py`, `plot_entropy_aging_detailed.py`, `plot_entropy_by_age.py`, `plot_roc_depression_three_models.py`, `publication_figures.py` | All declared outputs are absent. `nhanes_ultimate_logistic.py` and `publication_figures.py` can use synthetic/mock inputs; such outputs are not evidence of NHANES results. |

## Script-by-script map

### Historical P01-workstream candidates at the KAN-169 snapshot (not current authority)

#### `build_final_full_cohort.py`
- **Purpose:** Build the final cohort from demographics, depression questionnaire, monitor summary, and daily wear files; derive PHQ-9 fields and historical wear filters.
- **Inputs:** `DEMO_H.xpt`, `DPQ_H.xpt`, `PAXHD_H.xpt`, `PAXDAY_H.xpt`; optional `PAXRAW_*.xpt`. Defaults to `nhanes_2013_2014_raw/` (or `NHANES_RAW_DIR`).
- **Outputs:** `final_full_cohort.csv` by default.
- **Downstream data consumers:** `final_cohort_paper_analysis.py`, `final_cohort_phq9_ols_models.py`, `plot_roc_depression_three_models.py` consume the cohort file.
- **Current claims:** UNKNOWN; no output or manuscript linkage present.
- **Future canonical target:** `scripts/build_final_full_cohort.py`; KAN-169 leaves the original in place because `BASE_DIR`-relative input/output defaults would change on relocation.

#### `final_cohort_paper_analysis.py`
- **Purpose:** Compute PAXMIN 10-minute P01 and WASO proxy, derive sleep covariates, fit depression logistic models, and compare ROC curves.
- **Inputs:** `final_full_cohort.csv`; `PAXMIN_H.xpt`; optional `PAXHR_H.xpt`; `DEMO_H.xpt`, `BMX_H.xpt`, `SLQ_H.xpt` under `NHANES_RAW_DIR`/legacy raw path.
- **Outputs:** `final_paper_results.txt`, `outputs/final_cohort_pam_metrics.csv`, `outputs/roc_p01_vs_composite.png`.
- **Downstream data consumers:** `final_cohort_phq9_ols_models.py` and `plot_roc_depression_three_models.py` read the PAM cache. It imports `roc_plot_style.py`.
- **Current claims:** UNKNOWN; no output or manuscript linkage present.
- **Future canonical target:** `scripts/final_cohort_paper_analysis.py`; not copied because moving it changes root-relative paths and its helper import context.

#### `final_cohort_phq9_ols_models.py`
- **Purpose:** Fit base and extended OLS models for continuous PHQ-9 versus P01 and covariates.
- **Inputs:** `final_full_cohort.csv`, `outputs/final_cohort_pam_metrics.csv`, DEMO_H, BMX_H, and SLQ_H through the legacy raw path.
- **Outputs:** `outputs/phq9_ols_p01_comparison.txt`.
- **Downstream callers:** None found; `gen_tex/methods.tex`, `gen_tex/results.tex`, and `sections/table1.tex` are related fragments, not executable callers or verified results.
- **Current claims:** UNKNOWN; no report or complete manuscript present.
- **Future canonical target:** `scripts/final_cohort_phq9_ols_models.py`; no copy because `BASE_DIR` controls its legacy inputs and output.

#### `nhanes_paxmin_resolution_instability.py`
- **Purpose:** Compare nocturnal P01 at 5, 10, 30, and 60-minute resolutions; optionally report PHQ-9 regressions.
- **Inputs:** `PAXMIN_H.xpt`, `outputs/processed_data_physics_ultimate.csv`; optional DPQ_H, BMX_H, DEMO_H. Imports `roc_plot_style.py`.
- **Outputs:** `outputs/paxmin_resolution_metrics.csv`, `paxmin_resolution_correlations.csv`, `paxmin_resolution_scatter_vs_60m.png`, and `paxmin_resolution_phq9_regression.txt`.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; no output or manuscript linkage present.
- **Future canonical target:** `scripts/nhanes_paxmin_resolution_instability.py`; not copied because it uses root-relative paths and a root helper import.

#### `plot_roc_depression_three_models.py`
- **Purpose:** Plot ROC curves for P01-only, demographics/BMI-only, and combined models.
- **Inputs:** `final_full_cohort.csv`, `outputs/final_cohort_pam_metrics.csv`, BMX_H under the legacy raw path. Imports `roc_plot_style.py`.
- **Outputs:** `outputs/roc_comparison.png`.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; no figure or manuscript linkage present.
- **Future canonical target:** `scripts/plot_roc_depression_three_models.py`; not copied because it relies on root-relative paths and the root helper import.

### Legacy analyses and figures (`HISTORICAL_ANALYSIS_OR_OUTPUT`)

#### `nhanes_logistic_validation.py`
- **Purpose:** Legacy logistic validation for `sleep_problem_reported` with entropy and transition predictors; emits a forest plot.
- **Inputs:** `outputs/processed_data_physics_ultimate.csv`, falling back to `outputs/processed_data_physics_deep.csv`.
- **Outputs:** `outputs/fig12_forest_plot.png` plus console report.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; processed inputs and figure are absent.
- **Future canonical target:** `scripts/legacy/nhanes_logistic_validation.py`; no duplicate because `BASE_DIR`-relative input/output paths would change.

#### `nhanes_physica_physics.py`
- **Purpose:** Legacy hourly-activity entropy, Markov transition, spectral gap, mixing, EPR, statistical and figure pipeline.
- **Inputs:** `outputs/processed_data_refined.csv` (Cureus fallback) and root `PAXHR_H.xpt`.
- **Outputs:** `outputs/processed_data_physics_final.csv`; figures `fig5_entropy_boxplot.png`, `fig5b_entropy_by_age.png`, `fig6_markov_scatter.png`, `fig7_spectral_gap_comparison.png`, `fig8_stationary_dist_plot.png`, `fig9_spectral_epr_scatter.png`, `fig10_age_stratified_mixing.png`.
- **Downstream data consumers:** Entropy plotting scripts can read the generated `processed_data_physics_final.csv` if higher-priority inputs are absent.
- **Current claims:** UNKNOWN; predecessor data and outputs are absent.
- **Future canonical target:** `scripts/legacy/nhanes_physica_physics.py`; no copy because paths are rooted at the original script directory.

#### `nhanes_physica_ultimate.py`
- **Purpose:** Extend processed physics features with EPR, day/night transitions, PCA, and figures.
- **Inputs:** `outputs/processed_data_physics_deep.csv` (Cureus fallback) and root `PAXHR_H.xpt`.
- **Outputs:** `outputs/processed_data_physics_ultimate.csv`; `fig9_EPR_boxplot.png`, `fig10_TimeVarying_Transitions.png`, `fig11_Eigenvector_PhaseSpace.png`.
- **Downstream data consumers:** Legacy validation, threshold, logistic, PAXMIN resolution, plotting, and publication scripts read the `processed_data_physics_ultimate.csv` name.
- **Current claims:** UNKNOWN; deep input and outputs are absent.
- **Future canonical target:** `scripts/legacy/nhanes_physica_ultimate.py`; no copy because input paths depend on the original script directory.

#### `nhanes_potential_landscape_minute.py`
- **Purpose:** Sequential PAXMIN nocturnal potential landscape and transition entropy by participant.
- **Inputs:** `PAXMIN_H.xpt` at the project root or under the legacy raw directory.
- **Outputs:** `outputs/interim_results.csv`, `outputs/error_log.txt`, `outputs/potential_landscape_minute_results.csv`.
- **Downstream callers:** `nhanes_potential_landscape_minute_parallel.py` imports calculation functions directly.
- **Current claims:** UNKNOWN; no input data or output present.
- **Future canonical target:** `scripts/legacy/nhanes_potential_landscape_minute.py`; no copy to avoid a second import target and changed `BASE_DIR`.

#### `nhanes_potential_landscape_minute_parallel.py`
- **Purpose:** Chunk PAXMIN into per-participant Parquet staging, compute potential metrics in parallel, write result Parquet.
- **Inputs:** PAXMIN_H, optionally `outputs/processed_data_physics_ultimate.csv` for target SEQNs; imports `nhanes_potential_landscape_minute.py`.
- **Outputs:** `outputs/paxmin_staging_parts/`, `outputs/paxmin_staging_parquet/`, `outputs/potential_landscape_minute_parallel.parquet`, `outputs/failed_log.txt`.
- **Downstream callers:** None found; it is a direct consumer of the sequential module's functions.
- **Current claims:** UNKNOWN; no input or output present.
- **Future canonical target:** `scripts/legacy/nhanes_potential_landscape_minute_parallel.py`; no duplicate because a separate copy would break its local import relationship and root-relative paths.

#### `nhanes_strict_real_analysis.py`
- **Purpose:** Legacy logistic analysis and hourly empirical potential figures, requiring actual XPTs at runtime.
- **Inputs:** `outputs/processed_data_physics_ultimate.csv`, root PAXHR_H, BMX_H, DPQ_H.
- **Outputs:** `outputs/processed_data_final_master.csv`, `fig13_real_forest_plot.png`, `fig14_empirical_potential.png`.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; processed input/output absent; the `sleep_problem_reported` lineage is unresolved.
- **Future canonical target:** `scripts/legacy/nhanes_strict_real_analysis.py`; no duplicate because `BASE_DIR` paths change.

#### `nhanes_threshold_logistic_comparison.py`
- **Purpose:** Legacy hourly P01 threshold comparison (median, Q1, Q3) with logistic models.
- **Inputs:** `outputs/processed_data_physics_ultimate.csv`, PAXHR_H, BMX_H, DPQ_H.
- **Outputs:** Console summary only.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; no report or linked manuscript present.
- **Future canonical target:** `scripts/legacy/nhanes_threshold_logistic_comparison.py`; no copy because it reads root-relative paths.

#### `nhanes_threshold_robustness.py`
- **Purpose:** Legacy hourly P01 Q1/Q3 threshold robustness tests.
- **Inputs:** `outputs/processed_data_physics_ultimate.csv` and PAXHR_H.
- **Outputs:** Console summary only.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; no report or linked manuscript present.
- **Future canonical target:** `scripts/legacy/nhanes_threshold_robustness.py`; no copy because it reads root-relative paths.

#### `nhanes_ultimate_logistic.py`
- **Purpose:** Legacy logistic model and forest plot for sleep problem, BMI, PHQ-9, age, gender, and night P01.
- **Inputs:** `outputs/processed_data_physics_ultimate.csv`; root BMX_H and DPQ_H if present. The script uses synthetic BMI/PHQ-9 fallbacks when those XPTs are absent.
- **Outputs:** `outputs/processed_data_final_master.csv`, `outputs/fig13_ultimate_forest_plot.png`.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; no output present. A fallback-generated result would not establish NHANES evidence.
- **Future canonical target:** `scripts/legacy/nhanes_ultimate_logistic.py`; no copy because it is an unverified legacy alternative with root-relative paths.

#### `plot_entropy_aging_detailed.py`
- **Purpose:** Legacy age/entropy distribution and regression figure.
- **Inputs:** First available of `outputs/processed_data_physics_deep.csv`, `processed_data_physics_ultimate.csv`, `processed_data_physics_final.csv`.
- **Outputs:** `outputs/fig5c_entropy_aging_detailed.png`.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; input and figure absent.
- **Future canonical target:** `scripts/legacy/plot_entropy_aging_detailed.py`; no copy because its input/output roots are script-relative.

#### `plot_entropy_by_age.py`
- **Purpose:** Legacy entropy stratification by age.
- **Inputs:** First available of `outputs/processed_data_physics_deep.csv`, `processed_data_physics_ultimate.csv`, `processed_data_physics_final.csv`.
- **Outputs:** `outputs/fig5b_entropy_by_age.png`.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; input and figure absent.
- **Future canonical target:** `scripts/legacy/plot_entropy_by_age.py`; no copy because its input/output roots are script-relative.

#### `publication_figures.py`
- **Purpose:** Legacy publication-style forest, potential-landscape, and transition-entropy figures.
- **Inputs:** `outputs/processed_data_physics_ultimate.csv`; optional root BMX_H, DPQ_H, PAXHR_H. Figure 1 and 2 have synthetic/mock fallbacks when real data are unavailable.
- **Outputs:** `outputs/pub_fig1_forest_log.png`, `pub_fig2_potential_landscape.png`, `pub_fig3_transition_entropy.png`.
- **Downstream callers:** None found.
- **Current claims:** UNKNOWN; no figure assets or parent manuscript present. Synthetic/mock fallbacks are not NHANES evidence.
- **Future canonical target:** `scripts/legacy/publication_figures.py`; no copy because it uses root-relative files and contains unverified fallback behavior.

## Helper authority

#### `roc_plot_style.py` (`ANALYSIS_HELPER`)
- **Purpose:** Shared Matplotlib style and ROC color constants.
- **Inputs/outputs:** No data inputs; configures Matplotlib style.
- **Downstream importers:** `final_cohort_paper_analysis.py`, `nhanes_paxmin_resolution_instability.py`, `plot_roc_depression_three_models.py`.
- **Current claims:** Not a result-producing script.
- **Future canonical target:** `src/nhanes_p01/roc_plot_style.py`; left at root in KAN-169 because all importers still use `from roc_plot_style ...` and no caller migration was included.
