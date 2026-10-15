# Legacy artifact inventory — KAN-168

Inventory is of the freshly cloned worktree before bootstrap files were added. It covers every project path returned by recursive enumeration; `.git` internals are excluded. A name in the generated-output lists is a code reference, not evidence that the artifact exists.

## Existing repository files

### Analysis and plotting scripts (17)

- `build_final_full_cohort.py`
- `final_cohort_paper_analysis.py`
- `final_cohort_phq9_ols_models.py`
- `nhanes_logistic_validation.py`
- `nhanes_paxmin_resolution_instability.py`
- `nhanes_physica_physics.py`
- `nhanes_physica_ultimate.py`
- `nhanes_potential_landscape_minute.py`
- `nhanes_potential_landscape_minute_parallel.py`
- `nhanes_strict_real_analysis.py`
- `nhanes_threshold_logistic_comparison.py`
- `nhanes_threshold_robustness.py`
- `nhanes_ultimate_logistic.py`
- `plot_entropy_aging_detailed.py`
- `plot_entropy_by_age.py`
- `plot_roc_depression_three_models.py`
- `publication_figures.py`

Shared plotting helper: `roc_plot_style.py`.

### Other existing files

- `README.md`, `requirements.txt`, `LICENSE`, `.gitignore`
- `download_nhanes_2013_2014.sh`
- TeX fragments only: `gen_tex/methods.tex`, `gen_tex/results.tex`, `sections/table1.tex`

Initial inventory count: 26 files (18 Python files including the helper, one shell script, four root support files, and three TeX fragments).

## Artifact classes absent from the clone

- Notebooks: none (`.ipynb` files: 0).
- Source data: no CSV, XPT, Parquet, SAS, or other NHANES data files.
- Historical generated outputs: no `outputs/` directory or result files.
- Figures: no image/PDF figure files.
- Full manuscript: none. Three TeX fragments exist as listed above.
- Preprints, cover letters, journal highlights: none found.
- `Cureus/` directory: absent; it is only a legacy fallback path and ignore rule.
- Pre-existing raw-data root: absent in the clone. KAN-168 establishes `data/raw/` as the explicitly required empty future raw-data location; no source data are placed there.

`.gitignore` names `outputs/`, `Cureus/`, XPT files, `nhanes_2013_2014_raw/`, `final_full_cohort.csv`, `final_paper_results.txt`, and `TABLE*.txt`; these are ignore rules, not proof that corresponding files exist. The fetched clone had no ignored or untracked files.

## Legacy input/output paths referenced by code

Legacy source paths are inconsistent and are not migrated in KAN-168:

- Raw inputs: root-level XPT files; `nhanes_2013_2014_raw/`; or `$NHANES_RAW_DIR` in `build_final_full_cohort.py`, `final_cohort_paper_analysis.py`, `final_cohort_phq9_ols_models.py`, and related newer scripts.
- Processed inputs: `outputs/processed_data_refined.csv` (fallback `Cureus/outputs/processed_data_refined.csv`), `outputs/processed_data_physics_deep.csv` (fallback under `Cureus/outputs/`), `outputs/processed_data_physics_ultimate.csv`, `outputs/processed_data_physics_final.csv`, and `outputs/processed_data_final_master.csv`.
- Depression/P01 analysis inputs and cache: root `final_full_cohort.csv`, `outputs/final_cohort_pam_metrics.csv`.

## Output names declared by legacy code; none are present

- Cohort/results: `final_full_cohort.csv`, `final_paper_results.txt`, `outputs/phq9_ols_p01_comparison.txt`, `outputs/paxmin_resolution_metrics.csv`, `outputs/paxmin_resolution_correlations.csv`, `outputs/paxmin_resolution_phq9_regression.txt`, `outputs/potential_landscape_minute_results.csv`, `outputs/interim_results.csv`, `outputs/processed_data_physics_final.csv`, `outputs/processed_data_physics_ultimate.csv`, `outputs/processed_data_final_master.csv`.
- Parquet/cache/logs: `outputs/potential_landscape_minute_parallel.parquet`, `outputs/paxmin_staging_parts/part_*.parquet`, `outputs/paxmin_staging_parquet/<SEQN>.parquet`, `outputs/failed_log.txt`, `outputs/error_log.txt`, `outputs/final_cohort_pam_metrics.csv`.
- Figures: `outputs/roc_p01_vs_composite.png`, `outputs/roc_comparison.png`, `outputs/paxmin_resolution_scatter_vs_60m.png`, `outputs/fig5_entropy_boxplot.png`, `outputs/fig5b_entropy_by_age.png`, `outputs/fig5c_entropy_aging_detailed.png`, `outputs/fig6_markov_scatter.png`, `outputs/fig7_spectral_gap_comparison.png`, `outputs/fig8_stationary_dist_plot.png`, `outputs/fig9_spectral_epr_scatter.png`, `outputs/fig10_age_stratified_mixing.png`, `outputs/fig9_EPR_boxplot.png`, `outputs/fig10_TimeVarying_Transitions.png`, `outputs/fig11_Eigenvector_PhaseSpace.png`, `outputs/fig12_forest_plot.png`, `outputs/fig13_real_forest_plot.png`, `outputs/fig14_empirical_potential.png`, `outputs/fig13_ultimate_forest_plot.png`, `outputs/pub_fig1_forest_log.png`, `outputs/pub_fig2_potential_landscape.png`, and `outputs/pub_fig3_transition_entropy.png`.

## NHANES URLs hard-coded in tracked files

1. `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles` — base in `download_nhanes_2013_2014.sh`; the script appends the filenames recorded in `sources/manifest.yaml` except PAXMIN.
2. `https://ftp.cdc.gov/pub/NHANES/LargeDataFiles/PAXMIN_H.xpt` — PAXMIN source in the download script.
3. `https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&CycleBeginYear=2013` — link in `README.md`.
4. `https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013` — link in `nhanes_ultimate_logistic.py`.

## NHANES filenames and column names referenced by code

Code names (not independent component classifications): `DEMO_H.xpt`, `DPQ_H.xpt`, `BMX_H.xpt`, `SLQ_H.xpt`, `PAXHR_H.xpt`, `PAXMIN_H.xpt`, `PAXDAY_H.xpt`, `PAXHD_H.xpt`; optional legacy lookup `PAXRAW_H.xpt` / `PAXRAW_*.xpt`; and comment-only `PAX80_H`. Its official identity is recorded from the supplied specification as the NHANES 2013–2014 Physical Activity Monitor - Raw Data 80 Hz source distributed via CDC/NCHS FTP, but no tracked script reads it; requirement classification is `NOT_USED`, not a KAN-170 acquisition requirement. The optional PAXRAW path may refer to another cycle.

Exact code-to-source field trace and unresolved lineage are in `docs/DATA_PROVENANCE.md` and `config/variables.yaml`.

## KAN-169 artifact migration map

The 26-artifact baseline is verified against `git ls-tree -r a42dc1de8b07e7dae71ee3cea2cf29acf5ebf8fa` (26 paths). All 26 originals are Git-tracked and remain present. The migration is preservation-first: only the downloader and three partial TeX assets were copied; no analysis code was relocated or rewritten. Proposed destinations marked “planned” are not copies and do not supersede original paths.

Roles `ACTIVE_ANALYSIS` and `HISTORICAL_ANALYSIS_OR_OUTPUT` describe apparent code purpose, not scientific validation or verified contribution to current claims. No full manuscript, bibliography, images, result files, or current claim source is present; claim contribution is unknown until a manuscript/results lineage is established.

| ORIGINAL_PATH | GIT_TRACKED | FILE_TYPE | ROLE | DEPENDENCIES / CALLERS | CANONICAL_DESTINATION | ACTION | STATUS | NOTES |
|---|---|---|---|---|---|---|---|---|
| `.gitignore` | yes | Git ignore config | SUPPORT_OR_METADATA | Git ignore processing | `.gitignore` | LEAVE_IN_PLACE | updated in place | KAN-169 added an allowlist for the migration hash record so validation evidence can be tracked. |
| `LICENSE` | yes | Plain-text license | SUPPORT_OR_METADATA | Repository users | `LICENSE` | NO_ACTION | unchanged at root | Already appropriate as root metadata. |
| `README.md` | yes | Markdown documentation | SUPPORT_OR_METADATA | Repository users | `README.md` | LEAVE_IN_PLACE | updated in place | KAN-169 clarified that legacy paths remain pending a caller-aware code migration; not manuscript claim authority. |
| `requirements.txt` | yes | Python dependency manifest | SUPPORT_OR_METADATA | Python environment setup | `requirements.txt` | NO_ACTION | unchanged at root | KAN-168 dependency inventory retained; not copied. |
| `build_final_full_cohort.py` | yes | Python analysis script | ACTIVE_ANALYSIS | DEMO_H, DPQ_H, PAXDAY_H, PAXHD_H; optional PAXRAW; writes `final_full_cohort.csv`, consumed by the final-cohort analysis/OLS/ROC scripts | `scripts/build_final_full_cohort.py` (planned) | LEAVE_IN_PLACE | preserved at original path; copy deferred | `BASE_DIR`-relative raw/output defaults would change after relocation; no results or current manuscript establishes authority. |
| `final_cohort_paper_analysis.py` | yes | Python analysis script | ACTIVE_ANALYSIS | `final_full_cohort.csv`, PAXMIN_H, optional PAXHR_H, DEMO_H, BMX_H, SLQ_H; reads corrected PAM cache, writes a separate WASO proxy derivative, paper results, and legacy ROC figure | `scripts/final_cohort_paper_analysis.py` (planned) | LEAVE_IN_PLACE | preserved at original path; copy deferred | Legacy paper path; not used for KAN-173 corrected models or OOF ROC. Its WASO derivative no longer overwrites the corrected P01 cache. |
| `final_cohort_phq9_ols_models.py` | yes | Python analysis entry point | ACTIVE_ANALYSIS | Delegates to `kan173_corrected_analysis.py`; reads the corrected final cohort and KAN-172 cache through that runner | `scripts/final_cohort_phq9_ols_models.py` (planned) | LEAVE_IN_PLACE | preserved at original path; copy deferred | Former direct cache joins and text output retired in KAN-173; canonical outputs are under `outputs/models/`. |
| `nhanes_paxmin_resolution_instability.py` | yes | Python analysis script | ACTIVE_ANALYSIS | PAXMIN_H, `outputs/processed_data_physics_ultimate.csv`, optional DPQ/BMX/DEMO; writes metrics, correlations, scatter, regression report | `scripts/nhanes_paxmin_resolution_instability.py` (planned) | LEAVE_IN_PLACE | preserved at original path; copy deferred | Temporal-resolution comparison candidate; imports root `roc_plot_style`; no results are present. |
| `plot_roc_depression_three_models.py` | yes | Python analysis/figure script | ACTIVE_ANALYSIS | Final cohort, PAM cache, BMX_H; writes `outputs/roc_comparison.png` | `scripts/plot_roc_depression_three_models.py` (planned) | LEAVE_IN_PLACE | preserved at original path; copy deferred | Root-relative paths and `roc_plot_style` import; current contribution unverified. |
| `nhanes_logistic_validation.py` | yes | Python analysis/figure script | HISTORICAL_ANALYSIS_OR_OUTPUT | `outputs/processed_data_physics_ultimate.csv` or deep fallback; writes `fig12_forest_plot.png` | `scripts/legacy/nhanes_logistic_validation.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Legacy outcome/feature pipeline; relocation changes `BASE_DIR`; no output exists. |
| `nhanes_physica_physics.py` | yes | Python analysis/figure script | HISTORICAL_ANALYSIS_OR_OUTPUT | `processed_data_refined.csv` (Cureus fallback) and PAXHR_H; writes `processed_data_physics_final.csv` and physics figures | `scripts/legacy/nhanes_physica_physics.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Historical processed-data dependency is absent; root paths retained. |
| `nhanes_physica_ultimate.py` | yes | Python analysis/figure script | HISTORICAL_ANALYSIS_OR_OUTPUT | `processed_data_physics_deep.csv` (Cureus fallback) and PAXHR_H; writes ultimate CSV and figures | `scripts/legacy/nhanes_physica_ultimate.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Historical predecessor CSV is absent; downstream legacy scripts read its output name. |
| `nhanes_potential_landscape_minute.py` | yes | Python analysis script | HISTORICAL_ANALYSIS_OR_OUTPUT | PAXMIN_H; writes interim/error logs and minute potential-landscape CSV | `scripts/legacy/nhanes_potential_landscape_minute.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Direct PAXMIN reader; no result file exists. |
| `nhanes_potential_landscape_minute_parallel.py` | yes | Python analysis script | HISTORICAL_ANALYSIS_OR_OUTPUT | PAXMIN_H, processed ultimate CSV; imports `nhanes_potential_landscape_minute.py`; writes Parquet staging/results and failed log | `scripts/legacy/nhanes_potential_landscape_minute_parallel.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Direct module import and root-relative paths make an isolated copy unsafe. |
| `nhanes_strict_real_analysis.py` | yes | Python analysis/figure script | HISTORICAL_ANALYSIS_OR_OUTPUT | Processed ultimate CSV, PAXHR_H, BMX_H, DPQ_H; writes final-master CSV and empirical-potential/forest figures | `scripts/legacy/nhanes_strict_real_analysis.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Historical `sleep_problem_reported` lineage is absent; no output exists. |
| `nhanes_threshold_logistic_comparison.py` | yes | Python analysis script | HISTORICAL_ANALYSIS_OR_OUTPUT | Processed ultimate CSV, PAXHR_H, BMX_H, DPQ_H; prints median/Q1/Q3 logistic comparisons | `scripts/legacy/nhanes_threshold_logistic_comparison.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Alternative legacy threshold analysis; no report file or current claim trace. |
| `nhanes_threshold_robustness.py` | yes | Python analysis script | HISTORICAL_ANALYSIS_OR_OUTPUT | Processed ultimate CSV and PAXHR_H; prints Q1/Q3 threshold robustness | `scripts/legacy/nhanes_threshold_robustness.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Historical alternative; no results are present. |
| `nhanes_ultimate_logistic.py` | yes | Python analysis/figure script | HISTORICAL_ANALYSIS_OR_OUTPUT | Processed ultimate CSV; BMX_H/DPQ_H when present, synthetic fallback otherwise; writes final-master CSV and forest figure | `scripts/legacy/nhanes_ultimate_logistic.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Synthetic fallback disqualifies unverified output as evidence; no result exists. |
| `plot_entropy_aging_detailed.py` | yes | Python figure script | HISTORICAL_ANALYSIS_OR_OUTPUT | First available processed deep/ultimate/final CSV; writes `fig5c_entropy_aging_detailed.png` | `scripts/legacy/plot_entropy_aging_detailed.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Historical generated-data dependency is absent. |
| `plot_entropy_by_age.py` | yes | Python figure script | HISTORICAL_ANALYSIS_OR_OUTPUT | First available processed deep/ultimate/final CSV; writes `fig5b_entropy_by_age.png` | `scripts/legacy/plot_entropy_by_age.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Historical generated-data dependency is absent. |
| `publication_figures.py` | yes | Python figure/analysis script | HISTORICAL_ANALYSIS_OR_OUTPUT | Processed ultimate CSV; BMX_H/DPQ_H/PAXHR_H when present; writes publication figures 1–3 | `scripts/legacy/publication_figures.py` (planned) | LEAVE_IN_PLACE | preserved at original path; no duplicate created | Contains synthetic/mock fallback figure inputs; no figure assets are present. |
| `roc_plot_style.py` | yes | Python helper module | ANALYSIS_HELPER | Imported by final-cohort analysis, PAXMIN resolution, and three-model ROC scripts | `src/nhanes_p01/roc_plot_style.py` (planned) | LEAVE_IN_PLACE | root helper preserved as sole imported copy | Caller imports and script-relative paths are not changed in KAN-169. |
| `download_nhanes_2013_2014.sh` | yes | Bash acquisition script | DATA_ACQUISITION | CDC NHANES XPT endpoints; writes to caller-selected directory | `scripts/legacy/download_nhanes_2013_2014.sh` | COPY | copy present; original retained; SHA-256 matches | Copied without execution; KAN-170 acquisition was not started. |
| `gen_tex/methods.tex` | yes | TeX manuscript fragment | HISTORICAL_MANUSCRIPT | Refers to `final_cohort_phq9_ols_models.py`; no parent manuscript found | `manuscript/archive/legacy/methods.tex` | ARCHIVE_COPY | archive copy present; original retained; SHA-256 matches | Partial scaling text; no full/current manuscript authority. |
| `gen_tex/results.tex` | yes | TeX manuscript fragment | HISTORICAL_MANUSCRIPT | Regression coefficient-scaling text; no parent manuscript found | `manuscript/archive/legacy/results.tex` | ARCHIVE_COPY | archive copy present; original retained; SHA-256 matches | Partial results wording, not a full manuscript or verified result. |
| `sections/table1.tex` | yes | TeX table-note fragment | HISTORICAL_MANUSCRIPT | Refers to PHQ-9 OLS scaling; no parent table/manuscript found | `manuscript/archive/legacy/table1.tex` | ARCHIVE_COPY | archive copy present; original retained; SHA-256 matches | Partial table footnote; no current submission asset. |

The exact machine-readable copy map is `docs/LEGACY_MIGRATION_MAP.csv`; hash evidence is `outputs/validation/legacy_migration_hashes.txt`. No original was deleted or overwritten.
