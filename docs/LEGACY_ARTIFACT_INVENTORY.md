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
