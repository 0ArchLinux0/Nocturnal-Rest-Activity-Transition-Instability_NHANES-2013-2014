# Repository audit — KAN-168

## Status

- Repository preflight passes for the authoritative bound repository; local `main` and `origin/main` were synchronized at the supplied base HEAD before this continuation.
- The initial legacy inventory contains 26 files: 17 analysis scripts, one Python helper, one download script, four support files, and three TeX fragments. No raw/processed datasets, historical result files, figures, notebooks, full manuscript, preprint, cover letter, or highlights are present.
- The supplied canonical tree has been created. Empty canonical directories are retained with `.gitkeep` markers; raw data remain absent.
- `PAX80_H` is `NOT_USED`: it is not opened/read by tracked scripts and appears only in explanatory comments. `PAXMIN_H` is `REQUIRED` for minute-level P01 and potential-landscape paths.
- Eight source endpoints responded HTTP 200 to HEAD. `SOURCE_ENDPOINTS_RESOLVED=8`, `SOURCE_CONTENTS_VERIFIED=0`, and `SOURCE_REQUIREMENTS_UNRESOLVED=0`. No source dataset was downloaded.
- No scientific analysis was executed, no manuscript was rewritten, and no P01 correction was made.

## Runtime and verified dependency inventory

The canonical runtime is isolated Python 3.12.14 at `.venv/bin/python`; system Python was not modified. The established dependency system is `requirements.txt` (already documented by the repository README); SHAP was removed because no tracked Python file imports it. No exact-version lock was generated.

| Dependency | Classification | Evidence in tracked Python |
|---|---|---|
| NumPy | `REQUIRED_ANALYSIS` | Numerical operations in cohort, transition, and feature pipelines. |
| pandas | `REQUIRED_RUNTIME` | Tabular I/O and transformations across the pipelines. |
| SciPy | `REQUIRED_ANALYSIS` | Statistics, linear algebra, and KDE paths. |
| scikit-learn | `REQUIRED_ANALYSIS` | Scaling, PCA, ROC metrics, and model helpers. |
| statsmodels | `REQUIRED_ANALYSIS` | Logistic/OLS models and regression summaries. |
| Matplotlib | `REQUIRED_FIGURES` | Figure generation across analysis and plotting scripts. |
| Seaborn | `REQUIRED_FIGURES` | Publication/age-stratified figure scripts. |
| pyreadstat | `REQUIRED_ANALYSIS` | XPT loading; directly imported by the minute-level pipeline. |
| tqdm | `REQUIRED_ANALYSIS` | Parallel PAXMIN staging script. |
| pyarrow | `REQUIRED_ANALYSIS` | Pandas Parquet read/write paths in the parallel pipeline. |
| PyYAML | `NOT_USED` | No import in tracked Python; YAML was parsed only by the bootstrap validation tool. |
| SHAP | `NOT_USED` | No import or usage in tracked Python. |

The listed runtime/analysis/figure dependencies were installed into `.venv` from the revised `requirements.txt` and import-checked. No source data were needed for this environment validation.

## Provenance and remaining limits

Legacy processed inputs such as `outputs/processed_data_refined.csv` and `outputs/processed_data_physics_deep.csv` are absent and lack producer/source lineage in the clone. `sleep_problem_reported` is not equated with `depression_suspected`. This is a legacy lineage gap for KAN-169 mapping, not an unresolved raw-source acquisition requirement in the manifest.

The exact code-derived variable/source map is in `docs/DATA_PROVENANCE.md` and `config/variables.yaml`. HEAD status is endpoint evidence only; KAN-170 must verify file contents, schema/codebook variables, and checksums after authorized acquisition.

## Boundary

No legacy file was moved, deleted, or overwritten. `data/raw/` is empty. KAN-168 is limited to repository structure, inventory, source provenance, and environment/dependency setup. KAN-169 can begin after this bootstrap commit; KAN-170 may acquire the eight recorded code-driven sources, excluding PAX80_H. KAN-171 waits for stable legacy mapping and source provenance.
