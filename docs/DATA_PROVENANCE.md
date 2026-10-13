# Data provenance and variable trace — KAN-168

## Evidence level

This trace is derived from current code references, not local XPT/CSV contents. No source or historical data are present. Eight exact XPT endpoints returned HTTP 200 to HEAD-only requests; `source_contents_verified=0` because no source bytes, schemas, or variables were checked. `source_requirements_unresolved=0` for the code-driven acquisition set. The names below reproduce code literally and do not independently validate codebook definitions.

## Direct source-file and variable references in code

| Code filename/path reference | Exact fields read or referenced | Current code use |
|---|---|---|
| `DEMO_H.xpt` | `SEQN`, `RIDAGEYR`, `RIAGENDR`, `RIDRETH3`, `WTMEC2YR`; downstream also reads `INDFMPIR` | Cohort demographics and join key; downstream aliases include `Age`/`Sex` and `PIR`. |
| `DPQ_H.xpt` | `SEQN`, `DPQ010`, `DPQ020`, `DPQ030`, `DPQ040`, `DPQ050`, `DPQ060`, `DPQ070`, `DPQ080`, `DPQ090` | `build_final_full_cohort.py` computes `PHQ9_Score` from all nine items (7/9 treated missing; score is only retained when all nine contribute) and `depression_suspected = 1` at score >=10. Other legacy scripts sum selected or detected DPQ fields with different missing-value handling. These are historical code definitions, not a canonical scoring decision. |
| `BMX_H.xpt` | `SEQN`, `BMXBMI` | BMI input in legacy logistic/OLS code; alias `BMI` appears in the newer OLS script. |
| `SLQ_H.xpt` | `SEQN`, `SLD010H` | Sleep-hours input; `final_cohort_phq9_ols_models.py` aliases it `TST` and marks 77/99 missing. |
| `PAXDAY_H.xpt` | `SEQN`, `PAXDAYD`, `PAXDAYWD`, `PAXVMD`, `PAXQFD` | `build_final_full_cohort.py` applies its historical day/wear rules and derives valid-day/weekend flags. |
| `PAXHD_H.xpt` | `SEQN`, optionally retained when present: `PAXSTS`, `PAXSENID`, `PAXFDAY`, `PAXLDAY` | Monitor-summary join and optional `PAXSTS == 1` filter in `build_final_full_cohort.py`. |
| `PAXHR_H.xpt` | `SEQN`, `PAXDAYH`, `PAXSSNHP`; activity column selected from `PAXMTSH` or `PAXINTEN` | Legacy hourly entropy/Markov/physics and publication figure inputs. Selection order in code is `PAXMTSH`, then `PAXINTEN`; availability was not inspected. |
| `PAXMIN_H.xpt` | `SEQN`, `PAXDAYM`, `PAXSSNMP`, `PAXMTSM`, `PAXQFM`; `PAXPREDM` in WASO analysis | Minute-level instability and WASO code. `final_cohort_paper_analysis.py` derives `P01_10m` from `PAXMTSM` after its `PAXQFM` filter and derives `WASO_proxy_min` from `PAXPREDM`. The minute potential-landscape scripts use `PAXMTSM` and `PAXQFM`. |

`PAXRAW_H.xpt` / `PAXRAW_*.xpt` is an optional legacy lookup described by its own code comment as potentially belonging to another cycle; it is not a canonical source. `PAX80_H` appears only in explanatory prose in `build_final_full_cohort.py`; there is no tracked input path, reader, CLI argument, or analysis call for it. Classify PAX80_H as `NOT_USED` in the current pipeline. Its official identity is recorded from the supplied specification, but it is not a KAN-170 acquisition requirement. `PAXMIN_H` is a separate minute-level summary directly consumed by the minute-level P01 and potential-landscape scripts.

## Derived and historical analysis variables

| Analysis variable | Trace available in current code | Provenance status |
|---|---|---|
| `PHQ9_Score`, `depression_suspected` | `DPQ_H.xpt:DPQ010`–`DPQ090` in `build_final_full_cohort.py`; all-nine-item rule and >=10 threshold are explicitly in code | Code trace exists; canonical scoring/outcome not asserted. |
| `Age`, `Sex` | `final_full_cohort.csv` is built with `DEMO_H.xpt:RIDAGEYR` and `RIAGENDR`; separate legacy pipelines consume lowercase `age` and `gender` from processed CSVs | New cohort path trace exists; provenance of the missing legacy lowercase fields is not independently recorded. `Sex`/`Gender` are not assumed interchangeable labels beyond the code's rename. |
| `BMI` / `BMXBMI` | `BMX_H.xpt:BMXBMI` | Code trace exists; file absent. |
| `PIR` | `DEMO_H.xpt:INDFMPIR` | Code trace exists in newer OLS script; file absent. |
| `TST` | `SLQ_H.xpt:SLD010H` | Code trace exists in newer OLS script; file absent. |
| `P01_10m`, `P01`, `Night_P_01` | `PAXMIN_H.xpt:PAXMTSM`, filtered using `PAXQFM`, indexed/ordered using `PAXDAYM` and `PAXSSNMP`; legacy hourly `P_01`/`P_01_Night` use `PAXHR_H.xpt` and selected activity column `PAXMTSH` or `PAXINTEN` | Algorithm is present in legacy code; no data/results; not canonical. The code paths currently point to project root, `nhanes_2013_2014_raw/`, or `$NHANES_RAW_DIR`, not yet to `data/raw/`. |
| `WASO_proxy_min` | `PAXMIN_H.xpt:PAXPREDM`, with minute/day indices and valid-minute filtering from other PAXMIN fields | Code trace exists; label semantics/schema not checked against data dictionary. |
| `Entropy`, `P_01`, `P_10`, `Spectral_Gap`, `Mixing_Time`, `pi0`, `pi1`, `EPR`, `P_01_Day`, `P_01_Night`, `P_10_Day`, `P_10_Night`, PCA fields | Some are computed by `nhanes_physica_physics.py` / `nhanes_physica_ultimate.py` from `PAXHR_H.xpt` activity and/or their processed predecessor CSVs; the latter's names/columns are referenced in code | The activity inputs are traceable as above. `outputs/processed_data_refined.csv` and `outputs/processed_data_physics_deep.csv` are absent and have no producer/provenance artifact in this clone; column lineage is unresolved. |
| `sleep_problem_reported` | Read from legacy processed CSVs such as `outputs/processed_data_refined.csv`, `outputs/processed_data_physics_deep.csv`, and `outputs/processed_data_physics_ultimate.csv` | Source producer and source field are absent. Do not equate it with `depression_suspected` without evidence. |
| `Day_P_10`, `Night_P_01`, `Entropy`, `EPR`, and other fields read from `processed_data_physics_ultimate.csv` | Legacy consumers include `nhanes_logistic_validation.py`, threshold/plot scripts, and `publication_figures.py` | The named CSV is not present; its complete source-column lineage cannot be verified from this clone. |

## Local path decision and KAN-170 requirements

`data/raw/` is the single canonical raw-data root. The code-driven source set is in `sources/manifest.yaml`: eight endpoints are resolved, zero source contents are verified, and zero current source requirements remain unresolved. `PAXMIN_H` is required by the current minute-level pipeline; `PAX80_H` is not used and must not be added to the acquisition set. Existing scripts have not yet been redirected to `data/raw/`; that migration belongs to KAN-169. KAN-170 should record retrieval time, final URL, source page/codebook version, file size, format/schema checks, and SHA-256 for every acquired file. Do not use or migrate the optional `PAXRAW_H` path without cycle verification.
