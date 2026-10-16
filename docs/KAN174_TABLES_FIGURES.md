# KAN-174 corrected tables and figures

## Scope

Presentation-only follow-up to accepted KAN-173 results, whose source commit is `ce9ddafa213ba06306a9d071ae4ecf8ed838c90b`. The generator consumes KAN-173 result files and the provenance-validated corrected P01 cache. It does not run the PAXMIN pass, recalculate P01, change cohort rules, refit models, or edit manuscript text. `outputs/tables_figures_manifest.json` records the source commit, canonical analysis manifest, source files, artifact role, placement, and verification status.

## Assets and placement

| Asset | Placement | Purpose |
| --- | --- | --- |
| `outputs/tables/table1_corrected.csv` and `.tex` | Primary | Corrected analytic cohort descriptors overall and by PHQ-9 group. Continuous rows show available N and mean (SD); categorical rows show n (%). |
| `outputs/tables/table2_primary_corrected.csv` and `.tex` | Primary | Corrected Model 1 and Model 2 estimates from `primary_models_hc3.csv`, including N, P01 beta, HC3 SE, 95% CI, p, R2, adjusted R2, effect per +0.01 P01, and a footnote specifying model covariates, P01 scale/effect conversion, standardization, and HC3 inference. |
| `outputs/figures/roc_corrected_oof.png` and `.pdf` | Primary exploratory figure | Three out-of-fold ROC curves from `roc_corrected_oof_predictions.csv`; legend reports canonical OOF AUC and 95% CI, and an in-plot annotation reports the paired combined-versus-baseline AUC difference and CI from `roc_corrected.csv`. |
| `outputs/figures/resolution_sensitivity_effect.png` and `.pdf` | Primary figure | Resolution-specific estimates and HC3 intervals, plotted as beta × 0.01 for 5-, 10-, 30-, and 60-minute P01 resolutions. |
| `outputs/tables/tableS1_survey_weighted.csv` and `.tex` | Supplement | Survey-weighted Model 1 and Model 2 sensitivity estimates from `survey_weighted_models.csv`, with design variables and degrees of freedom stated. |
| `outputs/tables/tableS2_resolution_sensitivity.csv` and `.tex` | Supplement | P01 distribution and HC3 association estimates by temporal resolution from `resolution_sensitivity.csv`. The resolutions are presented separately and are not treated as interchangeable. |

All table CSV values are generated from canonical source files, not typed manuscript values. LaTeX tables reproduce their paired CSV contents.

## Table 1 source and interpretation limits

The descriptive frame uses the accepted corrected cohort, the existing participant-level corrected P01 cache, and small NHANES XPT files for DEMO, DPQ, BMX, SLQ, PAXHD, and PAXDAY as needed. When `final_full_cohort.csv` is available it supplies the existing participant-level cohort frame; otherwise the generator's fallback uses the small cohort/protocol files and the existing KAN-173 cohort-building helpers. Neither path opens PAXMIN.

`SLD010H` is labeled self-reported usual weekday sleep duration, not total sleep time (TST). `WASO_proxy_min` is labeled as an actigraphy-derived proxy, not validated WASO. KAN-173's committed participant-level outputs did not retain its values; KAN-174 therefore marks the row unavailable and does not recompute it. PHQ-9 grouping is limited to records with a complete PHQ-9 score.

## Deliberately omitted assets

- No cohort-flow figure: the supplied cohort counts do not form a strict sequential attrition funnel. The manifest records `skipped_not_appropriate` with the reason; no flow diagram was manufactured.
- No potential-landscape figure was generated or designated primary. Any historical potential-landscape file was left untouched and is excluded from the KAN-174 asset manifest.

## Verification

Run from the repository root with the project `.venv` active and Poppler's `pdftotext` utility available:

```bash
python scripts/make_corrected_tables_figures.py
python scripts/verify_corrected_assets.py
```

The verifier checks displayed estimates and intervals against the canonical KAN-173 CSVs, checks the ROC AUCs computed from saved OOF predictions against the canonical summary, extracts the ROC PDF text to verify each displayed OOF AUC/95% CI and paired delta/CI against that summary, validates Table 1 against the corrected cohort and P01 cache, checks the CSV/LaTeX pairs and variable-label caveats, and records `numbers_verified: true` for each generated asset. PNG dimensions and non-empty PDF/PNG outputs are checked by the verifier; the rendered PNGs were also visually inspected for readable labels and overlap. The recorded verification status is `PASS`.

KAN-174 changed tables and figures only. Scientific results, cohort membership, P01 values, and manuscript text were not changed.
