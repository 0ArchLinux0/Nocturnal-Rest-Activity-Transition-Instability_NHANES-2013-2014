# Submission Readiness

## Frozen revision
KAN-176 preflight started at `cafcd2844be10501c15f476f0ec7ab64718c29d6` on branch `main`; `origin/main` matched and the worktree was clean. This is the final package revision from KAN-175. KAN-168 through KAN-175 are documented in `docs/ANALYSIS_CHANGELOG.md` and accepted reports.

## Source integrity
All eight canonical XPT checksums recomputed from `data/raw/` matched `sources/checksums.sha256` and `sources/manifest.yaml`. `PAXMIN_H.xpt` SHA-256: `4b1aec371ebfca456ff2d386bc21444ddcb9ac06fdc8f13c3ed8589413d053bd`. All raw XPT files are gitignored; none are tracked.

## P01 algorithm integrity
The corrected cache `outputs/final_cohort_pam_metrics.csv` passed provenance validation against PAXMIN SHA-256, algorithm `kan172-p01-clock-contiguity-v1`, and method fingerprint `7c88634eb2f616c565d57bc1e2669c4a3552ae20d1792f10fca30b6992cabe64`. The cache covers exactly 4,882 cohort SEQNs and all 4,882 have finite P01_10m. Exact adjacency, daylight-gap exclusion, QC-invalid exclusion, incomplete-block exclusion, and midnight-edge synthetic checks passed. Full P01 regeneration was not required; the validated cache was reused for downstream reproduction.

## Corrected results
KAN-173 outputs were regenerated in `/tmp/nhanes-p01-kan176-repro/models` from the validated cache and compared to the canonical result files using `rtol=1e-10`, `atol=1e-12`; all six required outputs matched. Corrected cohort N=4,882. Model 1 HC3: N=4,508, beta=10.1266279289, SE=1.4134154035, 95% CI [7.3556398303, 12.8976160275], p=9.0678952184e-13. Model 2 HC3: N=4,048, beta=5.7227824413, SE=1.6543514236, 95% CI [2.4793415188, 8.9662233638], p=0.0005473225375. No numerical drift was detected beyond floating-point tolerance.

## Survey sensitivity
Reproduced survey Model 2 beta=5.8668977012, p=0.0199977558. The survey analysis remains a sensitivity analysis; it does not change interpretation of the primary unweighted HC3 estimates.

## Tables and figures
`make_corrected_tables_figures.py` and `verify_corrected_assets.py` ran in an isolated temporary project mirror using canonical KAN-173 inputs. All 12 generated assets passed number/source checks. The OOF ROC and resolution-effect figures were visually inspected. No potential-landscape primary asset or interchangeability claim is present. No cohort-flow figure was generated because the supplied counts are not a sequential attrition funnel. Canonical scientific assets were not overwritten.

## Manuscript package
The canonical generator rebuilt `Manuscript_JADR_FINAL.docx`, `Cover_Letter_JADR_FINAL.docx`, and `Highlights_JADR_FINAL.docx`. The number guard passed: six files scanned, zero obsolete-number hits, zero prohibited-claim hits, canonical table checks passed. All three DOCX archives passed integrity checks and rendered to PDF: manuscript 9 pages, cover letter 1 page, highlights 1 page. Both corrected figures are embedded and visually checked in manuscript PDF pages 5 and 6. Rendered manuscript text contains Kroenke, PAXMIN_H, NHANES Analytic Guidelines, Hori, Leaning, John, and Zhao.

## Reference audit
`manuscript/current/REFERENCE_AUDIT.md` records **REFERENCE_AUDIT=PASS** for all nine references, verified against PubMed and official CDC/NCHS sources.

## Reproducibility
A clean Python 3.12.14 venv at `/tmp/nhanes-p01-kan176-venv` installed `requirements.txt`; its `pip check` reported no broken requirements. The complete suite passed (16 tests), the P01 synthetic suite passed (14 tests), and `compileall` passed for `scripts/` and `src/`. The system Python 3.14.4 `pip check` reports an unrelated system package issue (`python-debian` requires missing `charset-normalizer`); the isolated clean validation environment passes. Cross-artifact values are recorded in `outputs/validation/submission_number_crosscheck.csv`; checksums for canonical models, tables, figures, and final DOCX files are in `outputs/validation/final_artifact_checksums.sha256` and verified.

## Remaining limitations
The cross-sectional design does not establish temporal order or causality. PHQ-9 is questionnaire-based symptom burden, not a clinician-established diagnosis. PAXPREDM-derived WASO is an unvalidated proxy; SLD010H is self-reported weekday/workday sleep duration. Incremental OOF discrimination is modest and external validation is absent. These limits are disclosed in the manuscript and do not block submission.

## Blockers
None.

READY FOR SUBMISSION
