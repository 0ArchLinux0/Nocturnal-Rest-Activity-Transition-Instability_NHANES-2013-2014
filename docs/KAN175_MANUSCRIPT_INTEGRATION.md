# KAN-175 corrected manuscript integration

## Source and authority

The manuscript rewrite began from `manuscript/current/KAN175_SOURCE_RECOVERY.md`, the recovered text snapshot of the prior manuscript, cover letter, and highlights. That file is historical evidence only; it was not treated as scientific authority. Corrected claims use the accepted KAN-173 run manifest/results and KAN-174 asset manifest/report. No scientific analysis was rerun or changed in KAN-175.

Canonical inputs: `outputs/models/kan173_run_manifest.json`, `docs/KAN173_CORRECTED_RESULTS.md`, `outputs/tables_figures_manifest.json`, and `docs/KAN174_TABLES_FIGURES.md`. Primary tables: `outputs/tables/table1_corrected.csv` and `outputs/tables/table2_primary_corrected.csv`. Supplementary tables: `outputs/tables/tableS1_survey_weighted.csv` and `outputs/tables/tableS2_resolution_sensitivity.csv`. Corrected figures: `outputs/figures/roc_corrected_oof.pdf` and `outputs/figures/resolution_sensitivity_effect.pdf` (submission package embeds the paired PNGs).

## Manuscript changes

The journal-neutral source `manuscript/current/Manuscript_corrected.md` is the scientific core. It replaces the abstract, introduction, corrected-algorithm methods, cohort/PHQ-9 definitions, statistical analysis, results, discussion, conclusion, declarations, and references. Results are structured as corrected cohort, primary association, survey-weighted sensitivity, temporal-resolution sensitivity, and exploratory cross-validated discrimination.

Removed from the primary scientific story: thermodynamic/potential-landscape framing, GMM/native-minute account, historical regression values, in-sample ROC statements, WASO-null assertion, resolution agreement/interchangeability assertions, and claims of diagnostic, screening, causal, or validated clinical use. The corrected P01 method describes reconstructed local time, validity predicate, complete clock-aligned blocks, participant median threshold, night grouping, exact adjacency, and numerator/denominator. SLD010H is identified as self-reported weekday/workday sleep duration; the PAXPREDM measure remains an unvalidated WASO proxy. Survey analysis is reported as a sensitivity analysis and unweighted HC3 is not presented as nationally representative.

Primary Table 1 and Table 2 and supplementary Tables S1/S2 are transcribed from the verified KAN-174 CSV assets in the core source. Corrected OOF ROC and resolution-effect images are referenced in the core. No recovered DOCX figure was reused.

## Reference audit and automated checks

`manuscript/current/REFERENCE_AUDIT.md` lists each reference, its supported claim, DOI/URL, verification status, and action. Independent verification was completed on 2026-10-16 against PubMed and official CDC/NCHS NHANES pages. The Kroenke PHQ-9 metadata/DOI were confirmed, the generic PAM citation was replaced with the exact PAXMIN_H documentation page, the generic portal citation was replaced with the official 2011–2016 NHANES analytic-guidelines PDF, and DPQ_H/SLQ_H metadata were confirmed. Reference audit status is PASS.

`scripts/verify_manuscript_numbers.py` scans corrected manuscript and submission text/DOCX against the banned historical numbers/claims and checks canonical corrected table inputs. Run with `python scripts/verify_manuscript_numbers.py`. The DOCX generator is `scripts/build_kan175_manuscript.py`; it reads the core Markdown and creates the JADR package.

## Outputs

- Core: `manuscript/current/Manuscript_corrected.md`
- Reference audit: `manuscript/current/REFERENCE_AUDIT.md`
- Manuscript DOCX: `manuscript/submission/Manuscript_JADR_FINAL.docx`
- Cover letter DOCX: `manuscript/submission/Cover_Letter_JADR_FINAL.docx`
- Highlights DOCX: `manuscript/submission/Highlights_JADR_FINAL.docx`
- JADR Markdown sources: `manuscript/submission/Cover_Letter_JADR_FINAL.md`, `manuscript/submission/Highlights_JADR_FINAL.md`
- Number guard: `scripts/verify_manuscript_numbers.py`
- DOCX generator: `scripts/build_kan175_manuscript.py`

Number verification completed: PASS. Six manuscript-source/submission text and DOCX files scanned; obsolete-number hits=0; prohibited-claim hits=0; canonical table structure/value checks passed. LibreOffice converted all three DOCX files to PDF; the manuscript rendered as an 8-page letter-size PDF. Extracted text and PDF image inspection confirmed both corrected figures and manuscript tables are present. Reference verification is now PASS. The corrected Markdown bibliography was updated after independent source verification; the submission DOCX must be rebuilt once from the updated core before KAN-175 can be closed as PASS. The original KAN-175 package commit was `0f96e99f53d98da87e152bcbf878c64dc0e85fcb`; the reference-audit/core-source follow-up commits were `4fdeb877875e8ded4b3ce325e2cbf59e2966e616` and `569b7d053f942269ea1581ffdc291164b123de13`.
