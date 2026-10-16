# Psychiatry Research retarget plan

## Decision

The frozen scientific package at `af7dee9dab20673dab006a2b823cf435e25d7bf4` is retargeted from *Journal of Affective Disorders Reports* to *Psychiatry Research*.

Hard publication constraint: **author publication fee = 0**.

Use the journal's **subscription / non-open-access route**. Do not select paid gold open access unless the author explicitly changes this decision.

## Scientific freeze

Do not alter accepted scientific analysis, cohort rules, P01 construction, model outputs, tables, figures, or interpretation.

Canonical readiness gate:
- `docs/SUBMISSION_READINESS.md` = `READY FOR SUBMISSION`
- final reproducibility commit before retarget: `af7dee9dab20673dab006a2b823cf435e25d7bf4`

## Psychiatry Research submission requirements to enforce

Current guide-for-authors requirements verified for retargeting:

- Provide editable source files; Word `.doc/.docx` is acceptable and PDF is not the manuscript source.
- Include complete title-page information.
- Full-Length Articles should not exceed 5,000 words; shorter manuscripts around 3,000 words are preferred.
- Abstract: concise/factual and <=250 words.
- Supply 1–7 English keywords.
- Highlights are required as a separate editable file.
- Highlights: 3–5 bullets, each <=85 characters including spaces.
- A graphical abstract is encouraged, not required.
- Figures/artwork should also be supplied as separate files with logical names (e.g., Figure_1, Figure_2).
- Tables must remain editable text.
- Use the journal reference style: author-year citations in text; alphabetized reference list (not numbered/Vancouver).
- Include declarations for competing interests, funding, data availability, and author contributions.
- Because generative AI was used in manuscript preparation, include a section titled `Declaration of generative AI and AI-assisted technologies in the writing process` before the references, with human-review/responsibility language consistent with the journal guide.
- Preserve the corrected cross-sectional/complementary-measure framing; no diagnostic, screening, validated-biomarker, causal, or clinical-utility claims.
- Human-subjects statement should identify NHANES 2013–2014 as continuation of NCHS ERB Protocol #2011-17 and state that participant consent was obtained under the parent survey protocol.
- Prepare several potential reviewer suggestions with institutional emails; exclude recent collaborators/colleagues and obvious conflicts.

## Deliverables

Create under:

`manuscript/submission/psychiatry-research/`

Required:
- `Manuscript_Psychiatry_Research_FINAL.docx`
- `Cover_Letter_Psychiatry_Research_FINAL.docx`
- `Highlights_Psychiatry_Research_FINAL.docx`
- `Figure_1_ROC.pdf` and/or accepted raster counterpart
- `Figure_2_Resolution.pdf` and/or accepted raster counterpart
- `SUBMISSION_CHECKLIST.md`
- `submission_manifest.json`

Optional:
- graphical abstract only if it improves the package without delaying submission.

## Fee guard

Before actual submission:

- confirm target journal = `Psychiatry Research`
- confirm publishing choice = `Subscription` / non-OA
- confirm author APC/publication fee = `0`
- do not approve a paid open-access option

## Tracking

Jira: KAN-423 — Retarget frozen manuscript to Psychiatry Research — zero-fee subscription route.

Final retarget gate:

`READY_TO_SUBMIT_PSYCHIATRY_RESEARCH`

or blockers.
