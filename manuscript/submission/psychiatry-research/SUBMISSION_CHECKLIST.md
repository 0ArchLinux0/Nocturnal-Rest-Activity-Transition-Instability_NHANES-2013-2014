# Psychiatry Research submission checklist

## Frozen scientific package

- [x] Target: **Psychiatry Research**; article type: Full-Length Article / Original Research.
- [x] Scientific freeze: `af7dee9dab20673dab006a2b823cf435e25d7bf4`.
- [x] Retarget started from `204d92b92aa4e976d94ca3eb3edd39946372560a`.
- [x] Editorial pass KAN-425 started from `0f1160927095fb5ec55498bf8c0d0b9678b2f6bd` on clean `main`.
- [x] No new analysis or scientific-number change; title, cohort/P01 definitions, and KAN-173/KAN-174 results retained.

## Manuscript and files

- [x] Editable manuscript DOCX and Markdown source include complete title-page information.
- [x] Standalone portal Title Page added: `Title_Page_Psychiatry_Research.docx` with matching Markdown source; all accepted metadata verified.
- [x] Standalone portal Abstract added: `Abstract_Psychiatry_Research.docx` with matching Markdown source; verbatim comparison with the final manuscript abstract passed at 160 words; no author details included.
- [x] Abstract: 160 words; unstructured; <=250-word limit.
- [x] Main text: 2,662 words; Introduction: 627; Discussion: 1,080. Counts use whitespace-delimited words, including in-text citations, excluding headings, table rows, table/figure captions, declarations, and references. Main text meets the preferred 2,500–3,200 range and the 5,000-word ceiling.
- [x] Six English keywords; allowed range 1–7.
- [x] Four major sections: Introduction, Methods, Results, Discussion; former conclusion integrated into Discussion.
- [x] Author-year in-text citations; 21 alphabetized references, including 12 verified additions; zero numbered reference citations. `manuscript/current/REFERENCE_AUDIT.md` records source identifiers, claim support, and limitations for every reference.
- [x] Ethics statement identifies NCHS ERB Protocol #2011-17 continuation, parent-protocol consent, and public de-identified secondary analysis; no local approval number added.
- [x] Funding, competing interests, data availability, consent, CRediT, and generative AI declarations present; AI declaration appears before References.
- [x] Primary tables are editable Word tables; supplementary tables are provided in a separate editable DOCX and Markdown source.
- [x] Table 1 gives available-case denominators from the existing descriptive table and distinguishes PHQ-9 groups from the overall cohort and model samples. WASO is omitted from Table 1 without inventing a descriptive value; Model 2's proxy remains disclosed in Methods.
- [x] ROC and temporal-resolution figures remain byte-identical, embedded and supplied separately as PDF and PNG files.
- [x] Cover letter rendered and visually checked as a one-page document.
- [x] Highlights are separate and editable: 5 bullets; character counts including spaces: 58, 48, 73, 46, 61 (each <=85).
- [x] Reviewer suggestions include two candidates with institutional emails verified against university profiles; repository-level conflict screen completed. Editorial office must confirm undisclosed conflicts before invitation.
- [x] `GRAPHICAL_ABSTRACT=SKIPPED_OPTIONAL` (not required by journal guidance).

## Verification — 17 October 2026

- [x] Earlier recursive number-guard pass: 17 files scanned with `OBSOLETE_NUMBER_HITS=0` and `PROHIBITED_CLAIM_HITS=0`. Final KAN-424 pass: 21 files scanned with the same zero-hit results.
- [x] Complete asset verifier passed against the frozen results; only its manifest-write destination was redirected to a temporary file. All 27 snapshotted scientific artifacts retained their SHA-256 hashes.
- [x] `INTERNAL_REVISION_LANGUAGE_HITS=0` across public Markdown, DOCX and rendered PDFs. Broad substring matches were reviewed: threshold/fold are scientific terms, older occurs in a reviewer publication title, and Gold occurs in the publishing-route instructions. Issue IDs, freeze provenance and audit labels in this checklist/manifest are internal-only.
- [x] `REFERENCE_AUDIT=PASS`; `NUMBERED_REFERENCE_CITATIONS=0`.
- [x] Earlier package pass: all four then-listed DOCX archives passed integrity checks; editable tables: 2 main and 2 supplementary. Supplementary scientific content is unchanged.
- [x] LibreOffice PDFs visually inspected: manuscript 13 pages; cover letter, highlights and supplement one page each. Tables, both figures, expanded prose and reference pages render cleanly.
- [x] Full-Length rationale is explicit in the cover letter. The portal/editor has not been queried; if Letter is required, stop and report `ARTICLE_TYPE_REDIRECT=YES`, without automatic compression.
- [x] Standalone Title Page and Abstract DOCX archives passed ZIP integrity checks. Both rendered to one-page LibreOffice PDFs and were visually inspected for truncation, Unicode and math-symbol rendering.
- [x] The KAN-424 metadata patch changed the final manuscript Markdown and DOCX only for the author affiliation and generative-AI disclosure; title, abstract wording, scientific text and results remain unchanged.
- [x] Jira and Notion updates: **UNAVAILABLE** in this session; final SHA and readiness are reported in the delivery response.


## Final portal metadata patch — KAN-424

- [x] `AFFILIATION=School of Medicine, Sapporo Medical University, Sapporo, Japan`; `AFFILIATION_OLD_HITS=0`; `AFFILIATION_NEW_HITS=10` across submission Markdown, DOCX and rendered PDFs.
- [x] `AI_DECLARATION=PASS`; `AI_DECLARATION_TOOL=OpenAI ChatGPT`; `AI_DECLARATION_PURPOSE=language and structural editing`. The disclosure assigns final review and responsibility to the author.
- [x] The Full-Length rationale uses “explicitly defined”; `PRESPECIFIED_LANGUAGE_HITS=0`.
- [x] Accepted abstract is unchanged: exact source match, 160 words. Title unchanged.
- [x] `SCIENTIFIC_RESULTS_CHANGED=NO`; `NEW_ANALYSES_RUN=NO`. Frozen model/table/figure outputs remain byte-identical; 25 output hashes checked. Abstract, highlights, supplement and separate figures remain byte-identical.
- [x] `NUMBER_VERIFICATION=PASS`; 21 files scanned; obsolete-number hits: 0; prohibited-claim hits: 0.
- [x] Regenerated manuscript, title-page and cover-letter DOCX archives pass ZIP integrity checks. PDFs rendered and visually inspected: manuscript 13 pages, title page 1 page, cover letter 1 page; table/figure layout and Unicode/math rendering retained.
- [x] `FEE_ROUTE=SUBSCRIPTION`; `AUTHOR_PUBLICATION_FEE_TARGET=0`; `PAID_GOLD_OA_ALLOWED=NO`. Article type and readiness status unchanged.

## Zero-fee publishing guard — REQUIRED

```text
TARGET_JOURNAL=Psychiatry Research
PUBLISHING_MODEL=Subscription
OPEN_ACCESS_SELECTED=NO
AUTHOR_PUBLICATION_FEE_TARGET=0
PAID_GOLD_OA_ALLOWED=NO
```

Do not select or approve paid Gold Open Access. At portal submission, confirm the Subscription/non-OA selection and author publication fee remains zero before completing the publishing-choice step.

## Readiness

**READY_TO_SUBMIT_PSYCHIATRY_RESEARCH_AFTER_EDITORIAL_PASS** — subject to final author review, portal fee-route/article-type confirmation and the editor's reviewer-conflict check. A graphical abstract is intentionally skipped as optional. Upload the files listed in `PORTAL_COPYPASTE.md`; inspect the portal-generated proof before author-controlled submission.
