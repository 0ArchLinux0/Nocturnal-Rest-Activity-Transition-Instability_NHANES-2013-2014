# Analysis changelog

## 2026-10-14 — KAN-168 repository bootstrap

- Recorded repository baseline and completed code/file inventory.
- Added source endpoint metadata records and code-derived variable mapping; no NHANES data were downloaded or opened.
- Added baseline, inventory, audit, provenance, source, and configuration files; established the explicitly requested empty `data/raw/` root.
- No analysis script was run or changed. No scientific results, cohort definition, P01 calculation, or manuscript text were changed. Existing historical numerical results, if later recovered, remain noncanonical until regenerated.
- Legacy migration is prepared for KAN-169 but not performed. Source acquisition is prepared for KAN-170 but not performed.
- At the initial pass, the supplied directory specification was unavailable; this was resolved by the explicit tree in the continuation request and is no longer a blocker.

## KAN-168 continuation — canonical tree and source requirement resolution

- Created the supplied canonical directory tree without moving or deleting legacy files; retained empty canonical directories with `.gitkeep`.
- Classified PAX80_H as `NOT_USED` from code evidence: current analysis reads minute-level `PAXMIN_H` and does not open PAX80_H. PAX80_H is excluded from KAN-170 acquisition.
- Updated source semantics: eight endpoints resolved by HEAD response; zero source contents verified; zero code-driven source requirements unresolved.
- Provisioned Python 3.12.14 in `.venv`, installed and import-checked only verified dependencies, and removed unused SHAP from the established requirements file.
- No scientific analysis, source-data download, P01 correction, or manuscript rewrite was performed.

## 2026-10-14 — KAN-169 structural-only legacy mapping

- Reconciled all 26 original tracked artifacts against the pre-bootstrap Git tree and updated the legacy inventory, artifact migration map, and script authority map.
- Preserved the acquisition script under `scripts/legacy/` and copied three TeX fragments to `manuscript/archive/legacy/`; all originals remain in place. SHA-256 evidence records byte-identical copies.
- Left root analysis scripts and their helper at their original paths: their `__file__`-relative data/output paths and local imports make relocation or duplicate runnable copies unsafe before caller migration. No script authority was silently changed.
- No algorithm changed.
- No variable definition changed.
- No statistical result changed.
- No manuscript scientific text changed.
- No source data downloaded.
- No P01 boundary correction performed.

## 2026-10-15 — KAN-172 corrected P01 temporal adjacency and block validity

- Implemented one canonical PAXFTIME/PAXSSNMP-based P01 engine for complete clock-aligned 5-, 10-, 30-, and 60-minute blocks. Valid minutes require `PAXQFM==0`, finite nonnegative `PAXMTSM`, and `PAXTSM==60`; overnight edges require exact block-start adjacency within a shared night ID.
- Preserved the cohort rules and PAXDAYM 1–9 analysis window. Migrated the five active scripts to `data/raw/` defaults, removed the resolution script's silent first-200-SEQN fallback, and added source/algorithm/method fingerprint checks for P01 caches.
- Completed the bounded full PAXMIN pass: 4,785 cohort participants and 19,140 participant-resolution rows. All old/corrected values were paired and finite at each resolution. Mean corrected-minus-legacy P01 deltas at 5/10/30/60 minutes were −0.008821, −0.013493, −0.031426, and −0.054073. At 10 minutes, mean absolute delta was 0.013800 and Pearson r was 0.984837.
- Wrote the participant comparison, resolution summary, run manifest, and corrected provenance-validated P01 cache under `outputs/`. The cache is keyed to PAXMIN SHA-256 `4b1aec371ebfca456ff2d386bc21444ddcb9ac06fdc8f13c3ed8589413d053bd` and method fingerprint `7c88634eb2f616c565d57bc1e2669c4a3552ae20d1792f10fca30b6992cabe64`.
- No OLS, logistic, or ROC analysis was run; no publication figure or manuscript was changed. Prior P01-dependent conclusions remain unverified pending the scoped KAN-173 work.

## 2026-10-15 — KAN-173 corrected NHANES P01 rerun

- Rebuilt the full cohort on PAXDAYD 2–8 with the existing ≥4 adequate-day, ≥600 valid-minute/day, and ≥1 weekend-day criteria. The corrected cohort is N=4,882 versus 4,785 in the prior KAN-172 cohort. Extended the provenance-validated KAN-172 P01 cache to all target SEQNs, including 127 SEQNs missing from the prior cache.
- Canonicalized PHQ-9 to DPQ010–DPQ090 only; 7/9 and invalid responses are missing, all nine items are required, and PHQ9_Score ranges 0–27. Added PHQ9_ge_10 for the secondary binary ROC analysis.
- Defined `self_reported_weekday_sleep_hours`, `sleep_disorder_history`, and nonvalidated `WASO_proxy_min`. The WASO proxy groups 22:00–06:00 full nights, excludes invalid and unknown/non-wear minute records, and was used only under that proxy name.
- Corrected primary OLS/HC3 results: Model 1 N=4,508, P01 beta=10.12663 (HC3 95% CI [7.35564, 12.89762], p=9.07e-13); Model 2 N=4,048, beta=5.72278 (HC3 95% CI [2.47934, 8.96622], p=5.47e-4). Both directions and significance are preserved against the supplied historical references; historical Ns were not supplied.
- Ran survey-weighted `svyglm` sensitivity models, corrected 5/10/30/60-minute resolution models, and stratified five-fold out-of-fold ROC with paired bootstrap intervals. Combined demographics+P01 AUC=0.63152 versus baseline 0.61328, ΔAUC=0.01824 (95% CI [0.00823, 0.03001]).
- Scientific impact is `ATTENUATED_BUT_SUPPORTED`: the corrected Model 2 effect is materially smaller than its historical reference but remains positive and significant. Added machine-readable outputs under `outputs/models/` and detailed results in `docs/KAN173_CORRECTED_RESULTS.md`. No manuscript text or figures were changed.
