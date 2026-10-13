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
