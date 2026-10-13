# Source records and KAN-170 acquisition requirements

`manifest.yaml` contains the eight code-driven XPT source records. Their URLs returned HTTP 200 to HEAD-only requests, so `SOURCE_ENDPOINTS_RESOLVED=8`; no source content was downloaded or inspected, so `SOURCE_CONTENTS_VERIFIED=0`. The current source requirement set has no unresolved raw-source records (`SOURCE_REQUIREMENTS_UNRESOLVED=0`). A HEAD response is not content verification, schema verification, or checksum evidence.

## Acquisition requirements for KAN-170

1. Acquire only the eight XPT records listed in `manifest.yaml`, after confirming official file pages/codebooks and exact variables for the intended analysis.
2. Store approved originals only under `data/raw/`; preserve upstream filenames and avoid duplicate root or `nhanes_2013_2014_raw/` copies.
3. Record retrieval time, final URL, source page/codebook/version, HTTP status, file size, and SHA-256. Verify format and required columns before marking contents verified.
4. Do not acquire PAX80_H: the current tracked analysis does not read it. It is a distinct 80 Hz source, not a synonym for the minute-level PAXMIN_H data.
5. Do not treat the optional PAXRAW_H lookup as a current-cycle source; its code comment warns it may refer to another cycle.

## Legacy code compatibility

Current scripts still read root-level files, `nhanes_2013_2014_raw/`, or `$NHANES_RAW_DIR`. KAN-169 owns migration of those readers to `data/raw/`; this bootstrap did not change analysis scripts. Do not run the legacy download script until KAN-170 validates and acquires the approved set.
