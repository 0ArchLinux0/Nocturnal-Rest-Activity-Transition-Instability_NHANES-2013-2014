#!/usr/bin/env bash
# =============================================================================
# Re-download NHANES 2013-2014 public files used for PAM / sleep / covariates.
# Official CDC sources only. Run locally (we cannot push GB of data into your Mac).
#
# Usage:
#   chmod +x download_nhanes_2013_2014.sh
#   ./download_nhanes_2013_2014.sh                    # saves to ./nhanes_2013_2014_raw
#   ./download_nhanes_2013_2014.sh /path/to/folder  # custom output directory
#
# Options (env):
#   SKIP_MINUTE=1   Skip PAXMIN_H.xpt (~8.7 GB) if you only need hourly + questionnaires.
# =============================================================================
set -euo pipefail

OUT_DIR="${1:-$(pwd)/nhanes_2013_2014_raw}"
mkdir -p "$OUT_DIR"

# wwwn.cdc.gov mirrors; large minute file is on FTP (same as NHANES data page).
BASE="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles"
PAXMIN_URL="https://ftp.cdc.gov/pub/NHANES/LargeDataFiles/PAXMIN_H.xpt"

download_one() {
  local url="$1"
  local fname="$2"
  local dest="${OUT_DIR}/${fname}"
  if [[ -s "${dest}" ]]; then
    echo ">>> skip (already have): ${fname}"
    echo ""
    return 0
  fi
  echo ">>> ${fname}"
  echo "    ${url}"
  # -C - resume partial download; -L follow redirects; -f fail on HTTP error
  curl -fL --continue-at - --retry 5 --retry-delay 15 \
    --connect-timeout 60 \
    -o "${dest}.part" "${url}"
  mv "${dest}.part" "${dest}"
  echo "    saved: ${dest}"
  echo ""
}

echo "Output directory: ${OUT_DIR}"
echo ""

# --- Physical Activity Monitor (Examination) ---
# PAXMIN_H: 1-minute MIMS (required for nhanes_potential_landscape_minute.py)
if [[ "${SKIP_MINUTE:-0}" != "1" ]]; then
  download_one "${PAXMIN_URL}" "PAXMIN_H.xpt"
else
  echo ">>> Skipping PAXMIN_H.xpt (SKIP_MINUTE=1)"
  echo ""
fi

download_one "${BASE}/PAXHR_H.xpt" "PAXHR_H.xpt"
download_one "${BASE}/PAXDAY_H.xpt" "PAXDAY_H.xpt"
download_one "${BASE}/PAXHD_H.xpt" "PAXHD_H.xpt"

# --- Demographics & body (BMI) ---
download_one "${BASE}/DEMO_H.xpt" "DEMO_H.xpt"
download_one "${BASE}/BMX_H.xpt" "BMX_H.xpt"

# --- Questionnaires typical for your manuscript pipeline ---
download_one "${BASE}/DPQ_H.xpt" "DPQ_H.xpt"
download_one "${BASE}/SLQ_H.xpt" "SLQ_H.xpt"

echo "Done. Files in: ${OUT_DIR}"
echo ""
echo "Minute pipeline: copy or symlink PAXMIN_H.xpt next to scripts, e.g.:"
echo "  ln -sf \"${OUT_DIR}/PAXMIN_H.xpt\" \"$(pwd)/PAXMIN_H.xpt\""
echo ""
echo "Read .xpt in Python, e.g.:"
echo "  import pandas as pd"
echo "  df = pd.read_sas('${OUT_DIR}/PAXHR_H.xpt', format='xport')"
echo ""
echo "Minute file is large (~8.7 GB). If download fails, re-run the same command;"
echo "curl will resume from the partial file (.part is renamed when complete)."
