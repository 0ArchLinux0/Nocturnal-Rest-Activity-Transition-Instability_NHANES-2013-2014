#!/usr/bin/env python3
"""Fail closed on historical numbers/claims and reconcile manuscript estimates to canonical CSVs."""
import csv
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "manuscript/current/Manuscript_corrected.md", *sorted((ROOT / "manuscript/submission").rglob("*.md")), *sorted((ROOT / "manuscript/submission").rglob("*.docx"))]
OBSOLETE = ("4422", "10.707", "9.105", "0.624", "0.643", "0.470")
PROHIBITED = ("native 1-minute", "clinical utility", "screening tool", "risk stratification", "validated biomarker", "total sleep time", "\bTST\b", "Sleep_Medication", "potential landscape", "thermodynamic")

def text_from(path):
    if path.suffix == ".docx":
        with zipfile.ZipFile(path) as z:
            root = ElementTree.fromstring(z.read("word/document.xml"))
        return " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
    return path.read_text(encoding="utf-8")

def main():
    hits = []
    for path in FILES:
        text = text_from(path)
        for value in OBSOLETE:
            if re.search(rf"(?<![\d.]){re.escape(value)}(?![\d.])", text):
                hits.append(f"{path.relative_to(ROOT)}: obsolete number {value}")
        for claim in PROHIBITED:
            if re.search(claim, text, re.I):
                hits.append(f"{path.relative_to(ROOT)}: prohibited/review claim {claim}")
    primary = list(csv.DictReader((ROOT / "outputs/tables/table2_primary_corrected.csv").open(encoding="utf-8", newline="")))
    survey = list(csv.DictReader((ROOT / "outputs/tables/tableS1_survey_weighted.csv").open(encoding="utf-8", newline="")))
    roc = list(csv.DictReader((ROOT / "outputs/models/roc_corrected.csv").open(encoding="utf-8", newline="")))
    resolution = list(csv.DictReader((ROOT / "outputs/tables/tableS2_resolution_sensitivity.csv").open(encoding="utf-8", newline="")))
    checks = [len(primary) == 2 and [r["N"] for r in primary] == ["4508", "4048"], len(survey) == 2 and survey[1]["N"] == "4048", len(roc) == 3 and abs(float(roc[0]["AUC_OOF"]) - .61328) < .00001 and abs(float(roc[1]["AUC_OOF"]) - .59089) < .00001 and abs(float(roc[2]["AUC_OOF"]) - .63152) < .00001, len(resolution) == 4 and [r["Resolution (min)"] for r in resolution] == ["5", "10", "30", "60"]]
    if not all(checks):
        hits.append("canonical table/manifest values did not match expected corrected source structure")
    if hits:
        print("NUMBER_GUARD=FAIL")
        print("\n".join(hits))
        print(f"OBSOLETE_NUMBER_HITS={sum(any(v in h for v in OBSOLETE) for h in hits)}")
        print(f"PROHIBITED_CLAIM_HITS={sum(any(re.search(p, h, re.I) for p in PROHIBITED) for h in hits)}")
        return 1
    print("NUMBER_GUARD=PASS")
    print(f"FILES_SCANNED={len(FILES)}")
    print("OBSOLETE_NUMBER_HITS=0")
    print("PROHIBITED_CLAIM_HITS=0")
    print("CANONICAL_CORRECTED_TABLES=PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
