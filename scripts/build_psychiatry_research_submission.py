#!/usr/bin/env python3
"""Build editable Psychiatry Research DOCX files from the retargeted sources."""
from pathlib import Path
import zipfile

from build_kan175_manuscript import render_docx

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "manuscript" / "submission" / "psychiatry-research"


def add_page_breaks(path: Path) -> None:
    markers = {
        "3. Results": "Heading2",
        "Table 2. Primary HC3 OLS models for continuous PHQ-9.": "Normal",
    }
    temporary = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as target:
        for entry in source.infolist():
            payload = source.read(entry.filename)
            if entry.filename == "word/document.xml":
                document = payload.decode("utf-8")
                for text, style in markers.items():
                    matches = []
                    start = 0
                    while (begin := document.find("<w:p>", start)) != -1:
                        end = document.find("</w:p>", begin) + len("</w:p>")
                        paragraph = document[begin:end]
                        if text in paragraph:
                            matches.append((begin, end, paragraph))
                        start = end
                    if len(matches) != 1:
                        raise RuntimeError(f"Could not uniquely locate {text!r} in the generated DOCX")
                    begin, end, paragraph = matches[0]
                    old_properties = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
                    new_properties = f'<w:pPr><w:pStyle w:val="{style}"/><w:pageBreakBefore/></w:pPr>'
                    if old_properties not in paragraph:
                        raise RuntimeError(f"Could not add page break before {text!r}")
                    document = document[:begin] + paragraph.replace(old_properties, new_properties, 1) + document[end:]
                payload = document.encode("utf-8")
            target.writestr(entry, payload)
    temporary.replace(path)

def main() -> None:
    sources = {
        "Manuscript_Psychiatry_Research_FINAL.md": "Manuscript_Psychiatry_Research_FINAL.docx",
        "Cover_Letter_Psychiatry_Research_FINAL.md": "Cover_Letter_Psychiatry_Research_FINAL.docx",
        "Highlights_Psychiatry_Research_FINAL.md": "Highlights_Psychiatry_Research_FINAL.docx",
        "Supplementary_Tables_Psychiatry_Research_FINAL.md": "Supplementary_Tables_Psychiatry_Research_FINAL.docx",
    }
    for source_name, output_name in sources.items():
        source = PACKAGE / source_name
        output = PACKAGE / output_name
        render_docx(source.read_text(encoding="utf-8"), output, PACKAGE)
        if output_name == "Manuscript_Psychiatry_Research_FINAL.docx":
            add_page_breaks(output)
    print("Created:", *(name for name in sources.values()))


if __name__ == "__main__":
    main()
