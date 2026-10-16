#!/usr/bin/env python3
"""Build the KAN-175 JADR submission DOCX files from the corrected Markdown sources."""
from __future__ import annotations

import re
import struct
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "submission"
OUT.mkdir(parents=True, exist_ok=True)

COVER = """# Cover Letter — Journal of Affective Disorders Reports

16 October 2026

Editorial Office
Journal of Affective Disorders Reports

**Re: Original Research submission — “Nocturnal Rest-to-Activity Transition Probability and Depressive Symptoms in NHANES 2013–2014: A Cross-Sectional Actigraphy Study”**

Dear Editors,

I submit the enclosed manuscript for consideration as an Original Research article in the *Journal of Affective Disorders Reports*. This cross-sectional secondary analysis of NHANES 2013–2014 examines the association between depressive symptom burden and nocturnal rest-to-activity transition probability (P01), a complementary actigraphy-derived behavioral measure.

P01 was constructed from temporally reconstructed PAM minute data using complete 10-minute activity blocks and exact temporal adjacency within a fixed overnight window. In the primary HC3 OLS analysis, the association was positive in both the demographic-adjusted model (N=4,508) and extended model (N=4,048). A survey-design sensitivity analysis using the MEC examination weight, strata, and primary sampling units supported the direction of the extended-model result. Exploratory stratified five-fold cross-validation yielded a modest incremental AUC of 0.018 for demographics plus P01 versus demographics alone (paired bootstrap 95% CI 0.008–0.030).

The manuscript interprets these findings as cross-sectional associations, not causal or diagnostic evidence. It does not propose P01 for clinical adoption or diagnosis. We believe the work may interest readers studying depressive symptoms, sleep-related behavior, and population actigraphy while maintaining these limits.

The work is original, is not under consideration elsewhere, and uses publicly available NHANES data. I am the sole author and have approved the submitted version. I declare no competing interests and no specific funding for this study.

Thank you for your consideration.

Sincerely,

Minjun Park
Department of Medicine
Sapporo Medical University
Sapporo, Japan
a25m080@sapmed.ac.jp
"""

HIGHLIGHTS = """# Highlights

- Corrected nocturnal P01 was positively associated with continuous PHQ-9.
- The association remained positive in survey-weighted sensitivity analysis.
- P01 associations were positive across four tested block resolutions.
- Adding P01 increased out-of-fold AUC by 0.018 (95% CI 0.008–0.030).
- Incremental cross-validated discrimination was modest.
"""

# XML helpers and a small deterministic Markdown-to-OOXML writer. No external
# converter is required; the generated files are standard WordprocessingML DOCX.
def run(text: str, bold: bool = False, italic: bool = False) -> str:
    props = ("<w:rPr>" + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + "</w:rPr>") if bold or italic else ""
    return f"<w:r>{props}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"

def inline(text: str) -> str:
    parts = re.split(r"(\*\*.*?\*\*|\*[^*]+\*|`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            out.append(run(part[2:-2], bold=True))
        elif part.startswith("*") and part.endswith("*"):
            out.append(run(part[1:-1], italic=True))
        elif part.startswith("`") and part.endswith("`"):
            out.append(run(part[1:-1]))
        elif part:
            out.append(run(part))
    return "".join(out)

def paragraph(text: str, style: str = "Normal") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{inline(text)}</w:p>'

def table_xml(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    cols = len(rows[0])
    grid = "".join('<w:gridCol w:w="%d"/>' % (9000 // cols) for _ in range(cols))
    body = []
    for ri, row in enumerate(rows):
        if all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in row):
            continue
        cells = []
        for cell in row:
            shade = '<w:shd w:fill="D9EAF7"/>' if ri == 0 else ""
            cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{9000 // cols}" w:type="dxa"/>{shade}</w:tcPr><w:p>{inline(cell.strip())}</w:p></w:tc>')
        body.append("<w:tr>" + "".join(cells) + "</w:tr>")
    return '<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr><w:tblGrid>' + grid + "</w:tblGrid>" + "".join(body) + "</w:tbl>"

def image_paragraph(image_id: int, descr: str, width: int, height: int) -> str:
    cx = 5_800_000
    cy = int(cx * height / width)
    return (f'<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{image_id}" name="Figure {image_id}" descr="{escape(descr)}"/>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
            f'<pic:nvPicPr><pic:cNvPr id="{image_id}" name="Figure {image_id}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="rId{image_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')

def render_docx(markdown: str, output: Path, source_dir: Path) -> None:
    lines = markdown.splitlines()
    body = []
    media: list[tuple[int, Path]] = []
    rels = []
    rid = 10
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                image_path = (source_dir / match.group(2)).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                data = image_path.read_bytes()
                width, height = struct.unpack(">II", data[16:24])
                image_id = len(media) + 1
                rels.append(f'<Relationship Id="rId{image_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/figure{image_id}.png"/>')
                media.append((image_id, image_path))
                body.append(image_paragraph(image_id, match.group(1), width, height))
                body.append(paragraph(match.group(1), "Caption"))
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append([x.strip() for x in lines[i].strip().strip("|").split("|")])
                i += 1
            body.append(table_xml(table_lines))
            continue
        if line.startswith("### "):
            body.append(paragraph(line[4:], "Heading3"))
        elif line.startswith("## "):
            body.append(paragraph(line[3:], "Heading2"))
        elif line.startswith("# "):
            body.append(paragraph(line[2:], "Title" if not body else "Heading1"))
        elif line.startswith("- "):
            body.append(paragraph(line[2:], "ListBullet"))
        elif re.match(r"^\d+\. ", line):
            body.append(paragraph(re.sub(r"^\d+\. ", "", line), "Normal"))
        else:
            chunks = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("#", "|", "![", "- ")) and not re.match(r"^\d+\. ", lines[i].strip()):
                chunks.append(lines[i].strip())
                i += 1
            body.append(paragraph(" ".join(chunks)))
            continue
        i += 1
    document = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>''' + "".join(body) + '''<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:sz w:val="25"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/></w:pPr><w:rPr><w:b/><w:sz w:val="23"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:rPr><w:i/><w:sz w:val="20"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="360"/><w:contextualSpacing/></w:pPr></w:style></w:styles>'''
    content_types = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
    rel_root = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>' + "".join(rels) + "</Relationships>"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        zf.writestr("word/document.xml", document)
        zf.writestr("word/styles.xml", styles)
        zf.writestr("word/_rels/document.xml.rels", rel_root)
        for image_id, path in media:
            zf.write(path, f"word/media/figure{image_id}.png")


def main() -> None:
    core_path = ROOT / "manuscript/current/Manuscript_corrected.md"
    core = core_path.read_text(encoding="utf-8")
    render_docx(core, OUT / "Manuscript_JADR_FINAL.docx", core_path.parent)
    (OUT / "Cover_Letter_JADR_FINAL.md").write_text(COVER, encoding="utf-8")
    (OUT / "Highlights_JADR_FINAL.md").write_text(HIGHLIGHTS, encoding="utf-8")
    render_docx(COVER, OUT / "Cover_Letter_JADR_FINAL.docx", OUT)
    render_docx(HIGHLIGHTS, OUT / "Highlights_JADR_FINAL.docx", OUT)
    print("Created:", *(str(path.relative_to(ROOT)) for path in sorted(OUT.glob("*_FINAL.docx"))))

if __name__ == "__main__":
    main()
