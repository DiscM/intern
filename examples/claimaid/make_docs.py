"""Generate appeal packets from leak-free test cases.

For N cases (half overturned, half upheld in the dataset), write the real
case facts into a health-plan denial letter plus an IMR application as:
  packets/<id>.pdf        (fpdf2, Latin-1 safe)
  packets/<id>.txt        (plain text)
  packets/<id>_scan.pdf   (Pillow image pages, no text layer, slightly crooked)

The letter carries the ReferenceID. The outcome and the reviewer findings
are never written into a packet.
"""
from __future__ import annotations

import argparse
import io
import os
import random

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

import data

PACKETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packets")


def _latin1(s: str) -> str:
    return s.encode("latin-1", "replace").decode("latin-1")


def letter_text(case: dict) -> str:
    d, t = case["diagnosis"], case["treatment"]
    return (
        f"HEALTH PLAN DETERMINATION LETTER\n"
        f"Reference ID: {case['case_id']}\n"
        f"Report year: {case['year']}\n"
        f"Review type: {case['review_type']}\n"
        f"Turnaround: {case['urgency']}\n"
        f"\n"
        f"Dear Member,\n"
        f"\n"
        f"We have completed our review of the treatment you requested:\n"
        f"  Treatment: {t['subcategory']} ({t['category']})\n"
        f"  Diagnosis: {d['subcategory']} ({d['category']})\n"
        f"  Patient age range: {case['patient_age_range']}\n"
        f"\n"
        f"After review, the health plan has decided not to cover the "
        f"requested treatment under your current benefits.\n"
        f"\n"
        f"CASE SUMMARY (facts only)\n"
        f"{'-' * 40}\n"
        f"{case['case_summary']}\n"
        f"{'-' * 40}\n"
        f"\n"
        f"INDEPENDENT MEDICAL REVIEW APPLICATION\n"
        f"If you disagree with this determination, you may file an "
        f"Independent Medical Review (IMR) with the Department of "
        f"Managed Health Care. Enclose this letter and your medical "
        f"records with the application.\n"
    )


def pick_cases(n: int) -> list[dict]:
    cases = data.eval_cases()
    pos = [c for c in cases if c["overturned"]]
    neg = [c for c in cases if not c["overturned"]]
    random.Random(42).shuffle(pos)
    random.Random(43).shuffle(neg)
    half = n // 2
    chosen = pos[: half] + neg[: n - half]
    random.Random(44).shuffle(chosen)
    return chosen


def write_pdf(case: dict, path: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in _latin1(letter_text(case)).split("\n"):
        pdf.cell(0, 5, _latin1(line), new_x="LMARGIN", new_y="NEXT")
    pdf.output(path)


def write_scan(case: dict, path: str, tilt: float) -> None:
    """A croked 'scan': text rendered as image pages, no text layer."""
    text = _latin1(letter_text(case))
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", 20)
    except OSError:
        font = ImageFont.load_default()
    # paginate at ~46 lines per page (image lines, not wraps)
    lines = text.split("\n")
    per_page = 46
    pages = [lines[i:i + per_page] for i in range(0, len(lines), per_page)] or [[""]]
    pdf = FPDF()
    first = True
    for pi, page_lines in enumerate(pages):
        img = Image.new("L", (1240, 1754), 255)  # ~A4 @150dpi
        draw = ImageDraw.Draw(img)
        y = 60
        for ln in page_lines:
            draw.text((70, y), ln, fill=20, font=font)
            y += 26
        img = img.rotate(tilt * (1 if pi % 2 == 0 else -1),
                         expand=True, fillcolor=255)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        if first:
            pdf.add_page()
            first = False
        else:
            pdf.add_page()
        pdf.image(buf, x=10, y=10, w=190)
    pdf.output(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    if not data.available() and not os.path.exists(data.DATA_CSV):
        print(f"downloading {data.DATA_CSV} ...")
        data.download()
    if not data.available():
        print("dataset not available")
        return 1
    os.makedirs(PACKETS, exist_ok=True)
    cases = pick_cases(args.n)
    for c in cases:
        base = os.path.join(PACKETS, c["case_id"])
        text = letter_text(c)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(text)
        write_pdf(c, base + ".pdf")
        write_scan(c, base + "_scan.pdf", tilt=2.0)
        print(f"packets/{c['case_id']}.pdf  .txt  _scan.pdf")
    print(f"wrote {len(cases)} packets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
