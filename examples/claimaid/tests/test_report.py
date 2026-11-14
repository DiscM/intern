"""Checks for the self-contained report.html (report.py).

Run: .venv/bin/python tests/test_report.py
"""
from __future__ import annotations

import html
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import report  # noqa: E402

FAILURES = 0


def check(name, ok, detail=""):
    global FAILURES
    print(("PASS: " if ok else "FAIL: ") + name + (f"  [{detail}]"
                                                   if detail and not ok
                                                   else ""))
    if not ok:
        FAILURES += 1


def main() -> int:
    name = 'a<b>&"x.txt'
    row_ok = {
        "name": name, "status": "ok", "strength": "strong",
        "p_overturn": 0.72, "next_step": "fast-track overturn",
        "patient_note": "Strong appeal: similar cases were often "
                        "overturned. Appealing is worth it.",
        "decision": "overturn", "needs_human": False,
        "treatment": "Surgery & Procedures", "diagnosis": "Oncology",
        "review_type": "Medical Necessity", "case_id": "MN17-26586",
        "truth": "upheld", "correct": False, "reasons": "none",
        "votes": "clinical_reviewer: overturn", "summary": "Patient <a> "
                                                          "requests & more",
        "cited_fields": "treatment_subcategory; case_id",
        "images_sent": 0, "drex_calls": 11, "input_tokens": 5000,
        "image_tokens": 0, "cost_usd": 0.00015, "seconds": 3.2,
        "error": "", "file": name,
    }
    row_failed = {
        "name": "bad.pdf", "status": "failed", "strength": "",
        "p_overturn": 0.0, "next_step": "", "patient_note": "",
        "decision": "", "needs_human": False, "treatment": "",
        "diagnosis": "", "review_type": "", "case_id": "", "truth": "",
        "correct": "", "reasons": "", "votes": "", "summary": "",
        "cited_fields": "", "images_sent": 0, "drex_calls": 0,
        "input_tokens": 0, "image_tokens": 0, "cost_usd": 0.0,
        "seconds": 1.0, "error": "extract failed: <boom>", "file": "bad.pdf",
    }
    rows = [dict(row_ok), dict(row_failed)]

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "report.html")
        report.write_report(path, rows,
                            {"date": "2026-11-14 12:00",
                             "model": "drex-max-v1.1", "mock": True,
                             "seconds": 42.0})
        with open(path, encoding="utf-8") as f:
            html_text = f.read()

        css = html_text.split("</style>")[0]
        check("one file: no scripts, no external assets",
              "<script" not in html_text.lower() and "src=" not in html_text
              and "<link" not in html_text.lower() and "url(" not in css)
        ext = re.findall(r"(?:src|href)=[\"']?(\S+?)[\"'\s>]", html_text)
        ext = [u for u in ext if u.startswith("http")]
        check("no external URLs except links",
              all(u.startswith(("http://", "https://")) for u in ext)
              and len(ext) == 0,
              f"ext={ext}")
        check("values escaped with html.escape",
              html.escape(name) in html_text and name not in html_text
              and "extract failed: &lt;boom&gt;" in html_text)
        check("dark-mode block present",
              "@media (prefers-color-scheme: dark)" in html_text
              and "--bg:" in html_text)
        check("table in an overflow-x:auto container",
              re.search(r"\.table-wrap\s*\{[^}]*overflow-x:\s*auto",
                        html_text) is not None
              and html_text.find("table-wrap")
              < html_text.find("<table"))
        check("mock banner shown",
              "MOCK MODE: random answers" in html_text)
        check("footer disclaimer present",
              "Decision support only. It never denies a case alone"
              in html_text)

        # optional: real browser, 390 px wide
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("SKIP: playwright not installed (390 px check skipped)")
        else:
            try:
                with sync_playwright() as pw:
                    b = pw.chromium.launch()
                    pg = b.new_page(viewport={"width": 390, "height": 800})
                    pg.goto("file://" + os.path.abspath(path))
                    sw = pg.evaluate("document.body.scrollWidth")
                    check("390 px wide: no horizontal page scroll",
                          sw <= 390, f"scrollWidth={sw}")
                    b.close()
            except Exception as e:
                print(f"SKIP: playwright check unavailable ({e})")

    if FAILURES:
        print(f"{FAILURES} check(s) failed")
        return 1
    print("report checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
