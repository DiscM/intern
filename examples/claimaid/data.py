"""IMR appeal-strength project: real California DMHC IMR data, leak-free.

Rules:
- Leakage: Findings, DaysToReview, DaysToAdopt, PatientGender are never
  given to any model. to_case() drops them.
- Download goes through requests (urllib gets HTTP 403 on the S3 redirect),
  writing to a temp file first, then renaming.
"""
from __future__ import annotations

import csv
import html
import json
import os
import random
import re
import tempfile

import requests

CSV_URL = (
    "https://data.chhs.ca.gov/dataset/b79b3447-4c10-4ae6-84e2-1076f83bb24e"
    "/resource/3340c5d7-4054-4d03-90e0-5f44290ed095"
    "/download/independent-medical-review-imr-determinations-trends.csv"
)
DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "imr.csv")

# Finding text that must be stripped before a case is leak-free.
_SUMMARY_START = "Nature of Statutory Criteria/Case Summary:"
_SUMMARY_END = "Findings:"
_FORBIDDEN = re.compile(
    r"overturn|upheld|uphold|reviewer|medically necessary|"
    r"likely to be more beneficial|not indicated|is indicated|"
    r"is supported|not supported|appropriate",
    re.I,
)

ROWS: list[dict] | None = None
TRAIN_IDX: list[int] | None = None
TEST_IDX: list[int] | None = None


def download(path: str = DATA_CSV, timeout: int = 600) -> str:
    """Download the IMR CSV via requests to a temp file, then rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with requests.get(CSV_URL, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return path


def extract_summary(findings: str) -> str | None:
    """Leak-free case summary: text between the Summary header and Findings:
    (after html.unescape), or None if it is too short or leaks a conclusion."""
    if not findings:
        return None
    text = html.unescape(findings)
    i = text.find(_SUMMARY_START)
    if i < 0:
        return None
    j = text.find(_SUMMARY_END, i)
    if j < 0:
        return None
    s = text[i + len(_SUMMARY_START):j].strip()
    if len(s) < 150 or _FORBIDDEN.search(s):
        return None
    return s


def to_case(row: dict) -> dict:
    """A leak-free case: no Findings, no Days*, no PatientGender."""
    case = {
        "case_id": (row.get("ReferenceID") or "").strip(),
        "year": _as_number(row.get("ReportYear")),
        "review_type": (row.get("Type") or "").strip(),
        "urgency": (row.get("IMRType") or "").strip(),
        "patient_age_range": (row.get("AgeRange") or "").strip(),
        "diagnosis": {
            "category": (row.get("DiagnosisCategory") or "").strip(),
            "subcategory": (row.get("DiagnosisSubCategory") or "").strip(),
        },
        "treatment": {
            "category": (row.get("TreatmentCategory") or "").strip(),
            "subcategory": (row.get("TreatmentSubCategory") or "").strip(),
        },
        "case_summary": extract_summary(row.get("Findings") or ""),
        "overturned": (row.get("Determination") or "").strip().startswith("Overturned"),
    }
    return case


def _as_number(v) -> float | int | None:
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return None


def _load(force: bool = False) -> bool:
    """Load, split and index the dataset once. False on failure."""
    global ROWS, TRAIN_IDX, TEST_IDX
    if ROWS is not None and not force:
        return True
    if not os.path.exists(DATA_CSV):
        return False
    with open(DATA_CSV, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)]
    if not rows:
        return False
    idx = list(range(len(rows)))
    random.Random(42).shuffle(idx)
    cut = int(len(rows) * 0.7)
    ROWS, TRAIN_IDX, TEST_IDX = rows, idx[:cut], idx[cut:]
    return True


def available() -> bool:
    """True if the dataset loads; otherwise the app runs without the Actuary."""
    return _load()


def find(case_id: str) -> dict:
    if not _load():
        raise KeyError(f"case not found: {case_id!r} (dataset not loadable)")
    for i in (TRAIN_IDX or []) + (TEST_IDX or []):
        if (ROWS[i].get("ReferenceID") or "").strip() == case_id:
            return to_case(ROWS[i])
    raise KeyError(f"case not found: {case_id!r}")


def eval_cases() -> list[dict]:
    """Leak-free benchmark cases: test rows that have a case summary."""
    _load()
    return [to_case(ROWS[i]) for i in TEST_IDX if to_case(ROWS[i])["case_summary"]]


def train_cases() -> list[dict]:
    _load()
    return [to_case(ROWS[i]) for i in TRAIN_IDX]


def case_to_fields(case: dict) -> dict:
    """Nested case -> flat NDI fields."""
    return {
        "case_id": case["case_id"],
        "year": case["year"],
        "review_type": case["review_type"],
        "urgency": case["urgency"],
        "patient_age_range": case["patient_age_range"] or "unknown",
        "diagnosis_category": case["diagnosis"]["category"],
        "diagnosis_subcategory": case["diagnosis"]["subcategory"],
        "treatment_category": case["treatment"]["category"],
        "treatment_subcategory": case["treatment"]["subcategory"],
        "case_summary": case["case_summary"] or "",
    }


def fields_to_case(fields: dict) -> dict:
    """Flat NDI fields -> nested case."""
    return {
        "case_id": fields.get("case_id") or "",
        "year": fields.get("year"),
        "review_type": fields.get("review_type") or "",
        "urgency": fields.get("urgency") or "",
        "patient_age_range": fields.get("patient_age_range") or "unknown",
        "diagnosis": {
            "category": fields.get("diagnosis_category") or "",
            "subcategory": fields.get("diagnosis_subcategory") or "",
        },
        "treatment": {
            "category": fields.get("treatment_category") or "",
            "subcategory": fields.get("treatment_subcategory") or "",
        },
        "case_summary": fields.get("case_summary") or "",
        "overturned": None,  # unknown until matched to the dataset
    }


def _train_rates() -> dict:
    """Overturn rates computed on train rows only."""
    _load()
    groups: dict[str, dict[str, list[int]]] = {
        "treatment_subcategory": {},
        "treatment_category": {},
        "diagnosis_subcategory": {},
        "diagnosis_category": {},
        "review_type": {},
        "urgency": {},
    }
    keymap = {
        "treatment_subcategory": "TreatmentSubCategory",
        "treatment_category": "TreatmentCategory",
        "diagnosis_subcategory": "DiagnosisSubCategory",
        "diagnosis_category": "DiagnosisCategory",
        "review_type": "Type",
        "urgency": "IMRType",
    }
    n = k = 0
    for i in TRAIN_IDX:
        row = ROWS[i]
        y = 1 if (row.get("Determination") or "").startswith("Overturned") else 0
        n += 1
        k += y
        for gname, col in keymap.items():
            v = (row.get(col) or "").strip() or "unknown"
            cells = groups[gname].setdefault(v, [0, 0])
            cells[0] += 1
            cells[1] += y
    base = k / n if n else None
    return {"base_rate": base, "groups": groups}


def history_for(fields: dict, min_rows: int = 20) -> dict:
    """Train overturn rates for this case's field values (>=20 train rows)."""
    if not _load():
        return {}
    stats = _train_rates()
    col = {
        "treatment_subcategory": "TreatmentSubCategory",
        "treatment_category": "TreatmentCategory",
        "diagnosis_subcategory": "DiagnosisSubCategory",
        "diagnosis_category": "DiagnosisCategory",
        "review_type": "Type",
        "urgency": "IMRType",
    }
    out = {"base_rate": stats["base_rate"]}
    for gname, cname in col.items():
        v = str(fields.get(gname) or "").strip() or "unknown"
        if gname in ("review_type", "urgency"):
            v = (fields.get(cname) or "").strip() or "unknown"
        n, k = stats["groups"].get(gname, {}).get(v, (0, 0))
        if n >= min_rows:
            out[gname] = {"value": v, "overturn_rate": k / n, "n_train": n}
    return out


def rule_score(fields: dict) -> float:
    """Train overturn rate of the TreatmentSubCategory (base rate below 10 rows)."""
    if not _load():
        return 0.5
    stats = _train_rates()
    v = (fields.get("treatment_subcategory") or "").strip() or "unknown"
    n, k = stats["groups"]["treatment_subcategory"].get(v, (0, 0))
    if n < 10:
        return stats["base_rate"] or 0.5
    return k / n


def _main() -> None:
    if not os.path.exists(DATA_CSV):
        print(f"downloading {DATA_CSV} ...")
        download()
    if not _load():
        print(f"dataset not loadable: {DATA_CSV}")
        raise SystemExit(1)
    evals = eval_cases()
    k = sum(1 for c in evals if c["overturned"])
    print(f"cases loaded: {len(ROWS)}")
    print(f"train rows: {len(TRAIN_IDX)}  test rows: {len(TEST_IDX)}")
    print(f"base rate (train): {sum(1 for i in TRAIN_IDX if (ROWS[i].get('Determination') or '').startswith('Overturned'))/len(TRAIN_IDX):.3f}")
    print(f"leak-free test (benchmark) cases: {len(evals)} (overturned: {k})")


if __name__ == "__main__":
    _main()
