"""AppTest checks for app.py (mock mode, both case sources).

Run: .venv/bin/python tests/test_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["NACE_MOCK"] = "1"
os.environ.pop("NACE_API_KEY", None)
os.environ.pop("DREX_MODEL", None)
os.environ.pop("NACE_BASE_URL", None)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

FAILURES = 0


def check(name, ok, detail=""):
    global FAILURES
    print(("PASS: " if ok else "FAIL: ") + name + (f"  [{detail}]"
                                                   if detail and not ok
                                                   else ""))
    if not ok:
        FAILURES += 1


def run_source(label: str, value: str) -> None:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    check(f"{label}: app runs without exceptions",
          len(at.exception) == 0,
          str([str(e.value) for e in at.exception]))
    at.radio[0].set_value(value)
    at.run()
    check(f"{label}: source switch without exceptions",
          len(at.exception) == 0,
          str([str(e.value) for e in at.exception]))
    btn = next(b for b in at.button
               if "Convene the committee" in (b.label or ""))
    btn.click()
    at.run()
    check(f"{label}: convene without exceptions",
          len(at.exception) == 0,
          str([str(e.value) for e in at.exception]))
    subs = [s.value for s in at.subheader]
    check(f"{label}: 'Appeal strength' shown", "Appeal strength" in subs,
          f"subheaders={subs}")


def main() -> int:
    run_source("document", "Appeal packet (document)")
    run_source("test case", "Real IMR test case (no NDI)")
    if FAILURES:
        print(f"{FAILURES} check(s) failed")
        return 1
    print("app checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
