#!/usr/bin/env python3
"""doctor.py — environment check.

Prints OK / WARN / FAIL for each item, with the next step when needed.
The API key is only reported as present or absent; it is never printed.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
from pathlib import Path

FAILS = 0


def report(status: str, label: str, detail: str = "") -> None:
    global FAILS
    if status == "FAIL":
        FAILS += 1
    line = f"{status}: {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def _ver_ok(v: str, need: tuple[int, int]) -> bool:
    m = re.match(r"(\d+)\.(\d+)", v or "")
    if not m:
        return False
    return (int(m.group(1)), int(m.group(2))) >= need


def main() -> int:
    # Python version
    ok = sys.version_info >= (3, 10)
    report("OK" if ok else "FAIL", "Python version",
           f"{sys.version.split()[0]} (need 3.10+)"
           + ("" if ok else "; next: install Python 3.10 or newer"))

    # Packages (distribution name: import name, minimum version)
    from importlib import metadata
    pkgs = {"requests": ("requests", None),
            "fpdf2": ("fpdf", None),
            "pillow": ("PIL", (10, 1)),
            "pypdfium2": ("pypdfium2", None),
            "streamlit": ("streamlit", (1, 50))}
    for name, (mod, need) in pkgs.items():
        try:
            importlib.import_module(mod)
            try:
                v = metadata.version(name)
            except metadata.PackageNotFoundError:
                v = str(getattr(importlib.import_module(mod),
                                "__version__", "unknown"))
            ok = need is None or _ver_ok(v, need)
            report("OK" if ok else "WARN", f"package {name}",
                   v if ok else f"have {v}, need >= "
                                f"{need[0]}.{need[1]}; next: "
                                f".venv/bin/pip install -U {name}")
        except ImportError:
            report("FAIL", f"package {name}",
                   "missing; next: .venv/bin/pip install -r requirements.txt")

    # Dataset
    import data
    ok = data.available()
    report("OK" if ok else "FAIL", "dataset data/imr.csv",
           f"{len(data.eval_cases())} leak-free test cases" if ok
           else "missing or unreadable; next: .venv/bin/python data.py "
                "(downloads it)")

    # Packets
    pdir = Path("packets")
    ok = pdir.is_dir() and any(pdir.iterdir())
    report("OK" if ok else "WARN", "packets/",
           "sample packets present" if ok
           else "empty; next: .venv/bin/python make_docs.py")

    # Key (only whether it exists — never the value)
    import nace
    nace._load_dotenv()
    has_key = bool(os.environ.get("NACE_API_KEY"))
    report("OK" if has_key else "WARN", "API key",
           "present (value not printed)" if has_key
           else "absent; mock mode only. Next: put NACE_API_KEY in .env")
    if not has_key:
        report("WARN", "live checks", "skipped (no key)")
        return FAILS
    nace.configure()

    # GET /v1/models
    try:
        models = nace.list_models()
        report("OK", "GET /v1/models", f"{len(models)} models")
    except nace.NaceError as e:
        report("FAIL", "GET /v1/models",
               f"{e}; next: check the key, the network and NACE_BASE_URL")
        return FAILS

    # One small Drex call
    try:
        r = nace.systemone({"probe": True},
                           {"ping": {"type": "noul",
                                     "instructions": "Reply yes."}})
        report("OK", "small Drex call",
               f"model={r['model']} input_tokens="
               f"{r['usage'].get('input_tokens')}")
    except nace.NaceError as e:
        report("FAIL", "small Drex call", str(e))

    # One NDI extract (a small text file)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "probe.txt")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("A patient requests a treatment. The health plan "
                     "denied the request as not medically necessary.")
        try:
            up = nace.upload(f)
            fl = up["result"]["file"]
            job = nace.extract(
                {"type": "workspace_file",
                 "workspace_id": fl["workspace_id"],
                 "file_id": fl["file_id"]},
                nace.ndi_schema(), nace.extract_instructions())
            ok = job.get("status") == "succeeded"
            report("OK" if ok else "FAIL", "NDI extract",
                   "succeeded" if ok else f"status={job.get('status')}")
        except nace.NaceError as e:
            report("FAIL", "NDI extract", str(e))

    print("doctor: " + ("all good" if FAILS == 0 else f"{FAILS} FAIL(s)"))
    return FAILS


if __name__ == "__main__":
    raise SystemExit(main())
