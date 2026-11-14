#!/usr/bin/env python3
"""Batch pipeline: run ClaimAid over a folder of documents.

It tells a patient how strong their appeal is, before they give up; it
triages appeals for human reviewers, and it never denies a case alone.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import data
import nace
import committee
import report

SUPPORTED = {
    ".pdf", ".xlsx", ".xlsm", ".xls", ".csv", ".parquet",
    ".docx", ".doc", ".docm", ".png", ".jpg", ".jpeg", ".tiff", ".tif",
    ".webp", ".bmp", ".pptx", ".ppt", ".html", ".htm", ".txt", ".log",
    ".xml", ".vsdx", ".vsd", ".md", ".mdx", ".json", ".eml", ".msg",
    ".oft", ".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".m4a",
    ".flac", ".mpga", ".oga", ".ogg",
}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
GB, MB = 1 << 30, 1 << 20


def size_limit(path: Path) -> int:
    ext = path.suffix.lower()
    if ext in (".csv", ".parquet", ".txt"):
        return 1 * GB
    if ext == ".json":
        return 512 * MB
    return 256 * MB


def collect(paths: list[str]) -> tuple[list[Path], list[tuple[str, str]]]:
    """Supported, size-ok, non-hidden files. Returns (docs, skipped)."""
    docs: list[Path] = []
    skipped: list[tuple[str, str]] = []

    def consider(p: Path) -> None:
        ext = p.suffix.lower()
        if ext not in SUPPORTED:
            skipped.append((str(p), f"unsupported type {ext or '(none)'}"))
            return
        try:
            if p.stat().st_size > size_limit(p):
                skipped.append((str(p), "too large"))
                return
        except OSError as e:
            skipped.append((str(p), f"unreadable: {e}"))
            return
        docs.append(p)

    for src in paths:
        p = Path(src)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(p)
                if any(part.startswith(".") for part in rel.parts):
                    skipped.append((str(f), "hidden"))
                    continue
                consider(f)
        elif p.is_file():
            consider(p)
        else:
            skipped.append((str(p), "not found"))
    return docs, skipped


def prompt_key() -> str | None:
    """Hidden key prompt in an interactive terminal. Enter = mock mode."""
    if not sys.stdin.isatty():
        return None
    print("No API key found.")
    try:
        key = getpass.getpass("NACE_API_KEY (press Enter for mock mode): ")
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not key:
        return None
    try:
        save = input("Save key to .env (permissions 600)? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return key
    if save.strip().lower().startswith("y"):
        env = Path(".env")
        lines = [l for l in env.read_text().splitlines()
                 if not l.startswith("NACE_API_KEY=")]
        lines.append(f"NACE_API_KEY={key}")
        env.write_text("\n".join(lines) + "\n")
        os.chmod(env, 0o600)
        print("saved to .env")
    return key


def extract_doc(path: Path, cache_dir: Path | None) -> dict:
    """NDI extract with a live-mode disk cache keyed by file hash."""
    with open(path, "rb") as f:
        blob = f.read()
    sha = hashlib.sha256(blob).hexdigest()
    if cache_dir is not None:
        hit = cache_dir / f"{sha}.json"
        if hit.exists():
            with open(hit, encoding="utf-8") as f:
                return json.load(f)
    up = nace.upload(nace.pdf_to_image(str(path)) or str(path))
    f = up["result"]["file"]
    job = nace.extract_stable(
        {"type": "workspace_file",
         "workspace_id": f["workspace_id"], "file_id": f["file_id"]},
        nace.ndi_schema(), nace.extract_instructions())
    if job.get("status") != "succeeded":
        raise nace.NaceError(f"document extract failed: {job.get('status')}")
    out = {"result": job["result"], "file": f}
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(hit, "w", encoding="utf-8") as f:
            json.dump(out, f)
    return out


def attach_images(stem: str, attach_dir: str | None) -> tuple[list, list]:
    """Images whose names start with the document's stem (at most 8)."""
    if not attach_dir:
        return [], []
    d = Path(attach_dir)
    if not d.is_dir():
        return [], []
    imgs = sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS
                  and p.stem.startswith(stem))[:8]
    urls = [nace.image_to_data_url(str(p)) for p in imgs]
    return urls, []


def match_outcome(case_id, stem: str):
    for cand in (case_id, stem):
        cand = (str(cand) if cand else "").strip()
        if not cand:
            continue
        try:
            return data.find(cand)["overturned"]
        except KeyError:
            continue
    return None


def process(path: Path, i: int, n: int, args, cache_dir: Path | None) -> dict:
    """One document: extract, run the committee, one printed line, a row."""
    t0 = time.monotonic()
    row = {k: "" for k in (
        "name", "status", "strength", "p_overturn", "next_step",
        "patient_note", "decision", "needs_human", "treatment", "diagnosis",
        "review_type", "case_id", "truth", "correct", "reasons", "votes",
        "summary", "cited_fields", "images_sent", "drex_calls",
        "input_tokens", "image_tokens", "cost_usd", "seconds", "error",
        "file")}
    row["name"] = path.name
    row["file"] = str(path)
    try:
        job = extract_doc(path, cache_dir)
        res_j = job["result"]
        fields = res_j["data"]
        cited = []
        for fd in res_j["fields"]:
            if fd.get("status") != "found" or fd.get("path") == "case_summary":
                continue
            pgs = sorted({c.get("location", {}).get("page")
                          for c in fd.get("citations") or []
                          if c.get("location", {}).get("page") is not None})
            label = fd["path"]
            if pgs:
                label += " (p. " + ", ".join(str(p) for p in pgs) + ")"
            cited.append(label)
        case = data.fields_to_case(fields)
        outcome = match_outcome(fields.get("case_id"),
                                path.stem.replace("_scan", ""))
        images, image_refs = attach_images(path.stem, args.attach)
        ext = path.suffix.lower()
        if (ext in IMG_EXTS or ext == ".pdf") and not images:
            # the document is itself an image (or a PDF rendered to one):
            # reuse the uploaded file
            f = job["file"]
            image_refs = [{"type": "workspace_file",
                           "workspace_id": f["workspace_id"],
                           "file_id": f["file_id"]}]
        res = committee.run(
            case, images=images or None, image_refs=image_refs or None,
            debate_rounds=args.debate, threshold=args.threshold)
        last = list(res["rounds"].values())[-1]
        votes = "; ".join(f"{k}: {v.get('vote', {}).get('choice', '?')}"
                          for k, v in last.items())
        row.update({
            "status": "ok",
            "strength": res["strength"],
            "p_overturn": res["overturn_prob"],
            "next_step": res["next_step"],
            "patient_note": res["patient_note"],
            "decision": res["final"],
            "needs_human": res["needs_human"],
            "treatment": case["treatment"]["subcategory"],
            "diagnosis": case["diagnosis"]["subcategory"],
            "review_type": case["review_type"],
            "case_id": case["case_id"],
            "truth": ("overturned" if outcome else "upheld")
                      if outcome is not None else "",
            "correct": (res["final"] == "overturn") == outcome
                        if outcome is not None else "",
            "reasons": "; ".join(res["reasons"]),
            "votes": votes,
            "summary": (case["case_summary"] or "")[:120],
            "cited_fields": "; ".join(cited),
            "images_sent": res["images_sent"],
            "drex_calls": res["calls"],
            "input_tokens": res["input_tokens"],
            "image_tokens": res["image_tokens"],
            "cost_usd": res["cost_usd"],
            "seconds": round(time.monotonic() - t0, 2),
        })
        mark = ""
        if outcome is not None:
            mark = " ✅" if row["correct"] else " ❌"
        icon = "⚡ fast-track" if not res["needs_human"] else "👤 human review"
        print(f"[{i}/{n}] {path.name} {res['strength']} "
              f"P={res['overturn_prob'] * 100:.0f}% {icon}{mark}")
    except Exception as e:  # one failing document does not stop the batch
        row["status"] = "failed"
        row["error"] = str(e)
        row["seconds"] = round(time.monotonic() - t0, 2)
        print(f"[{i}/{n}] {path.name} FAILED {e}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="files and folders")
    ap.add_argument("--mock", action="store_true", help="mock mode")
    ap.add_argument("--key", help="API key (beats env and .env)")
    ap.add_argument("--model", help="Drex model id")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--debate", type=int, default=1,
                    help="debate rounds: 0 or 1 (default 1)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/")
    ap.add_argument("--attach", help="folder of extra images")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not reuse cached extractions")
    ap.add_argument("--open", action="store_true",
                    help="open report.html in the browser")
    args = ap.parse_args()

    nace.configure(api_key=args.key, mock=args.mock or None,
                   model=args.model)
    mock = nace.is_mock()
    if not mock:
        if nace.configure()["api_key"] is None:
            # --mock not given and no key from --key, env or .env
            if not sys.stdin.isatty():
                print("No API key. Set NACE_API_KEY (or add it to .env), "
                      "pass --key, or use --mock.")
                return 1
            key = prompt_key()
            nace.configure(api_key=key, mock=key is None, model=args.model)
            mock = nace.is_mock()
            if mock:
                print("no key entered — mock mode (random answers)")
    if not mock:
        try:
            nace.list_models()  # fail fast before any document
        except nace.NaceError as e:
            if e.status in (401, 403):
                print("The API key was refused.")
                return 1
            print(f"fail-fast: cannot reach the model list: {e}")
            return 1

    if not args.paths:
        if sys.stdin.isatty():
            try:
                folder = input("Folder: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 1
            args.paths = [folder] if folder else []
        if not args.paths:
            print("no documents to process")
            return 1

    docs, skipped = collect(args.paths)
    for p, why in skipped:
        print(f"skipped: {p} ({why})")
    if not docs:
        print("nothing to process")
        return 1

    out_dir = Path(args.out) / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = None if (mock or args.no_cache) else \
        Path(args.out) / ".extract_cache"

    t0 = time.monotonic()
    rows = [None] * len(docs)

    def worker(idx):
        rows[idx] = process(docs[idx], idx + 1, len(docs), args, cache_dir)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(worker, range(len(docs))))
    seconds = time.monotonic() - t0

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cols = ["name", "status", "strength", "p_overturn", "next_step",
            "patient_note", "decision", "needs_human", "treatment",
            "diagnosis", "review_type", "case_id", "truth", "correct",
            "reasons", "votes", "summary", "cited_fields", "images_sent",
            "drex_calls", "input_tokens", "image_tokens", "cost_usd",
            "seconds", "error", "file"]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    report_path = report.write_report(
        str(out_dir / "report.html"), rows,
        {"date": time.strftime("%Y-%m-%d %H:%M"),
         "model": nace.configure()["model"],
         "mock": mock, "seconds": seconds})

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"{len(rows)} document(s): {n_ok} ok, "
          f"{len(rows) - n_ok} failed — {out_dir}")
    print("Decision support only. It never denies a case alone.")
    if args.open:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(report_path))
    return 1 if n_ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
