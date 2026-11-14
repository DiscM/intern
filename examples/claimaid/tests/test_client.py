#!/usr/bin/env python3
"""Offline test: run the real nace.py and committee.py against a local fake
Nace server that follows the API rules (401, 429 retry-after-ms, 500 once,
upload grant + queued job, extract 202 then poll-to-succeeded).

Prints one PASS/FAIL line per check; exits 1 on any failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

PORT = 0
LOCK = threading.Lock()
LOG = {
    "systemone_attempts": 0,
    "systemone_images": [],   # number of images per systemone request
    "upload_bodies": [],
    "extract_headers": {},
    "upload_headers": {},
    "polls": {"job-up-1": 0, "job-ex-1": 0},
    "models_calls": 0,
}

EXTRACT_DATA = {
    "case_id": "MN17-26586",
    "year": 2017,
    "review_type": "Medical Necessity",
    "urgency": "Standard",
    "patient_age_range": "11 to 17",
    "diagnosis_category": "Mental Health",
    "diagnosis_subcategory": "Behavioral",
    "treatment_category": "Mental Health",
    "treatment_subcategory": "Intensive Outpt Prog",
    "case_summary": ("A parent requested coverage for intensive outpatient "
                     "program services for a child with a behavioral "
                     "condition, including individual and family therapy "
                     "sessions, after an earlier level of care was "
                     "completed. " * 2).strip(),
}


def extract_fields() -> list:
    fields = []
    for name in ("case_id", "year", "review_type", "urgency",
                 "patient_age_range", "diagnosis_category",
                 "diagnosis_subcategory", "treatment_category",
                 "treatment_subcategory", "case_summary"):
        value = EXTRACT_DATA[name]
        fields.append({
            "path": name,
            "value": value,
            "status": "found" if value is not None else "not_found",
            "confidence": 0.95,
            "citations": [{
                "location": {"kind": "image", "page": 1, "polygons": []},
                "source_text": str(value)[:120],
            }] if name == "case_summary" else [],
        })
    fields[0]["citations"] = [{
        "location": {"kind": "image", "page": 2, "polygons": []},
        "source_text": "MN17-26586",
    }]
    return fields


def _valid_image_item(it) -> bool:
    if isinstance(it, dict):
        return (it.get("type") == "workspace_file"
                and "workspace_id" in it and "file_id" in it)
    if isinstance(it, str):
        if it.startswith("data:image/"):
            return it.startswith(("data:image/png;base64,",
                                  "data:image/jpeg;base64,",
                                  "data:image/webp;base64,"))
        return len(it) > 16  # raw base64
    return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code: int, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def _authed(self) -> bool:
        if not self.path.startswith("/v1/"):
            return True  # the upload URL is a separate (signed) endpoint
        return self.headers.get("Authorization") == "Bearer test-key"

    def do_POST(self):
        if not self._authed():
            self._json(401, {"error": "unauthorized"})
            return
        path = self.path.split("?")[0]
        body = self._body()
        if path == "/v1/systemone":
            with LOCK:
                LOG["systemone_attempts"] += 1
                n = LOG["systemone_attempts"]
            if n == 1:
                self._json(429, {"error": "slow down"},
                           {"retry-after-ms": "120"})
                return
            if n == 2:
                self._json(500, {"error": "boom"})
                return
            req = json.loads(body)
            images = req.get("images") or []
            with LOCK:
                LOG["systemone_images"].append(list(images))
            if len(body) > 4_194_304:
                self._json(413, {"error": "body too large"})
                return
            if images:
                if not req.get("model", "").startswith("drex-max"):
                    self._json(422, {"error": "images unsupported by model"})
                    return
                if len(images) > 8:
                    self._json(422, {"error": "more than 8 images"})
                    return
                if not all(_valid_image_item(i) for i in images):
                    self._json(422, {"error": "invalid image item"})
                    return
            answers = {}
            for qid, q in (req.get("questions") or {}).items():
                t = q.get("type")
                if t == "noul":
                    answers[qid] = {"noul": 0.8}
                elif t == "choice":
                    labels = list((q.get("criteria") or {}).keys())
                    answers[qid] = {"choice": labels[0],
                                    "probabilities": {labels[0]: 1.0}}
                elif t == "score":
                    ncrit = len(q.get("criteria") or [])
                    answers[qid] = {
                        "score": 1.0, "legend": q.get("criteria"),
                        "probabilities": {str(i): 1.0 / ncrit
                                          for i in range(ncrit)}}
            self._json(200, {"model": req.get("model"), "answers": answers,
                             "usage": {"input_tokens": 500},
                             "evaluation_time_ms": 10,
                             "request_id": f"r-{n}"})
        elif path == "/v1/documents/upload-grants":
            self._json(200, {
                "workspace_id": "ws-test",
                "upload_url": f"http://127.0.0.1:{PORT}/upload",
                "token": "t-123",
                "expires_at": 9999999999,
                "max_bytes": 104857600,
            })
        elif path == "/upload":
            LOG["upload_bodies"].append(body)
            LOG["upload_headers"] = dict(self.headers)
            self._json(200, {"id": "job-up-1", "status": "queued"})
        elif path == "/v1/documents/extract":
            LOG["extract_headers"] = dict(self.headers)
            self._json(202, {"id": "job-ex-1", "status": "running"})
        else:
            self._json(404, {"error": "not found"})

    def do_GET(self):
        if not self._authed():
            self._json(401, {"error": "unauthorized"})
            return
        if self.path.startswith("/v1/models"):
            LOG["models_calls"] += 1
            self._json(200, {"models": [
                {"id": "drex-max-v1.1"}, {"id": "drex-max-finance"}]})
            return
        if self.path.startswith("/v1/documents/jobs/"):
            jid = self.path.rsplit("/", 1)[1]
            with LOCK:
                LOG["polls"][jid] = LOG["polls"].get(jid, 0) + 1
                n = LOG["polls"][jid]
            if jid == "job-up-1":
                if n < 2:
                    self._json(200, {"id": jid, "status": "running"})
                else:
                    self._json(200, {"id": jid, "status": "succeeded",
                                     "result": {"file": {
                                         "workspace_id": "ws-test",
                                         "file_id": "f-1"}}})
            elif jid == "job-ex-1":
                if n < 2:
                    self._json(200, {"id": jid, "status": "running"})
                else:
                    self._json(200, {"id": jid, "status": "succeeded",
                                     "result": {"data": EXTRACT_DATA,
                                                "fields": extract_fields()}})
            else:
                self._json(404, {"error": "no such job"})
        else:
            self._json(404, {"error": "not found"})


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    global PORT
    PORT = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    os.environ["NACE_BASE_URL"] = f"http://127.0.0.1:{PORT}"
    os.environ["NACE_API_KEY"] = "test-key"
    os.environ["NACE_MOCK"] = ""
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    os.environ.pop("DREX_MODEL", None)

    import data
    import nace
    import committee

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}: {name}"
              + (f"  [{detail}]" if detail and not ok else ""))

    # 1. wrong key -> 401, no retry, clear error
    nace.configure(api_key="wrong-key")
    err = None
    try:
        nace.systemone({}, {"a": {"type": "noul", "instructions": "x"}})
    except nace.NaceError as e:
        err = e
    check("wrong key rejected with 401 (no retry)",
          err is not None and err.status == 401 and
          LOG["systemone_attempts"] == 0,
          f"err={err!r} attempts={LOG['systemone_attempts']}")

    # 2. 429 (retry-after-ms) then 500 once, then 200
    nace.configure(api_key="test-key")
    r = nace.systemone({"case": {"x": 1}},
                       {"a": {"type": "noul", "instructions": "x"}})
    check("429 retried (retry-after-ms), 500 retried once, then 200",
          r["answers"]["a"]["noul"] == 0.8 and
          LOG["systemone_attempts"] == 3,
          f"attempts={LOG['systemone_attempts']}")

    # 3-5. upload grant -> queued job -> polled; multipart metadata; extract
    with tempfile.TemporaryDirectory() as td:
        letter = os.path.join(td, "letter.txt")
        with open(letter, "w") as f:
            f.write("Reference ID: MN17-26586\n" + "facts " * 60)
        job = nace.upload(letter)
        check("upload grant -> queued job polled to succeeded",
              job.get("status") == "succeeded" and
              job["result"]["file"]["file_id"] == "f-1",
              f"job={job}")
        body = LOG["upload_bodies"][0]
        idx = body.find(b'name="metadata"')
        hdr_start = body.rfind(b"Content-Disposition", 0, idx)
        hdr_end = body.find(b"\r\n\r\n", idx)
        meta_hdr = body[hdr_start:hdr_end] if hdr_start > 0 else b""
        file_ok = b'name="file"; filename="' in body
        check("metadata is a plain text field (no filename=)",
              idx > 0 and b"filename" not in meta_hdr and file_ok,
              meta_hdr.decode("latin-1", "replace"))
        check("upload sends X-Upload-Token",
              LOG["upload_headers"].get("X-Upload-Token") == "t-123",
              f"hdr={LOG['upload_headers'].get('X-Upload-Token')!r}")
        job = nace.extract(
            {"type": "workspace_file", "workspace_id": "ws-test",
             "file_id": "f-1"},
            nace.ndi_schema(), nace.extract_instructions())
        check("extract 202 then poll to succeeded",
              job.get("status") == "succeeded" and
              job["result"]["data"]["case_id"] == "MN17-26586",
              f"job={job.get('status')}")
        check("Idempotency-Key sent on extract",
              LOG["extract_headers"].get("Idempotency-Key", "").startswith(
                  "idem-"),
              f"hdr={LOG['extract_headers'].get('Idempotency-Key')!r}")

        # 6. schema shape
        schema = nace.ndi_schema()
        s = json.dumps(schema)
        flat = all(p.get("type") in ("string", "number") and
                   "properties" not in p and "format" not in p
                   for p in schema["properties"].values())
        check("schema flat: string/number only, no integer/nested/$ref/anyOf",
              flat and '"integer"' not in s and '"$ref"' not in s and
              '"anyOf"' not in s and '"format"' not in s)

        # 7. committee loads the file back nested, with citations
        rc = committee.load_case(letter)
        case = rc["case"]
        check("case comes back nested (diagnosis/treatment dicts)",
              isinstance(case.get("diagnosis"), dict) and
              isinstance(case.get("treatment"), dict) and
              case["treatment"]["subcategory"] == EXTRACT_DATA[
                  "treatment_subcategory"])
        check("citations returned with pages",
              bool(rc["citations"]) and all(
                  "page" in c["location"] for c in rc["citations"]),
              f"citations={rc['citations']}")
        import data as _data
        check("outcome matched by case_id",
              rc["outcome"] == _data.find(EXTRACT_DATA["case_id"])["overturned"],
              f"outcome={rc['outcome']}")

        # 8. full committee run against the fake server
        res = committee.run(case)
        check("committee run completes against fake server (11 calls)",
              res["calls"] == 11 and 0.0 <= res["overturn_prob"] <= 1.0,
              f"calls={res['calls']}")

        # 9. image helpers
        from PIL import Image
        bmp = os.path.join(td, "big.bmp")
        Image.new("RGB", (3000, 3000), (200, 30, 30)).save(bmp)
        u_bmp = nace.image_to_data_url(bmp)
        n_bytes = int((len(u_bmp) - len("data:image/jpeg;base64,")) * 3 / 4)
        check("3000x3000 BMP -> JPEG data URL under 300 KiB budget",
              u_bmp.startswith("data:image/jpeg;base64,") and n_bytes <= 300_000,
              f"n_bytes={n_bytes}")
        png = os.path.join(td, "small.png")
        Image.new("RGB", (64, 64), (1, 2, 3)).save(png)
        u_png = nace.image_to_data_url(png)
        check("small PNG stays a PNG data URL",
              u_png.startswith("data:image/png;base64,"))
        notimg = os.path.join(td, "notimg.png")
        with open(notimg, "wb") as f:
            f.write(b"definitely not an image")
        try:
            nace.image_to_data_url(notimg)
            check("unreadable image raises a clear error", False)
        except nace.NaceError:
            check("unreadable image raises a clear error", True)

        # 10. drex-v2.0 with images refuses before any request
        n_before = len(LOG["systemone_images"])
        nace.configure(api_key="test-key", model="drex-v2.0")
        try:
            nace.systemone({}, {"a": {"type": "noul",
                                       "instructions": "x"}}, images=[u_png])
            check("drex-v2.0 with images raises before any request", False)
        except nace.NaceError:
            check("drex-v2.0 with images raises before any request",
                  len(LOG["systemone_images"]) == n_before)
        nace.configure(api_key="test-key")

        # 11. 9 images are cut to 8
        n_before = len(LOG["systemone_images"])
        c9 = data.find("MN17-26586")
        committee.run(c9, images=[u_png] * 9, debate_rounds=0)
        seen = [len(i) for i in LOG["systemone_images"][n_before:]]
        check("9 images are cut to 8",
              len(seen) == 6 and 8 in seen and max(seen) <= 8,
              f"seen={seen}")

        # 12. with 2 images, only the Clinical Reviewer's requests have them
        n_before = len(LOG["systemone_images"])
        committee.run(c9, images=[u_png, u_bmp], debate_rounds=0)
        seen = [len(i) for i in LOG["systemone_images"][n_before:]]
        check("with 2 images only the Clinical Reviewer gets them",
              len(seen) == 6 and sorted(seen) == [0, 0, 0, 0, 0, 2],
              f"seen={seen}")

        # 13. committee.py photo.jpg sends one workspace_file item
        photo = os.path.join(td, "photo.jpg")
        Image.new("RGB", (64, 64), (9, 9, 9)).save(photo)
        env = dict(os.environ, NACE_API_KEY="test-key",
                   NACE_BASE_URL=f"http://127.0.0.1:{PORT}", NACE_MOCK="",
                   NO_PROXY="127.0.0.1,localhost")
        n_before = len(LOG["systemone_images"])
        proc = subprocess.run(
            [sys.executable, "committee.py", photo], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=300)
        with_ref = [i for i in LOG["systemone_images"][n_before:]
                    if len(i) == 1 and i[0].get("type") == "workspace_file"]
        check("committee.py photo.jpg sends one workspace_file item",
              proc.returncode == 0 and len(with_ref) == 2,
              f"rc={proc.returncode} out={proc.stdout[-200:]} "
              f"ref_calls={len(with_ref)}")

        # 14. run.py end to end against the fake server
        outd = os.path.join(td, "runout")
        proc = subprocess.run(
            [sys.executable, "run.py", td, "--out", outd,
             "--workers", "2"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        runs = [d for d in os.listdir(outd) if d.startswith("run_")] \
            if os.path.isdir(outd) else []
        rd = os.path.join(outd, runs[0]) if runs else ""
        has3 = rd and all(os.path.exists(os.path.join(rd, f))
                          for f in ("results.jsonl", "results.csv",
                                    "report.html"))
        check("run.py end to end: rc 0 and 3 outputs in the run dir",
              proc.returncode == 0 and has3,
              f"rc={proc.returncode} out={proc.stdout[-200:]} "
              f"err={proc.stderr[-200:]} runs={runs}")

        # 15. a wrong key stops run.py with exit code 1
        bad_env = dict(env, NACE_API_KEY="bad-key")
        proc = subprocess.run(
            [sys.executable, "run.py", td, "--out",
             os.path.join(td, "runout2")],
            cwd=ROOT, env=bad_env, capture_output=True, text=True,
            timeout=300)
        check("run.py with a wrong key stops with exit code 1",
              proc.returncode == 1 and "refused" in proc.stdout,
              f"rc={proc.returncode} out={proc.stdout[-200:]}")

    server.shutdown()
    failed = [c for c in checks if not c[1]]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
