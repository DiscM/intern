"""Nace API client: Drex (systemone) and NDI document extract.

Rules implemented here:
- Authorization: Bearer <NACE_API_KEY>; key from environment or .env
  (the real environment wins). NACE_BASE_URL overrides the base URL.
- Retry on 429/503/529 (wait on retry-after-ms for Drex, Retry-After in
  seconds for NDI, plus jitter, up to 6 tries), retry a 500 once, backoff
  on network errors. Never retry 401, 402 or 422.
- Bodies are bytes/JSON, never open files, so a retry resends the same data.
- Mock mode (no key, or NACE_MOCK=1): no network calls; fake answers in the
  real shapes, stable for the same input (seed from a hash).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import tempfile
import time
from typing import Any

import requests

DEFAULT_BASE_URL = "https://console.nace.ai"
DEFAULT_MODEL = "drex-max-v1.1"
MAX_QUESTIONS = 512
MAX_OPTIONS = 255
MAX_BODY_BYTES = 4_194_304
COST_PER_MTOK = 0.03  # USD per million input tokens (estimate)
RETRY_STATUSES = (429, 503, 529)
NO_RETRY_STATUSES = (401, 402, 422)
VISION_MODELS = ("drex-max",)  # only drex-max-* accept images
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

_state: dict[str, Any] = {
    "api_key": None,
    "base_url": DEFAULT_BASE_URL,
    "model": DEFAULT_MODEL,
    "mock": False,
    "configured": False,
}

# mock-mode registry: file_id -> local path
_MOCK_UPLOADS: dict[str, str] = {}


class NaceError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def _load_dotenv() -> None:
    """Load .env; the real environment wins."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def configure(api_key: str | None = None, mock: bool | None = None,
              model: str | None = None) -> dict:
    _load_dotenv()
    if api_key is None:
        api_key = os.environ.get("NACE_API_KEY")
    if mock is None:
        mock = os.environ.get("NACE_MOCK") == "1" or not api_key
    _state.update(
        api_key=api_key,
        mock=bool(mock),
        model=model or os.environ.get("DREX_MODEL") or DEFAULT_MODEL,
        base_url=os.environ.get("NACE_BASE_URL") or DEFAULT_BASE_URL,
        configured=True,
    )
    return dict(_state)


def _ensure_configured() -> None:
    if not _state.get("configured"):
        configure()


def is_mock() -> bool:
    _ensure_configured()
    return _state["mock"]


def cost_usd(usage: dict) -> float:
    """Estimated cost: $0.03 per million input tokens (an estimate).
    TODO(verify): usage may also carry image_tokens; it is not clear if
    input_tokens already includes them, so the cost uses input_tokens only
    and callers log image_tokens separately."""
    return usage.get("input_tokens", 0) / 1_000_000 * COST_PER_MTOK


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def list_models() -> list:
    """GET /v1/models (fail-fast key check). Mock: the known models."""
    _ensure_configured()
    if _state["mock"]:
        return [DEFAULT_MODEL, "drex-max-finance"]
    resp = _request("GET", "/v1/models")
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        data = data.get("models", data.get("data", []))
    if isinstance(data, dict):
        return [m["id"] for m in data.get("models", [])]
    return [m if isinstance(m, str) else m.get("id", str(m)) for m in data]


def model_supports_vision(model: str | None = None) -> bool:
    """Only drex-max-* models accept images (others give 422)."""
    return (model or _state["model"] or "").startswith("drex-max")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def image_to_data_url(path: str, max_bytes: int = 300_000) -> str:
    """A small PNG/JPEG/WebP is kept as-is; anything else becomes JPEG
    (quality >= 60, shrunk 20% per step, until it fits or the short side
    reaches 400 px). Raises on unreadable images."""
    with open(path, "rb") as f:
        return data_url_from_bytes(f.read(), max_bytes, name=path)


def data_url_from_bytes(blob: bytes, max_bytes: int = 300_000,
                        name: str = "image") -> str:
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(blob))
    except Exception as e:
        raise NaceError(f"not a readable image: {name} ({e})") from e
    img.load()
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        raise NaceError(f"image too large: {w * h} pixels > {MAX_IMAGE_PIXELS}")
    fmt = (img.format or "").upper()
    ext = os.path.splitext(name)[1].lower()
    ext_ok = ext in (".png", ".jpg", ".jpeg", ".webp")
    if len(blob) <= max_bytes and fmt in ("PNG", "JPEG", "WEBP") and ext_ok:
        kind = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[fmt]
        return f"data:image/{kind};base64," + base64.b64encode(blob).decode()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    quality = 60
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= max_bytes or min(img.size) <= 400:
            url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            if len(url.encode()) > MAX_IMAGE_BYTES:
                raise NaceError(
                    f"image too large to send: {len(url.encode())} > "
                    f"{MAX_IMAGE_BYTES} bytes even after compression")
            return url
        img = img.resize((max(1, int(img.width * 0.8)),
                          max(1, int(img.height * 0.8))))


def pdf_to_image(path: str, scale: float = 2.0, quality: int = 85) -> str | None:
    """Render every page of a PDF into one stacked JPEG and return the
    temp file path, so NDI reads the document exactly like an image
    document (no text layer, same OCR/vision path). None for non-PDFs
    or when rendering is unavailable/fails."""
    if os.path.splitext(path)[1].lower() != ".pdf":
        return None
    try:
        import pypdfium2 as pdfium
        from PIL import Image
    except ImportError:
        return None

    def render(sc: float) -> list:
        pdf = pdfium.PdfDocument(path)
        try:
            return [pdf[i].render(scale=sc).to_pil().convert("RGB")
                    for i in range(len(pdf))]
        finally:
            pdf.close()

    try:
        imgs = render(scale)
        if sum(im.width * im.height for im in imgs) > 30_000_000:
            imgs = render(1.0)  # stay under the 40 MP per-image cap
        w = max(im.width for im in imgs)
        h = sum(im.height for im in imgs)
        canvas = Image.new("RGB", (w, h), "white")
        y = 0
        for im in imgs:
            canvas.paste(im, (0, y))
            y += im.height
        stem = os.path.splitext(os.path.basename(path))[0]
        tmp = os.path.join(tempfile.gettempdir(), f"drex_{stem}_pages.jpg")
        canvas.save(tmp, format="JPEG", quality=quality)
        return tmp
    except Exception:
        return None


def image_ref(source) -> dict | str:
    """A local path -> data URL; an NDI workspace_file source -> an image
    item referencing the already-uploaded file (no second upload)."""
    if isinstance(source, dict):
        if source.get("type") == "workspace_file" and "workspace_id" in source \
                and "file_id" in source:
            return source
        raise NaceError(f"bad image item: {source!r}")
    return image_to_data_url(source)


def _retry_wait_ms(resp: requests.Response) -> float:
    v = resp.headers.get("retry-after-ms")
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    v = resp.headers.get("Retry-After")
    if v:
        try:
            return float(v) * 1000.0
        except ValueError:
            pass
    return 1000.0


def _request(method: str, path: str, *, absolute: bool = False, auth: bool = True,
             headers: dict | None = None, json_body: Any = None,
             data: Any = None, files: Any = None,
             max_retries: int = 6, timeout: int = 180) -> requests.Response:
    """One request with the retry rules. Returns the response (any status)."""
    _ensure_configured()
    url = path if absolute else _state["base_url"].rstrip("/") + path
    hdrs = dict(headers or {})
    if auth:
        hdrs["Authorization"] = "Bearer " + (_state["api_key"] or "")

    tries = 0
    five_tried = False
    while True:
        tries += 1
        try:
            resp = requests.request(
                method, url, headers=hdrs,
                json=json_body, data=data, files=files, timeout=timeout,
            )
        except requests.RequestException as e:
            if tries > max_retries:
                raise NaceError(f"network error after {tries} tries: {e}") from e
            time.sleep(min(30.0, 0.5 * 2 ** (tries - 1)) + random.uniform(0, 0.25))
            continue

        code = resp.status_code
        if code in NO_RETRY_STATUSES:
            raise NaceError(f"HTTP {code}: {resp.text[:300]}", status=code)
        if code in RETRY_STATUSES:
            if tries > max_retries:
                raise NaceError(f"HTTP {code} after {tries} tries", status=code)
            wait = _retry_wait_ms(resp) / 1000.0
            time.sleep(wait + random.uniform(0, 0.1 * wait + 0.05))
            continue
        if code == 500 and not five_tried:
            five_tried = True  # retry a 500 once
            continue
        return resp


# ---------------------------------------------------------------------------
# Drex: /v1/systemone
# ---------------------------------------------------------------------------

def _mock_seed(method: str, path: str, **kw: Any) -> int:
    raw = json.dumps(
        {"m": method, "p": path, **kw}, sort_keys=True, default=str,
    ).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _mock_systemone(body: dict) -> dict:
    seed = _mock_seed("POST", "/v1/systemone", **body)
    rng = random.Random(seed)
    answers = {}
    for qid, q in body.get("questions", {}).items():
        qtype = q.get("type")
        if qtype == "noul":
            answers[qid] = {"noul": round(rng.random(), 3)}
        elif qtype == "choice":
            labels = list((q.get("criteria") or {}).keys())[:MAX_OPTIONS] or ["a"]
            probs = [rng.random() for _ in labels]
            tot = sum(probs) or 1.0
            probs = [p / tot for p in probs]
            answers[qid] = {
                "choice": labels[rng.randrange(len(labels))],
                "probabilities": {l: round(p, 3) for l, p in zip(labels, probs)},
            }
        elif qtype == "score":
            crit = list(q.get("criteria") or [])[:MAX_OPTIONS]
            probs = [rng.random() for _ in crit]
            tot = sum(probs) or 1.0
            probs = [p / tot for p in probs]
            answers[qid] = {
                "score": round(sum(p * i for i, p in enumerate(probs)), 3),
                "legend": crit,
                "probabilities": {str(i): round(p, 3) for i, p in enumerate(probs)},
            }
    n_q = len(body.get("questions", {}))
    n_img = len(body.get("images") or [])
    # the number of images changes the fake answers (they are in the seed)
    usage = {"input_tokens": 300 + 120 * n_q + 800 * n_img
             + rng.randrange(0, 100),
             "output_tokens": 50 + 20 * n_q,
             "image_tokens": 800 * n_img}
    return {
        "model": body.get("model"),
        "answers": answers,
        "usage": usage,
        "evaluation_time_ms": 250 + rng.randrange(0, 900),
        "request_id": f"mock-{seed:016x}",
    }


def systemone(state: Any, questions: dict, model: str | None = None,
              images: list | None = None) -> dict:
    """One Drex call. state: any JSON. questions: {id: typed question}.
    images: at most 8 items (data URL, base64 bytes, or a workspace_file
    ref). Named "Image 1".."Image N" in text, in array order."""
    _ensure_configured()
    if len(questions) > MAX_QUESTIONS:
        raise NaceError(f"too many questions: {len(questions)} > {MAX_QUESTIONS}")
    for q in questions.values():
        crit = q.get("criteria")
        if crit is not None and len(crit) > MAX_OPTIONS:
            raise NaceError(f"too many options: {len(crit) if crit else 0}"
                            f" > {MAX_OPTIONS}")
    images = (images or [])[:MAX_IMAGES]
    if images and not (model or _state["model"]).startswith("drex-max"):
        raise NaceError(
            f"images are only supported by drex-max models, not "
            f"{model or _state['model']!r}; refusing before the request")
    for it in images:
        if isinstance(it, dict):
            ok = it.get("type") == "workspace_file" and "workspace_id" in it \
                and "file_id" in it
        elif isinstance(it, str) and it.startswith("data:image/"):
            ok = it.startswith(("data:image/png;base64,",
                                "data:image/jpeg;base64,",
                                "data:image/webp;base64,"))
        elif isinstance(it, (str, bytes)) and it:
            ok = True  # raw base64
        else:
            ok = False
        if not ok:
            raise NaceError(f"invalid image item: {str(it)[:80]!r}")
    body = {"model": model or _state["model"], "state": state,
            "questions": questions}
    if images:
        body["images"] = images
    if _state["mock"]:
        return _mock_systemone(body)
    raw = json.dumps(body, sort_keys=True).encode()
    if len(raw) > MAX_BODY_BYTES:
        raise NaceError(f"body too large: {len(raw)} > {MAX_BODY_BYTES}")
    resp = _request("POST", "/v1/systemone", json_body=body)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# NDI: documents upload + extract
# ---------------------------------------------------------------------------

_TEXT_EXTS = {".txt", ".md", ".html", ".eml", ".csv", ".json", ".xml"}


def ndi_schema() -> dict:
    """Flat NDI schema: string and number only, no format/$ref/anyOf."""
    from data import available as _avail, _load, ROWS  # noqa: local import

    ranges = ["unknown"]
    if _avail() and ROWS:
        seen = []
        for row in ROWS:
            v = (row.get("AgeRange") or "").strip()
            if v and v not in seen:
                seen.append(v)
        ranges = seen + ["unknown"]
    props = {
        "case_id": {"type": "string"},
        "year": {"type": "number"},
        "review_type": {"type": "string",
                        "enum": ["Medical Necessity",
                                 "Experimental/Investigational",
                                 "Urgent Care"]},
        "urgency": {"type": "string", "enum": ["Standard", "Expedited"]},
        "patient_age_range": {"type": "string", "enum": ranges},
        "diagnosis_category": {"type": "string"},
        "diagnosis_subcategory": {"type": "string"},
        "treatment_category": {"type": "string"},
        "treatment_subcategory": {"type": "string"},
        "case_summary": {"type": "string"},
    }
    return {"type": "object", "properties": props,
            "required": ["treatment_subcategory", "case_summary"]}


def extract_instructions() -> str:
    return ("Extract only the facts stated in this health-plan document. "
            "Do NOT include any reviewer decision, conclusion or outcome in "
            "case_summary.")


def _mock_upload_job(grant_path: str, size: int, local_path: str) -> dict:
    sha = hashlib.sha256(grant_path.encode()).hexdigest()
    file_id = "f-" + sha[:12]
    ws = "ws-" + sha[12:24]
    _MOCK_UPLOADS[file_id] = local_path
    return {
        "id": "job-mock-" + sha[:16],
        "status": "succeeded",
        "result": {"file": {"workspace_id": ws, "file_id": file_id}},
    }


def upload(path: str) -> dict:
    """Upload a local file via upload grant. Returns the finished job."""
    _ensure_configured()
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        blob = f.read()
    sha = hashlib.sha256(blob).hexdigest()
    grant_path = f"{sha[:16]}_{os.path.basename(path)}"

    if _state["mock"]:
        job = _mock_upload_job(grant_path, len(blob), path)
        return job if job["status"] == "succeeded" else wait_job(job)

    resp = _request("POST", "/v1/documents/upload-grants",
                    json_body={"path": grant_path})
    resp.raise_for_status()
    grant = resp.json()
    max_bytes = grant.get("max_bytes")
    if max_bytes is not None and len(blob) > max_bytes:
        raise NaceError(f"file too large: {len(blob)} > max_bytes {max_bytes}")
    meta = {"path": grant_path}
    if len(blob) >= 32 * 1024 * 1024:
        meta["total_size_bytes"] = len(blob)
    files = {"file": (os.path.basename(path), blob)}
    # TODO(verify): the signed upload_url likely authenticates via
    # X-Upload-Token only; we do NOT also send the Bearer key there.
    resp = _request("POST", grant["upload_url"], absolute=True, auth=False,
                    headers={"X-Upload-Token": grant["token"]},
                    data={"metadata": json.dumps(meta)}, files=files)
    resp.raise_for_status()
    job = resp.json()
    return job if job.get("status") in ("succeeded", "failed", "cancelled") \
        else wait_job(job)


def _mock_extract(body: dict) -> dict:
    src = body.get("source", {})
    file_id = src.get("file_id", "")
    local = _MOCK_UPLOADS.get(file_id)
    summary = "Mock extract: document text unavailable in mock mode."
    data: dict[str, Any] = {
        "case_id": None, "year": None, "review_type": None, "urgency": None,
        "patient_age_range": None, "diagnosis_category": None,
        "diagnosis_subcategory": None, "treatment_category": None,
        "treatment_subcategory": "unknown",
        "case_summary": summary,
    }
    citations = []
    if local and os.path.exists(local):
        ext = os.path.splitext(local)[1].lower()
        if ext in _TEXT_EXTS:
            try:
                text = open(local, encoding="utf-8", errors="replace").read()
            except OSError:
                text = ""
            summary = text[:800]
            data["case_summary"] = summary
            citations = [{
                "location": {"kind": "text", "page": 1, "polygons": []},
                "source_text": summary[:200],
            }]
    fields = []
    schema = body.get("schema", {})
    for name, prop in (schema.get("properties") or {}).items():
        value = data.get(name)
        status = "found" if value not in (None, "", "unknown") else "not_found"
        if name == "case_summary" and value:
            status = "found"
        fields.append({
            "path": name,
            "value": value,
            "status": "found" if status == "found" or (name == "case_summary" and value) else "not_found",
            "confidence": 0.9 if status == "found" else None,
            "citations": citations if name == "case_summary" else [],
        })
    seed = _mock_seed("mock-extract", json.dumps(body, sort_keys=True, default=str))
    return {
        "id": f"job-mock-{seed:016x}",
        "status": "succeeded",
        "result": {"data": data, "fields": fields},
    }


def extract(source: dict, schema: dict, instructions: str,
            wait_seconds: int = 60, idem: str | None = None) -> dict:
    """NDI extract. Returns the finished job (result.data + fields)."""
    _ensure_configured()
    body = {
        "source": source,
        "schema": schema,
        "instructions": instructions,
        "parse_mode": "medium",
        "citations": {"enabled": True, "include_source_text": True},
    }
    raw = json.dumps(body, sort_keys=True).encode()
    idem = idem or ("idem-" + hashlib.sha256(raw).hexdigest())
    if _state["mock"]:
        return _mock_extract(body)
    resp = _request("POST", f"/v1/documents/extract?wait_seconds={wait_seconds}",
                    headers={"Idempotency-Key": idem}, json_body=body)
    if resp.status_code == 200:
        job = resp.json()
    elif resp.status_code == 202:
        job = wait_job(resp.json())
    else:
        raise NaceError(f"extract HTTP {resp.status_code}: {resp.text[:300]}",
                        status=resp.status_code)
    return job if job.get("status") in ("succeeded", "failed", "cancelled") \
        else wait_job(job)


def _pick_value(values: list) -> Any:
    """Majority value across extraction samples; ties prefer the
    specific value (a real value over None/'unknown'); then longest
    (for truncated free text); then first."""
    def key(v: Any) -> str:
        return json.dumps(v, sort_keys=True, default=str)

    counts: dict[str, int] = {}
    for v in values:
        counts[key(v)] = counts.get(key(v), 0) + 1
    best = max(counts.values())
    cands = [v for v in values if counts[key(v)] == best]

    def generic(v: Any) -> bool:
        return v is None or str(v).strip().lower() == "unknown"

    specific = [v for v in cands if not generic(v)]
    cands = specific or cands
    return max(cands, key=lambda v: (len(str(v)) if v is not None else -1,
                                     -values.index(v)))


def extract_stable(source: dict, schema: dict, instructions: str,
                   samples: int = 3, wait_seconds: int = 60) -> dict:
    """NDI extract, sampled and merged.

    NDI flips individual fields between runs on byte-identical uploads
    (e.g. patient_age_range 'unknown' <-> '41 to 50'), and the committee
    amplifies that into very different overturn probabilities. Run the
    extract `samples` times (distinct idempotency keys) and merge field
    by field, so the case the committee sees is stable. Returns the
    finished-job shape (result.data + result.fields)."""
    tag = hashlib.sha256(
        json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
    results = []
    for i in range(max(1, samples)):
        job = extract(source, schema, instructions, wait_seconds,
                      idem=f"stable-{tag}-{i}")
        if job.get("status") == "succeeded":
            results.append(job["result"])
    if not results:
        raise NaceError("all extract samples failed")
    if len(results) == 1:
        return {"status": "succeeded", "result": results[0]}

    by_path: dict[str, list[dict]] = {}
    for res in results:
        for fd in res.get("fields", []):
            by_path.setdefault(fd["path"], []).append(fd)
    fields: list[dict] = []
    data: dict[str, Any] = {}
    for path, fs in by_path.items():
        chosen = _pick_value([fd.get("value") for fd in fs])
        src = next((fd for fd in fs if fd.get("value") == chosen), fs[0])
        fields.append(dict(src))
        data[path.lstrip("/")] = chosen
    out = dict(results[0])
    out["data"] = data
    out["fields"] = fields
    return {"status": "succeeded", "result": out}


def wait_job(job: dict, timeout_s: int = 600) -> dict:
    """Poll GET /v1/documents/jobs/{id} until succeeded/failed/cancelled."""
    _ensure_configured()
    delay = 1.0
    started = time.monotonic()
    while True:
        status = job.get("status")
        if status in ("succeeded", "failed", "cancelled"):
            return job
        if time.monotonic() - started > timeout_s:
            raise NaceError(f"job {job.get('id')} timed out after {timeout_s}s")
        if _state["mock"]:
            return job
        time.sleep(delay)
        resp = _request("GET", f"/v1/documents/jobs/{job['id']}")
        resp.raise_for_status()
        job = resp.json()
        delay = min(10.0, delay * 2)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    configure()
    qs = {
        "a": {"type": "noul", "instructions": "Is it medically necessary?"},
        "b": {"type": "choice", "instructions": "Denial reason?",
              "criteria": {"not covered": "x", "experimental": "y"}},
        "c": {"type": "score", "instructions": "Strength of evidence?",
              "criteria": ["weak", "some", "strong", "decisive"]},
    }
    r = systemone({"case": {"x": 1}}, qs)
    print("model:", r["model"], "| request_id:", r["request_id"])
    for k, v in r["answers"].items():
        print(k, "=>", json.dumps(v))
    print(f"cost: ${cost_usd(r['usage']):.6f} (estimate)")
