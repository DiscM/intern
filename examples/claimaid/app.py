"""ClaimAid — Streamlit app.

It tells a patient how strong their appeal is, before they give up.
It triages appeals for human reviewers, and it never denies a case alone.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import data  # noqa: E402
import nace  # noqa: E402
import committee  # noqa: E402

st.set_page_config(page_title="ClaimAid", layout="wide")

nace.configure()
_is_mock = nace.is_mock()
_model = nace.configure()["model"]

AGENT_DISPLAY = {
    "clinical_reviewer": "Clinical Reviewer",
    "evidence_reviewer": "Evidence Reviewer",
    "plan_policy_analyst": "Plan Policy Analyst",
    "patient_advocate": "Patient Advocate",
    "actuary": "Actuary",
}
AGENT_AVATAR = {
    "clinical_reviewer": "clinical",
    "evidence_reviewer": "evidence",
    "plan_policy_analyst": "policy",
    "patient_advocate": "advocate",
    "actuary": "actuary",
}
PILL_HTML = {
    "strong": "#2e7d32",
    "uncertain": "#ef6c00",
    "weaker": "#616161",
}


@st.cache_data(show_spinner=False)
def _packets() -> list[str]:
    d = ROOT / "packets"
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.iterdir()
                  if p.suffix.lower() in (".pdf", ".txt", ".jpg", ".jpeg",
                                          ".png", ".webp")
                  and not p.name.startswith("."))


@st.cache_data(show_spinner="Reading the document…")
def _load_document(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    # PDFs are rendered to a stacked page image first, so NDI reads them
    # on the same image path as image documents (no text-layer shortcut).
    norm = nace.pdf_to_image(path)
    up = nace.upload(norm or path)
    f = up["result"]["file"]
    job = nace.extract_stable(
        {"type": "workspace_file",
         "workspace_id": f["workspace_id"], "file_id": f["file_id"]},
        nace.ndi_schema(), nace.extract_instructions())
    if job.get("status") != "succeeded":
        raise nace.NaceError(f"document extract failed: {job.get('status')}")
    res = job["result"]
    fields = res["data"]
    case = data.fields_to_case(fields)
    outcome = None
    stem = Path(path).stem.replace("_scan", "")
    for cand in (fields.get("case_id"), stem):
        cand = (str(cand) if cand else "").strip()
        if not cand:
            continue
        try:
            outcome = data.find(cand)["overturned"]
            break
        except KeyError:
            continue
    return {
        "case": case,
        "fields": fields,
        "citations": [c for fd in res["fields"]
                      for c in fd.get("citations", [])],
        "field_status": {fd["path"]: fd.get("status")
                         for fd in res["fields"]},
        "outcome": outcome,
        "image_ref": ({"type": "workspace_file",
                       "workspace_id": f["workspace_id"],
                       "file_id": f["file_id"]}
                      if norm is not None
                      or Path(path).suffix.lower()
                      in (".png", ".jpg", ".jpeg", ".webp") else None),
    }


@st.cache_data(show_spinner=False)
def _pdf_preview(path: str, pages: int = 3) -> list[bytes]:
    try:
        import pypdfium2 as pdfium
        import io
        pdf = pdfium.PdfDocument(path)
        out = []
        for i in range(min(pages, len(pdf))):
            img = pdf[i].render(scale=1.5).to_pil()
            try:
                out.append(img.tobytes("png", "PNG"))
            except OSError:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                out.append(buf.getvalue())
        pdf.close()
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
banner = ROOT / "assets" / "banner.png"
if banner.exists():
    st.image(str(banner), width="stretch")
st.title("ClaimAid")
st.caption("It tells a patient how strong their appeal is, before they "
           "give up.")
st.caption("It triages appeals for human reviewers, and it never denies "
           "a case alone.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"**Mode:** "
                f"{'mock (random answers)' if _is_mock else 'live · ' + _model}")
    debate = st.slider("Debate rounds", 0, 2, 1)
    threshold = st.slider("Overturn threshold", 0.05, 1.0, 0.5, 0.05)
    confidence = st.slider("Human-review confidence", 0.50, 0.95, 0.65,
                           0.01)
    source = st.radio("Case source",
                      ["Appeal packet (document)",
                       "Real IMR test case (no NDI)"])
    st.caption("Data: California DMHC IMR Determinations, Trend "
               "(CC-BY). The model never sees the outcome, the Findings "
               "text, review days or PatientGender.")

# ---------------------------------------------------------------------------
# Case input
# ---------------------------------------------------------------------------
case_obj = None
citations: list = []
field_status: dict = {}
outcome = None
image_ref = None
name_disp = ""

if source == "Appeal packet (document)":
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("Upload an appeal packet",
                              type=["pdf", "png", "jpg", "jpeg", "webp",
                                    "txt", "docx", "csv", "html", "eml",
                                    "json", "xml"],
                              key="up_doc")
    with c2:
        packs = _packets()
        pick = st.selectbox("…or pick from packets/", packs, index=0,
                            format_func=lambda p: Path(p).name,
                            key="pick_doc")
    doc_path = None
    if up is not None:
        name_disp = up.name
        target = ROOT / "packets" / "_upload_tmp" / up.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(up.getvalue())
        doc_path = str(target)
    elif pick:
        doc_path = pick
        name_disp = Path(pick).name
    if doc_path:
        try:
            info = _load_document(doc_path)
            case_obj = info["case"]
            citations = info["citations"]
            field_status = info["field_status"]
            outcome = info["outcome"]
            image_ref = info["image_ref"]
            if not name_disp:
                name_disp = doc_path
            if doc_path.lower().endswith(".pdf"):
                imgs = _pdf_preview(doc_path)
                if imgs:
                    cols = st.columns(len(imgs))
                    for col, b in zip(cols, imgs):
                        col.image(b, width="stretch")
                if nace.model_supports_vision():
                    st.caption(
                        "The PDF was rendered to page image(s); NDI and "
                        "the Clinical Reviewer both read it as an image, "
                        "exactly like an image document.")
                else:
                    st.warning(
                        f"Model {_model} cannot read images; the Clinical "
                        "Reviewer will only get the extracted text.")
        except Exception as e:
            st.error(f"Could not read the document: {e}")
else:
    cases = data.eval_cases()
    labels = {c["case_id"]: c for c in cases}
    sel = st.selectbox(
        "IMR test case", list(labels), index=0,
        format_func=lambda i: (f"{i} · "
                               f"{labels[i]['treatment']['subcategory']} · "
                               f"{labels[i]['diagnosis']['subcategory']}"),
        key="pick_case")
    case_obj = labels[sel]
    outcome = case_obj["overturned"]
    name_disp = case_obj["case_id"]

# extracted case: summary info box + expander
if case_obj and case_obj.get("case_summary"):
    st.info(f"Summary: {case_obj['case_summary'][:600]}")
with st.expander("Extracted case (JSON + citations)"):
    st.json(case_obj or {})
    if _is_mock:
        st.caption("Mock mode: no real citations are produced.")
    elif not citations:
        st.caption("No citations returned.")
    else:
        icon = {"found": "✅", "not_found": "❌", "ambiguous": "⚠️"}
        for c in citations:
            loc = c.get("location", {})
            st.markdown(
                f"{icon.get(field_status.get('case_summary', ''), '📄')} "
                f"`case_summary` · page {loc.get('page', '?')} — "
                f"{c.get('source_text', '')[:120]}")

# ---------------------------------------------------------------------------
# Extra images
# ---------------------------------------------------------------------------
img_up = st.file_uploader(
    "Extra images (optional, Drex Max vision)",
    type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
    key="up_imgs")
image_urls: list[str] = []
if img_up:
    shown = img_up[:8]
    cols = st.columns(max(1, min(8, len(shown))))
    for col, im in zip(cols, shown):
        col.image(im, width=96)
    if nace.model_supports_vision():
        try:
            for im in shown:
                image_urls.append(
                    nace.data_url_from_bytes(im.read(), name=im.name))
        except nace.NaceError as e:
            st.error(f"Image problem: {e}")
            image_urls = []
        if image_urls:
            st.caption(f"{len(image_urls)} image(s) sent to the "
                       "Clinical Reviewer")
    else:
        st.warning(f"Model {_model} cannot read images; they will not be "
                   "sent.")
    if len(img_up) > 8:
        st.caption("Only the first 8 images are used.")

# ---------------------------------------------------------------------------
# Convene
# ---------------------------------------------------------------------------
_prev_source = st.session_state.get("prev_source")
if _prev_source is not None and _prev_source != source:
    st.session_state["result"] = None
st.session_state["prev_source"] = source

if st.button("Convene the committee", width="stretch") and case_obj:
    try:
        # The document's own image (PDF rendered pages or the image
        # itself) travels via image_ref; extras go via image_urls.
        refs = [image_ref] if image_ref and nace.model_supports_vision() \
            else None
        res = committee.run(
            case_obj,
            images=image_urls or None,
            image_refs=refs,
            debate_rounds=debate, threshold=threshold,
            confidence=confidence)
        st.session_state["result"] = res
    except Exception as e:
        st.error(f"The committee could not be convened: {e}")

res = st.session_state.get("result")
if res and case_obj:
    fields = data.case_to_fields(case_obj)
    with st.expander("Actuary history (train-set overturn rates)"):
        if data.available():
            h = data.history_for(fields)
            st.write({k: (f"{v['overturn_rate']:.2f} (n={v['n_train']})"
                          if isinstance(v, dict) else v)
                      for k, v in h.items()})
        else:
            st.caption("No local dataset available; the Actuary sits out.")

    r1 = res["rounds"].get("round1", {})
    for rnd_name, answers in res["rounds"].items():
        st.markdown(f"**{rnd_name.title()}**")
        cols = st.columns(len(answers))
        for col, (aname, ans) in zip(cols, answers.items()):
            av = ROOT / "assets" / f"{AGENT_AVATAR.get(aname, '')}.png"
            if av.exists():
                col.image(str(av), width=40)
            col.markdown(f"**{AGENT_DISPLAY.get(aname, aname)}**")
            vote = ans.get("vote", {}).get("choice", "?")
            color = "green" if vote == "overturn" else "red"
            col.markdown(f"<span style='color:{color};font-weight:600'>"
                         f"{vote.upper()}</span>", unsafe_allow_html=True)
            if rnd_name != "round1" and aname in r1:
                prev = r1[aname].get("vote", {}).get("choice")
                if prev and prev != vote:
                    col.caption(f"↩ was {prev}")
            for k, v in ans.items():
                if k == "vote":
                    continue
                if "noul" in v:
                    col.caption(f"{k}: {v['noul']:.2f}")
                elif "score" in v:
                    leg = v.get("legend") or []
                    lvl = int(round(v["score"]))
                    lab = leg[lvl] if 0 <= lvl < len(leg) \
                        else f"score {v['score']:.1f}"
                    col.caption(f"{k}: {lab}")
                elif "choice" in v:
                    col.caption(f"{k}: {v['choice']}")

    st.subheader("Appeal strength")
    color = PILL_HTML.get(res["strength"], PILL_HTML["weaker"])
    c1, c2, c3 = st.columns([1, 2, 2])
    c1.markdown(
        f"<span style='background:{color};color:white;border-radius:999px;"
        f"padding:4px 14px;font-weight:600'>{res['strength']}</span>",
        unsafe_allow_html=True)
    c1.write(f"leans {res['final']}")
    c2.metric("P(overturn)", f"{res['overturn_prob'] * 100:.0f}%")
    st.progress(max(0.0, min(1.0, res["overturn_prob"])))
    st.info(res["patient_note"])
    if res["needs_human"]:
        st.warning("Next step: **human review** — " +
                   "; ".join(res["reasons"]))
    else:
        st.success("Next step: **fast-track overturn**")
    st.caption("Decision support only. It never denies a case alone.")
    if outcome is None:
        st.write("Actual outcome: unknown (case not in the public dataset)")
    else:
        correct = (res["final"] == "overturn") == outcome
        st.write(("✅" if correct else "❌") + " Actual outcome: "
                 + ("Overturned" if outcome else "Upheld")
                 + " (public IMR dataset)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Drex calls", res["calls"])
    m2.metric("Input tokens", res["input_tokens"])
    m3.metric("Image tokens", res["image_tokens"])
    m4.metric("Cost (estimate)", f"${res['cost_usd']:.6f}")
    m5.metric("Time", f"{res['wall_ms'] / 1000:.1f} s")
    chav = ROOT / "assets" / "chair.png"
    if chav.exists():
        st.image(str(chav), width=64)
