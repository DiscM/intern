"""Five-agent Drex committee with a Drex chair, for IMR appeal strength.

Framing: for patients, "It tells a patient how strong their appeal is, before
they give up."; for reviewers, "It triages appeals for human reviewers, and it
never denies a case alone."
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import data
import nace

VOTE_Q = {
    "type": "choice",
    "instructions": "Vote on this appeal: will an independent medical review "
                    "overturn the health plan's denial?",
    "criteria": {"overturn": "the denial should be overturned",
                 "uphold": "the denial should be upheld"},
}

NOTES = {
    "strong": "Strong appeal: similar cases were often overturned. Appealing is worth it.",
    "uncertain": "Uncertain: it could go either way. Appeal, and add records or a letter from your doctor.",
    "weaker": "Weaker on paper, but you can still appeal. New medical records or a letter from your doctor can change the outcome.",
}


def _cl_clinical(_fields, _has_data, n_images=0):
    qs = {
        "medically_necessary": {
            "type": "noul",
            "instructions": "Is the requested treatment medically necessary "
                            "for this patient's condition?"},
        "standard_of_care": {
            "type": "score",
            "instructions": "How does the requested treatment compare to the "
                            "standard of care?",
            "criteria": ["far below standard", "below standard",
                         "meets standard", "exceeds standard"]},
    }
    if n_images:
        qs["images_support"] = {
            "type": "noul",
            "instructions": (f"Image 1 to Image {n_images} are documents or "
                             "photos sent with the appeal. Do the images "
                             "support the request?")}
    return qs


def _cl_evidence(_fields, _has_data, n_images=0):
    return {
        "evidence_strength": {
            "type": "score",
            "instructions": "How strong is the evidence supporting the "
                            "requested treatment for this condition?",
            "criteria": ["weak", "moderate", "strong", "decisive"]},
        "experimental": {
            "type": "noul",
            "instructions": "Is the requested treatment experimental or "
                            "investigational?"},
    }


def _cl_policy(_fields, _has_data, n_images=0):
    return {
        "denial_reason": {
            "type": "choice",
            "instructions": "What is the health plan's stated reason for "
                            "denying this treatment?",
            "criteria": {"not medically necessary": "denied as not medically necessary",
                         "not covered": "not covered by plan benefits",
                         "experimental": "experimental or investigational",
                         "documentation": "inadequate documentation"}},
        "denial_valid": {
            "type": "noul",
            "instructions": "Is the plan's stated denial reason, as given, "
                            "valid?"},
    }


def _cl_advocate(_fields, _has_data, n_images=0):
    return {
        "harm_if_denied": {
            "type": "score",
            "instructions": "How serious is the harm to the patient if this "
                            "treatment is denied?",
            "criteria": ["minimal", "moderate", "serious",
                         "severe or life-threatening"]},
        "alternatives_tried": {
            "type": "noul",
            "instructions": "Have lower-level or alternative treatments "
                            "already been tried?"},
    }


def _cl_actuary(fields, has_data, n_images=0):
    if has_data:
        h = data.history_for(fields)
        bits = [f"{k}={v['overturn_rate']:.2f} (n={v['n_train']})"
                for k, v in h.items() if k != "base_rate"]
        instr = ("Train-set overturn rates: base "
                 f"{h.get('base_rate', 0):.2f}; "
                 f"{' ; '.join(bits) or 'no specific history'}. "
                 "What does the history of similar cases suggest for this appeal?")
    else:
        instr = ("No historical case data is available. What would case "
                 "history generally suggest for this appeal?")
    return {"history": {
        "type": "score",
        "instructions": instr,
        "criteria": ["strongly uphold", "uphold", "neutral",
                     "overturn", "strongly overturn"]}}


AGENTS = [
    {"name": "clinical_reviewer",
     "role": "You are a board-certified clinical reviewer for independent medical review.",
     "questions": _cl_clinical},
    {"name": "evidence_reviewer",
     "role": "You are an evidence review specialist who grades the strength of medical evidence.",
     "questions": _cl_evidence},
    {"name": "plan_policy_analyst",
     "role": "You analyze health plan policy and stated denial reasons.",
     "questions": _cl_policy},
    {"name": "patient_advocate",
     "role": "You represent the patient's clinical interests and safety.",
     "questions": _cl_advocate},
    {"name": "actuary",
     "role": "You are an actuary who reasons from the overturn history of similar cases.",
     "questions": _cl_actuary},
]

CHAIR_QUESTIONS = {
    "decision": {"type": "choice",
                 "instructions": "Weigh the case and the agents' findings. "
                                 "Should this denial be overturned?",
                 "criteria": {"overturn": "overturn the denial",
                               "uphold": "uphold the denial"}},
    "overturn": {"type": "noul",
                 "instructions": "What is the probability that an independent "
                                 "medical review overturns this denial?"},
    "needs_human": {"type": "noul",
                    "instructions": "Does this case require a human reviewer "
                                    "before any decision is made?"},
}


def _compact(answers: dict) -> str:
    parts = [f"vote={answers.get('vote', {}).get('choice', '?')}"]
    for k, v in answers.items():
        if k == "vote":
            continue
        if "noul" in v:
            parts.append(f"{k}={v['noul']:.2f}")
        elif "score" in v:
            parts.append(f"{k}={v['score']:.1f}")
        elif "choice" in v:
            parts.append(f"{k}={v['choice']}")
    return "; ".join(parts)


def run(case: dict, images: list | None = None, image_refs: list | None = None,
        debate_rounds: int = 1, threshold: float = 0.5,
        confidence: float = 0.65) -> dict:
    """Run the committee on one leak-free case. Returns the full result.

    images: data URLs (max 8) sent to the Clinical Reviewer.
    image_refs: workspace_file refs from a document that is itself an image.
    debate_rounds: 0 or 1 (1 = the default second exchange round).
    """
    t0 = time.perf_counter()
    nace._ensure_configured()  # never re-derive mock/key mid-run
    fields = data.case_to_fields(case)
    has_data = data.available()
    agents = [a for a in AGENTS if has_data or a["name"] != "actuary"]
    images = (images or [])[:nace.MAX_IMAGES]
    img_refs = (image_refs or [])[:nace.MAX_IMAGES]
    n_images = len(images) + len(img_refs)

    calls = 0
    input_tokens = 0
    image_tokens = 0

    def call(agent: dict, state: dict) -> dict:
        nonlocal calls, input_tokens, image_tokens
        is_clinical = agent["name"] == "clinical_reviewer"
        qs = dict(agent["questions"](fields, has_data, n_images if is_clinical
                                     else 0))
        qs["vote"] = VOTE_Q
        if is_clinical and n_images:
            state = dict(state)
            state["attachments"] = (
                f"Image 1 to Image {n_images} are documents or photos sent "
                "with the appeal.")
        agent_images = (img_refs + images) if is_clinical else None
        r = nace.systemone(state, qs, images=agent_images)
        calls += 1
        input_tokens += r["usage"].get("input_tokens", 0)
        image_tokens += r["usage"].get("image_tokens", 0)
        return r["answers"]

    def call_chair(state: dict) -> dict:
        nonlocal calls, input_tokens, image_tokens
        # the chair never gets images
        r = nace.systemone(state, CHAIR_QUESTIONS)
        calls += 1
        input_tokens += r["usage"].get("input_tokens", 0)
        image_tokens += r["usage"].get("image_tokens", 0)
        return r["answers"]

    def state_extra(agent: dict, prev: dict | None) -> dict:
        st = {"case": fields, "round": 1}
        if prev is None:
            st["role"] = agent["role"]
        else:
            st["instruction"] = ("Keep your view unless their findings show "
                                 "something you missed.")
            st["peers"] = {n: _compact(ans) for n, ans in prev.items()
                           if n != agent["name"]}
        return st

    last: dict = {}
    rounds_out = {}
    for rnd in range(1 + max(0, debate_rounds)):
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            futs = {a["name"]: ex.submit(call, a, state_extra(a, None if rnd == 0
                                                              else last))
                    for a in agents}
            last = {name: f.result() for name, f in futs.items()}
        rounds_out[f"round{rnd + 1}"] = last

    chair = call_chair({"case": fields,
                        "findings": {n: _compact(a) for n, a in last.items()}})

    p = float(chair["overturn"]["noul"])
    final = "overturn" if p >= threshold else "uphold"

    reasons = []
    if final == "uphold":
        reasons.append("final decision is uphold")
    votes = [a.get("vote", {}).get("choice") for a in last.values()]
    if "overturn" in votes and "uphold" in votes:
        reasons.append("agents disagree on the vote")
    if max(p, 1 - p) < confidence:
        reasons.append(f"confidence {max(p, 1 - p):.2f} below {confidence:.2f}")
    # TODO(verify): threshold for the chair's needs_human noul (0.5 used).
    if float(chair["needs_human"]["noul"]) >= 0.5:
        reasons.append("chair requested human review")
    needs_human = bool(reasons)

    strength = "strong" if p >= 0.65 else "uncertain" if p >= 0.35 else "weaker"
    return {
        "rounds": rounds_out,
        "chair": chair,
        "final": final,
        "overturn_prob": p,
        "strength": strength,
        "patient_note": NOTES[strength],
        "needs_human": needs_human,
        "reasons": reasons,
        "next_step": "human review" if needs_human else "fast-track overturn",
        "calls": calls,
        "input_tokens": input_tokens,
        "image_tokens": image_tokens,
        "images_sent": len(images) + len(img_refs),
        "image_refs": img_refs,
        "cost_usd": nace.cost_usd({"input_tokens": input_tokens}),
        "wall_ms": int((time.perf_counter() - t0) * 1000),
    }


# ---------------------------------------------------------------------------
# Case loading (dataset id or document file)
# ---------------------------------------------------------------------------

def load_case(src: str) -> dict:
    """id -> dataset case; file -> NDI upload + extract; else File not found.

    PDFs are rendered to a stacked page image first, so NDI reads them on
    the same image path as image documents (and the Clinical Reviewer
    reuses that same upload as its image)."""
    if os.path.exists(src):
        norm = nace.pdf_to_image(src)
        up = nace.upload(norm or src)
        f = up["result"]["file"]
        job = nace.extract_stable(
            {"type": "workspace_file",
             "workspace_id": f["workspace_id"], "file_id": f["file_id"]},
            nace.ndi_schema(), nace.extract_instructions())
        if job.get("status") != "succeeded":
            raise nace.NaceError(f"document extract failed: {job.get('status')}")
        result = job["result"]
        fields = result["data"]
        citations = [c for fd in result["fields"] for c in fd.get("citations", [])]
        case = data.fields_to_case(fields)
        outcome = None
        stem = os.path.splitext(os.path.basename(src))[0].replace("_scan", "")
        for cand in (fields.get("case_id"), stem):
            cand = (cand or "").strip()
            if not cand:
                continue
            try:
                outcome = data.find(cand)["overturned"]
                break
            except KeyError:
                continue
        image_ref = None
        ext = os.path.splitext(src)[1].lower()
        if norm is not None or ext in (".png", ".jpg", ".jpeg",
                                       ".webp"):
            # the document is itself an image (or a PDF rendered to one):
            # reuse the uploaded file (no second upload)
            image_ref = {"type": "workspace_file",
                         "workspace_id": f["workspace_id"],
                         "file_id": f["file_id"]}
        return {"case": case, "citations": citations,
                "outcome": outcome, "source": "file", "image_ref": image_ref}
    looks_like_path = os.sep in src or os.path.splitext(src)[1].lower() in (
        ".pdf", ".txt", ".md", ".html", ".eml", ".csv", ".json",
        ".xml", ".png", ".jpg", ".jpeg")
    if looks_like_path:
        raise FileNotFoundError(f"File not found: {src}")
    case = data.find(src)  # KeyError: "case not found: ..."
    return {"case": case, "citations": [], "outcome": case["overturned"],
            "source": "id", "image_ref": None}


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: committee.py <id|file> [image ...]")
        return 1
    nace.configure()
    try:
        rc = load_case(argv[0])
    except (FileNotFoundError, KeyError, nace.NaceError) as e:
        print(str(e.args[0]) if isinstance(e, KeyError) else str(e))
        return 1
    case = rc["case"]
    images: list[str] = []
    for im in argv[1:]:
        if not os.path.exists(im):
            print(f"File not found: {im}")
            return 1
        try:
            images.append(nace.image_to_data_url(im))
        except nace.NaceError as e:
            print(str(e))
            return 1
    image_refs = [rc["image_ref"]] if rc["image_ref"] else []
    n_images = len(images) + len(image_refs)
    if n_images and not nace.model_supports_vision():
        print(f"warning: model {nace.configure()['model']} cannot read "
              "images; running without them")
        images, image_refs, n_images = [], [], 0
    res = run(case, images=images or None,
              image_refs=image_refs or None)

    print(f"Case {case['case_id'] or '(unnamed)'}: "
          f"{case['treatment']['subcategory']} | "
          f"year {case['year']} | {case['review_type']} | {case['urgency']}")
    if n_images:
        print(f"Clinical Reviewer got {n_images} image(s)")
    for rnd, answers in res["rounds"].items():
        print(f"{rnd.title()}:")
        for name, a in answers.items():
            print(f"  {name}: {_compact(a)}")
    ch = res["chair"]
    print(f"Chair: decision={ch['decision']['choice']} "
          f"P(overturn)={res['overturn_prob']:.2f} "
          f"needs_human={ch['needs_human']['noul'] >= 0.5}")
    print(f"Final: {res['final']} | strength={res['strength']} "
          f"| P(overturn)={res['overturn_prob']:.2f}")
    print(f"Note: {res['patient_note']}")
    print(f"Next step: {res['next_step']}")
    print("Reasons: " + ("; ".join(res["reasons"]) or "none"))
    if rc["source"] == "file" and rc["citations"]:
        pages = ", ".join(str(c["location"].get("page"))
                          for c in rc["citations"])
        print(f"Citations: {len(rc['citations'])} on page(s) {pages}")
    if rc["outcome"] is None:
        print("Actual outcome: unknown (case not in the public dataset)")
    else:
        print("Actual outcome: "
              + ("Overturned" if rc["outcome"] else "Upheld")
              + " (public IMR dataset)")
    print("Decision support only. It never denies a case alone.")
    print(f"calls={res['calls']} input_tokens={res['input_tokens']} "
          f"image_tokens={res['image_tokens']} "
          f"cost=${res['cost_usd']:.6f} (estimate) wall_ms={res['wall_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
