# ClaimAid

It tells a patient how strong their appeal is, before they give up.
It triages appeals for human reviewers. It never denies a case alone.

A health plan denied a patient's treatment. ClaimAid reads the appeal
packet and convenes a committee of five Drex agents that vote and
debate. A Drex chair gives P(overturn) — the chance that an
independent medical review overturns the denial. The output is an
appeal strength, a patient note and a next step.

## Quick start

All commands run from this folder. Python 3.10+ is the only
prerequisite.

1. **One case, from the terminal** (the first run makes `.venv` and
   installs the packages):

   ```sh
   .venv/bin/python committee.py MN17-26586
   ```

   It prints the agent votes, the appeal strength, a patient note and
   a next step. `MN17-26586` is an id from the public dataset; a file
   path (PDF, TXT, …) works the same way. Add image files to include
   photos in the review:

   ```sh
   .venv/bin/python committee.py packets/MN16-24179.pdf photo.jpg
   ```

2. **A folder of documents** (the batch pipeline):

   ```sh
   ./run.sh <folder>        # e.g. ./run.sh packets
   ```

   It writes `results/run_<date_time>/` with `results.jsonl`,
   `results.csv` and `report.html`. Open `report.html` in a browser.

3. **The web app**:

   ```sh
   .venv/bin/streamlit run app.py
   ```

   It opens at http://localhost:8501. Choose a packet from `packets/`
   or a dataset case, add optional images, and click
   "Convene the committee". PDF packets are rendered to page
   image(s) first, so NDI and the Clinical Reviewer read them exactly
   like image documents (see "The NDI flow").

Other tools:

- `.venv/bin/python make_docs.py` — sample packets in `packets/`.
- `.venv/bin/python benchmark.py --n 100` — method comparison on the
  test set.
- `.venv/bin/python doctor.py` — environment check.

**Mock mode:** try everything without a key or API calls:

```sh
NACE_MOCK=1 .venv/bin/python committee.py MN17-26586
./run.sh packets --mock
```

Mock answers are random. Never report them as results.

**API key:** put `NACE_API_KEY` in `.env` (see `.env.example`). The
environment wins over `.env`. The key is never printed.

## The data and the leakage rules

The dataset is the California DMHC "IMR Determinations, Trend"
dataset, license CC-BY, about 42,749 rows. The label is `True` when
`Determination` starts with "Overturned". The split is 70/30 with seed
42. The benchmark set is the test rows with a clean summary (about
633 cases).

Leakage rules: the model never gets the `Findings` text,
`DaysToReview`, `DaysToAdopt` or `PatientGender`. `case_summary` is
the text between "Nature of Statutory Criteria/Case Summary:" and
"Findings:", and rows that already name the outcome are removed.

Baselines on the 633 leak-free benchmark cases:

- rule (train-set rate of the TreatmentSubCategory): accuracy about
  63%, AUC about 0.66.

## The NDI flow

Before upload, every document is normalized so the same document
extracts the same way no matter how it was saved:

- PDFs are rendered by `pdf_to_image()` (pypdfium2) into one stacked
  JPEG (scale 2.0, quality 85; re-rendered at 1.0 if the stack would
  exceed 30 megapixels). NDI then reads the document on the same
  image path it reads image documents — the PDF's text layer is
  never used.
- Extraction is sampled: `extract_stable()` runs the extract 3 times
  (distinct idempotency keys) and merges field by field — majority
  value, ties prefer a specific value over `unknown`/null, and for
  free text the longest (least truncated) wins. This is needed
  because NDI flips individual fields between runs on
  byte-identical uploads, and the committee amplifies that into very
  different P(overturn).

All three entry points (web app, `committee.py`, `run.py`) use the
same normalization and the same stable extraction.

1. `POST /v1/documents/upload-grants` with a relative path
   (`<sha256[:16]>_<file name>`).
2. `POST` the file to the signed `upload_url` with `X-Upload-Token`
   and `metadata` as a plain text field.
3. `POST /v1/documents/extract?wait_seconds=60` with a flat schema
   (string and number only) and an `Idempotency-Key` (three times,
   via `extract_stable`).
4. Poll `GET /v1/documents/jobs/{id}` until it is finished.

The extraction gives the case fields and citations. `run.py` caches
each (merged) extraction in `<out>/.extract_cache/<sha256>.json`
(live mode only).

## The image rules

Only the `drex-max` models accept images. The code refuses images
before the request on any other model.

- Top-level `"images"` array, at most 8 items.
- Items: a data URL (`data:image/png|jpeg|webp;base64,...`) or a
  `workspace_file` reference to an already uploaded file.
- PNG, JPEG or WebP only. At most 16 MiB per image and 40 megapixels
  per image. The whole body must fit in 4,194,304 bytes.
- `image_to_data_url()` converts other formats and big files to JPEG:
  it lowers the quality (to at least 60) and shrinks the size by 20%
  per step, down to a 400 px short side.
- By default only the Clinical Reviewer gets images. A PNG, JPEG or
  WebP document reuses its own upload (no second upload); a PDF
  reuses its rendered page image the same way, so the Clinical
  Reviewer sees exactly what NDI extracted from.
- `usage` can carry `image_tokens`. The cost is computed from
  `input_tokens` only. TODO(verify): it is not clear if
  `input_tokens` already includes `image_tokens`.

## Costs

The cost is an estimate at $0.03 per million input tokens. It is
always labeled an estimate.

## The report

`report.html` is one self-contained file: inline CSS, no scripts, no
external assets, a dark-mode theme, and a table that scrolls at
narrow widths. Strengths are green (strong), orange (uncertain) or
gray (weaker) — never red.

## Live checks

Run these only with a real key (no `NACE_MOCK`):

```sh
python doctor.py
python run.py packets
python committee.py packets/MN16-24179.pdf /tmp/p1.jpg
python benchmark.py --n 100
```

### Live results (2026-11-14, model drex-max-v1.1)

- `doctor.py`: all OK (Python 3.14.6, packages, data, packets, key,
  `GET /v1/models` 8 models, small Drex call, NDI extract).
- `run.py packets`: 12/12 ok. Citations in the report show page
  numbers (e.g. `/case_id (p. 1)`).
- `committee.py MN17-26586 /tmp/p1.jpg` (live):
  `input_tokens=4947 image_tokens=216`. Without the image:
  `input_tokens=4554 image_tokens=0`. The difference (393) matches the
  image tokens plus the extra prompt text, but it is still not certain
  which side counts the image tokens. TODO(verify): whether
  `input_tokens` already includes `image_tokens`.
- `benchmark.py --n 100` (live):

| method | n | acc | prec | rec | F1 | AUC | $/case |
|---|---|---|---|---|---|---|---|
| majority | 100 | 0.450 | 0.450 | 1.000 | 0.621 | 0.500 | 0.000000* |
| rule | 100 | 0.650 | 0.667 | 0.444 | 0.533 | 0.631 | 0.000000* |
| drex_single | 100 | 0.690 | 0.694 | 0.556 | 0.617 | 0.755 | 0.000008* |
| drex_single_h | 100 | 0.700 | 0.703 | 0.578 | 0.634 | 0.708 | 0.000014* |
| committee | 100 | 0.650 | 0.609 | 0.622 | 0.615 | 0.704 | 0.000144* |

\* cost is an estimate at $0.03 per million input tokens.

### API differences found

None so far: all calls matched the documented shapes.
