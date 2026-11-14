"""Self-contained ClaimAid report (report.html).

One file: inline CSS, no scripts, no external fonts or images, every value
escaped with html.escape. Dark mode via CSS variables; no horizontal page
scroll at 390 px (the table scrolls in an overflow-x:auto container).
"""
from __future__ import annotations

import html

PILL = {
    "strong": ("pill-strong", "strong"),
    "uncertain": ("pill-uncertain", "uncertain"),
    "weaker": ("pill-weaker", "weaker"),
}

CSS = """
:root {
  --bg: #f6f7f9; --card: #ffffff; --text: #1a1d21; --muted: #5c6370;
  --line: #dde1e7; --green: #2e7d32; --orange: #ef6c00; --gray: #616161;
  --bar-bg: #e3e6ea; --note: #eef4fb; --warn: #fff3e0; --ok: #e6f4ea;
  --bar-fill: #3949ab;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --card: #1e2127; --text: #e8eaed; --muted: #9aa0a6;
    --line: #33383f; --green: #81c995; --orange: #f2b263; --gray: #9aa0a6;
    --bar-bg: #2a2f36; --note: #1d2735; --warn: #33270f; --ok: #1c2e22;
    --bar-fill: #8c9eff;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--text);
       font: 14px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
       sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 18px 0 8px; }
.sub { color: var(--muted); margin: 0 0 12px; }
.meta { color: var(--muted); font-size: 12px; margin-bottom: 12px; }
.mock { font-weight: bold; color: var(--orange); margin: 8px 0; }
.cards { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.card { background: var(--card); border: 1px solid var(--line);
        border-radius: 8px; padding: 8px 12px; min-width: 90px; }
.card .v { font-size: 18px; font-weight: 600; }
.card .l { font-size: 11px; color: var(--muted); }
.table-wrap { overflow-x: auto; background: var(--card);
              border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; min-width: 640px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid
         var(--line); vertical-align: top; white-space: nowrap; }
td.wrap, .det { white-space: normal; }
td.name { white-space: normal; min-width: 160px; max-width: 260px; }
td.name .d { font-size: 11px; color: var(--muted); }
.pill { display: inline-block; border-radius: 999px; padding: 2px 10px;
        font-size: 12px; font-weight: 600; color: #fff; }
.pill-strong { background: var(--green); }
.pill-uncertain { background: var(--orange); }
.pill-weaker { background: var(--gray); }
.bar { background: var(--bar-bg); border-radius: 4px; height: 8px;
       width: 90px; margin-top: 4px; }
.bar i { display: block; height: 8px; border-radius: 4px;
         background: var(--bar-fill); }
.step { font-size: 13px; }
details { min-width: 220px; max-width: 420px; }
summary { cursor: pointer; color: var(--muted); }
.det { border-left: 3px solid var(--line); padding-left: 8px; margin: 6px 0; }
.note { background: var(--note); border-radius: 6px; padding: 8px;
        margin: 6px 0; }
.warnbox { background: var(--warn); border-radius: 6px; padding: 8px;
           margin: 6px 0; }
.okbox { background: var(--ok); border-radius: 6px; padding: 8px; margin: 6px 0; }
.err { color: var(--orange); font-weight: 600; }
footer { margin-top: 16px; color: var(--muted); font-size: 12px; }
"""


def esc(v) -> str:
    return html.escape(str(v if v is not None else ""))


def _card(label: str, value) -> str:
    return (f'<div class="card"><div class="v">{esc(value)}</div>'
            f'<div class="l">{esc(label)}</div></div>')


def _bar(p: float) -> str:
    pct = max(0.0, min(1.0, p)) * 100
    return (f'<div class="bar"><i style="width:{pct:.0f}%"></i></div>')


def _row(r: dict) -> str:
    if r["status"] == "failed":
        return (f'<tr><td class="name">{esc(r["name"])}'
                f'<div class="d">failed</div></td>'
                f'<td colspan="4"><span class="err">{esc(r["error"])}</span>'
                f'</td></tr>')
    cls, label = PILL.get(r["strength"], PILL["weaker"])
    step = ("⚡ fast-track overturn" if not r["needs_human"]
            else "👤 human review")
    truth = ""
    if r["truth"] in ("overturned", "upheld"):
        mark = "✅" if r["correct"] else "❌"
        truth = (f'<div class="det">actual outcome: {mark} '
                 f'{esc(r["truth"])}</div>')
    votes = esc(r["votes"]) if r["votes"] else "—"
    cited = esc(r["cited_fields"]) if r["cited_fields"] else "—"
    details = (
        f'<details><summary>details</summary>'
        f'<div class="note">{esc(r["patient_note"])}</div>'
        f'<div class="det">summary: {esc(r["summary"] or "—")}</div>'
        f'<div class="det">reasons: {esc(r["reasons"] or "none")}</div>'
        f'<div class="det">votes: {votes}</div>'
        f'<div class="det">calls {esc(r["drex_calls"])} · tokens '
        f'{esc(r["input_tokens"])} · image tokens {esc(r["image_tokens"])}'
        f' · images {esc(r["images_sent"])} · {esc(r["seconds"])} s</div>'
        f'<div class="det">cited fields: {cited}</div>'
        f'{truth}</details>')
    return (
        f'<tr>'
        f'<td class="name">{esc(r["name"])}<div class="d">'
        f'{esc(r["treatment"])} · {esc(r["diagnosis"])}</div></td>'
        f'<td><span class="pill {cls}">{label}</span></td>'
        f'<td>{r["p_overturn"] * 100:.0f}%{_bar(r["p_overturn"])}</td>'
        f'<td class="step">{step}</td>'
        f'<td>${r["cost_usd"]:.6f}</td>'
        f'<td class="wrap">{details}</td>'
        f'</tr>')


def write_report(path: str, rows: list[dict], meta: dict) -> str:
    """Write report.html from result rows. Returns the path."""
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    n_strong = sum(1 for r in ok if r["strength"] == "strong")
    n_unc = sum(1 for r in ok if r["strength"] == "uncertain")
    n_weak = sum(1 for r in ok if r["strength"] == "weaker")
    n_fast = sum(1 for r in ok if not r["needs_human"])
    known = [r for r in ok if r["truth"] in ("overturned", "upheld")]
    acc = (sum(1 for r in known if r["correct"]) / len(known) * 100
           if known else None)
    cost = sum(r["cost_usd"] for r in ok)

    cards = "".join([
        _card("appeals", len(ok)),
        _card("strong", n_strong),
        _card("uncertain", n_unc),
        _card("weaker", n_weak),
        _card("fast-track", n_fast),
        _card("human review", len(ok) - n_fast),
        _card("errors", len(failed)),
    ])
    if acc is not None:
        cards += _card("accuracy (known)", f"{acc:.0f}%")
    cards += _card("estimated cost", f"${cost:.4f}")
    cards += _card("time", f"{meta.get('seconds', 0):.0f} s")

    mock = ('<p class="mock">MOCK MODE: random answers</p>'
            if meta.get("mock") else "")
    body_rows = "".join(_row(r) for r in rows)
    out = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClaimAid — Appeal Strength Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>ClaimAid — Appeal Strength Report</h1>
<p class="sub">How strong is each appeal, and who should look at it next.
Decision support only: it never denies a case alone.</p>
<p class="meta">{esc(meta.get("date", ""))} · model: {esc(meta.get("model", ""))}</p>
{mock}
<div class="cards">{cards}</div>
<div class="table-wrap">
<table>
<thead><tr><th>document</th><th>strength</th><th>P(overturn)</th>
<th>next step</th><th>cost</th><th>details</th></tr></thead>
<tbody>
{body_rows}
</tbody>
</table>
</div>
<footer>Decision support only. It never denies a case alone: every case
that leans "uphold" goes to a human reviewer. A weaker appeal can still
win, especially with new medical records or a letter from the doctor.</footer>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    return path
