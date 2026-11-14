"""Benchmark: how well each method predicts the public IMR outcome.

Methods: majority, rule, drex_single, drex_single_h, committee.
Metrics: accuracy, precision, recall, F1, AUC (by rank, hand-written),
$/case, ms/case. Mock runs are marked, never reported as real results.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import data
import nace
import committee

SINGLE_Q = {
    "answer": {
        "type": "noul",
        "instructions": "Will an independent medical review overturn this "
                        "health plan's denial?",
    },
}


def _predict_drex(case: dict, with_history: bool) -> tuple[bool, float, int]:
    fields = data.case_to_fields(case)
    state = {"case": fields}
    if with_history and data.available():
        state["history"] = data.history_for(fields)
    r = nace.systemone(state, SINGLE_Q)
    p = float(r["answers"]["answer"]["noul"])
    return p >= 0.5, p, r["usage"].get("input_tokens", 0)


def run_method(method: str, cases: list[dict], workers: int) -> dict:
    nace.configure()
    n = len(cases)
    base_rate = data.rule_score({})  # base rate when category has <10 rows
    preds: list[bool] = []
    scores: list[float] = []
    tokens = 0
    t0 = time.perf_counter()

    if method == "majority":
        label = base_rate >= 0.5
        preds = [label] * n
        scores = [base_rate] * n
    elif method == "rule":
        for c in cases:
            s = data.rule_score(data.case_to_fields(c))
            preds.append(s >= 0.5)
            scores.append(s)
    elif method in ("drex_single", "drex_single_h"):
        def one(c):
            return _predict_drex(c, method == "drex_single_h")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ok, p, tok in ex.map(one, cases):
                preds.append(ok)
                scores.append(p)
                tokens += tok
    elif method == "committee":
        def one(c):
            res = committee.run(c)
            return res["final"] == "overturn", res["overturn_prob"], res["input_tokens"]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ok, p, tok in ex.map(one, cases):
                preds.append(ok)
                scores.append(p)
                tokens += tok
    else:
        raise ValueError(f"unknown method: {method}")

    wall_ms = (time.perf_counter() - t0) * 1000
    labels = [c["overturned"] for c in cases]
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fp = sum(1 for y, p in zip(labels, preds) if (not y) and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and (not p))
    acc = sum(1 for y, p in zip(labels, preds) if y == p) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "method": method,
        "n": n,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": _auc(scores, labels),
        "cost_per_case": nace.cost_usd({"input_tokens": tokens}) / n,
        "ms_per_case": wall_ms / n,
        "wall_ms": wall_ms,
    }


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    """AUC by hand, from ranks (Mann-Whitney U with tied ranks)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j + 2) / 2  # average of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    sum_pos = sum(ranks[i] for i, y in enumerate(labels) if y)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--methods", nargs="+",
                    default=["majority", "rule", "drex_single",
                             "drex_single_h", "committee"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cases = data.eval_cases()
    if len(cases) < args.n:
        print(f"warning: only {len(cases)} leak-free test cases, using all")
    cases = cases[: args.n]
    nace.configure()
    mock = nace.is_mock()

    results = [run_method(m, cases, args.workers) for m in args.methods]

    if mock:
        print("NOTE: mock mode (NACE_MOCK=1 / no key) - these numbers are NOT real results.")
    print()
    print("| method | n | acc | prec | rec | F1 | AUC | $/case | ms/case |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "-"
        print(f"| {r['method']} | {r['n']} | {r['accuracy']:.3f} "
              f"| {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} "
              f"| {auc} | {r['cost_per_case']:.6f}* | {r['ms_per_case']:.0f} |")
    print("* cost is an estimate at $0.03 per million input tokens.")

    os.makedirs("results", exist_ok=True)
    out = {
        "mock": mock,
        "n": len(cases),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }
    path = os.path.join("results", f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
