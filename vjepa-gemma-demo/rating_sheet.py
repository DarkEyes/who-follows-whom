#!/usr/bin/env python3
"""Blind-vs-framed comparison of Gemma descriptions.

Export (writes a shuffled, condition-hidden rating sheet):
    python rating_sheet.py export results/game/game.json

Fill the 'rating' column in results/game/rating_sheet.csv:
    0 = no interaction described
    1 = some interaction
    2 = clear coordinated interaction

Score (unblinds, computes discrimination per condition + sign test):
    python rating_sheet.py score results/game/game.json

Automatic keyword scoring without human ratings:
    python rating_sheet.py auto results/game/game.json
"""

import csv
import json
import os
import random
import re
import sys

FIELDS = {
    "event": ("framed", "top"),
    "event_control": ("framed", "control"),
    "event_blind": ("blind", "top"),
    "event_control_blind": ("blind", "control"),
}

COORD = re.compile(r"\b(mutual|both|simultan|shortly after|followed|"
                   r"mirror|in sync|synchron|together|coordinat)\b", re.I)
NOTHING = re.compile(r"\b(nothing notable|no notable|unremarkable|"
                     r"unchanged|no significant)\b", re.I)


def collect(path):
    d = json.load(open(path))
    items = []
    for w in d["windows"]:
        for e in w["edges"]:
            for k, (cond, role) in FIELDS.items():
                if k in e and not e[k].startswith("["):
                    items.append({
                        "start": w["start"],
                        "pair": "/".join(e["pair"]),
                        "z": e["z"],
                        "condition": cond,
                        "role": role,
                        "text": e[k],
                    })
    return items


def export(path):
    items = collect(path)
    random.Random(0).shuffle(items)
    out = os.path.join(os.path.dirname(path), "rating_sheet.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "description", "rating"])
        for i, it in enumerate(items):
            w.writerow([i, it["text"], ""])
    key = os.path.join(os.path.dirname(path), "rating_key.json")
    json.dump(items, open(key, "w"), indent=2)
    print(f"wrote {out} ({len(items)} rows, shuffled, conditions hidden)")
    print(f"wrote {key} (do not open until rating is done)")


def _summarize(scored):
    from collections import defaultdict
    by = defaultdict(list)
    for it, s in scored:
        by[(it["condition"], it["role"])].append(s)
    print(f"{'condition':>8} {'role':>8} {'n':>4} {'mean':>6}")
    means = {}
    for k in sorted(by):
        m = sum(by[k]) / len(by[k])
        means[k] = m
        print(f"{k[0]:>8} {k[1]:>8} {len(by[k]):>4} {m:>6.2f}")
    for cond in ("framed", "blind"):
        if (cond, "top") in means and (cond, "control") in means:
            delta = means[(cond, "top")] - means[(cond, "control")]
            print(f"discrimination Delta({cond}) = {delta:+.2f}")
    # per-window sign test for blind condition
    top = {(it["start"]): s for it, s in scored
           if it["condition"] == "blind" and it["role"] == "top"}
    ctl = {(it["start"]): s for it, s in scored
           if it["condition"] == "blind" and it["role"] == "control"}
    common = sorted(set(top) & set(ctl))
    if common:
        wins = sum(top[t] > ctl[t] for t in common)
        losses = sum(top[t] < ctl[t] for t in common)
        n = wins + losses
        if n:
            from math import comb
            p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n
            print(f"blind sign test: top>control in {wins}/{n} decisive "
                  f"windows (one-sided p={p:.3f})")


def score(path):
    key = os.path.join(os.path.dirname(path), "rating_key.json")
    sheet = os.path.join(os.path.dirname(path), "rating_sheet.csv")
    items = json.load(open(key))
    ratings = {}
    with open(sheet) as f:
        for row in csv.DictReader(f):
            if row["rating"].strip() != "":
                ratings[int(row["id"])] = float(row["rating"])
    scored = [(items[i], r) for i, r in ratings.items()]
    print(f"{len(scored)}/{len(items)} rated")
    _summarize(scored)


def auto(path):
    items = collect(path)
    scored = []
    for it in items:
        s = 0.0
        if COORD.search(it["text"]):
            s = 2.0
        if NOTHING.search(it["text"]):
            s = 0.0
        scored.append((it, s))
    print("automatic keyword scoring (coarse):")
    _summarize(scored)


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    {"export": export, "score": score, "auto": auto}[cmd](path)
