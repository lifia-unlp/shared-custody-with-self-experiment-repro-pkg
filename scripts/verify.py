#!/usr/bin/env python3
"""Compare output/values.json (produced by analysis.py) with paper_values.json.

Prints a table of paper value, reproduced value and match flag, writes
output/verification.md, and exits non-zero if any value differs beyond its
tolerance. Values are compared as printed in the manuscript, so tolerances
follow the number of decimals the paper shows.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    paper = json.load(open(os.path.join(ROOT, "paper_values.json")))
    got = json.load(open(os.path.join(ROOT, "output", "values.json")))
    lines = ["# Verification against the manuscript", "",
             "| Key | Paper | Reproduced | Match | Note |", "|---|---|---|---|---|"]
    bad = documented = 0
    for k, spec in paper.items():
        if k.startswith("_"):
            continue
        if k not in got:
            lines.append(f"| {k} | {spec['paper']} | (missing) | NO | |")
            bad += 1
            continue
        v = got[k]
        ok = abs(v - spec["paper"]) <= spec["tol"] + 1e-12
        if ok:
            flag, note = "yes", spec.get("where", "")
        elif "expected_diff" in spec:
            flag, note = "differs (documented)", spec["expected_diff"]
            documented += 1
        else:
            flag, note = "NO", spec.get("where", "")
            bad += 1
        lines.append(f"| {k} | {spec['paper']} | {v:.6g} | {flag} | {note} |")
    n = sum(1 for k in paper if not k.startswith("_"))
    lines += ["", f"{n - bad - documented} of {n} values match; {documented} differ for a documented "
              f"reason (see notes); {bad} differ unexpectedly."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(ROOT, "output", "verification.md"), "w") as f:
        f.write(text)
    print(text)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
