#!/usr/bin/env python3
"""CPU-only extraction sanity for the re-extracted (kreuzberg 4.9) manual KB:
every numeric ground-truth fact must be findable via lilbee search before any
GPU pod exists. Catches extraction regressions (the rc7/rc9 fuse-table class)
at prep time for cents.

Reads ground_truth.json's real shape: {q: {must_contain_facts: [{id, desc,
patterns: [regex...]}]}}. LILBEE_DATA must point at the re-extracted manual
KB working dir.

Usage: search_sanity.py ground_truth.json
"""

import json
import re
import subprocess
import sys

QUERIES = [
    "maximum trailer weight towing",
    "gross combination weight",
    "trailer towing precautions cooling",
    "jump start battery cable connection order",
    "engine compartment fuse amp rating",
    "power distribution box fuses",
    "automatic transmission fluid check",
]


def search(q: str) -> str:
    # --json: the table view truncates chunk text with ellipses, hiding the
    # very numbers this gate exists to verify
    out = subprocess.run(["lilbee", "--json", "search", q, "--top-k", "10"],
                         capture_output=True, text=True)
    if out.returncode != 0 or "no embed model server" in (out.stdout + out.stderr).lower():
        sys.exit(f"SEARCH_SANITY_FAIL: lilbee search errored: {(out.stderr or out.stdout)[-300:]}")
    payload = json.loads(out.stdout)
    joined = " ".join(r.get("chunk", "") for r in payload.get("results", []))
    # PDF chunks wrap mid-phrase; patterns must not die on line breaks
    return re.sub(r"\s+", " ", joined).lower()


def main() -> None:
    gt = json.load(open(sys.argv[1]))
    corpus = "\n".join(search(q) for q in QUERIES)
    if not corpus.strip():
        sys.exit("SEARCH_SANITY_FAIL: all searches returned nothing")
    failures = []
    for qkey, q in gt.items():
        for fact in q.get("must_contain_facts", []):
            pats = fact.get("patterns", [])
            # numeric facts must surface in search hits; prose facts are graded
            # from real answers in the canary
            if not any(any(c.isdigit() for c in p) for p in pats):
                continue
            bounded = [rf"\b{p}\b" if p.isdigit() else p for p in pats]
            if not any(re.search(p, corpus, re.I) for p in bounded):
                failures.append(f"{qkey}/{fact.get('id')}")
    if failures:
        print(f"SEARCH_SANITY_FAIL missing facts: {failures}")
        sys.exit(1)
    print("SEARCH_SANITY_OK")


if __name__ == "__main__":
    main()
