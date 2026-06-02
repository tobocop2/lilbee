"""Demo-readiness gate: does the freshly built index ground every probe?

Searches each probe against a running ``lilbee serve`` and fails (exit 1) unless
every probe's expected file surfaces. ingest_corpus.sh runs this before any model
work so a stale or empty corpus is caught here, not three giants into the matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lilbee_http import LilbeeClient
from score_retrieval import budget_for_context, load_probes, score_pass

# Generous fixed top_k for the gate: we only ask "is the expected file reachable
# at all", not "is it optimally ranked" (that is tune_run's job per model).
_GATE_TOP_K = 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--token", required=True)
    ap.add_argument("--probes", type=Path, default=Path(__file__).with_name("probes.toml"))
    ap.add_argument("--n-ctx", type=int, default=8192)
    args = ap.parse_args()

    client = LilbeeClient(base_url=args.base_url, token=args.token)
    probes = load_probes(args.probes)
    results = {p.query: client.search(p.query, top_k=_GATE_TOP_K) for p in probes}
    score = score_pass(probes, results, budget_for_context(args.n_ctx))

    for o in score.outcomes:
        mark = "ok " if o.hit else "MISS"
        where = ", ".join(o.citations) if o.hit else f"missing {', '.join(o.missed)}"
        print(f"  [{mark}] {o.query}  ->  {where}")
    print(f"recall={score.recall:.0%}  mrr={score.mrr:.2f}")

    if score.recall < 1.0:
        print("GATE FAIL: index does not ground every probe; not demo-ready.", file=sys.stderr)
        return 1
    print("GATE PASS: index grounds every probe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
