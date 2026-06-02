"""Scoring for the opencode-demo retrieval probes.

Pure functions over already-fetched ``/api/search`` results: no I/O, no server,
no model. ``tune_run.py`` imports these to score each sweep step; the gate in
``ingest_corpus.sh`` calls the ``--gate`` CLI to fail a non-grounded index before
any GPU hours are spent.

A probe is "hit" when every one of its expected path substrings appears in the
returned ``source`` fields. The per-probe token cost is the size of the whole
search response (all returned chunk contents), because that response is what gets
injected into the chat as the tool result. ``within_budget`` compares that cost to
the chat model's retrieval budget, the constraint that turns an over-wide ``top_k``
into the bb-xdic context overflow.

Run ``python score_retrieval.py --selftest`` to check the scoring logic locally
without a server.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A chunk of English/code is roughly four characters per token. The sweep only
# needs the budget comparison to be monotonic, not exact, so a constant divisor
# is enough to rank settings and to catch a top_k that would overflow the window.
CHARS_PER_TOKEN = 4

# Fraction of a model's context window we allow a single retrieval tool response
# to occupy. The rest is reserved for the system prompt, the running conversation,
# and the model's own generation. Deliberately conservative: an over-budget search
# response is exactly what produced the bb-xdic "exceeds context window" failure.
TOOL_BUDGET_FRACTION = 0.35


@dataclass(frozen=True)
class Probe:
    """One demo question and the real files that must surface to answer it."""

    query: str
    expect: tuple[str, ...]


@dataclass
class ProbeOutcome:
    """How one probe fared at a given retrieval setting."""

    query: str
    hit: bool
    missed: list[str] = field(default_factory=list)
    first_rank: int | None = None
    response_tokens: int = 0
    within_budget: bool = True
    citations: list[str] = field(default_factory=list)


@dataclass
class SweepScore:
    """Aggregate score for a full pass over the probe set."""

    recall: float
    mrr: float
    max_response_tokens: int
    all_within_budget: bool
    outcomes: list[ProbeOutcome]

    def is_better_than(self, other: SweepScore | None) -> bool:
        """Greedy comparison: budget-fit first, then recall, then rank quality.

        A setting that pushes any probe response over the model's budget is never
        an improvement, even if it nominally retrieves more, because the chat turn
        that follows would overflow. Among budget-safe settings, higher recall
        wins; ties break on mean reciprocal rank (expected file nearer the top).
        """
        if other is None:
            return True
        if self.all_within_budget != other.all_within_budget:
            return self.all_within_budget
        if self.recall != other.recall:
            return self.recall > other.recall
        return self.mrr > other.mrr


def load_probes(path: Path) -> list[Probe]:
    """Parse ``probes.toml`` into ``Probe`` records."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    probes = raw.get("probe", [])
    return [Probe(query=p["query"], expect=tuple(p["expect"])) for p in probes]


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def budget_for_context(n_ctx: int) -> int:
    """Token budget a single retrieval response may occupy for an ``n_ctx`` model."""
    return int(n_ctx * TOOL_BUDGET_FRACTION)


def _response_tokens(results: list[dict[str, Any]]) -> int:
    total = 0
    for doc in results:
        for ex in doc.get("excerpts", []):
            total += estimate_tokens(ex.get("content", ""))
    return total


def _citation(doc: dict[str, Any]) -> str:
    source = doc.get("source", "?")
    excerpts = doc.get("excerpts") or []
    if not excerpts:
        return source
    start = excerpts[0].get("line_start")
    end = excerpts[0].get("line_end")
    if start and end:
        return f"{source}:{start}-{end}"
    return source


def score_probe(probe: Probe, results: list[dict[str, Any]], budget: int) -> ProbeOutcome:
    """Score one probe's search results against its expected files and the budget."""
    sources = [doc.get("source", "") for doc in results]
    missed = [
        needle for needle in probe.expect if not any(needle in src for src in sources)
    ]
    first_rank: int | None = None
    for idx, src in enumerate(sources, start=1):
        if any(needle in src for needle in probe.expect):
            first_rank = idx
            break
    tokens = _response_tokens(results)
    hit_docs = [doc for doc in results if any(n in doc.get("source", "") for n in probe.expect)]
    return ProbeOutcome(
        query=probe.query,
        hit=not missed,
        missed=missed,
        first_rank=first_rank,
        response_tokens=tokens,
        within_budget=tokens <= budget,
        citations=[_citation(doc) for doc in hit_docs],
    )


def aggregate(outcomes: list[ProbeOutcome]) -> SweepScore:
    """Combine per-probe outcomes into a single comparable score."""
    if not outcomes:
        return SweepScore(0.0, 0.0, 0, True, [])
    hits = sum(1 for o in outcomes if o.hit)
    recall = hits / len(outcomes)
    rr = [1.0 / o.first_rank for o in outcomes if o.first_rank]
    mrr = sum(rr) / len(outcomes)
    max_tokens = max(o.response_tokens for o in outcomes)
    all_safe = all(o.within_budget for o in outcomes)
    return SweepScore(recall, mrr, max_tokens, all_safe, outcomes)


def score_pass(
    probes: list[Probe], results_by_query: dict[str, list[dict[str, Any]]], budget: int
) -> SweepScore:
    """Score a complete pass: one results list per probe query."""
    outcomes = [
        score_probe(p, results_by_query.get(p.query, []), budget) for p in probes
    ]
    return aggregate(outcomes)


def _selftest() -> int:
    """Exercise the scoring + greedy-comparison logic without a server."""
    probes = [
        Probe("a", ("server/chat_dispatch/dispatch.py",)),
        Probe("b", ("server/chat_completions_api/canonical.py",)),
    ]
    big = "x" * (CHARS_PER_TOKEN * 100)
    thin = {
        "a": [{"source": "lilbee/server/chat_dispatch/dispatch.py",
               "excerpts": [{"content": big, "line_start": 10, "line_end": 20}]}],
        "b": [{"source": "lilbee/server/other.py", "excerpts": [{"content": big}]}],
    }
    rich = {
        "a": thin["a"],
        "b": [{"source": "lilbee/server/chat_completions_api/canonical.py",
               "excerpts": [{"content": big, "line_start": 5, "line_end": 9}]}],
    }
    budget = 1000
    thin_score = score_pass(probes, thin, budget)
    rich_score = score_pass(probes, rich, budget)
    assert thin_score.recall == 0.5, thin_score.recall
    assert rich_score.recall == 1.0, rich_score.recall
    assert rich_score.is_better_than(thin_score)
    # Budget overflow must lose to a budget-safe, lower-recall pass.
    overflow = {
        "a": [{"source": "lilbee/server/chat_dispatch/dispatch.py",
               "excerpts": [{"content": "y" * (CHARS_PER_TOKEN * 5000)}]}],
        "b": rich["b"],
    }
    over_score = score_pass(probes, overflow, budget)
    assert over_score.recall == 1.0 and not over_score.all_within_budget
    assert not over_score.is_better_than(rich_score), "overflow must not beat safe full recall"
    assert rich_score.outcomes[1].citations == ["lilbee/server/chat_completions_api/canonical.py:5-9"]
    print("score_retrieval selftest: OK")
    return 0


def _gate(probes_path: Path, results_path: Path, n_ctx: int) -> int:
    """Fail (exit 1) unless every probe's expected file is present.

    ``results_path`` is a JSON object mapping each probe query to its raw
    ``/api/search`` result list. Used by ingest_corpus.sh as the demo-readiness
    gate on the freshly built index.
    """
    probes = load_probes(probes_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    score = score_pass(probes, results, budget_for_context(n_ctx))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", action="store_true", help="Run the demo-readiness gate.")
    ap.add_argument("--probes", type=Path, default=Path(__file__).with_name("probes.toml"))
    ap.add_argument("--results", type=Path, help="JSON: {query: [search results]}")
    ap.add_argument("--n-ctx", type=int, default=8192)
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.gate:
        if args.results is None:
            ap.error("--gate requires --results")
        return _gate(args.probes, args.results, args.n_ctx)
    ap.error("nothing to do: pass --selftest or --gate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
