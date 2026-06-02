"""The recorded fine-tuning session: lilbee tuning retrieval to one chat model.

This is the program VHS records for a fine-tuning reel. It runs a greedy
forward-selection sweep over lilbee's query-time retrieval knobs against the
shared corpus, narrating each step: the baseline (some probes miss), each knob it
tries (recall climbing or "no gain"), the context-budget clamp that keeps the
retrieved pool inside this model's window, and a before/after of the cited chunks.

It does real work through real product routes (``GET /api/search`` +
``PATCH /api/config``), and on the way it persists the winning knobs to a per-model
``query-knobs.toml``. So running the reel IS the precompute: once model serving is
solved, every opencode demo boots already-tuned.

Tuning needs no chat model loaded, only the embedder + reranker and this model's
``n_ctx`` for the budget clamp, so every model's reel is recordable up front.

``--fake`` swaps in a deterministic in-process search so the narrative, the clamp,
and the artifact can be verified locally with no server and no models.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from score_retrieval import (
    Probe,
    SweepScore,
    budget_for_context,
    load_probes,
    score_pass,
)

GREEN = "\033[32m"
RED = "\033[31m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

BAR_WIDTH = 24
TOP_K_FLOOR = 6  # never clamp below this; fewer chunks stops grounding the answer

# Ordered greedy moves over query-time knobs only (no reindex). Each is tried on
# top of the accepted-so-far config and kept only if it improves the score.
MOVES: list[tuple[str, dict[str, Any]]] = [
    ("wider candidate pool", {"top_k": 16}),
    ("rerank the top candidates", {"rerank_candidates": 32}),
    ("allow more chunks per source", {"diversity_max_per_source": 8}),
    ("accept fuzzier matches", {"max_distance": 0.85}),
    ("HyDE query expansion", {"hyde": True}),
]

# Baseline values for the tuned keys when the server can't report them (fake mode).
BASELINE: dict[str, Any] = {
    "top_k": 5,
    "rerank_candidates": 0,
    "diversity_max_per_source": 3,
    "max_distance": 0.7,
    "hyde": False,
}

SearchFn = Callable[[str, int], list[dict[str, Any]]]


@dataclass
class Tuner:
    probes: list[Probe]
    budget: int
    search: SearchFn
    apply: Callable[[dict[str, Any]], None]
    pace: float

    def _pause(self, mult: float = 1.0) -> None:
        if self.pace:
            time.sleep(self.pace * mult)

    def _measure(self, knobs: dict[str, Any]) -> SweepScore:
        self.apply(knobs)
        results = {p.query: self.search(p.query, knobs["top_k"]) for p in self.probes}
        return score_pass(self.probes, results, self.budget)

    def run(self, model: str, n_ctx: int) -> tuple[dict[str, Any], SweepScore]:
        _banner(model, n_ctx, len(self.probes), self.budget)
        current = dict(BASELINE)
        best = self._measure(current)
        _section("BASELINE  (lilbee defaults)")
        _render_pass(best)
        _render_bar("recall", best.recall)
        self._pause(1.6)

        _section("TUNING  (each knob kept only if it helps)")
        for name, delta in MOVES:
            trial = {**current, **delta}
            score = self._measure(trial)
            if score.is_better_than(best):
                current, best = trial, score
                _move_line(name, delta, accepted=True, recall=best.recall, reason="kept")
            else:
                self.apply(current)  # revert the rejected trial on the server
                # Distinguish "wouldn't help" from "would overflow this model's
                # window" -- the latter is the bb-xdic constraint made visible.
                over = not score.all_within_budget and score.recall > best.recall
                reason = "over budget" if over else "no gain"
                _move_line(name, delta, accepted=False, recall=best.recall, reason=reason)
            self._pause()

        best = self._clamp_to_budget(current, best, model, n_ctx)
        return current, best

    def _clamp_to_budget(
        self, current: dict[str, Any], best: SweepScore, model: str, n_ctx: int
    ) -> SweepScore:
        """Shrink top_k until every probe response fits this model's window."""
        if best.all_within_budget:
            return best
        _section(f"CONTEXT CLAMP  (fit {model}'s {n_ctx:,}-token window)")
        while not best.all_within_budget and current["top_k"] > TOP_K_FLOOR:
            current["top_k"] = max(TOP_K_FLOOR, current["top_k"] - 2)
            best = self._measure(current)
            fit = "fits" if best.all_within_budget else "still over"
            print(f"  top_k -> {current['top_k']:>2}   "
                  f"max response {best.max_response_tokens:>5} tok / {self.budget} budget  "
                  f"{DIM}({fit}){RESET}")
            self._pause(0.8)
        return best


def _banner(model: str, n_ctx: int, n_probes: int, budget: int) -> None:
    print(f"{BOLD}{CYAN}lilbee{RESET}  tuning retrieval for "
          f"{BOLD}{model}{RESET}  ({n_ctx:,}-token context)")
    print(f"{DIM}corpus: the lilbee repo   probes: {n_probes} grounded code questions   "
          f"budget: {budget:,} tok/response{RESET}\n")


def _section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")


def _short(query: str, width: int = 46) -> str:
    return query if len(query) <= width else query[: width - 1] + "…"


def _render_pass(score: SweepScore) -> None:
    for o in score.outcomes:
        if o.hit:
            cite = o.citations[0] if o.citations else ""
            over = "" if o.within_budget else f" {RED}[over budget]{RESET}"
            print(f"  {GREEN}✓{RESET} {_short(o.query):<46} {DIM}{cite}{RESET}{over}")
        else:
            miss = ", ".join(o.missed)
            print(f"  {RED}✗{RESET} {_short(o.query):<46} {DIM}missing {miss}{RESET}")


def _render_bar(label: str, frac: float) -> None:
    filled = round(frac * BAR_WIDTH)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    color = GREEN if frac >= 0.999 else (YELLOW if frac >= 0.5 else RED)
    print(f"  {label:<6} {color}[{bar}]{RESET} {frac:.0%}\n")


def _move_line(name: str, delta: dict[str, Any], *, accepted: bool, recall: float, reason: str) -> None:
    knob = ", ".join(f"{k}={_fmt(v)}" for k, v in delta.items())
    if accepted:
        tag = f"{GREEN}+ kept{RESET}"
    elif reason == "over budget":
        tag = f"{YELLOW}- over budget{RESET}"
    else:
        tag = f"{DIM}- no gain{RESET}"
    filled = round(recall * BAR_WIDTH)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    barcolor = GREEN if recall >= 0.999 else YELLOW
    print(f"  {tag:<16} {name:<30} {DIM}{knob}{RESET}")
    print(f"  {'':<16} recall {barcolor}[{bar}]{RESET} {recall:.0%}")


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _write_artifact(path: Path, knobs: dict[str, Any], model: str, n_ctx: int) -> None:
    lines = [
        f"# Tuned retrieval knobs for {model} (n_ctx={n_ctx}) on the lilbee corpus.",
        "# Produced by tune_run.py; merged into the shared config.toml before launch.",
        "",
    ]
    lines += [f"{k} = {_fmt(v)}" for k, v in knobs.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_search(knobs_holder: dict[str, dict[str, Any]], probes: list[Probe]) -> SearchFn:
    """Deterministic search whose recall climbs as the right knobs are set.

    Models, loosely, a corpus where each probe needs a different knob to surface
    its file. Response size scales with top_k so a small-context model triggers
    the budget clamp. For local verification only; the real reel uses LilbeeClient.
    """
    snippet = "x" * 600  # ~150 tokens per returned chunk

    def unlocked(probe_idx: int, k: dict[str, Any]) -> bool:
        return [
            True,
            k["top_k"] >= 16,
            k["rerank_candidates"] >= 32,
            k["diversity_max_per_source"] >= 8,
            k["max_distance"] >= 0.85 or k["hyde"],
        ][probe_idx]

    by_query = {p.query: i for i, p in enumerate(probes)}

    def search(query: str, top_k: int) -> list[dict[str, Any]]:
        idx = by_query[query]
        k = knobs_holder["current"]
        docs: list[dict[str, Any]] = []
        if unlocked(idx, k):
            docs.append({
                "source": "lilbee/" + probes[idx].expect[0],
                "excerpts": [{"content": snippet, "line_start": 12, "line_end": 40}],
            })
        for d in range(top_k - len(docs)):
            docs.append({"source": f"lilbee/other/distractor_{d}.py",
                         "excerpts": [{"content": snippet}]})
        return docs

    return search


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-ctx", type=int, required=True)
    ap.add_argument("--probes", type=Path, default=Path(__file__).with_name("probes.toml"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--token", default="")
    ap.add_argument("--token-file", type=Path, help="Read the token from a file (keeps it off "
                    "the recorded command line).")
    ap.add_argument("--fake", action="store_true", help="Local verification: no server.")
    ap.add_argument("--pace", type=float, default=1.3, help="Seconds between narration steps.")
    args = ap.parse_args()

    probes = load_probes(args.probes)
    budget = budget_for_context(args.n_ctx)

    if args.fake:
        holder: dict[str, dict[str, Any]] = {"current": dict(BASELINE)}
        search = _fake_search(holder, probes)

        def apply(knobs: dict[str, Any]) -> None:
            holder["current"] = dict(knobs)
    else:
        from lilbee_http import LilbeeClient

        token = args.token
        if args.token_file:
            token = args.token_file.read_text(encoding="utf-8").strip()
        client = LilbeeClient(base_url=args.base_url, token=token)

        def apply(knobs: dict[str, Any]) -> None:
            try:
                client.patch_config(knobs)
            except Exception as exc:  # noqa: BLE001 - keep the reel running, note it
                print(f"  {DIM}(settings note: {exc}){RESET}")

        def search(query: str, top_k: int) -> list[dict[str, Any]]:
            return client.search(query, top_k=top_k, scope="raw")

    tuner = Tuner(probes=probes, budget=budget, search=search, apply=apply, pace=args.pace)
    winner, final = tuner.run(args.model, args.n_ctx)

    _section("RESULT")
    knob_str = "  ".join(f"{k}={_fmt(v)}" for k, v in winner.items())
    print(f"  {DIM}{knob_str}{RESET}")
    _render_bar("recall", final.recall)
    _write_artifact(args.out, winner, args.model, args.n_ctx)
    print(f"  {GREEN}saved{RESET} {args.out}")
    print(f"  {DIM}opencode will boot already-tuned for {args.model}.{RESET}")

    _section("AFTER  (grounded, with citations)")
    _render_pass(final)
    return 0 if final.recall >= 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
