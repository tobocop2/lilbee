#!/usr/bin/env python3
"""Turn the raw measure.py files into the distance tables behind the canary threshold.

    analyze.py <dir with raw JSON files> <out.json>

Distance is 1 - cosine. A pair's value is the WORST canary, because the rule compares per canary.
The retrieval value is an oracle that does not use the canaries: the top 10 of real documents for
real queries, when the index comes from one run and the queries from another.
"""

from __future__ import annotations

import base64
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

TOP_K = 10


def decode(items: list[str]) -> np.ndarray:
    rows = [np.frombuffer(base64.b64decode(item), dtype="<f4").astype(np.float64) for item in items]
    return np.vstack(rows)


def unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def worst_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("nan")
    return float((1.0 - (unit(left) * unit(right)).sum(axis=1)).max())


def overlap_at_k(docs_a, queries_a, queries_b) -> float:
    docs = unit(docs_a)
    truth = np.argsort(-(unit(queries_a) @ docs.T), axis=1)[:, :TOP_K]
    other = np.argsort(-(unit(queries_b) @ docs.T), axis=1)[:, :TOP_K]
    hits = [len(set(t.tolist()) & set(o.tolist())) / TOP_K for t, o in zip(truth, other)]
    return float(np.mean(hits))


def cross_overlap(a: dict, b: dict) -> float | None:
    """Index built by run a, queries embedded by run b, against queries embedded by run a."""
    if "corpus" not in a or "queries" not in b:
        return None
    return overlap_at_k(a["corpus_m"], a["queries_m"], b["queries_m"])


def mixed_overlap(a: dict, b: dict) -> float | None:
    """A library re-indexed half by run a and half by run b, against the whole of run a."""
    if "corpus" not in a or "corpus" not in b:
        return None
    n = min(len(a["corpus_m"]), len(b["corpus_m"]))
    half = n // 2
    mixed = np.vstack([a["corpus_m"][:half], b["corpus_m"][half:n]])
    docs = unit(a["corpus_m"][:n])
    truth = np.argsort(-(unit(a["queries_m"]) @ docs.T), axis=1)[:, :TOP_K]
    got = np.argsort(-(unit(a["queries_m"]) @ unit(mixed).T), axis=1)[:, :TOP_K]
    return float(np.mean([len(set(t.tolist()) & set(g.tolist())) / TOP_K for t, g in zip(truth, got)]))


def main() -> int:
    raw = Path(sys.argv[1])
    records: dict[tuple[str, str, str], dict] = {}
    errors: list[str] = []
    for path in sorted(raw.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if "error" in rec:
            errors.append(f"{path.stem}: {rec['error'][:160]}")
            continue
        rec["m"] = decode(rec["vectors"])
        if "corpus" in rec:
            rec["corpus_m"] = decode(rec["corpus"])
            rec["queries_m"] = decode(rec["queries"])
        records[(rec["engine"], rec["model"], rec["scenario"])] = rec

    engines = sorted({k[0] for k in records})
    models = sorted({k[1] for k in records})
    scenarios = sorted({k[2] for k in records})
    out: dict[str, object] = {"engines": engines, "models": models, "errors": errors}

    same_engine: dict[str, dict[str, float]] = defaultdict(dict)
    for (engine, model, scenario), rec in records.items():
        if scenario in ("base", "wrongpool"):
            continue
        base = records.get((engine, model, "base"))
        if base is not None:
            same_engine[scenario][f"{engine} {model}"] = worst_distance(base["m"], rec["m"])
    out["same_engine_worst"] = {
        s: {"worst": max(v.values()), "at": max(v, key=v.get), "n": len(v)}
        for s, v in same_engine.items()
    }

    cross_engine: dict[str, dict] = {}
    for model in models:
        for left, right in combinations(engines, 2):
            a, b = records.get((left, model, "base")), records.get((right, model, "base"))
            if a is None or b is None:
                continue
            cross_engine[f"{model} | {left} vs {right}"] = {
                "distance": worst_distance(a["m"], b["m"]),
                "overlap_at_10": cross_overlap(a, b),
                "mixed_overlap_at_10": mixed_overlap(a, b),
            }
    out["cross_engine"] = cross_engine

    cross_quant: dict[str, dict] = {}
    for engine in engines:
        for left, right in combinations(models, 2):
            a, b = records.get((engine, left, "base")), records.get((engine, right, "base"))
            if a is None or b is None or a["m"].shape != b["m"].shape:
                continue
            kind = "same_family" if left.split(".")[0].split("-Q")[0] == right.split(".")[0].split("-Q")[0] else "other_family"
            cross_quant[f"{engine} | {left} vs {right}"] = {
                "kind": kind,
                "distance": worst_distance(a["m"], b["m"]),
                "overlap_at_10": cross_overlap(a, b),
                "mixed_overlap_at_10": mixed_overlap(a, b),
            }
    out["cross_model"] = cross_quant

    pooling: dict[str, float] = {}
    for (engine, model, scenario), rec in records.items():
        if scenario == "wrongpool" and (engine, model, "base") in records:
            pooling[f"{engine} {model}"] = worst_distance(records[(engine, model, "base")]["m"], rec["m"])
    out["wrong_pooling"] = pooling

    per_canary: dict[str, list[float]] = {}
    for key, value in cross_engine.items():
        model, pair = key.split(" | ")
        left, right = pair.split(" vs ")
        a, b = records[(left, model, "base")]["m"], records[(right, model, "base")]["m"]
        per_canary[key] = (1.0 - (unit(a) * unit(b)).sum(axis=1)).tolist()
    out["cross_engine_per_canary"] = per_canary
    out["device_proof"] = {
        f"{k[0]} {k[1]}": r["device_proof"] for k, r in records.items() if k[2] == "base"
    }
    out["scenario_names"] = scenarios
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"records={len(records)} errors={len(errors)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
