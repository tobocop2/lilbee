"""Command-line entry point for the lilbee-vs-RAGFlow benchmark.

Subcommands mirror the run stages: preregister, collect, score-ir, answer,
score-ragas, stats, report. Heavy scorers (pytrec_eval, ragas) are imported
lazily inside the modules they live in, so the CLI loads without them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from evals.benchmark import ir_metrics, stats
from evals.benchmark.collectors import (
    DEFAULT_TARGET_DOCS,
    LilbeeCollector,
    RagflowCollector,
    collect_run,
    load_queries,
)
from evals.benchmark.manifest import Manifest
from evals.benchmark.ragas_tier import (
    RagasJudge,
    Sample,
    make_ragas_evaluator,
    score_ragas,
)
from evals.benchmark.report import render_report
from evals.retrieval.checkpoint import JsonlCheckpoint, load_jsonl

DEFAULT_METRICS = ["nDCG@10", "Recall@20", "MRR@10"]
ASK_ROUTE = "/api/ask"
ASK_TIMEOUT_SECONDS = 600.0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _cmd_preregister(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    fingerprint = manifest.freeze(args.out)
    print(f"froze manifest {manifest.run_id} -> {args.out} ({fingerprint[:12]})")
    return 0


def _build_collector(args: argparse.Namespace) -> LilbeeCollector | RagflowCollector:
    if args.system == "lilbee":
        return LilbeeCollector(args.base_url, run_tag=args.run_tag, target_docs=args.target_docs)
    if not args.api_key or not args.dataset_id:
        raise ValueError("ragflow collection needs --api-key and at least one --dataset-id")
    return RagflowCollector(
        args.base_url,
        args.api_key,
        args.dataset_id,
        run_tag=args.run_tag,
        target_docs=args.target_docs,
    )


def _cmd_collect(args: argparse.Namespace) -> int:
    queries = load_queries(args.queries)
    collector = _build_collector(args)
    hits = collect_run(
        collector,
        queries,
        args.run,
        args.checkpoint,
        on_query=lambda qid: print(f"[{args.run_tag}] {qid}", flush=True),
    )
    print(f"collected {len(queries)} queries, {len(hits)} hits -> {args.run}")
    return 0


def _cmd_score_ir(args: argparse.Namespace) -> int:
    from evals.benchmark.runfile import read_run, run_to_pytrec

    qrels = json.loads(args.qrels.read_text())
    run = run_to_pytrec(read_run(args.run))
    scores = ir_metrics.score_run(qrels, run, args.metrics)
    _write_jsonl(
        args.out,
        [
            {
                "dataset": args.dataset,
                "run_tag": args.run_tag,
                "aggregated": scores.aggregated,
                "per_query": scores.per_query,
            }
        ],
    )
    print(f"scored {args.run_tag} on {args.dataset}: {scores.aggregated} -> {args.out}")
    return 0


def _ask(client: httpx.Client, base_url: str, question: str, top_k: int) -> tuple[str, list[str]]:
    response = client.post(
        f"{base_url.rstrip('/')}{ASK_ROUTE}", json={"question": question, "top_k": top_k}
    )
    response.raise_for_status()
    body = response.json()
    contexts = [chunk["chunk"] for chunk in body.get("sources", [])]
    return body["answer"], contexts


def _cmd_answer(args: argparse.Namespace) -> int:
    queries = load_queries(args.queries)
    references = json.loads(args.ground_truth.read_text()) if args.ground_truth else {}
    checkpoint = JsonlCheckpoint(args.out, "query_id")
    client = httpx.Client(timeout=ASK_TIMEOUT_SECONDS)
    for query_id, text in queries.items():
        if query_id in checkpoint:
            continue
        answer, contexts = _ask(client, args.base_url, text, args.top_k)
        checkpoint.append(
            {
                "query_id": query_id,
                "question": text,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": references.get(query_id, ""),
            }
        )
        print(f"[{args.arm}] {query_id}", flush=True)
    print(f"answered {len(queries)} queries -> {args.out}")
    return 0


def _load_samples(path: Path) -> list[Sample]:
    return [
        Sample(
            question=row["question"],
            answer=row["answer"],
            contexts=list(row["contexts"]),
            ground_truth=row.get("ground_truth", ""),
        )
        for row in load_jsonl(path)
    ]


def _cmd_score_ragas(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    judge = RagasJudge(
        model=manifest.models.judge,
        base_url=args.judge_base_url,
        temperature=manifest.temperature,
    )
    evaluate_fn = make_ragas_evaluator(judge)
    metrics = args.metrics
    scored: dict[str, Any] = {}
    for label, path in (("arm_a", args.samples_a), ("arm_b", args.samples_b)):
        samples = _load_samples(path)
        scored[label] = (samples, score_ragas(samples, metrics, evaluate_fn=evaluate_fn))
    (samples_a, scores_a), (samples_b, scores_b) = scored["arm_a"], scored["arm_b"]
    rows = [
        {
            "row_type": "ragas",
            "metric": metric,
            "arm_a": scores_a.means[metric],
            "arm_b": scores_b.means[metric],
            "n_a": scores_a.scored[metric],
            "n_b": scores_b.scored[metric],
        }
        for metric in scores_a.means
    ]
    _append_jsonl(args.out, rows)
    print(
        f"scored {len(samples_a)} and {len(samples_b)} answers with RAGAS "
        f"(judge {manifest.models.judge}) -> {args.out}"
    )
    return 0


def _load_metrics_file(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"no metrics rows in {path}")
    return rows[0]


def resolve_comparison(
    manifest: Manifest, file_a: dict[str, Any], file_b: dict[str, Any]
) -> tuple[str, str, str]:
    """Validate two metrics files against the manifest, return (dataset, arm_a, arm_b).

    The arm labels are the metrics files' own run_tags, not free text, so the
    stamped comparison cannot claim a pairing other than the one that produced
    the numbers. Both files must be the same declared dataset, and the run_tags
    must be the manifest's two declared arms; otherwise the fingerprint would
    attest to a study the manifest never froze, or (on zero query overlap) a
    cross-dataset mismatch would masquerade as a genuine null.
    """
    dataset_a, dataset_b = file_a["dataset"], file_b["dataset"]
    if dataset_a != dataset_b:
        raise ValueError(
            f"metrics files are different datasets ('{dataset_a}' vs '{dataset_b}'); "
            "a paired comparison must be within one dataset"
        )
    arm_a, arm_b = file_a["run_tag"], file_b["run_tag"]
    manifest.require_declared_comparison(arm_a, arm_b, dataset_a)
    return dataset_a, arm_a, arm_b


def _cmd_stats(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    file_a = _load_metrics_file(args.metrics_a)
    file_b = _load_metrics_file(args.metrics_b)
    dataset, arm_a, arm_b = resolve_comparison(manifest, file_a, file_b)
    rows: list[dict[str, Any]] = [
        {
            "row_type": "meta",
            "run_id": manifest.run_id,
            "fingerprint": manifest.fingerprint(),
            "arm_a": arm_a,
            "arm_b": arm_b,
        }
    ]
    for metric in manifest.metrics:
        per_query_a = file_a["per_query"].get(metric, {})
        per_query_b = file_b["per_query"].get(metric, {})
        result = stats.compare(
            metric,
            per_query_a,
            per_query_b,
            resamples=manifest.stats.bootstrap_resamples,
            seed=manifest.stats.seed,
            alpha=manifest.stats.alpha,
        )
        row = {"row_type": "ir", "dataset": dataset, **result.to_dict()}
        rows.append(row)
    _append_jsonl(args.out, rows)
    print(f"wrote {len(rows) - 1} paired IR comparisons -> {args.out}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = render_report(load_jsonl(args.results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")
    return 0


def _add_preregister(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("preregister", help="validate and freeze the manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_preregister)


def _add_collect(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("collect", help="build a TREC run file for one arm")
    parser.add_argument("--system", choices=("lilbee", "ragflow"), required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--target-docs",
        type=int,
        default=DEFAULT_TARGET_DOCS,
        help="distinct parent documents to collect per query; both arms use the same depth",
    )
    parser.add_argument("--api-key", default="", help="ragflow only")
    parser.add_argument(
        "--dataset-id", action="append", default=[], help="ragflow only, repeatable"
    )
    parser.set_defaults(handler=_cmd_collect)


def _add_score_ir(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("score-ir", help="score a run file against qrels with pytrec_eval")
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_score_ir)


def _add_answer(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("answer", help="generate answers for the RAGAS tier")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_answer)


def _add_score_ragas(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("score-ragas", help="score both arms' generated answers with RAGAS")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--samples-a", type=Path, required=True, help="arm A's answers; both arms are required"
    )
    parser.add_argument(
        "--samples-b", type=Path, required=True, help="arm B's answers; both arms are required"
    )
    parser.add_argument(
        "--judge-base-url",
        required=True,
        help="OpenAI-compatible endpoint serving the manifest's judge model",
    )
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_score_ragas)


def _add_stats(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("stats", help="paired bootstrap CI and randomization test")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics-a", type=Path, required=True)
    parser.add_argument("--metrics-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_stats)


def _add_report(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("report", help="render results.jsonl as markdown")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals.benchmark", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for register in (
        _add_preregister,
        _add_collect,
        _add_score_ir,
        _add_answer,
        _add_score_ragas,
        _add_stats,
        _add_report,
    ):
        register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, RuntimeError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
