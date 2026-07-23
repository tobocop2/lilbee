"""Command-line entry point for the lilbee retrieval benchmark.

Subcommands mirror the run stages: preregister, fetch, collect, score-ir,
answer, score-ragas, stats, report. Heavy dependencies (ir_measures,
ir_datasets, ragas) are imported lazily inside the modules they live in, so the
CLI loads without them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from evals.benchmark import metrics, stats
from evals.benchmark.collectors import (
    DEFAULT_TARGET_DOCS,
    LilbeeCollector,
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
from evals.benchmark.stats import DEFAULT_SEED
from evals.cli_support import append_jsonl, render_to_file, write_jsonl
from evals.deps import scorer_versions
from evals.retrieval.checkpoint import JsonlCheckpoint, load_items, load_jsonl

DEFAULT_METRICS = ["nDCG@10", "Recall@20", "MRR@10"]
ASK_ROUTE = "/api/ask"
ASK_TIMEOUT_SECONDS = 600.0


def _cmd_preregister(args: argparse.Namespace) -> int:
    manifest = Manifest.load(args.manifest)
    # Freezing is also the moment to warn that a run will be rescorable but not
    # rebuildable. The guard is not fatal here so a template can still be frozen,
    # but a real run that leaves the build unrecorded is told so before any data
    # moves, when it is still cheap to fill in.
    try:
        manifest.require_reproducible()
    except ValueError as exc:
        print(f"warning: {exc}", file=sys.stderr)
    fingerprint = manifest.freeze(args.out)
    print(f"froze manifest {manifest.run_id} -> {args.out} ({fingerprint[:12]})")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    queries = load_queries(args.queries)
    collector = LilbeeCollector(args.base_url, run_tag=args.run_tag, target_docs=args.target_docs)
    hits = collect_run(
        collector,
        queries,
        args.run,
        args.checkpoint,
        on_query=lambda qid: print(f"[{args.run_tag}] {qid}", flush=True),
    )
    print(f"collected {len(queries)} queries, {len(hits)} hits -> {args.run}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Materialize every dataset the manifest declares, from ir_datasets."""
    from evals.benchmark.collectors import write_queries
    from evals.benchmark.datasets import iter_documents, load_ir_dataset

    manifest = Manifest.load(args.manifest)
    for spec in manifest.datasets:
        out = args.out / spec.name
        out.mkdir(parents=True, exist_ok=True)
        # Queries and qrels are small enough to hold; the corpus is not. MS
        # MARCO's 8.8M passages as a dict were killed by the container's
        # out-of-memory monitor, so documents stream straight to disk.
        _, queries, qrels = load_ir_dataset(spec.loader, documents=False)
        write_queries(out / "queries.jsonl", queries)
        # TREC qrels, not JSON: the published artifact should be the format
        # every other IR tool reads, since these files travel with the report.
        (out / "qrels.trec").write_text(
            "".join(
                f"{qid} 0 {doc_id} {grade}\n"
                for qid, judged in sorted(qrels.items())
                for doc_id, grade in sorted(judged.items())
            )
        )
        written = 0
        with (out / "corpus.jsonl").open("w") as handle:
            for doc_id, title, text in iter_documents(spec.loader):
                handle.write(json.dumps({"doc_id": doc_id, "title": title, "text": text}) + "\n")
                written += 1
                # A silent hour is indistinguishable from a hang on a corpus
                # this size, and the run is watched from a tmux pane.
                if written % 1_000_000 == 0:
                    print(f"  {spec.name}: {written:,} passages written", flush=True)
        print(
            f"{spec.name}: {written:,} docs, {len(queries):,} queries, "
            f"{len(qrels):,} judged -> {out}"
        )
    return 0


def _cmd_score_ir(args: argparse.Namespace) -> int:
    from evals.benchmark.runfile import read_qrels, read_run

    qrels = read_qrels(args.qrels)
    run = read_run(args.run)
    scores = metrics.score_run(qrels, run, args.metrics)
    # Pool coverage travels with the scores. Every metric above treats an unjudged
    # document as non-relevant, so how much of the run the labels actually cover
    # is what says whether a delta is a finding or an artefact of the pool.
    judged = metrics.judged_at_k(qrels, run)
    write_jsonl(
        args.out,
        [
            {
                "dataset": args.dataset,
                "run_tag": args.run_tag,
                "aggregated": scores["aggregated"],
                "judged_at_k": judged,
                "judged_depth": metrics.JUDGED_DEPTH,
                "per_query": scores["per_query"],
            }
        ],
    )
    print(
        f"scored {args.run_tag} on {args.dataset}: {scores['aggregated']} "
        f"(judged@{metrics.JUDGED_DEPTH} {judged:.1%}) -> {args.out}"
    )
    if judged == 0.0:
        print(
            "warning: no retrieved document in the top "
            f"{metrics.JUDGED_DEPTH} carries a judgment on any topic. Every metric "
            "above is therefore zero by construction. This is the signature of a "
            "document-id mismatch between the run file and the qrels, not of a bad "
            "system; check that both name documents the same way before reporting.",
            file=sys.stderr,
        )
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
    checkpoint = JsonlCheckpoint(
        args.out,
        "query_id",
        fingerprint={"arm": args.arm, "base_url": args.base_url, "top_k": args.top_k},
    )
    client = httpx.Client(timeout=ASK_TIMEOUT_SECONDS)
    for query_id, text in queries.items():
        if query_id in checkpoint:
            continue
        answer, contexts = _ask(client, args.base_url, text, args.top_k)
        checkpoint.append(
            {
                "query_id": query_id,
                "arm": args.arm,
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
        for row in load_items(path)
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
    # Stamped at score time, not at freeze time. ragas is a fast-moving product
    # whose metric prompts change between releases, so "which ragas produced
    # this number" is a question the requirements pin cannot answer for a run
    # that has already finished.
    rows.append(
        {
            "row_type": "versions",
            "judge_model": manifest.models.judge,
            "judge_base_url": args.judge_base_url,
            "scorers": scorer_versions(),
        }
    )
    append_jsonl(args.out, rows)
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
    missing = [
        metric
        for metric in manifest.metrics
        if metric not in file_a["per_query"] or metric not in file_b["per_query"]
    ]
    if missing:
        # An absent metric would otherwise reach compare() as two empty vectors
        # and come back n=0, delta 0.0, CI [0,0], p=1.0, which renders as a
        # measured tie between the arms rather than as data that was never
        # scored. score-ir takes its own --metrics, so this is a live mismatch.
        raise ValueError(
            f"metrics {sorted(missing)} are declared in manifest '{manifest.run_id}' "
            "but absent from the scored files; re-run score-ir with "
            f"--metrics {' '.join(manifest.metrics)} rather than reporting them as ties"
        )
    for metric in manifest.metrics:
        per_query_a = file_a["per_query"][metric]
        per_query_b = file_b["per_query"][metric]
        result = stats.compare(
            metric,
            per_query_a,
            per_query_b,
            resamples=manifest.stats.bootstrap_resamples,
            seed=manifest.stats.seed,
            alpha=manifest.stats.alpha,
        )
        # Each row carries its own arm pair, not just the file's first meta row.
        # A results file accumulates several comparisons (BH runs across all of
        # them), and an ablation's comparisons do not share one arm pair, so a
        # single file-level label would print one comparison's scores under
        # another's arm names.
        row = {
            "row_type": "ir",
            "dataset": dataset,
            "arm_a": arm_a,
            "arm_b": arm_b,
            # Pool coverage for each arm, carried from score-ir so the report can
            # state how much of each run the labels covered beside the delta.
            "judged_a": file_a.get("judged_at_k"),
            "judged_b": file_b.get("judged_at_k"),
            **result.to_dict(),
        }
        rows.append(row)
    append_jsonl(args.out, rows)
    print(f"wrote {len(rows) - 1} paired IR comparisons -> {args.out}")
    return 0


def _cmd_score_ragchecker(args: argparse.Namespace) -> int:
    """Cross-check the RAGAS tier, and split the result by which half moved."""
    from evals.benchmark.ragchecker_tier import (
        RagCheckerJudge,
        attribution,
        make_ragchecker_evaluator,
        score_ragchecker,
    )

    manifest = Manifest.load(args.manifest)
    judge = RagCheckerJudge(model=manifest.models.judge, base_url=args.judge_base_url)
    evaluate_fn = make_ragchecker_evaluator(judge)
    scored = {}
    for label, path in (("arm_a", args.samples_a), ("arm_b", args.samples_b)):
        samples = _load_samples(path)
        query_ids = [row["query_id"] for row in load_items(path)]
        scored[label] = score_ragchecker(samples, query_ids, evaluate_fn=evaluate_fn)
    deltas = attribution(scored["arm_a"], scored["arm_b"])
    append_jsonl(
        args.out,
        [
            {
                "row_type": "ragchecker",
                "arm_a": scored["arm_a"].to_dict(),
                "arm_b": scored["arm_b"].to_dict(),
                **deltas,
            }
        ],
    )
    print(
        f"retriever side moved {deltas['retriever_delta']:+.4f}, generator side "
        f"{deltas['generator_delta']:+.4f} -> {args.out}"
    )
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Grade a public human-rated set with this judge and report the agreement.

    No annotator is involved: SummEval's ratings were produced by three experts
    years before this harness existed, so the judge is measured against labels
    nobody here could have tuned.
    """
    import random

    from evals.benchmark.calibration import calibrate, load_summeval
    from evals.retrieval.blinding import BlindRow
    from evals.retrieval.judging import judge_rows
    from evals.retrieval.llm import judge_backend, warm_chat

    pairs = load_summeval(limit=args.articles)
    judge = judge_backend()
    warm_chat(judge.chat)
    # Reuses the blind grading path, so the judge scores these exactly as it
    # scores a real run: same rubric, same presentations, same checkpointing.
    rows = [
        BlindRow(
            gid=pair.pair_id,
            question="Summarise the ground material.",
            source="summeval",
            ground=pair.ground,
            answer=pair.response,
            variant=index % 2,
        )
        for index, pair in enumerate(pairs)
    ]
    random.Random(args.seed).shuffle(rows)
    graded = judge_rows(rows, judge.llm, args.work_dir / "calibration_grades.jsonl")
    results = calibrate(graded, pairs)
    append_jsonl(args.out, [{"row_type": "calibration", **r.to_dict()} for r in results])
    for result in results:
        print(
            f"{result.dimension}: spearman {result.spearman:+.3f} vs expert ceiling "
            f"{result.expert_ceiling:.3f} ({result.fraction_of_ceiling:.0%} of it), "
            f"n={result.n}"
        )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    return render_to_file(args.results, args.out, render_report)


def _add_preregister(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("preregister", help="validate and freeze the manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_preregister)


def _add_collect(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("collect", help="build a TREC run file for one arm")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--target-docs",
        type=int,
        default=DEFAULT_TARGET_DOCS,
        help="distinct parent documents to collect per query; every arm uses the same depth",
    )
    parser.set_defaults(handler=_cmd_collect)


def _add_fetch(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("fetch", help="materialize the manifest's datasets from ir_datasets")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_fetch)


def _add_score_ir(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("score-ir", help="score a run file against qrels with ir_measures")
    parser.add_argument("--qrels", type=Path, required=True, help="TREC qrels file")
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


def _add_score_ragchecker(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "score-ragchecker",
        help="claim-level cross-check of the answer tier, split retriever vs generator",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples-a", type=Path, required=True)
    parser.add_argument("--samples-b", type=Path, required=True)
    parser.add_argument(
        "--judge-base-url",
        required=True,
        help="OpenAI-compatible endpoint serving the manifest's judge model",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_score_ragchecker)


def _add_calibrate(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("calibrate", help="measure the judge against SummEval's expert ratings")
    parser.add_argument("--articles", type=int, default=None, help="cap articles, not pairs")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, required=True)
    parser.set_defaults(handler=_cmd_calibrate)


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
        _add_fetch,
        _add_collect,
        _add_score_ir,
        _add_answer,
        _add_score_ragas,
        _add_stats,
        _add_score_ragchecker,
        _add_calibrate,
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
