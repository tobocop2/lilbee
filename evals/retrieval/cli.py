"""Command-line entry point: questions, answer, judge, score, report."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from evals.retrieval.answers import AnswerRow, answer_questions, make_http_client
from evals.retrieval.blinding import BlindAssignment, build_blind_rows, unblind
from evals.retrieval.checkpoint import load_items, load_jsonl
from evals.retrieval.judging import DIMENSIONS, judge_rows
from evals.retrieval.llm import judge_backend, lilbee_chat_fn, warm_chat
from evals.retrieval.questions import (
    COUNT_QUESTIONS,
    DEFAULT_SEED,
    KNOWN_ITEM_QUESTIONS,
    TOPICAL_QUESTIONS,
    Question,
    build_questions,
)
from evals.retrieval.report import render_report
from evals.retrieval.scoring import build_results

GID_MAP_FILE = "gid_map.json"
PREFAILED_FILE = "prefailed.json"
JUDGE_META_FILE = "judge_meta.json"
BLIND_ROWS_FILE = "blind_rows.jsonl"
GRADES_FILE = "grades.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _load_questions(path: Path) -> list[Question]:
    return [Question.from_dict(row) for row in load_jsonl(path)]


def _load_answer_arm(path: Path) -> tuple[str, dict[str, AnswerRow]]:
    rows = [AnswerRow.from_dict(row) for row in load_items(path)]
    if not rows:
        raise ValueError(f"no answer rows in {path}")
    return rows[0].arm, {row.qid: row for row in rows}


def _load_arms(args: argparse.Namespace) -> dict[str, dict[str, AnswerRow]]:
    arm_a, answers_a = _load_answer_arm(args.answers_a)
    arm_b, answers_b = _load_answer_arm(args.answers_b)
    if arm_a == arm_b:
        raise ValueError(f"both answer files carry the arm label '{arm_a}'")
    return {arm_a: answers_a, arm_b: answers_b}


def _cmd_questions(args: argparse.Namespace) -> int:
    lancedb_dir = args.lancedb_dir or args.data_root / "data" / "lancedb"
    chat = lilbee_chat_fn()
    warm_chat(chat)
    questions = build_questions(
        lancedb_dir,
        chat,
        topical=args.topical,
        known_item=args.known_item,
        count=args.count,
        seed=args.seed,
    )
    _write_jsonl(args.out, [question.to_dict() for question in questions])
    print(f"wrote {len(questions)} questions -> {args.out}")
    return 0


def _cmd_answer(args: argparse.Namespace) -> int:
    questions = _load_questions(args.questions)
    rows = answer_questions(
        questions,
        args.base_url,
        args.arm,
        args.out,
        top_k=args.top_k,
        client=make_http_client(),
    )
    failed = sum(1 for row in rows if row.error)
    print(f"answered {len(rows)} new questions, {failed} errors -> {args.out}")
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    questions = _load_questions(args.questions)
    answers_by_arm = _load_arms(args)
    noise_arm = next(reversed(answers_by_arm))
    blind = build_blind_rows(questions, answers_by_arm, noise_arm, random.Random(args.seed))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / GID_MAP_FILE).write_text(
        json.dumps({gid: a.to_dict() for gid, a in blind.assignments.items()}, indent=1)
    )
    (args.work_dir / PREFAILED_FILE).write_text(json.dumps(blind.prefailed, indent=1))
    _write_jsonl(args.work_dir / BLIND_ROWS_FILE, [row.to_dict() for row in blind.rows])
    judge = judge_backend()
    warm_chat(judge.chat)
    grades = judge_rows(blind.rows, judge.chat, args.work_dir / GRADES_FILE)
    # The noise arm and judge identity are recorded here so scoring cannot pick a
    # different noise arm than the one that was actually graded twice, and so a
    # finished run says which model produced its grades.
    (args.work_dir / JUDGE_META_FILE).write_text(
        json.dumps(
            {"noise_arm": noise_arm, "judge_model": judge.model, "judge_base_url": judge.base_url},
            indent=1,
        )
    )
    unreturned = len(blind.rows) - sum(1 for row in blind.rows if row.gid in grades)
    print(
        f"{len(blind.rows)} blind rows judged by {judge.model} ({unreturned} unreturned), "
        f"{len(blind.prefailed)} prefailed -> {args.work_dir}"
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    questions = _load_questions(args.questions)
    answers_by_arm = _load_arms(args)
    assignments = {
        gid: BlindAssignment.from_dict(data)
        for gid, data in json.loads((args.work_dir / GID_MAP_FILE).read_text()).items()
    }
    judge_meta = json.loads((args.work_dir / JUDGE_META_FILE).read_text())
    noise_arm = judge_meta["noise_arm"]
    if noise_arm not in answers_by_arm:
        raise ValueError(
            f"the judge pass graded '{noise_arm}' twice, but the arms given here are "
            f"{sorted(answers_by_arm)}; pass the same --answers-a/--answers-b as judge did"
        )
    prefailed = json.loads((args.work_dir / PREFAILED_FILE).read_text())
    judged = {
        record["gid"]: {k: v for k, v in record.items() if k != "gid"}
        for record in load_jsonl(args.work_dir / GRADES_FILE)
    }
    # The noise floor is measured only over rows a judge actually graded twice.
    # Prefailed rows were never judged; both of their replicates are mechanically
    # set to zero below, which would register as perfect agreement and pull the
    # floor toward zero, making every real cross-arm delta look like signal.
    judged_only = unblind(assignments, judged)
    scored = dict(judged)
    for gid in prefailed:
        scored[gid] = dict.fromkeys(DIMENSIONS, 0)
    results = build_results(
        questions,
        answers_by_arm,
        unblind(assignments, scored),
        noise_arm,
        judged=judged_only,
        judge_model=judge_meta.get("judge_model", ""),
    )
    _write_jsonl(args.out, results)
    print(f"wrote {len(results)} result rows -> {args.out}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = render_report(load_jsonl(args.results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(f"wrote {args.out}")
    return 0


def _add_arm_io_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers-a", type=Path, required=True)
    parser.add_argument(
        "--answers-b", type=Path, required=True, help="the arm judged twice for the noise floor"
    )
    parser.add_argument("--work-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals.retrieval", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    questions = subparsers.add_parser("questions", help="generate the question battery")
    questions.add_argument("--data-root", type=Path, required=True)
    questions.add_argument(
        "--lancedb-dir",
        type=Path,
        default=None,
        help="override the default <data-root>/data/lancedb",
    )
    questions.add_argument("--out", type=Path, required=True)
    questions.add_argument("--topical", type=int, default=TOPICAL_QUESTIONS)
    questions.add_argument("--known-item", type=int, default=KNOWN_ITEM_QUESTIONS)
    questions.add_argument("--count", type=int, default=COUNT_QUESTIONS)
    questions.add_argument("--seed", type=int, default=DEFAULT_SEED)
    questions.set_defaults(handler=_cmd_questions)

    answer = subparsers.add_parser("answer", help="answer the battery against one server")
    answer.add_argument("--questions", type=Path, required=True)
    answer.add_argument("--base-url", required=True)
    answer.add_argument("--arm", required=True, help="label recorded on every row")
    answer.add_argument("--out", type=Path, required=True)
    answer.add_argument("--top-k", type=int, default=0)
    answer.set_defaults(handler=_cmd_answer)

    judge = subparsers.add_parser("judge", help="blind-judge both arms' answers")
    _add_arm_io_arguments(judge)
    judge.add_argument("--seed", type=int, default=DEFAULT_SEED)
    judge.set_defaults(handler=_cmd_judge)

    score = subparsers.add_parser("score", help="unblind grades and write results.jsonl")
    _add_arm_io_arguments(score)
    score.add_argument("--out", type=Path, required=True)
    score.set_defaults(handler=_cmd_score)

    report = subparsers.add_parser("report", help="render results.jsonl as markdown")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.set_defaults(handler=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
