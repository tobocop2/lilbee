"""The command line of the harness: ``python -m tools.qa.crawl_parity <command>``."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from tools.qa.crawl_parity import report, selftest
from tools.qa.crawl_parity.corpus import load_synthetic
from tools.qa.crawl_parity.model import Mode
from tools.qa.crawl_parity.pipeline import Plan, Yardstick, run
from tools.qa.crawl_parity.retrieval import RetrievalConfig
from tools.qa.crawl_parity.sides import load_sides
from tools.qa.crawl_parity.thresholds import THRESHOLDS_FILE, load_thresholds
from tools.qa.crawl_parity.truth import BrowserName
from tools.qa.crawl_parity.verdict import EXPECTED_FILE, load_expected

EXIT_PASS = 0
EXIT_FAIL = 1
RETRIEVAL_SECTION = "retrieval"


def _csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def _retrieval(sides_file: Path) -> RetrievalConfig | None:
    """The retrieval section of a sides file, when it has one."""
    with sides_file.open("rb") as handle:
        table = tomllib.load(handle).get(RETRIEVAL_SECTION)
    if table is None:
        return None
    return RetrievalConfig(Path(table["python"]), str(table["embedding_model"]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crawl_parity", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("run", "judge the candidate: PASS or FAIL"),
        ("selftest", "plant known defects and show that each one is reported"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--sides", required=True, type=Path, help="sides.toml")
        command.add_argument("--work", required=True, type=Path, help="directory for run files")
    judge_command = commands.add_parser(
        "judge", help="set a saved run against expected.toml again, without a new crawl"
    )
    judge_command.add_argument("--report", required=True, type=Path, help="a run's report")
    judge_command.add_argument("--expected", default=EXPECTED_FILE, type=Path)
    run_command = commands.choices["run"]
    run_command.add_argument("--report", required=True, type=Path, help="directory for the report")
    run_command.add_argument("--modes", default="http,browser", type=_csv)
    run_command.add_argument("--seeds", default="", type=_csv, help="default: every seed")
    run_command.add_argument("--skip", default="", type=_csv, help="yardsticks to leave out")
    run_command.add_argument(
        "--browser", default=BrowserName.CHROMIUM.value, help="ground truth browser"
    )
    run_command.add_argument("--thresholds", default=THRESHOLDS_FILE, type=Path)
    run_command.add_argument("--expected", default=EXPECTED_FILE, type=Path)
    run_command.add_argument(
        "--sample-seed", default=0, type=int, help="seed of the question sample"
    )
    return parser


def _run(arguments: argparse.Namespace) -> int:
    corpus = load_synthetic()
    skipped = {Yardstick(name) for name in arguments.skip}
    plan = Plan(
        corpus=corpus,
        sides=load_sides(arguments.sides),
        thresholds=load_thresholds(arguments.thresholds),
        expected=load_expected(arguments.expected),
        work=arguments.work,
        modes=tuple(Mode(name) for name in arguments.modes),
        yardsticks=frozenset(Yardstick) - skipped,
        seeds=tuple(arguments.seeds) or tuple(seed.name for seed in corpus.seeds),
        retrieval=_retrieval(arguments.sides),
        browser=BrowserName(arguments.browser),
        sample_seed=arguments.sample_seed,
    )
    outcome = run(plan)
    assert outcome.verdict is not None  # set by pipeline.run
    path = report.write(plan, outcome, arguments.report)
    print(report.verdict_line(outcome.verdict))
    print(f"report: {path}")
    return EXIT_PASS if outcome.verdict.passed else EXIT_FAIL


def _selftest(arguments: argparse.Namespace) -> int:
    results = selftest.run(
        load_sides(arguments.sides), _retrieval(arguments.sides), load_thresholds(), arguments.work
    )
    for result in results:
        print(result.line())
    counts = Counter(result.result for result in results)
    failed = counts[selftest.Result.FAIL]
    print(
        f"SELFTEST {'PASS' if not failed else 'FAIL'}: {counts[selftest.Result.PASS]} passed, "
        f"{failed} failed, {counts[selftest.Result.SKIPPED]} skipped"
    )
    return EXIT_PASS if not failed else EXIT_FAIL


def _judge(arguments: argparse.Namespace) -> int:
    verdict = report.rejudge(arguments.report, load_expected(arguments.expected))
    print(report.verdict_line(verdict))
    print(f"verdict: {arguments.report / report.VERDICT_FILE}")
    return EXIT_PASS if verdict.passed else EXIT_FAIL


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code."""
    arguments = _parser().parse_args(argv)
    handlers = {"run": _run, "selftest": _selftest, "judge": _judge}
    return handlers[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
