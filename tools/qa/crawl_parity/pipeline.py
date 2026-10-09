"""One run: replay the corpus, drive each side at each layer, measure, give the verdict."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from tools.qa.crawl_parity import parity, retrieval, speed
from tools.qa.crawl_parity.converter import convert_corpus
from tools.qa.crawl_parity.corpus import Corpus
from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode, Page, Side
from tools.qa.crawl_parity.replay import Replay
from tools.qa.crawl_parity.sides import CrawlRequest, CrawlResult, SideConfig, run_crawl
from tools.qa.crawl_parity.thresholds import LeftBehindLimits, Thresholds
from tools.qa.crawl_parity.truth import (
    CAPTURE_VERSION,
    BrowserName,
    Target,
    Truth,
    TruthCache,
    capture,
)
from tools.qa.crawl_parity.verdict import Expected, Verdict, judge

TRUTH_CACHE_FILE = "truth-cache.json"
SPEED_SEED = "f"
_RANDOM_SUFFIX = re.compile(r"[A-Za-z0-9_]{6,}$")


class Yardstick(StrEnum):
    """The measurements a run can make."""

    PARITY = "parity"
    RETRIEVAL = "retrieval"
    LEFT_BEHIND = "left-behind"
    SPEED = "speed"


KINDS = {
    Yardstick.PARITY: (
        Kind.PAGE_LOST,
        Kind.PAGE_EXTRA,
        Kind.TEXT_LOST,
        Kind.TEXT_ADDED,
        Kind.STRUCTURE,
        Kind.VISIBLE_MISSED,
    ),
    Yardstick.RETRIEVAL: (Kind.RECALL,),
    Yardstick.LEFT_BEHIND: (Kind.LEFT_BEHIND,),
    Yardstick.SPEED: (Kind.SLOWER,),
}


@dataclass(frozen=True)
class Plan:
    """Everything one run is given."""

    corpus: Corpus
    sides: dict[Side, SideConfig]
    thresholds: Thresholds
    expected: list[Expected]
    work: Path
    modes: tuple[Mode, ...]
    yardsticks: frozenset[Yardstick]
    seeds: tuple[str, ...]
    retrieval: retrieval.RetrievalConfig | None = None
    browser: BrowserName = BrowserName.CHROMIUM
    sample_seed: int = 0

    def layers(self, side: Side) -> list[Layer]:
        """The layers *side* can run, lowest first."""
        return [layer for layer in Layer if self.sides[side].has(layer)]

    def top_layer(self) -> Layer:
        """The highest layer every configured side can run; the layer the verdict is about."""
        common = [layer for layer in Layer if all(c.has(layer) for c in self.sides.values())]
        return common[-1]

    def covers_whole_corpus(self) -> bool:
        """Whether every seed of the corpus is crawled."""
        return set(self.seeds) == {seed.name for seed in self.corpus.seeds}


@dataclass(frozen=True)
class SpeedOutcome:
    """The samples of both sides for one mode and what the comparison said."""

    mode: Mode
    samples: dict[Side, list[speed.Sample]]
    comparison: speed.SpeedComparison | None


@dataclass(frozen=True)
class RetrievalOutcome:
    """Recall for each side in one mode."""

    mode: Mode
    asked: int
    results: dict[Side, retrieval.RetrievalResult]


@dataclass
class Outcome:
    """Everything one run measured."""

    truths: dict[str, Truth] = field(default_factory=dict)
    pages: dict[tuple[Side, Layer, Mode], dict[str, Page]] = field(default_factory=dict)
    crawls: list[CrawlResult] = field(default_factory=list)
    comparisons: list[parity.Comparison] = field(default_factory=list)
    speeds: list[SpeedOutcome] = field(default_factory=list)
    retrievals: list[RetrievalOutcome] = field(default_factory=list)
    differences: list[Difference] = field(default_factory=list)
    unrecorded: Counter[str] = field(default_factory=Counter)
    not_measured: list[str] = field(default_factory=list)
    ran: set[tuple[Kind, Mode]] = field(default_factory=set)
    verdict: Verdict | None = None


def _targets(corpus: Corpus, replay: Replay, browser: BrowserName) -> dict[str, Target]:
    """One target for each record, seeds first, as a reader who starts at a seed.

    The key changes when any body of the corpus changes.
    """
    digest = hashlib.sha256(f"{CAPTURE_VERSION}:{browser}".encode())
    for path, record in sorted(corpus.records.items()):
        digest.update(path.encode() + record.response.body)
    seeds = [seed.path for seed in corpus.seeds]
    ordered = [*seeds, *(path for path in corpus.records if path not in seeds)]
    return {path: Target(replay.url(path), f"{path}:{digest.hexdigest()}") for path in ordered}


def temp_pattern(name: str) -> str:
    """A temp entry's name with its random ending replaced, so that runs give one signature."""
    return _RANDOM_SUFFIX.sub("*", name)


def left_behind_differences(
    candidate: CrawlResult, oracle: CrawlResult | None, limits: LeftBehindLimits
) -> list[Difference]:
    """What the candidate's run left; threads are counted against the oracle's run."""
    found: list[Difference] = []

    def add(feature: str, amount: float, detail: str) -> None:
        found.append(
            Difference(
                Kind.LEFT_BEHIND,
                candidate.mode,
                candidate.layer,
                feature,
                amount=amount,
                detail=detail,
            )
        )

    processes = Counter(process.name for process in candidate.left.processes)
    if sum(processes.values()) > limits.processes_max:
        for name, count in sorted(processes.items()):
            add(f"process:{name}", count, f"{count} alive after the run ended")
        ports = [port for process in candidate.left.processes for port in process.listening_ports]
        if ports:
            add("listening-socket", len(ports), f"ports {sorted(ports)}")
    entries = Counter(temp_pattern(entry) for entry in candidate.left.temp_entries)
    if sum(entries.values()) > limits.temp_entries_max:
        for pattern, count in sorted(entries.items()):
            add(f"temp:{pattern}", count, f"{count} left in the run's temp directory")
    if candidate.threads_started is not None:
        baseline = oracle.threads_started if oracle and oracle.threads_started is not None else 0
        excess = candidate.threads_started - baseline
        if excess > limits.threads_max:
            add("threads", excess, f"{candidate.threads_started} started, oracle {baseline}")
    return found


class _Run:
    """The state of one run while it is made."""

    def __init__(self, plan: Plan, replay: Replay) -> None:
        self.plan = plan
        self.replay = replay
        self.outcome = Outcome()
        self._by_run: dict[tuple[Side, Layer, Mode, str], CrawlResult] = {}

    def _note_unrecorded(self) -> None:
        self.outcome.unrecorded.update(self.replay.unrecorded())

    def capture_truth(self) -> None:
        cache = TruthCache(self.plan.work / TRUTH_CACHE_FILE)
        targets = _targets(self.plan.corpus, self.replay, self.plan.browser)
        self.replay.reset()
        self.replay.answer_as_to_a_patient_reader(True)
        self.outcome.truths = capture(targets, self.plan.browser, cache)
        self.replay.answer_as_to_a_patient_reader(False)
        self._note_unrecorded()

    def crawl(self, mode: Mode) -> None:
        """Every seed at every crawling layer on every side."""
        for side, config in self.plan.sides.items():
            for layer in self.plan.layers(side):
                if layer is Layer.CONVERTER:
                    continue
                merged = self.outcome.pages.setdefault((side, layer, mode), {})
                for name in self.plan.seeds:
                    seed = self.plan.corpus.seed(name)
                    if mode not in seed.modes:
                        continue
                    self.replay.reset()
                    work = self.plan.work / "crawl" / f"{side}-{layer}-{mode}-{name}"
                    result = run_crawl(
                        config, layer, CrawlRequest(self.replay.url(seed.path), mode), work
                    )
                    self._note_unrecorded()
                    self.outcome.crawls.append(result)
                    self._by_run[(side, layer, mode, name)] = result
                    for path, page in result.pages.items():
                        if path not in merged or page.markdown:
                            merged[path] = page

    def convert(self) -> None:
        """The converter layer of every side that has one, kept under each mode: it has none."""
        for side, config in self.plan.sides.items():
            if not config.has(Layer.CONVERTER):
                continue
            pages = convert_corpus(
                config, self.plan.corpus, self.replay.origin, self.plan.work / f"convert-{side}"
            )
            for mode in self.plan.modes:
                self.outcome.pages[(side, Layer.CONVERTER, mode)] = pages

    def compare(self, mode: Mode) -> None:
        """Yardstick A at every common layer, with the top layer's differences attributed."""
        top = self.plan.top_layer()
        candidate = self.outcome.pages
        if Side.ORACLE not in self.plan.sides:
            self.outcome.differences.extend(
                parity.against_truth(
                    candidate[(Side.CANDIDATE, top, mode)],
                    self.outcome.truths,
                    top,
                    mode,
                    self.plan.thresholds.truth,
                )
            )
            return
        by_layer: dict[Layer, list[Difference]] = {}
        for layer in Layer:
            keys = (Side.ORACLE, layer, mode), (Side.CANDIDATE, layer, mode)
            if not all(key in self.outcome.pages for key in keys):
                continue
            comparison = parity.compare(
                self.outcome.pages[keys[0]],
                self.outcome.pages[keys[1]],
                self.outcome.truths,
                layer,
                mode,
                self.plan.thresholds.parity,
            )
            self.outcome.comparisons.append(comparison)
            by_layer[layer] = comparison.differences
        self.outcome.differences.extend(parity.attribute_to_lowest_layer(by_layer, top))

    def left_behind(self) -> None:
        for (side, layer, mode, name), result in self._by_run.items():
            if side is Side.CANDIDATE:
                oracle = self._by_run.get((Side.ORACLE, layer, mode, name))
                self.outcome.differences.extend(
                    left_behind_differences(result, oracle, self.plan.thresholds.left_behind)
                )

    def speed(self, mode: Mode) -> None:
        top = self.plan.top_layer()
        limits = self.plan.thresholds.speed
        seed = self.plan.corpus.seed(SPEED_SEED)
        request = CrawlRequest(self.replay.url(seed.path), mode)
        samples = {
            side: speed.measure(
                config,
                top,
                request,
                limits.repeats,
                self.plan.work / "speed" / f"{side}-{mode}",
                self.replay.reset,
            )
            for side, config in self.plan.sides.items()
        }
        comparison = None
        if Side.ORACLE in samples and Side.CANDIDATE in samples:
            comparison = speed.compare(
                samples[Side.ORACLE], samples[Side.CANDIDATE], limits, top, mode
            )
            self.outcome.differences.extend(comparison.differences)
            if not comparison.compared:
                self.outcome.not_measured.append(f"speed, {mode} mode: {comparison.reason}")
        self.outcome.speeds.append(SpeedOutcome(mode, samples, comparison))

    def retrieval(self, mode: Mode) -> None:
        config = self.plan.retrieval
        if config is None:
            self.outcome.not_measured.append("retrieval: no [retrieval] section in the sides file")
            return
        limits = self.plan.thresholds.retrieval
        top = self.plan.top_layer()
        asked = retrieval.questions(self.outcome.truths, limits.sample_size, self.plan.sample_seed)
        results = {
            side: retrieval.measure(
                self.outcome.pages[(side, top, mode)],
                asked,
                config,
                limits.top_k,
                self.plan.work / "retrieval" / f"{side}-{mode}",
            )
            for side in self.plan.sides
        }
        self.outcome.retrievals.append(RetrievalOutcome(mode, len(asked), results))
        if Side.ORACLE in results and Side.CANDIDATE in results:
            self.outcome.differences.extend(
                retrieval.compare(
                    results[Side.ORACLE], results[Side.CANDIDATE], asked, limits, top, mode
                )
            )


def _ran(plan: Plan) -> set[tuple[Kind, Mode]]:
    """What the run could have shown; nothing when it crawled only part of the corpus."""
    if not plan.covers_whole_corpus():
        return set()
    return {
        (kind, mode)
        for yardstick in plan.yardsticks
        for kind in KINDS[yardstick]
        for mode in plan.modes
    }


def run(plan: Plan) -> Outcome:
    """Make one run and return everything it measured, with the verdict."""
    with Replay(plan.corpus) as replay:
        state = _Run(plan, replay)
        state.capture_truth()
        state.convert()
        for mode in plan.modes:
            state.crawl(mode)
            if Yardstick.PARITY in plan.yardsticks:
                state.compare(mode)
            if Yardstick.SPEED in plan.yardsticks:
                state.speed(mode)
            if Yardstick.RETRIEVAL in plan.yardsticks:
                state.retrieval(mode)
        if Yardstick.LEFT_BEHIND in plan.yardsticks:
            state.left_behind()
    outcome = state.outcome
    outcome.ran = _ran(plan)
    outcome.verdict = judge(
        outcome.differences, plan.expected, outcome.ran, tuple(outcome.not_measured)
    )
    return outcome
