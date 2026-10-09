"""Self-tests: plant a known defect, show the harness names it; plant nothing, show it is silent."""

from __future__ import annotations

import hashlib
import statistics
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tools.qa.crawl_parity import parity, plants, retrieval, speed
from tools.qa.crawl_parity.corpus import Corpus, load_synthetic
from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode, Page, Side, Status
from tools.qa.crawl_parity.pipeline import TRUTH_CACHE_FILE, left_behind_differences
from tools.qa.crawl_parity.replay import Replay
from tools.qa.crawl_parity.sides import (
    CrawlRequest,
    CrawlResult,
    SideConfig,
    neutral_origin,
    run_crawl,
)
from tools.qa.crawl_parity.structure import Element
from tools.qa.crawl_parity.thresholds import Thresholds
from tools.qa.crawl_parity.tokens import markdown_tokens, tokenize
from tools.qa.crawl_parity.truth import (
    CAPTURE_VERSION,
    BrowserName,
    Target,
    Truth,
    TruthCache,
    capture,
)
from tools.qa.crawl_parity.verdict import Expected, judge

SEED = "f"
# Where the lilbee driver keeps lilbee's data, and the one file a crawl with no sync leaves.
LILBEE_DATA_DIR = "lilbee-data"
CRAWL_RECORD = "crawl_meta.json"
FIRST_RUN = "http://127.0.0.1:50001"
SECOND_RUN = "http://127.0.0.1:50002"
SENTENCE_PAGE = "/f/blockquote"
TABLE_PAGE = "/f/tablespans"
HIDDEN_PAGE = "/f/hidden"
# The seed that sets a cookie, then a page that needs it.
COOKIE_PATHS = ("/c/", "/c/need")
STATUS_OK = 200
MIN_PAGES = 30
LEAKY_DRIVER = "selftest_leaky.py"
INVISIBLE_WORDS = "zzplantedone zzplantedtwo zzplantedthree"
CONCURRENT_REQUESTS = 3
# A crawl that takes 1.25 times as long saves pages at 0.8 of the rate: a 20 percent slowdown.
PLANTED_TIME_FACTOR = 0.25
PLANTED_SLOWDOWN = 0.20
SLOWDOWN_TOLERANCE = 0.08
DROPPED_QUESTION_SHARE = 0.10
MS_PER_SECOND = 1000


class Result(StrEnum):
    """How one self-test ended."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class SelfTest:
    """One self-test: what it planted, and whether the harness answered as it must."""

    name: str
    result: Result
    detail: str

    @property
    def passed(self) -> bool:
        """A skipped self-test did not fail; its line says that it did not run."""
        return self.result is not Result.FAIL

    def line(self) -> str:
        """One printed line."""
        return f"{self.result:<7} {self.name}: {self.detail}"


def _check(name: str, ok: bool, detail: str) -> SelfTest:
    return SelfTest(name, Result.PASS if ok else Result.FAIL, detail)


def _signatures(differences: list[Difference]) -> str:
    return ", ".join(sorted({d.signature for d in differences})) or "no difference"


@dataclass
class _Bench:
    """Real output of the reference side, which every plant is applied to."""

    config: SideConfig
    layer: Layer
    corpus: Corpus
    replay: Replay
    thresholds: Thresholds
    work: Path
    truths: dict[str, Truth]
    first: CrawlResult
    second: CrawlResult

    def compare(self, candidate: dict[str, Page]) -> list[Difference]:
        """Yardstick A of the reference's first run against *candidate*."""
        comparison = parity.compare(
            self.first.saved(),
            candidate,
            self.truths,
            self.layer,
            Mode.HTTP,
            self.thresholds.parity,
        )
        return comparison.differences

    def request(self) -> CrawlRequest:
        return CrawlRequest(self.replay.url(self.corpus.seed(SEED).path), Mode.HTTP)


def _control_parity(bench: _Bench) -> SelfTest:
    differences = bench.compare(bench.second.saved())
    saved = len(bench.first.saved())
    ok = not differences and saved >= MIN_PAGES
    return _check(
        "control, parity: the reference against a second run of itself",
        ok,
        f"{saved} pages, {_signatures(differences)}",
    )


def _plant_sentence_deleted(bench: _Bench) -> SelfTest:
    markdown = bench.second.saved()[SENTENCE_PAGE].markdown or ""
    sentence = plants.plain_sentence(markdown)
    planted = plants.with_markdown(
        bench.second.saved(), SENTENCE_PAGE, plants.delete_sentence(markdown, sentence)
    )
    differences = bench.compare(planted)
    words = len(tokenize(sentence).words)
    lost = [
        d
        for d in differences
        if d.kind is Kind.TEXT_LOST and d.counts and d.feature != parity.SYMBOLS_FEATURE
    ]
    ok = (
        {d.page for d in differences} == {SENTENCE_PAGE}
        and sum(d.amount for d in lost) == words
        and not any(d.kind in {Kind.TEXT_ADDED, Kind.PAGE_LOST} for d in differences)
    )
    return _check(
        "plant, parity: one sentence deleted",
        ok,
        f"{words} words deleted on {SENTENCE_PAGE}; reported {_signatures(differences)}",
    )


def _plant_sentence_duplicated(bench: _Bench) -> SelfTest:
    markdown = bench.second.saved()[SENTENCE_PAGE].markdown or ""
    sentence = plants.plain_sentence(markdown)
    planted = plants.with_markdown(
        bench.second.saved(), SENTENCE_PAGE, plants.duplicate_sentence(markdown, sentence)
    )
    differences = bench.compare(planted)
    words = len(tokenize(sentence).words)
    added = [
        d
        for d in differences
        if d.kind is Kind.TEXT_ADDED and d.counts and d.feature != parity.SYMBOLS_FEATURE
    ]
    ok = (
        {d.page for d in differences} == {SENTENCE_PAGE}
        and sum(d.amount for d in added) == words
        and all(d.feature.startswith(parity.REPEAT) for d in added)
        and not any(d.kind is Kind.TEXT_LOST for d in differences)
    )
    return _check(
        "plant, parity: one sentence duplicated",
        ok,
        f"{words} words repeated on {SENTENCE_PAGE}; reported {_signatures(differences)}",
    )


def _plant_page_dropped(bench: _Bench) -> SelfTest:
    differences = bench.compare(plants.without_page(bench.second.saved(), SENTENCE_PAGE))
    ok = [(d.kind, d.page, d.counts) for d in differences] == [
        (Kind.PAGE_LOST, SENTENCE_PAGE, True)
    ]
    return _check("plant, parity: one page dropped", ok, f"reported {_signatures(differences)}")


def _plant_table_flattened(bench: _Bench) -> SelfTest:
    markdown = bench.second.saved()[TABLE_PAGE].markdown or ""
    flattened = plants.flatten_tables(markdown)
    differences = bench.compare(plants.with_markdown(bench.second.saved(), TABLE_PAGE, flattened))
    tables = [d for d in differences if d.kind is Kind.STRUCTURE and d.feature == Element.TABLE]
    same_words = markdown_tokens(markdown).word_counts() == markdown_tokens(flattened).word_counts()
    ok = bool(tables) and same_words and markdown != flattened
    ok = ok and not any(
        d.kind is Kind.TEXT_LOST and d.feature != parity.SYMBOLS_FEATURE for d in differences
    )
    return _check(
        "plant, parity: one table flattened, every word kept",
        ok,
        f"reported {_signatures(differences)}",
    )


def _normalisation(bench: _Bench, name: str, base: str, variant: str, deleted: str) -> SelfTest:
    """A variant under one normalisation is silent; the same variant with a deletion is not."""

    def differences(candidate: str) -> list[Difference]:
        pages = {"/p": Page("/p", base)}, {"/p": Page("/p", candidate)}
        return parity.compare(
            *pages, {}, bench.layer, Mode.HTTP, bench.thresholds.parity
        ).differences

    silent = differences(variant)
    loud = differences(deleted)
    ok = not silent and any(d.kind is Kind.TEXT_LOST for d in loud) and base != variant
    return _check(
        f"normalisation {name}: silent on the variant, loud on a deletion inside it",
        ok,
        f"variant: {_signatures(silent)}; deletion: {_signatures(loud)}",
    )


def _normalisations(bench: _Bench) -> list[SelfTest]:
    composed = unicodedata.normalize("NFC", "café déjà naïve words")
    decomposed = unicodedata.normalize("NFD", composed)
    return [
        _normalisation(
            bench,
            "markdown-syntax",
            "A **bold** claim with `code` kept.",
            "A __bold__ claim with ``code`` kept.",
            "A __bold__ with ``code`` kept.",
        ),
        _normalisation(
            bench, "unicode-nfc", composed, decomposed, decomposed.replace(" words", "")
        ),
        _normalisation(
            bench, "whitespace", "one two three four", "one   two\nthree\tfour", "one   two\nfour"
        ),
        _normalisation(
            bench,
            "replay-origin",
            neutral_origin(f"see {FIRST_RUN}/a for more words", FIRST_RUN) or "",
            neutral_origin(f"see {SECOND_RUN}/a  for more words", SECOND_RUN) or "",
            neutral_origin(f"see {SECOND_RUN}/a for words", SECOND_RUN) or "",
        ),
    ]


def _truth_checks(bench: _Bench) -> list[SelfTest]:
    truth = bench.truths[SENTENCE_PAGE]
    faithful = "\n\n".join(truth.sentences)
    control = parity.score(markdown_tokens(faithful), truth)
    sentence = max(truth.sentences, key=len)
    words = tokenize(sentence).words
    missing = parity.score(markdown_tokens(faithful.replace(sentence, "", 1)), truth)
    extra = parity.score(markdown_tokens(f"{faithful}\n\n{INVISIBLE_WORDS}"), truth)
    shouted = parity.score(markdown_tokens(faithful.upper()), truth)
    shouted_short = parity.score(markdown_tokens(faithful.replace(sentence, "", 1).upper()), truth)
    hidden = bench.truths[HIDDEN_PAGE]
    behind_cookie = bench.truths[COOKIE_PATHS[1]]
    hidden_only = set(hidden.hidden_words) - set(hidden.visible.words)
    return [
        _check(
            "control, ground truth: markdown that is the visible text",
            control.recall == 1.0 and not control.unexplained and bool(truth.visible.words),
            f"recall {control.recall:.3f}, {sum(control.unexplained.values())} unexplained words, "
            f"{len(truth.visible.words)} visible words",
        ),
        _check(
            "plant, ground truth: one visible sentence missing",
            sum(missing.missed.values()) == len(words) and missing.recall < 1.0,
            f"{len(words)} words removed, {sum(missing.missed.values())} reported missed, "
            f"recall {missing.recall:.3f}",
        ),
        _check(
            "plant, ground truth: three words the page does not show",
            sum(extra.unexplained.values()) == len(INVISIBLE_WORDS.split()) and extra.recall == 1.0,
            f"{sum(extra.unexplained.values())} unexplained words reported",
        ),
        _check(
            "normalisation visible-case: silent on another case, loud on a deletion inside it",
            shouted.recall == 1.0 and sum(shouted_short.missed.values()) == len(words),
            f"upper case recall {shouted.recall:.3f}; with a deletion "
            f"{sum(shouted_short.missed.values())} missed",
        ),
        _check(
            "control, ground truth: the browser separates hidden text from visible text",
            len(hidden_only) > 0 and any("V1" in word for word in hidden.visible.words),
            f"{len(hidden_only)} words only in hidden elements of {HIDDEN_PAGE}",
        ),
        _check(
            "control, ground truth: a page behind a cookie is read as by a reader from the seed",
            behind_cookie.scorable() and behind_cookie.status == STATUS_OK,
            f"{COOKIE_PATHS[1]} answered {behind_cookie.status} with "
            f"{len(behind_cookie.visible.words)} visible words",
        ),
    ]


def _left_behind_checks(bench: _Bench) -> list[SelfTest]:
    limits = bench.thresholds.left_behind
    clean = left_behind_differences(bench.first, bench.second, limits)
    leaky = SideConfig(
        Side.CANDIDATE, "planted", {Layer.CRAWLER: Path(sys.executable)}, LEAKY_DRIVER, ""
    )
    result = run_crawl(leaky, Layer.CRAWLER, bench.request(), bench.work / "leaky")
    planted = left_behind_differences(result, None, limits)
    features = {d.feature.split(":")[0] for d in planted}
    wanted = {"process", "listening-socket", "temp", "threads"}
    return [
        _check(
            "control, left behind: a reference crawl in HTTP mode",
            not clean and bench.first.left.is_empty(),
            f"{_signatures(clean)}; {len(bench.first.left.processes)} processes, "
            f"{len(bench.first.left.temp_entries)} temp entries",
        ),
        _check(
            "plant, left behind: a process, its listening socket, a temp directory, a thread",
            features == wanted and len(result.left.processes) == 1,
            f"reported {_signatures(planted)}; the planted process was stopped",
        ),
    ]


def _speed_checks(bench: _Bench) -> list[SelfTest]:
    limits = bench.thresholds.speed
    request = bench.request()

    def samples(label: str) -> list[speed.Sample]:
        return speed.measure(
            bench.config,
            bench.layer,
            request,
            limits.repeats,
            bench.work / label,
            bench.replay.reset,
        )

    base = samples("speed-base")
    again = samples("speed-again")
    requests = max(len(bench.replay.requests()), 1)
    control = speed.compare(base, again, limits, bench.layer, Mode.HTTP)
    typical = statistics.median(sample.crawl_seconds for sample in base) if base else 0.0
    delay_ms = PLANTED_TIME_FACTOR * typical * CONCURRENT_REQUESTS / requests * MS_PER_SECOND
    bench.replay.delay_every_response(delay_ms)
    slow = samples("speed-slow")
    bench.replay.delay_every_response(0.0)
    planted = speed.compare(base, slow, limits, bench.layer, Mode.HTTP)
    if not control.compared or not planted.compared:
        reason = control.reason if not control.compared else planted.reason
        return [SelfTest("speed self-tests", Result.FAIL, f"no comparison was made: {reason}")]
    slower = [d for d in planted.differences if d.feature == speed.PAGES_FEATURE]
    measured = slower[0].amount if slower else 0.0
    return [
        _check(
            "control, speed: the reference against a second set of runs",
            not control.differences,
            f"{_signatures(list(control.differences))}; {control.reason}; "
            + "; ".join(s.text() for s in again),
        ),
        _check(
            "plant, speed: every answer delayed for a 20 percent slowdown",
            bool(slower) and abs(measured - PLANTED_SLOWDOWN) <= SLOWDOWN_TOLERANCE,
            f"{delay_ms:.1f} ms added to each of {requests} answers; measured slowdown "
            f"{measured:.2f}; {planted.reason}",
        ),
    ]


def _retrieval_checks(bench: _Bench, config: retrieval.RetrievalConfig | None) -> list[SelfTest]:
    if config is None:
        return [
            SelfTest(
                "retrieval self-tests", Result.SKIPPED, "the sides file has no [retrieval] section"
            )
        ]
    limits = bench.thresholds.retrieval
    pages = bench.first.saved()
    truths = {path: truth for path, truth in bench.truths.items() if path in pages}
    asked = retrieval.questions(truths, limits.sample_size, seed=0)
    dropped = sorted({q.page for q in asked})[: max(1, round(len(asked) * DROPPED_QUESTION_SHARE))]
    kept = {path: page for path, page in pages.items() if path not in dropped}
    lost_questions = sum(1 for q in asked if q.page in dropped)

    def measured(label: str, side_pages: dict[str, Page]) -> retrieval.RetrievalResult:
        return retrieval.measure(side_pages, asked, config, limits.top_k, bench.work / label)

    base, again, short = (
        measured("r-base", pages),
        measured("r-again", pages),
        measured("r-short", kept),
    )
    control = retrieval.compare(base, again, asked, limits, bench.layer, Mode.HTTP)
    planted = retrieval.compare(base, short, asked, limits, bench.layer, Mode.HTTP)
    return [
        _check(
            "control, retrieval: two indexes of the same pages",
            not control and base.recall > 0,
            f"recall {base.recall:.3f} and {again.recall:.3f} on {len(asked)} questions; index "
            f"{base.index_seconds:.0f} s, search {base.search_seconds:.0f} s",
        ),
        _check(
            f"plant, retrieval: {len(dropped)} pages with {lost_questions} questions dropped",
            len(planted) == 1 and planted[0].kind is Kind.RECALL,
            f"recall {base.recall:.3f} against {short.recall:.3f}; reported {_signatures(planted)}",
        ),
    ]


def _verdict_checks(bench: _Bench) -> list[SelfTest]:
    planted = bench.compare(plants.without_page(bench.second.saved(), SENTENCE_PAGE))
    signature = planted[0].signature if planted else ""
    ran = {(Kind.PAGE_LOST, Mode.HTTP)}
    stale = Expected("page-lost/http/*/no-such-reason", "example#2")
    cases = {
        Status.NEW: judge(planted, [], ran),
        Status.KNOWN: judge(planted, [Expected(signature, "example#1")], ran),
        Status.ACCEPTED: judge(planted, [Expected(signature, "example#1", accepted=True)], ran),
    }
    statuses_ok = all(
        [finding.status for finding in verdict.findings] == [status]
        for status, verdict in cases.items()
    )
    passes = {status: verdict.passed for status, verdict in cases.items()}
    fixed = judge([], [stale], ran)
    not_run = judge([], [stale], set())
    return [
        _check(
            "verdict: a planted difference is NEW, KNOWN with an entry, ACCEPTED by the owner",
            statuses_ok
            and passes == {Status.NEW: False, Status.KNOWN: False, Status.ACCEPTED: True},
            f"passed: { ({str(k): v for k, v in passes.items()}) }",
        ),
        _check(
            "verdict: an unmatched entry is FIXED, and NOT-RUN when its yardstick did not run",
            len(fixed.fixed) == 1 and len(not_run.not_run) == 1 and not not_run.fixed,
            f"{len(fixed.fixed)} FIXED after a full run, {len(not_run.not_run)} NOT-RUN after none",
        ),
    ]


def _lilbee_driver_checks(bench: _Bench) -> list[SelfTest]:
    """With the sync replaced, a lilbee crawl leaves its crawl record and no index."""
    if bench.layer is not Layer.LILBEE:
        return []
    data = bench.first.work / "out" / LILBEE_DATA_DIR / "data"
    entries = sorted(entry.name for entry in data.iterdir()) if data.is_dir() else []
    return [
        _check(
            "control, lilbee driver: the crawl made no index, so no embedding model ran",
            entries == [CRAWL_RECORD],
            f"the data directory holds {entries}",
        )
    ]


def _reference(sides: dict[Side, SideConfig]) -> SideConfig:
    """The side the self-tests run: the oracle, or the candidate when there is no oracle."""
    return sides.get(Side.ORACLE) or sides[Side.CANDIDATE]


def _target(replay: Replay, corpus: Corpus, path: str) -> Target:
    digest = hashlib.sha256(corpus.records[path].response.body).hexdigest()
    return Target(replay.url(path), f"{CAPTURE_VERSION}:{path}:{digest}")


def _bench(
    config: SideConfig, thresholds: Thresholds, work: Path, replay: Replay, corpus: Corpus
) -> _Bench:
    layer = [layer for layer in Layer if config.has(layer)][-1]
    seed = corpus.seed(SEED)
    wanted = [*COOKIE_PATHS, *(path for path in corpus.html_pages() if path.startswith(seed.path))]
    targets = {path: _target(replay, corpus, path) for path in wanted}
    truths = capture(targets, BrowserName.CHROMIUM, TruthCache(work / TRUTH_CACHE_FILE))
    request = CrawlRequest(replay.url(seed.path), Mode.HTTP)
    replay.reset()
    first = run_crawl(config, layer, request, work / "first")
    replay.reset()
    second = run_crawl(config, layer, request, work / "second")
    return _Bench(config, layer, corpus, replay, thresholds, work, truths, first, second)


def run(
    sides: dict[Side, SideConfig],
    retrieval_config: retrieval.RetrievalConfig | None,
    thresholds: Thresholds,
    work: Path,
) -> list[SelfTest]:
    """Every self-test, on real output of the reference side."""
    corpus = load_synthetic()
    single: list[Callable[[_Bench], SelfTest]] = [
        _control_parity,
        _plant_sentence_deleted,
        _plant_sentence_duplicated,
        _plant_page_dropped,
        _plant_table_flattened,
    ]
    grouped: list[Callable[[_Bench], list[SelfTest]]] = [
        _normalisations,
        _truth_checks,
        _left_behind_checks,
        _verdict_checks,
        _speed_checks,
    ]
    with Replay(corpus) as replay:
        bench = _bench(_reference(sides), thresholds, work, replay, corpus)
        saved = min(len(bench.first.saved()), len(bench.second.saved()))
        if saved < MIN_PAGES:
            detail = f"{saved} pages saved, {MIN_PAGES} needed: {bench.first.stderr_tail[-300:]}"
            return [
                SelfTest("the reference crawl that every plant is applied to", Result.FAIL, detail)
            ]
        results = [check(bench) for check in single]
        results.extend(_lilbee_driver_checks(bench))
        for group in grouped:
            results.extend(group(bench))
        results.extend(_retrieval_checks(bench, retrieval_config))
    return results
