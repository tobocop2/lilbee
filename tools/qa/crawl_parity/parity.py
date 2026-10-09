"""Yardsticks A and B: candidate against oracle, and each side against what a reader sees."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from tools.qa.crawl_parity import structure
from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode, Page
from tools.qa.crawl_parity.thresholds import ParityLimits, TruthLimits
from tools.qa.crawl_parity.tokens import Tokens, markdown_tokens
from tools.qa.crawl_parity.truth import ATTRIBUTE_CONTEXT, HIDDEN_PREFIX, NO_CONTEXT, Truth

NO_TRUTH = "no-truth"
SYMBOLS_FEATURE = "symbols"
NO_REASON = "no-reason-given"
REPEAT = "repeat"
ORACLE_REPEAT = "oracle-repeat"
SAMPLE_WORDS = 8
FEATURE_WORDS = 3
REASON_CHARS = 50
_URL = re.compile(r"https?://\S+")
_DIGITS = re.compile(r"\d+")
_NOT_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class TruthScore:
    """How one side's markdown stands against the rendered page."""

    recall: float
    missed: Counter[str]
    hidden_held: int
    attribute_held: int
    unexplained: Counter[str]


@dataclass(frozen=True)
class PageMeasure:
    """The numbers of one page that both sides saved."""

    path: str
    words_lost: int
    words_added: int
    symbols_lost: int
    symbols_added: int
    structure_changed: int
    oracle_truth: TruthScore | None
    candidate_truth: TruthScore | None


@dataclass(frozen=True)
class Comparison:
    """Yardstick A for one layer and mode, with yardstick B's score for each page."""

    layer: Layer
    mode: Mode
    oracle_saved: int
    candidate_saved: int
    measures: list[PageMeasure] = field(default_factory=list)
    differences: list[Difference] = field(default_factory=list)

    @property
    def words_lost(self) -> int:
        """Word tokens lost, over every page both sides saved."""
        return sum(measure.words_lost for measure in self.measures)

    @property
    def words_added(self) -> int:
        """Word tokens added, over every page both sides saved."""
        return sum(measure.words_added for measure in self.measures)

    @property
    def structure_changed(self) -> int:
        """Structure descriptions that differ, over every page both sides saved."""
        return sum(measure.structure_changed for measure in self.measures)


def _fold(counts: Counter[str]) -> Counter[str]:
    folded: Counter[str] = Counter()
    for word, count in counts.items():
        folded[word.casefold()] += count
    return folded


def score(tokens: Tokens, truth: Truth) -> TruthScore:
    """Yardstick B for one page: visible words held, and words held that are not visible.

    A visible word the markdown holds in another letter case counts as held (``visible-case``):
    a style sheet can change the case a reader sees.
    """
    visible = truth.visible.word_counts()
    held = tokens.word_counts()
    missed = _fold(visible - held) - _fold(held - visible)
    extra = _fold(held - visible) - _fold(visible - held)
    hidden = _fold(truth.hidden_words)
    attribute = _fold(truth.attribute_words)
    total = sum(visible.values())
    unexplained = Counter(
        {w: c for w, c in extra.items() if w not in hidden and w not in attribute}
    )
    return TruthScore(
        recall=(total - sum(missed.values())) / total if total else 1.0,
        missed=missed,
        hidden_held=sum(count for word, count in extra.items() if word in hidden),
        attribute_held=sum(
            count for word, count in extra.items() if word in attribute and word not in hidden
        ),
        unexplained=unexplained,
    )


def reason_feature(reason: str | None) -> str:
    """A failure text with addresses and numbers taken out, as a signature feature."""
    if not reason:
        return NO_REASON
    text = _DIGITS.sub("N", _URL.sub("URL", reason)).lower()
    return _NOT_SLUG.sub("-", text).strip("-")[:REASON_CHARS] or NO_REASON


def _is_visible_context(context: str) -> bool:
    return not context.startswith(HIDDEN_PREFIX) and context not in {ATTRIBUTE_CONTEXT, NO_CONTEXT}


def _sample(words: Counter[str]) -> str:
    return " ".join(word for word, _ in words.most_common(SAMPLE_WORDS))


def _by_context(words: Counter[str], truth: Truth | None) -> dict[str, Counter[str]]:
    groups: dict[str, Counter[str]] = {}
    for word, count in words.items():
        context = truth.context_of(word) if truth else NO_TRUTH
        groups.setdefault(context, Counter())[word] = count
    return groups


@dataclass(frozen=True)
class _PageContext:
    """What every difference of one page shares."""

    path: str
    layer: Layer
    mode: Mode
    truth: Truth | None
    limits: ParityLimits

    def difference(
        self, kind: Kind, feature: str, amount: float, detail: str, counts: bool = True
    ) -> Difference:
        return Difference(kind, self.mode, self.layer, feature, self.path, amount, detail, counts)


def _split(changed: Counter[str], room: Counter[str]) -> tuple[Counter[str], Counter[str]]:
    """*changed* as the part the page's visible text accounts for (up to *room*), and the rest."""
    explained: Counter[str] = Counter()
    rest: Counter[str] = Counter()
    for word, count in changed.items():
        part = min(count, room[word])
        explained[word] = part
        rest[word] = count - part
    return +explained, +rest


def _lost_differences(
    lost: Counter[str], candidate: Counter[str], page: _PageContext
) -> list[Difference]:
    """Lost words, split by what a reader sees.

    A lost word counts when the page shows it more often than the candidate holds it. A word
    the oracle repeats beyond what the page shows does not count; a word the page does not
    show counts only when the limits say so.
    """
    if page.truth is None:
        return [page.difference(Kind.TEXT_LOST, NO_TRUTH, sum(lost.values()), _sample(lost))]
    visible_loss, rest = _split(lost, page.truth.visible.word_counts() - candidate)
    found = [
        page.difference(Kind.TEXT_LOST, context, sum(words.values()), _sample(words))
        for context, words in sorted(_by_context(visible_loss, page.truth).items())
    ]
    for context, words in sorted(_by_context(rest, page.truth).items()):
        shown = _is_visible_context(context)
        feature = f"{ORACLE_REPEAT}:{context}" if shown else context
        counts = not shown and page.limits.invisible_text_lost_counts
        found.append(
            page.difference(Kind.TEXT_LOST, feature, sum(words.values()), _sample(words), counts)
        )
    return found


def _added_feature(context: str, words: Counter[str]) -> str:
    if context == NO_CONTEXT:
        named = sorted(word.casefold() for word, _ in words.most_common(FEATURE_WORDS))
        return f"{NO_CONTEXT}:{'+'.join(named)}"
    return f"{REPEAT}:{context}" if _is_visible_context(context) else context


def _added_differences(
    added: Counter[str], oracle: Counter[str], page: _PageContext
) -> list[Difference]:
    """Added words, split by what a reader sees.

    An added word does not count when the page shows it more often than the oracle holds it:
    the candidate then holds visible text the oracle lacks. Every other added word counts.
    """
    if page.truth is None:
        return [page.difference(Kind.TEXT_ADDED, NO_TRUTH, sum(added.values()), _sample(added))]
    visible_gain, rest = _split(added, page.truth.visible.word_counts() - oracle)
    found = [
        page.difference(Kind.TEXT_ADDED, context, sum(words.values()), _sample(words), counts=False)
        for context, words in sorted(_by_context(visible_gain, page.truth).items())
    ]
    for context, words in sorted(_by_context(rest, page.truth).items()):
        found.append(
            page.difference(
                Kind.TEXT_ADDED, _added_feature(context, words), sum(words.values()), _sample(words)
            )
        )
    return found


def _within(differences: list[Difference], limit: int) -> list[Difference]:
    """Drop the counting differences when their total is within *limit*."""
    counted = sum(difference.amount for difference in differences if difference.counts)
    if counted > limit:
        return differences
    return [difference for difference in differences if not difference.counts]


def _symbol_differences(oracle: Tokens, candidate: Tokens, page: _PageContext) -> list[Difference]:
    lost = oracle.symbol_counts() - candidate.symbol_counts()
    added = candidate.symbol_counts() - oracle.symbol_counts()
    found = [
        page.difference(kind, SYMBOLS_FEATURE, sum(symbols.values()), _sample(symbols))
        for kind, symbols in ((Kind.TEXT_LOST, lost), (Kind.TEXT_ADDED, added))
        if symbols
    ]
    return _within(found, page.limits.symbols_changed_per_page_max)


def _structure_differences(oracle: str, candidate: str, page: _PageContext) -> list[Difference]:
    found: list[Difference] = []
    for delta in structure.compare(structure.signature(oracle), structure.signature(candidate)):
        amount = sum(delta.only_reference.values()) + sum(delta.only_other.values())
        detail = (
            f"oracle only: {_sample(delta.only_reference)} | "
            f"candidate only: {_sample(delta.only_other)}"
        )
        found.append(page.difference(Kind.STRUCTURE, delta.element.value, amount, detail))
    return _within(found, page.limits.structure_changed_per_page_max)


def _compare_page(
    oracle: str, candidate: str, page: _PageContext
) -> tuple[PageMeasure, list[Difference]]:
    oracle_tokens, candidate_tokens = markdown_tokens(oracle), markdown_tokens(candidate)
    lost = oracle_tokens.word_counts() - candidate_tokens.word_counts()
    added = candidate_tokens.word_counts() - oracle_tokens.word_counts()
    symbols = _symbol_differences(oracle_tokens, candidate_tokens, page)
    shape = _structure_differences(oracle, candidate, page)
    differences = [
        *_within(
            _lost_differences(lost, candidate_tokens.word_counts(), page) if lost else [],
            page.limits.words_lost_per_page_max,
        ),
        *_within(
            _added_differences(added, oracle_tokens.word_counts(), page) if added else [],
            page.limits.words_added_per_page_max,
        ),
        *symbols,
        *shape,
    ]
    scorable = page.truth is not None and page.truth.scorable()
    measure = PageMeasure(
        path=page.path,
        words_lost=sum(lost.values()),
        words_added=sum(added.values()),
        symbols_lost=int(sum(d.amount for d in symbols if d.kind is Kind.TEXT_LOST)),
        symbols_added=int(sum(d.amount for d in symbols if d.kind is Kind.TEXT_ADDED)),
        structure_changed=int(sum(d.amount for d in shape)),
        oracle_truth=score(oracle_tokens, page.truth) if scorable and page.truth else None,
        candidate_truth=score(candidate_tokens, page.truth) if scorable and page.truth else None,
    )
    return measure, differences


@dataclass(frozen=True)
class _RunContext:
    """The layer, mode, truths and limits of one comparison."""

    layer: Layer
    mode: Mode
    truths: dict[str, Truth]
    limits: ParityLimits


def _page_set_differences(
    oracle: dict[str, Page], candidate: dict[str, Page], context: _RunContext
) -> list[Difference]:
    """Pages only one side saved; the ground truth says whether a reader gets the page."""
    found: list[Difference] = []
    for path in sorted(set(oracle) | set(candidate)):
        ours, theirs = oracle.get(path), candidate.get(path)
        truth = context.truths.get(path)
        readable = truth is None or truth.scorable()
        if ours and ours.markdown and not (theirs and theirs.markdown):
            reason = reason_feature(theirs.error if theirs else "not returned")
            found.append(
                Difference(
                    Kind.PAGE_LOST, context.mode, context.layer, reason, path, counts=readable
                )
            )
        elif theirs and theirs.markdown and not (ours and ours.markdown):
            reason = reason_feature(ours.error if ours else "not returned")
            found.append(
                Difference(
                    Kind.PAGE_EXTRA, context.mode, context.layer, reason, path, counts=not readable
                )
            )
    return found


def compare(
    oracle: dict[str, Page],
    candidate: dict[str, Page],
    truths: dict[str, Truth],
    layer: Layer,
    mode: Mode,
    limits: ParityLimits,
) -> Comparison:
    """Yardstick A for the pages of one layer and mode, by corpus path."""
    run = _RunContext(layer, mode, truths, limits)
    oracle_saved = {path for path, page in oracle.items() if page.markdown}
    candidate_saved = {path for path, page in candidate.items() if page.markdown}
    comparison = Comparison(layer, mode, len(oracle_saved), len(candidate_saved))
    page_set = _page_set_differences(oracle, candidate, run)
    for kind, limit in (
        (Kind.PAGE_LOST, limits.pages_lost_max),
        (Kind.PAGE_EXTRA, limits.pages_extra_unreadable_max),
    ):
        comparison.differences.extend(_within([d for d in page_set if d.kind is kind], limit))
    for path in sorted(oracle_saved & candidate_saved):
        truth = truths.get(path)
        # A capture that failed says nothing about what is visible, so it is no truth at all.
        page = _PageContext(path, layer, mode, truth if truth and truth.usable() else None, limits)
        measure, differences = _compare_page(
            oracle[path].markdown or "", candidate[path].markdown or "", page
        )
        comparison.measures.append(measure)
        comparison.differences.extend(differences)
    return comparison


def against_truth(
    pages: dict[str, Page], truths: dict[str, Truth], layer: Layer, mode: Mode, limits: TruthLimits
) -> list[Difference]:
    """Yardstick B as a verdict, for a side with no oracle beside it."""
    found: list[Difference] = []
    for path, truth in sorted(truths.items()):
        if not truth.scorable():
            continue
        page = pages.get(path)
        if page is None or not page.markdown:
            reason = reason_feature(page.error if page else "not returned")
            found.append(Difference(Kind.PAGE_LOST, mode, layer, reason, path))
            continue
        result = score(markdown_tokens(page.markdown), truth)
        if result.recall < limits.visible_recall_min:
            for context, words in sorted(_by_context(result.missed, None).items()):
                found.append(
                    Difference(
                        Kind.VISIBLE_MISSED,
                        mode,
                        layer,
                        context,
                        path,
                        sum(words.values()),
                        _sample(words),
                    )
                )
        extra = sum(result.unexplained.values())
        if extra > limits.unexplained_words_per_page_max:
            found.append(
                Difference(
                    Kind.TEXT_ADDED,
                    mode,
                    layer,
                    NO_CONTEXT,
                    path,
                    extra,
                    _sample(result.unexplained),
                )
            )
    return found


def attribute_to_lowest_layer(
    by_layer: dict[Layer, list[Difference]], top: Layer
) -> list[Difference]:
    """The differences of *top*, each moved to the lowest layer that shows the same finding."""
    order = [layer for layer in Layer if layer in by_layer]
    attributed: list[Difference] = []
    for difference in by_layer[top]:
        lowest, seen = next(
            (layer, low)
            for layer in order
            for low in ([difference] if layer is top else by_layer[layer])
            if difference.same_finding(low)
        )
        attributed.append(
            Difference(
                difference.kind,
                difference.mode,
                lowest,
                seen.feature,
                difference.page,
                difference.amount,
                difference.detail,
                difference.counts,
            )
        )
    return attributed
