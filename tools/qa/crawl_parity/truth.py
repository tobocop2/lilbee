"""Ground truth: the text a reader sees on a replayed page, taken from a real browser."""

from __future__ import annotations

import contextlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict

from playwright.sync_api import BrowserContext, Error, Page, sync_playwright
from tools.qa.crawl_parity.tokens import Tokens, tokenize

NETWORK_IDLE_TIMEOUT_MS = 5000
NAVIGATION_TIMEOUT_MS = 20000
SETTLE_MS = 1000
STATUS_OK = 200
HIDDEN_PREFIX = "hidden:"
ATTRIBUTE_CONTEXT = "attribute"
NO_CONTEXT = "not-in-page"
# Raise this when the capture changes, so that no run reads a capture an older one saved.
CAPTURE_VERSION = "3"
EVALUATE_ATTEMPTS = 3
# What the browser says a document is, after its own sniffing, when a reader gets text.
READABLE_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain", "text/markdown"})

# Returns the rendered text, every text node with its two nearest ancestors and whether it is
# rendered, and the text of attributes a reader can meet (alternative text, titles, labels).
_CAPTURE = """
() => {
  const skipped = new Set(['SCRIPT', 'STYLE', 'TEMPLATE']);
  const nodes = [];
  const attributes = [];
  const visit = (root) => {
    for (const node of root.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.nodeValue;
        const parent = node.parentElement;
        if (!text.trim() || !parent) continue;
        const names = [];
        for (let el = parent; el && el !== document.documentElement && names.length < 2;
             el = el.parentElement) {
          if (el !== document.body) names.unshift(el.localName);
        }
        const shown = parent.checkVisibility(
          {visibilityProperty: true, contentVisibilityAuto: true});
        nodes.push([names.join('>') || 'body', shown, text]);
      } else if (node.nodeType === Node.ELEMENT_NODE && !skipped.has(node.tagName)) {
        for (const name of ['alt', 'title', 'aria-label', 'placeholder', 'value', 'label']) {
          const value = node.getAttribute(name);
          if (value) attributes.push(value);
        }
        visit(node);
        if (node.shadowRoot) visit(node.shadowRoot);
      }
    }
  };
  visit(document.documentElement);
  return {text: document.body ? document.body.innerText : '', nodes, attributes};
}
"""


class Capture(TypedDict):
    """What the capture script returns for one page."""

    text: str
    nodes: list[list[Any]]
    attributes: list[str]


class BrowserName(StrEnum):
    """The browser engines ground truth can be taken with."""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


@dataclass(frozen=True)
class Truth:
    """What a reader sees on one page, and where each word sits in the page."""

    path: str
    status: int
    renders_text: bool
    visible: Tokens = field(default_factory=lambda: Tokens((), ()))
    hidden_words: Counter[str] = field(default_factory=Counter)
    attribute_words: Counter[str] = field(default_factory=Counter)
    contexts: dict[str, str] = field(default_factory=dict)
    sentences: tuple[str, ...] = ()
    error: str = ""

    def usable(self) -> bool:
        """Whether the browser rendered this page as HTML, whatever its status."""
        return self.renders_text and not self.error

    def scorable(self) -> bool:
        """Whether a crawler is expected to save this page: an HTML answer with visible words."""
        return self.status == STATUS_OK and self.renders_text and bool(self.visible.words)

    def context_of(self, word: str) -> str:
        """Where *word* sits in the page."""
        return self.contexts.get(word, NO_CONTEXT)


def _contexts(nodes: list[list[Any]], attributes: Counter[str]) -> dict[str, str]:
    """For each word, the context that holds it most often; a rendered context wins a tie."""
    seen: dict[str, Counter[str]] = {}
    for context, shown, text in nodes:
        label = str(context) if shown else HIDDEN_PREFIX + str(context)
        for word in tokenize(str(text)).words:
            seen.setdefault(word, Counter())[label] += 1
    for word in attributes:
        seen.setdefault(word, Counter())[ATTRIBUTE_CONTEXT] += 1
    return {
        word: max(counts, key=lambda label: (counts[label], not label.startswith(HIDDEN_PREFIX)))
        for word, counts in seen.items()
    }


def _sentences(text: str) -> tuple[str, ...]:
    """The rendered lines of a page, each a candidate question for the retrieval yardstick."""
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def truth_from_capture(path: str, status: int, renders_text: bool, capture: Capture) -> Truth:
    """The truth record for one page from what the browser returned."""
    nodes = capture["nodes"]
    hidden: Counter[str] = Counter()
    for _context, shown, text in nodes:
        if not shown:
            hidden.update(tokenize(str(text)).words)
    attribute_words: Counter[str] = Counter()
    for value in capture["attributes"]:
        attribute_words.update(tokenize(str(value)).words)
    text = str(capture["text"])
    return Truth(
        path=path,
        status=status,
        renders_text=renders_text,
        visible=tokenize(text),
        hidden_words=hidden,
        attribute_words=attribute_words,
        contexts=_contexts(nodes, attribute_words),
        sentences=_sentences(text),
    )


@dataclass(frozen=True)
class Raw:
    """What the browser returned for one page, in a form that is saved as JSON."""

    status: int
    renders_text: bool
    data: Capture | None = None
    error: str = ""


@dataclass(frozen=True)
class Target:
    """One page to capture: its replay URL and a key that changes when its content does."""

    url: str
    key: str


class TruthCache:
    """Captures kept between runs in one JSON file, by target key."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, dict[str, Any]] = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        )

    def get(self, key: str) -> Raw | None:
        """The capture saved under *key*."""
        entry = self._entries.get(key)
        return Raw(**entry) if entry else None

    def put(self, key: str, raw: Raw) -> None:
        """Keep *raw* under *key*."""
        self._entries[key] = asdict(raw)

    def save(self) -> None:
        """Write every capture to the file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries), encoding="utf-8")


def _evaluate_when_still(page: Page, script: str) -> Any:
    """Evaluate *script*; a page that is navigating is waited for and asked again."""
    for _attempt in range(EVALUATE_ATTEMPTS - 1):
        try:
            return page.evaluate(script)
        except Error:
            page.wait_for_load_state("load", timeout=NAVIGATION_TIMEOUT_MS)
            page.wait_for_timeout(SETTLE_MS)
    return page.evaluate(script)


def _capture_raw(context: BrowserContext, url: str) -> Raw:
    page = context.new_page()
    try:
        response = page.goto(url, wait_until="load", timeout=NAVIGATION_TIMEOUT_MS)
        if response is None:
            return Raw(0, False, error="no response")
        # A page that never goes idle is captured as it stands after the settle time.
        with contextlib.suppress(Error):
            page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
        page.wait_for_timeout(SETTLE_MS)
        if _evaluate_when_still(page, "document.contentType") not in READABLE_TYPES:
            return Raw(response.status, False)
        return Raw(response.status, True, _evaluate_when_still(page, _CAPTURE))
    except Error as failure:
        return Raw(0, False, error=str(failure).splitlines()[0])
    finally:
        page.close()


def truth_of(path: str, raw: Raw) -> Truth:
    """The truth record of one capture."""
    if raw.data is None:
        return Truth(path, raw.status, raw.renders_text, error=raw.error)
    return truth_from_capture(path, raw.status, raw.renders_text, raw.data)


def capture(
    targets: dict[str, Target], browser_name: BrowserName, cache: TruthCache
) -> dict[str, Truth]:
    """The truth for each corpus path in *targets*; one browser serves every page not cached."""
    raws = {path: cache.get(target.key) for path, target in targets.items()}
    missing = [path for path, raw in raws.items() if raw is None]
    if missing:
        with sync_playwright() as playwright:
            engines = {
                BrowserName.CHROMIUM: playwright.chromium,
                BrowserName.FIREFOX: playwright.firefox,
                BrowserName.WEBKIT: playwright.webkit,
            }
            browser = engines[browser_name].launch()
            # One context for every page: a cookie an earlier page set is sent to a later one.
            context = browser.new_context()
            try:
                for path in missing:
                    raw = _capture_raw(context, targets[path].url)
                    raws[path] = raw
                    cache.put(targets[path].key, raw)
            finally:
                browser.close()
        cache.save()
    return {path: truth_of(path, raw) for path, raw in raws.items() if raw is not None}
