"""Properties of the file name a crawled page is saved under, and of a library re-crawled after
the name rule changed.

Each property holds for every address a generator writes:

- two different addresses never share a saved path, compared the way Windows and macOS compare
  names (letter case and Unicode normalization ignored);
- every spelling of one address shares one saved path;
- every segment of a saved path is a legal Windows name that prints, and the path fits 200 bytes;
- the label of a saved page shows no percent escape the address did not hold;
- a letter-and-digit segment of the address reads as written in the saved path.

The library properties crawl a library saved by the earlier name rule again and sync it into a
real store: every page keeps exactly one file and one index entry, whether or not its text
changed, no folder is left empty, and saving a page never reads the whole crawl metadata.

Which spellings name one address follows the key the crawler (crawlberg 1.11.1) gives a page: the
WHATWG URL standard as the Rust url crate 2.5.8 parses it, then the escape normalization of
RFC 3986 section 6.2.2, without the fragment, query parameters sorted by name. That crate keeps
a raw ``^`` in a path, where newer parsers escape it.

So an unreserved character and its escape are one address; a space, a control, a
non-ASCII character, and ``" < > ` { }`` are one address with their escape, because the parser
escapes them; any other printable ASCII character and its escape are two addresses. A host is one
address in any letter case, in Unicode and in punycode, and with or without its default port.
Raw tab, newline and backslash are left out because the parser removes them or reads a backslash
as ``/``. Dot segments are left out because the parser resolves them.
"""

from __future__ import annotations

import logging
import re
import string
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from lilbee.app.services import set_services
from lilbee.core.config import cfg
from lilbee.crawler import (
    CrawlMeta,
    CrawlResult,
    content_hash,
    crawl_and_save,
    load_crawl_metadata,
    save_crawl_metadata,
    url_to_filename,
)
from lilbee.crawler.migrate import move_saved_pages
from lilbee.data.ingest import sync
from lilbee.data.store import Store
from lilbee.retrieval.query.formatting import source_label
from tests.conftest import make_mock_services

_LAST_SEGMENT = "page"
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
# Characters that only ever appear escaped: written raw they are structure, start an escape,
# or the parser removes them.
_ESCAPED_ONLY = "/?#%\\\t\n\r"
# Printable ASCII characters the parser escapes, so the raw character and its escape are one.
_PARSER_ESCAPED = frozenset('"<>`{}')
# Printable ASCII characters whose escape names a different address than the character itself.
_KEPT = frozenset(set(string.punctuation) - _UNRESERVED - set(_ESCAPED_ONLY) - _PARSER_ESCAPED)
# The same two sets for a query, which the parser escapes differently.
_QUERY_PARSER_ESCAPED = frozenset("\"<>'")
_QUERY_KEPT = frozenset(
    set(string.punctuation) - _UNRESERVED - set(_ESCAPED_ONLY) - _QUERY_PARSER_ESCAPED
)
_WINDOWS_RESERVED_CHARACTERS = frozenset('<>:"/\\|?*')
_CONTROL_CHARACTERS = frozenset([*map(chr, range(0x20)), "\x7f"])
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in [*string.digits, "¹", "²", "³"]}
)
_MAX_SEGMENT_BYTES = 255
_MAX_PATH_BYTES = 200
_DOT_SEGMENTS = frozenset({".", ".."})
_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
# A private-use character no generated address holds, marking a kept escape in a caseless key.
_ESCAPED_KEPT_MARK = chr(0xE000)

# Each host and the spellings that name it.
_HOSTS = {
    "example": ("example.com", "EXAMPLE.com", "Example.COM:443"),
    "idn": ("bücher.example", "xn--bcher-kva.example", "BÜCHER.example"),
    "device": ("con.example", "CON.example"),
    "ipv6": ("[::1]", "[0:0::1]", "[0000::0001]:443"),
}

_COMBINING_ACUTE = unicodedata.lookup("COMBINING ACUTE ACCENT")
# Characters that make near-collisions likely: case pairs, a composed and a decomposed letter,
# the characters a name rule replaces or strips, controls, and characters that do not print.
_TRICKY = (
    "a",
    "A",
    "b",
    "_",
    ".",
    " ",
    "-",
    "~",
    ";",
    "=",
    "é",
    "É",
    "e",
    _COMBINING_ACUTE,
    "ß",
    "ü",
    "K",
    unicodedata.lookup("KELVIN SIGN"),
    "漢",
    "\x00",
    "\x01",
    "\x1b",
    "\x7f",
    "\x85",
    "\xa0",
    unicodedata.lookup("RIGHT-TO-LEFT OVERRIDE"),
    unicodedata.lookup("LINE SEPARATOR"),
    unicodedata.lookup("ZERO WIDTH NO-BREAK SPACE"),
    *sorted(_KEPT),
    *sorted(_PARSER_ESCAPED),
    *_ESCAPED_ONLY,
)
_DEVICE_SEGMENTS = ("con", "CON", "nul.txt", "Com1", "lpt9.md", "aux", "prn ", "COM¹", "conin$")


@dataclass(frozen=True)
class _Token:
    """One character of an address and whether it is an escape that names a different address."""

    char: str
    kept_escape: bool

    @property
    def identity(self) -> tuple[str, bool]:
        """What two spellings of one address share."""
        return (self.char, self.kept_escape)


def _token(char: str, escaped: bool) -> _Token:
    """The token for *char*, written escaped when *escaped* (always, for the escaped-only set)."""
    if char in _ESCAPED_ONLY:
        return _Token(char, kept_escape=True)
    return _Token(char, kept_escape=escaped and char in _KEPT)


def _query_token(char: str, escaped: bool) -> _Token:
    """The token for *char* in a query, where the parser escapes another set."""
    if char in _ESCAPED_ONLY:
        return _Token(char, kept_escape=True)
    return _Token(char, kept_escape=escaped and char in _QUERY_KEPT)


_chars = st.one_of(
    st.sampled_from(_TRICKY),
    st.sampled_from(_TRICKY),
    st.characters(codec="utf-8", exclude_characters=_ESCAPED_ONLY),
)
_tokens = st.builds(_token, _chars, st.booleans())
_segment = st.one_of(
    st.just([]),
    st.lists(_tokens, min_size=1, max_size=5),
    st.lists(_tokens, min_size=1, max_size=40),
    st.sampled_from(_DEVICE_SEGMENTS).map(lambda name: [_token(char, False) for char in name]),
).filter(lambda tokens: "".join(token.char for token in tokens) not in _DOT_SEGMENTS)
_last = st.one_of(
    st.just(tuple(_token(char, False) for char in _LAST_SEGMENT)),
    st.just(()),
    _segment.map(tuple),
)


@dataclass(frozen=True)
class _Address:
    """An address: a host, segments, a last segment (empty for a trailing slash), a query."""

    host: str
    segments: tuple[tuple[_Token, ...], ...]
    last: tuple[_Token, ...]
    query: tuple[_Token, ...] | None

    @property
    def identity(self) -> object:
        """Equal for two spellings of one address, different for two addresses."""
        return (
            self.host,
            tuple(tuple(token.identity for token in segment) for segment in self.segments),
            tuple(token.identity for token in self.last),
            None if self.query is None else tuple(token.identity for token in self.query),
        )


_addresses = st.builds(
    _Address,
    st.sampled_from(sorted(_HOSTS)),
    st.lists(_segment.map(tuple), min_size=1, max_size=3).map(tuple),
    _last,
    st.one_of(
        st.none(),
        st.lists(st.builds(_query_token, _chars, st.booleans()), min_size=1, max_size=8).map(tuple),
    ),
)


def _near(token: _Token) -> list[list[_Token]]:
    """Token runs a name rule could confuse with *token*: nothing, or a similar character."""
    return [
        [],
        [_token(token.char, not token.kept_escape)],
        [_token("_", False)],
        [_token("-", False)],
        [_token(token.char.swapcase(), False)],
        [token, _token(".", False)],
        [token, _token(" ", False)],
        [_token(token.char, False), _token(_COMBINING_ACUTE, False)],
    ]


def _edited(address: _Address, data: st.DataObject) -> _Address:
    """*address* with one token changed, so the two addresses are near each other."""
    parts = [list(segment) for segment in address.segments] + [list(address.last)]
    filled = [part for part in parts if part]
    assume(filled)
    part = data.draw(st.sampled_from(filled))
    index = data.draw(st.integers(0, len(part) - 1))
    part[index : index + 1] = data.draw(st.sampled_from(_near(part[index])))
    assume(all(_plain(tuple(p)) not in _DOT_SEGMENTS for p in parts))
    return _Address(address.host, tuple(map(tuple, parts[:-1])), tuple(parts[-1]), address.query)


def _plain(tokens: tuple[_Token, ...]) -> str:
    """The characters of *tokens*, with a kept escape marked so it differs from its character."""
    return "".join(_ESCAPED_KEPT_MARK + t.char if t.kept_escape else t.char for t in tokens)


def _caseless(name: str) -> str:
    """*name* as a case-insensitive, normalization-insensitive file system compares it."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def _escape(char: str, upper: bool) -> str:
    """*char* as percent escapes of its UTF-8 bytes."""
    escape = "".join(f"%{byte:02X}" for byte in char.encode("utf-8"))
    return escape if upper else escape.lower()


def _spell(tokens: tuple[_Token, ...], data: st.DataObject, kept: frozenset[str] = _KEPT) -> str:
    """One spelling of *tokens*: each character raw or escaped where that keeps the address."""
    spelled = []
    for token in tokens:
        if token.kept_escape:
            spelled.append(_escape(token.char, data.draw(st.booleans())))
        elif token.char in kept:
            spelled.append(token.char)
        elif data.draw(st.booleans()):
            spelled.append(_escape(token.char, data.draw(st.booleans())))
        else:
            spelled.append(token.char)
    return "".join(spelled)


def _url(address: _Address, data: st.DataObject) -> str:
    """One spelling of *address* as a URL, with or without a fragment."""
    host = data.draw(st.sampled_from(_HOSTS[address.host]))
    path = "/".join(_spell(part, data) for part in (*address.segments, address.last))
    url = f"https://{host}/{path}"
    if address.query is not None:
        url = f"{url}?{_spell(address.query, data, _QUERY_KEPT)}"
    # The parser strips a raw space or C0 control from the end of the whole address.
    assume(not url.endswith((" ", *map(chr, range(0x20)))))
    if data.draw(st.booleans()):
        url = f"{url}#{data.draw(st.sampled_from(['a', 'b', 'top', 'x%20y']))}"
    return url


def _compared(saved: str) -> str:
    """*saved* as a disk resolves it: empty segments collapse, no case, one normalization, no
    trailing dots or spaces."""
    return "/".join(_caseless(segment.rstrip(". ")) for segment in saved.split("/") if segment)


def _illegal_parts(segment: str) -> list[str]:
    """What makes *segment* an illegal Windows name or unsafe to print; empty when it is neither."""
    problems = [repr(char) for char in segment if char in _CONTROL_CHARACTERS]
    problems += [repr(char) for char in segment if char in _WINDOWS_RESERVED_CHARACTERS]
    problems += [repr(char) for char in segment if not (char.isprintable() or char == " ")]
    if segment in _DOT_SEGMENTS or not segment:
        problems.append("empty or dot segment")
    if segment.endswith((" ", ".")):
        problems.append("trailing space or dot")
    if segment.split(".")[0].rstrip(" ").upper() in _DEVICE_NAMES:
        problems.append("device name")
    if len(segment.encode("utf-8")) > _MAX_SEGMENT_BYTES:
        problems.append("longer than 255 bytes")
    return problems


_PROPERTY_SETTINGS = settings(max_examples=1000, deadline=None, derandomize=True, database=None)


class TestSavedNames:
    @_PROPERTY_SETTINGS
    @given(one=_addresses, data=st.data())
    def test_two_addresses_never_share_a_saved_path(self, one: _Address, data: st.DataObject):
        other = _edited(one, data)
        assume(one.identity != other.identity)
        one_url, other_url = _url(one, data), _url(other, data)
        assert _compared(url_to_filename(one_url)) != _compared(url_to_filename(other_url)), (
            one_url,
            other_url,
        )

    @_PROPERTY_SETTINGS
    @given(address=_addresses, data=st.data())
    def test_two_spellings_of_one_address_share_a_saved_path(
        self, address: _Address, data: st.DataObject
    ):
        one_url, other_url = _url(address, data), _url(address, data)
        assert url_to_filename(one_url) == url_to_filename(other_url), (one_url, other_url)

    @_PROPERTY_SETTINGS
    @given(address=_addresses, data=st.data())
    def test_every_segment_is_a_legal_windows_name_that_prints(
        self, address: _Address, data: st.DataObject
    ):
        url = _url(address, data)
        saved = url_to_filename(url)
        assert {
            seg: _illegal_parts(seg) for seg in saved.split("/") if _illegal_parts(seg)
        } == {}, url
        assert len(saved.encode("utf-8")) <= _MAX_PATH_BYTES, url

    @_PROPERTY_SETTINGS
    @given(address=_addresses, data=st.data())
    def test_a_label_shows_no_escape_the_address_did_not_hold(
        self, address: _Address, data: st.DataObject
    ):
        tokens = [t for part in (*address.segments, address.last) for t in part]
        assume(all(not t.kept_escape and t.char.isprintable() for t in tokens))
        url = _url(address, data)
        label = source_label(f"_web/{url_to_filename(url)}")
        assert not _ESCAPE.search(label), (url, label)

    @_PROPERTY_SETTINGS
    @given(
        word=st.text(
            st.characters(categories=("Ll", "Lu", "Lo", "Nd")), min_size=1, max_size=20
        ).map(lambda text: unicodedata.normalize("NFC", text))
    )
    def test_a_letter_and_digit_segment_reads_as_written(self, word: str):
        assume(word.upper() not in _DEVICE_NAMES)
        saved = url_to_filename(f"https://example.com/t/{word}")
        assert word in saved, (word, saved)


# Two addresses the crawler treats as two pages: each pair must keep two saved paths.
_DISTINCT = [
    pytest.param("https://x.test/doc;v=1", "https://x.test/doc;v=2", id="last-segment-params"),
    pytest.param("https://x.test/a;p/doc", "https://x.test/a;q/doc", id="segment-params"),
    pytest.param("https://x.test/t//x", "https://x.test/t/x", id="empty-segment"),
    pytest.param("https://x.test/t/x/", "https://x.test/t/x", id="trailing-slash"),
    pytest.param("https://x.test/t/CAFÉ/page", "https://x.test/t/café/page", id="non-ascii-case"),
    pytest.param(
        "https://x.test/t/café/page", f"https://x.test/t/cafe{_COMBINING_ACUTE}/page", id="nfc-nfd"
    ),
    pytest.param("https://x.test/t/Page", "https://x.test/t/page", id="ascii-case"),
    pytest.param("https://x.test/t/a.html", "https://x.test/t/a.htm", id="extension"),
    pytest.param("https://x.test/t/a", "https://x.test/t/a.md", id="extension-added"),
    pytest.param("https://x.test/t/", "https://x.test/t/index", id="index-name"),
    pytest.param("https://x.test/t/a:b", "https://x.test/t/a_b", id="reserved-and-replacement"),
    pytest.param("https://x.test/t/a%3Ab", "https://x.test/t/a:b", id="kept-escape"),
    pytest.param("https://x.test/a^b", "https://x.test/a%5Eb", id="caret-crawlberg-keeps"),
    pytest.param("http://x.test/t", "https://x.test/t", id="scheme"),
    pytest.param("https://x.test:8443/t", "https://x.test/t", id="port"),
]

_LONG = "https://x.test/" + "segment-of-a-long-address/" * 12 + "page"
# Two spellings of one page: each pair must share one saved path.
_SAME = [
    pytest.param(_LONG + "#a", _LONG + "#b", id="long-fragment"),
    pytest.param("https://x.test/t#a", "https://x.test/t", id="short-fragment"),
    pytest.param("https://bücher.test/t", "https://xn--bcher-kva.test/t", id="unicode-punycode"),
    pytest.param("https://X.Test/t", "https://x.test/t", id="host-case"),
    pytest.param("https://x.test:443/t", "https://x.test/t", id="default-port"),
    pytest.param("https://[0:0::1]/t", "https://[::1]/t", id="ipv6-spelling"),
    pytest.param("https://x.test/a%3ab", "https://x.test/a%3Ab", id="escape-case"),
    pytest.param("https://x.test/caf%C3%A9", "https://x.test/café", id="non-ascii-escape"),
    pytest.param("https://x.test/a%7Eb", "https://x.test/a~b", id="unreserved-escape"),
    pytest.param("https://x.test/a%22b", 'https://x.test/a"b', id="parser-escaped"),
    pytest.param("https://x.test/t?b=2&a=1", "https://x.test/t?a=1&b=2", id="query-order"),
]


@pytest.mark.parametrize(("one", "other"), _DISTINCT)
def test_two_pages_keep_two_saved_paths(one: str, other: str):
    assert _compared(url_to_filename(one)) != _compared(url_to_filename(other))


@pytest.mark.parametrize(("one", "other"), _SAME)
def test_two_spellings_of_one_page_share_a_saved_path(one: str, other: str):
    assert url_to_filename(one) == url_to_filename(other)


def test_a_long_address_is_cut_to_a_path_every_file_system_accepts():
    url = "https://example.com/" + "漢" * 120
    saved = url_to_filename(url)
    assert len(saved.encode("utf-8")) <= _MAX_PATH_BYTES
    assert saved.endswith(".md")
    assert url_to_filename("https://example.com/" + "%E6%BC%A2" * 120) == saved


# A library saved by the earlier name rule. Every address below got the same file name at main
# (90573e72c) and at ac2dcae13, measured by running each commit's url_to_filename. Written in
# this order, a later page overwrites an earlier page that shares its file.
_SITE = "http://site.test/s/"
_EARLIER_LIBRARY = [
    ("a:b", "site.test/s/a_b/index.md"),
    ("a_b", "site.test/s/a_b/index.md"),
    ("con", "site.test/s/con/index.md"),
    ("x..y", "site.test/s/x_y/index.md"),
    ("x_y", "site.test/s/x_y/index.md"),
    ("caf%C3%A9", "site.test/s/caf%C3%A9/index.md"),
    ("CAF%C3%89", "site.test/s/CAF%C3%89/index.md"),
    ("cafe%CC%81", "site.test/s/cafe%CC%81/index.md"),
    ("sp%20ace", "site.test/s/sp%20ace/index.md"),
    ("plain", "site.test/s/plain/index.md"),
    ("tail%20/x", "site.test/s/tail%20/x/index.md"),
    ("q%3Ar", "site.test/s/q%3Ar/index.md"),
    ("Report.v2.final.", "site.test/s/Report.v2.final./index.md"),
    ("dir.d/page", "site.test/s/dir.d/page/index.md"),
    ("doc;v=1", "site.test/s/doc/index.md"),
    ("doc;v=2", "site.test/s/doc/index.md"),
    ("t//x", "site.test/s/t//x/index.md"),
    ("t/x", "site.test/s/t/x/index.md"),
    ("Special:Random", "site.test/s/Special_Random/index.md"),
]
_FILLER = " ".join(f"Sentence {i} is ordinary text about rivers and maps." for i in range(30))


def _text(path: str, version: str) -> str:
    """The markdown of page *path* in *version*: a marker line, then filler."""
    return f"# Page\n\nBody of {path} in version {version}.\n\n{_FILLER}\n"


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    """A data root with a real store, a stub embedder, and a crawl that needs no browser."""
    import lilbee.data.ingest.pipeline as pipeline
    from lilbee.runtime.progress import CRAWL_TOTAL_UNKNOWN

    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.documents_dir.mkdir()
    cfg.data_dir.mkdir()
    cfg.crawl_sync_interval = 0
    monkeypatch.setattr(pipeline, "_max_concurrent", lambda: 1)
    monkeypatch.setattr("lilbee.crawler.crawler_available", lambda: True)
    monkeypatch.setattr("lilbee.crawler.crawlberg_fetcher.crawler_available", lambda: True)
    monkeypatch.setattr("lilbee.crawler.bootstrap.chromium_installed", lambda: True)
    monkeypatch.setattr(
        "lilbee.crawler.bootstrap.headless_shell_executable", lambda: tmp_path / "shell"
    )
    monkeypatch.setattr(
        "lilbee.crawler.sitemap._count_sitemap_urls", lambda *a, **kw: CRAWL_TOTAL_UNKNOWN
    )
    monkeypatch.setattr(
        "lilbee.crawler.url_filter.socket.getaddrinfo",
        lambda host, port, *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    store = Store(cfg)
    embedder = MagicMock()
    embedder.embedding_available.return_value = True
    embedder.embed_batch.side_effect = lambda texts, **kw: [
        np.full(cfg.embedding_dim, 0.1, dtype=np.float32) for _ in texts
    ]
    embedder.truncated_total = 0
    set_services(
        make_mock_services(
            store=store,
            embedder=embedder,
            worker_pool=MagicMock(),
            pool_runtime=MagicMock(),
            pool_health_ticker=MagicMock(),
        )
    )
    yield store
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _write_earlier_library(
    site: str = _SITE,
    pages: list[tuple[str, str]] = _EARLIER_LIBRARY,
    version: str = "1",
    crawled_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    """Save each of *pages* of *site* under its earlier name, as main does, with its entry."""
    web = cfg.documents_dir / "_web"
    meta = load_crawl_metadata()
    for path, earlier in pages:
        text = _text(path, version)
        (web / earlier).parent.mkdir(parents=True, exist_ok=True)
        (web / earlier).write_text(text, encoding="utf-8")
        meta[site + path] = CrawlMeta(earlier, content_hash(text), crawled_at)
    save_crawl_metadata(meta)


async def _crawl(
    monkeypatch: pytest.MonkeyPatch, site: str, paths: list[str], version: str
) -> None:
    """Crawl *site*, the crawler reporting each of *paths* with the text of *version*."""
    results = [CrawlResult(url=site + path, markdown=_text(path, version)) for path in paths]

    async def crawl(url: str, **kwargs: object) -> list[CrawlResult]:
        for result in results:
            await kwargs["on_result"](result)  # type: ignore[operator]
        return results

    monkeypatch.setattr("lilbee.crawler.runner.crawl_recursive", crawl)
    await crawl_and_save(site, depth=1)


async def _recrawl(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    """Crawl the site again, the crawler reporting every page with the text of *version*."""
    await _crawl(monkeypatch, _SITE, [path for path, _ in _EARLIER_LIBRARY], version)


def _library_state(store: Store, versions: dict[str, str]) -> dict[str, object]:
    """Files, index entries and crawl metadata, against pages *versions* (path -> text version)."""
    web = cfg.documents_dir / "_web"
    files = sorted(
        p.relative_to(cfg.documents_dir).as_posix() for p in web.rglob("*.md") if p.is_file()
    )
    sources = sorted(s["filename"] for s in store.get_sources())
    texts = {f: (cfg.documents_dir / f).read_text(encoding="utf-8") for f in files}
    lost = [
        path
        for path, version in versions.items()
        if sum(f"Body of {path} in version {version}." in t for t in texts.values()) != 1
    ]
    empty = sorted(
        d.relative_to(web).as_posix() for d in web.rglob("*") if d.is_dir() and not any(d.iterdir())
    )
    recorded = {f"_web/{entry.file}" for entry in load_crawl_metadata().values()}
    return {
        "files": len(files),
        "index entries": len(sources),
        "sources without a file": sorted(set(sources) - set(files)),
        "files without a source": sorted(set(files) - set(sources)),
        "files without a crawl entry": sorted(set(files) - recorded),
        "pages not in exactly one file": lost,
        "empty folders": empty,
    }


def _clean(pages: int) -> dict[str, object]:
    """The library state of *pages* pages, each in one file with one index and one crawl entry."""
    return {
        "files": pages,
        "index entries": pages,
        "sources without a file": [],
        "files without a source": [],
        "files without a crawl entry": [],
        "pages not in exactly one file": [],
        "empty folders": [],
    }


class TestRecrawledLibrary:
    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("version", ["1", "2"], ids=["same-text", "changed-text"])
    async def test_every_page_keeps_one_file_and_one_index_entry(
        self, library: Store, monkeypatch: pytest.MonkeyPatch, version: str
    ):
        _write_earlier_library()
        await sync(quiet=True)
        await _recrawl(monkeypatch, version)
        await sync(quiet=True)
        versions = {path: version for path, _ in _EARLIER_LIBRARY}
        assert _library_state(library, versions) == _clean(len(_EARLIER_LIBRARY))


# main saved /s/a.html as the file a.md, and /s/a.md%3F/z under the folder a.md%3F, whose slug
# is a.md: a folder of the new rule with the name of a file of the old one.
_CLASH_SITE = "http://clash.test/s/"
_CLASH_LIBRARY = [
    ("a.md%3F/z", "clash.test/s/a.md%3F/z/index.md"),
    ("a.html", "clash.test/s/a.md"),
    ("b", "clash.test/s/b/index.md"),
]


@pytest.mark.timeout(300)
async def test_a_folder_never_takes_the_name_of_an_earlier_saved_file(
    library: Store, monkeypatch: pytest.MonkeyPatch
):
    _write_earlier_library(_CLASH_SITE, _CLASH_LIBRARY)
    await sync(quiet=True)
    for _ in range(2):
        await _crawl(monkeypatch, _CLASH_SITE, [path for path, _ in _CLASH_LIBRARY], "1")
        await sync(quiet=True)
    versions = {path: "1" for path, _ in _CLASH_LIBRARY}
    assert _library_state(library, versions) == _clean(len(_CLASH_LIBRARY))


# Pages main gives a file of its own, with the address main keys them by. %7Euser keys
# another spelling of its canonical address.
_MAIN_PAGES = [
    ("plain", "site.test/s/plain/index.md"),
    ("caf%C3%A9", "site.test/s/caf%C3%A9/index.md"),
    ("dir.d/page", "site.test/s/dir.d/page/index.md"),
    ("Special:Random", "site.test/s/Special_Random/index.md"),
    ("%7Euser", "site.test/s/%7Euser/index.md"),
]
_OTHER_SITE = "https://other.test/"
_OTHER_PAGES = ["guide", "intro"]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("next_site", ["same-site", "other-site"])
async def test_a_page_main_saved_after_the_move_keeps_its_newer_text(
    library: Store, monkeypatch: pytest.MonkeyPatch, next_site: str
):
    main_paths = [path for path, _ in _MAIN_PAGES]
    await _crawl(monkeypatch, _SITE, main_paths, "1")
    await _crawl(monkeypatch, _OTHER_SITE, _OTHER_PAGES, "1")
    await sync(quiet=True)
    _write_earlier_library(_SITE, _MAIN_PAGES, "2", datetime.now(UTC).isoformat())
    await sync(quiet=True)
    if next_site == "same-site":
        await _crawl(monkeypatch, _SITE, main_paths, "2")
    else:
        await _crawl(monkeypatch, _OTHER_SITE, _OTHER_PAGES, "1")
    await sync(quiet=True)
    versions = {path: "2" for path in main_paths} | {path: "1" for path in _OTHER_PAGES}
    assert _library_state(library, versions) == _clean(len(versions))


@pytest.mark.timeout(300)
async def test_a_move_cut_short_before_its_record_keeps_every_entry(
    library: Store, monkeypatch: pytest.MonkeyPatch
):
    _write_earlier_library(_SITE, _MAIN_PAGES)
    await sync(quiet=True)

    def cut(meta: dict[str, CrawlMeta]) -> None:
        raise OSError(28, "No space left on device")

    with monkeypatch.context() as patched:
        patched.setattr("lilbee.crawler.save.save_crawl_metadata", cut)
        with pytest.raises(OSError, match="No space left"):
            move_saved_pages(library)
    await _crawl(monkeypatch, _OTHER_SITE, _OTHER_PAGES, "1")
    await sync(quiet=True)
    versions = {path: "1" for path, _ in _MAIN_PAGES} | {path: "1" for path in _OTHER_PAGES}
    assert _library_state(library, versions) == _clean(len(versions))


@pytest.mark.timeout(300)
async def test_a_page_written_after_its_last_sync_is_indexed_again_after_the_move(
    library: Store,
):
    _write_earlier_library(_SITE, _MAIN_PAGES)
    await sync(quiet=True)
    path, earlier = _MAIN_PAGES[0]
    (cfg.documents_dir / "_web" / earlier).write_text(_text(path, "2"), encoding="utf-8")
    move_saved_pages(library)
    await sync(quiet=True)
    chunks = library.get_chunks_by_source(f"_web/{url_to_filename(_SITE + path)}")
    indexed = " ".join(chunk.chunk for chunk in chunks)
    assert f"Body of {path} in version 2." in indexed
    assert f"Body of {path} in version 1." not in indexed


# A page of an earlier library whose address spells the saved name of a sibling page: main saved
# it as written, so it holds the name the move gives /s/x.
_HOSTILE = "http://hostile.test/s/"
_CRAFTED = url_to_filename(_HOSTILE + "x").rpartition("/")[2]
_X_EARLIER = "hostile.test/s/x/index.md"
_HOSTILE_FOLDER = [("x", _X_EARLIER), (f"{_CRAFTED}/y", f"hostile.test/s/{_CRAFTED}/y/index.md")]
_HOSTILE_FILE = [("x", _X_EARLIER), (_CRAFTED, f"hostile.test/s/{_CRAFTED}")]
_OTHER_LIBRARY = [("guide", "other.test/guide/index.md"), ("intro", "other.test/intro/index.md")]


def _at_their_names() -> dict[str, str]:
    """Each crawl entry whose file is not the name its address gives it."""
    return {
        address: entry.file
        for address, entry in load_crawl_metadata().items()
        if entry.file != url_to_filename(address)
    }


@pytest.mark.timeout(300)
async def test_a_folder_at_a_page_name_moves_out_before_the_page_moves_in(
    library: Store, monkeypatch: pytest.MonkeyPatch
):
    _write_earlier_library(_HOSTILE, _HOSTILE_FOLDER)
    _write_earlier_library(_OTHER_SITE, _OTHER_LIBRARY)
    await sync(quiet=True)
    for _ in range(2):
        await _crawl(monkeypatch, _OTHER_SITE, _OTHER_PAGES, "2")
    await _crawl(monkeypatch, _HOSTILE, [path for path, _ in _HOSTILE_FOLDER], "2")
    await sync(quiet=True)
    versions = {path: "2" for path, _ in _HOSTILE_FOLDER} | {path: "2" for path in _OTHER_PAGES}
    assert _library_state(library, versions) == _clean(len(versions))
    assert _at_their_names() == {}


@pytest.mark.timeout(300)
async def test_a_page_whose_name_holds_a_folder_the_record_does_not_know_stays_put(
    library: Store, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _write_earlier_library(_HOSTILE, _HOSTILE_FOLDER)
    _write_earlier_library(_OTHER_SITE, _OTHER_LIBRARY)
    meta = load_crawl_metadata()
    del meta[_HOSTILE + f"{_CRAFTED}/y"]
    save_crawl_metadata(meta)
    await sync(quiet=True)
    with caplog.at_level(logging.WARNING, logger="lilbee.crawler.migrate"):
        for _ in range(2):
            await _crawl(monkeypatch, _OTHER_SITE, _OTHER_PAGES, "2")
    await _crawl(monkeypatch, _HOSTILE, ["x"], "2")
    await sync(quiet=True)
    name = url_to_filename(_HOSTILE + "x")
    skipped = [r for r in caplog.records if _X_EARLIER in r.message and name in r.message]
    assert len(skipped) == 3
    assert _at_their_names() == {_HOSTILE + "x": _X_EARLIER}
    state = _library_state(library, {"x": "2"} | {path: "2" for path in _OTHER_PAGES})
    assert state["pages not in exactly one file"] == []
    assert state["sources without a file"] == []
    assert state["files without a crawl entry"] == [f"_web/hostile.test/s/{_CRAFTED}/y/index.md"]


@pytest.mark.timeout(300)
async def test_a_page_never_moves_over_the_file_of_another_page(
    library: Store, monkeypatch: pytest.MonkeyPatch
):
    _write_earlier_library(_HOSTILE, _HOSTILE_FILE)
    await sync(quiet=True)
    await _crawl(monkeypatch, _OTHER_SITE, _OTHER_PAGES, "1")
    await sync(quiet=True)
    versions = {path: "1" for path, _ in _HOSTILE_FILE} | {path: "1" for path in _OTHER_PAGES}
    assert _library_state(library, versions) == _clean(len(versions))
    assert _at_their_names() == {}
    await _crawl(monkeypatch, _HOSTILE, [path for path, _ in _HOSTILE_FILE], "1")
    await sync(quiet=True)
    assert _library_state(library, versions) == _clean(len(versions))


_NEWER_SITE = "http://newer.test/"
_NEWER_LIBRARY = [("p", "newer.test/p/index.md"), ("q", "newer.test/q/index.md")]


@pytest.mark.timeout(300)
async def test_a_page_kept_at_an_older_copy_takes_its_newer_text_on_the_next_crawl(
    library: Store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    paths = [path for path, _ in _NEWER_LIBRARY]
    await _crawl(monkeypatch, _NEWER_SITE, paths, "1")
    await sync(quiet=True)
    _write_earlier_library(_NEWER_SITE, _NEWER_LIBRARY, "2", datetime.now(UTC).isoformat())
    await sync(quiet=True)
    (cfg.documents_dir / "_web" / "newer.test/p/index.md").replace(tmp_path / "aside.md")
    await _crawl(monkeypatch, _OTHER_SITE, _OTHER_PAGES, "1")
    await _crawl(monkeypatch, _NEWER_SITE, paths, "2")
    await sync(quiet=True)
    versions = {path: "2" for path in paths} | {path: "1" for path in _OTHER_PAGES}
    assert _library_state(library, versions) == _clean(len(versions))


class _CountingMeta(dict):
    """Crawl metadata that counts each pass over all of its entries."""

    passes = 0

    def __iter__(self):
        type(self).passes += 1
        return super().__iter__()

    def items(self):
        type(self).passes += 1
        return super().items()

    def values(self):
        type(self).passes += 1
        return super().values()


@pytest.mark.parametrize("pages", [5, 40])
async def test_saving_a_page_does_not_read_the_whole_metadata(
    library: Store, monkeypatch: pytest.MonkeyPatch, pages: int
):
    paths = [f"p{n}" for n in range(pages)]
    others = {
        f"http://other.test/{n}": CrawlMeta(f"other.test/{n}/index.md", "h", "t")
        for n in range(2000)
    }
    stored = {_SITE + p: CrawlMeta(f"site.test/s/{p}/old.md", "old", "t") for p in paths}
    _CountingMeta.passes = 0
    monkeypatch.setattr(
        "lilbee.crawler.save.load_crawl_metadata", lambda: _CountingMeta(others | stored)
    )
    monkeypatch.setattr("lilbee.crawler.save.save_crawl_metadata", lambda meta: None)
    results = [CrawlResult(url=_SITE + p, markdown=_text(p, "2")) for p in paths]

    async def crawl(url: str, **kwargs: object) -> list[CrawlResult]:
        for result in results:
            await kwargs["on_result"](result)  # type: ignore[operator]
        return results

    monkeypatch.setattr("lilbee.crawler.runner.crawl_recursive", crawl)
    await crawl_and_save(_SITE, depth=1)
    passes = _CountingMeta.passes
    assert passes <= 3, f"{passes} passes over the metadata for {pages} pages"
