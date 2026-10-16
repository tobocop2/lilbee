"""Properties of the file name a crawled page is saved under, checked against generated addresses.

The name rule was adjusted twice by example, and each adjustment let a new kind of address
through. These state what must hold for every address a generator writes:

- two different addresses never share a saved path, compared the way Windows and macOS
  compare names;
- the raw and percent-encoded spellings of one address share one saved path;
- every segment of a saved path is a legal Windows name that prints safely.

An address here is a path of one to three generated segments under a fixed host, a fixed last
segment and an optional query. A generated character is written raw or as its UTF-8 escape:
for a space, a control character, a non-ASCII character or an unreserved character the two
spellings are one address (RFC 3986 section 6.2.2, RFC 3987 section 3.1); for any other
printable ASCII character they are two. Dot segments are left out because crawlberg resolves
them before it reports an address.
"""

from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from lilbee.crawler.save import url_to_filename

_PREFIX = "https://example.com/"
_LAST_SEGMENT = "page"
_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
# Printable ASCII characters whose escape names a different address than the character itself.
_KEPT = frozenset(set(string.punctuation) - _UNRESERVED - set("/?#%"))
# Characters that only ever appear escaped: written raw they are path structure or start an escape.
_ESCAPED_ONLY = "/?#%"
_WINDOWS_RESERVED_CHARACTERS = frozenset('<>:"/\\|?*')
_CONTROL_CHARACTERS = frozenset([*map(chr, range(0x20)), "\x7f"])
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"{port}{n}" for port in ("COM", "LPT") for n in [*string.digits, "¹", "²", "³"]}
)
_MAX_SEGMENT_BYTES = 255
_DOT_SEGMENTS = frozenset({".", ".."})
# A private-use character no generated address holds, marking a kept escape in a caseless key.
_ESCAPED_KEPT_MARK = chr(0xE000)

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
    "é",
    "É",
    "e",
    _COMBINING_ACUTE,
    "ß",
    "ü",
    "漢",
    "\x00",
    "\x01",
    "\t",
    "\n",
    "\x1b",
    "\x7f",
    "\x85",
    "\xa0",
    unicodedata.lookup("RIGHT-TO-LEFT OVERRIDE"),
    unicodedata.lookup("LINE SEPARATOR"),
    unicodedata.lookup("ZERO WIDTH NO-BREAK SPACE"),
    *sorted(_KEPT),
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


_chars = st.one_of(
    st.sampled_from(_TRICKY),
    st.sampled_from(_TRICKY),
    st.characters(codec="utf-8", exclude_characters="/?#%"),
)
_tokens = st.builds(_token, _chars, st.booleans())
_segment = st.one_of(
    st.lists(_tokens, min_size=1, max_size=5),
    st.lists(_tokens, min_size=1, max_size=40),
    st.sampled_from(_DEVICE_SEGMENTS).map(lambda name: [_token(char, False) for char in name]),
).filter(lambda tokens: "".join(token.char for token in tokens) not in _DOT_SEGMENTS)


@dataclass(frozen=True)
class _Address:
    """An address as a list of segments and an optional query, each a list of tokens."""

    segments: tuple[tuple[_Token, ...], ...]
    query: tuple[_Token, ...] | None

    @property
    def identity(self) -> object:
        """Equal for two spellings of one address, different for two addresses."""
        return (
            tuple(tuple(token.identity for token in segment) for segment in self.segments),
            None if self.query is None else tuple(token.identity for token in self.query),
        )

    @property
    def caseless(self) -> object:
        """Equal for two addresses that differ only in letter case or Unicode normalization."""
        return (
            tuple(_caseless(_plain(segment)) for segment in self.segments),
            None if self.query is None else tuple(token.identity for token in self.query),
        )


_addresses = st.builds(
    _Address,
    st.lists(_segment.map(tuple), min_size=1, max_size=3).map(tuple),
    st.one_of(st.none(), st.lists(_tokens, min_size=1, max_size=8).map(tuple)),
)


def _near(token: _Token) -> list[list[_Token]]:
    """Token runs a name rule could confuse with *token*: nothing, or a similar character."""
    return [
        [],
        [_token(token.char, not token.kept_escape)],
        [_token("_", False)],
        [_token(token.char.swapcase(), False)],
        [token, _token(".", False)],
        [token, _token(" ", False)],
        [_token(token.char, False), _token(_COMBINING_ACUTE, False)],
    ]


def _edited(address: _Address, data: st.DataObject) -> _Address:
    """*address* with one token changed, so the two addresses are near each other."""
    segments = [list(segment) for segment in address.segments]
    segment = data.draw(st.sampled_from(segments))
    index = data.draw(st.integers(0, len(segment) - 1))
    segment[index : index + 1] = data.draw(st.sampled_from(_near(segment[index])))
    assume(all(_plain(tuple(seg)) not in {"", *_DOT_SEGMENTS} for seg in segments))
    return _Address(tuple(map(tuple, segments)), address.query)


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


def _spell(tokens: tuple[_Token, ...], data: st.DataObject) -> str:
    """One spelling of *tokens*: each character raw or escaped where that keeps the address."""
    spelled = []
    for token in tokens:
        if token.kept_escape:
            spelled.append(_escape(token.char, data.draw(st.booleans())))
        elif token.char in _KEPT:
            spelled.append(token.char)
        elif data.draw(st.booleans()):
            spelled.append(_escape(token.char, data.draw(st.booleans())))
        else:
            spelled.append(token.char)
    return "".join(spelled)


def _url(address: _Address, data: st.DataObject) -> str:
    """One spelling of *address* as a URL."""
    path = "/".join(_spell(segment, data) for segment in address.segments)
    url = f"{_PREFIX}{path}/{_LAST_SEGMENT}"
    if address.query is not None:
        url = f"{url}?{_spell(address.query, data)}"
    return url


def _compared(saved: str) -> str:
    """*saved* as Windows and macOS compare it: no case, one normalization, no trailing dots."""
    return "/".join(_caseless(segment.rstrip(". ")) for segment in saved.split("/"))


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


class TestSavedNames:
    @settings(max_examples=1000, deadline=None)
    @given(one=_addresses, data=st.data())
    def test_two_addresses_never_share_a_saved_path(self, one: _Address, data: st.DataObject):
        other = _edited(one, data)
        assume(one.caseless != other.caseless)
        one_url, other_url = _url(one, data), _url(other, data)
        assert _compared(url_to_filename(one_url)) != _compared(url_to_filename(other_url)), (
            one_url,
            other_url,
        )

    @settings(max_examples=1000, deadline=None)
    @given(address=_addresses, data=st.data())
    def test_two_spellings_of_one_address_share_a_saved_path(
        self, address: _Address, data: st.DataObject
    ):
        one_url, other_url = _url(address, data), _url(address, data)
        assert url_to_filename(one_url) == url_to_filename(other_url), (one_url, other_url)

    @settings(max_examples=1000, deadline=None)
    @given(address=_addresses, data=st.data())
    def test_every_segment_is_a_legal_windows_name_that_prints(
        self, address: _Address, data: st.DataObject
    ):
        url = _url(address, data)
        saved = url_to_filename(url)
        assert {
            seg: _illegal_parts(seg) for seg in saved.split("/") if _illegal_parts(seg)
        } == {}, url


# An address and the path it is saved under, one for each part of the naming rule.
_SAVED_PATHS = [
    pytest.param("https://example.com/t/naïve", "example.com/t/naïve/index.md", id="letter"),
    pytest.param("https://example.com/t/na%c3%afve", "example.com/t/naïve/index.md", id="escape"),
    pytest.param("https://example.com/t/raw space", "example.com/t/raw space/index.md", id="space"),
    pytest.param("https://example.com/t/a\x01b", "example.com/t/a%01b/index.md", id="control"),
    pytest.param(
        "https://example.com/t/a%e2%80%aeb", "example.com/t/a%E2%80%AEb/index.md", id="bidi"
    ),
    pytest.param(
        "https://example.com/t/tail /x", "example.com/t/tail%20/x/index.md", id="space-end"
    ),
    pytest.param("https://example.com/t/dot./x", "example.com/t/dot%2E/x/index.md", id="dot-end"),
    pytest.param("https://example.com/a/../b", "example.com/a/%2E%2E/b/index.md", id="dot-segment"),
    pytest.param("https://example.com/t/a..b/x", "example.com/t/a..b/x/index.md", id="dot-run"),
    pytest.param("https://example.com/t/a:b", "example.com/t/a%3Ab/index.md", id="colon"),
    pytest.param(
        "https://example.com/t/a%3Ab", "example.com/t/a%253Ab/index.md", id="colon-escape"
    ),
    pytest.param("https://example.com/t/a%25b", "example.com/t/a%2525b/index.md", id="percent"),
    pytest.param("https://example.com/t/a%2Fb", "example.com/t/a%2Fb/index.md", id="slash-escape"),
    pytest.param("https://example.com/t/Con.txt", "example.com/t/%43on.md", id="device-name"),
    pytest.param("https://example.com/t/%ff", "example.com/t/%FF/index.md", id="not-utf-8"),
    pytest.param("https://bücher.example/t", "bücher.example/t/index.md", id="host-letter"),
]


@pytest.mark.parametrize(("url", "saved"), _SAVED_PATHS)
def test_an_address_is_saved_under_its_readable_legal_name(url: str, saved: str):
    assert url_to_filename(url) == saved


def test_a_long_address_is_cut_to_a_path_every_file_system_accepts():
    url = f"{_PREFIX}{'漢' * 120}"
    saved = url_to_filename(url)
    assert len(saved.encode("utf-8")) <= 200
    assert saved.startswith("example.com/漢")
    assert saved.endswith(".md")
    assert url_to_filename(_PREFIX + "%E6%BC%A2" * 120) == saved


# Two addresses that differ only in the case of a non-ASCII letter, or in composed against
# decomposed form: the readable names differ only where a case-insensitive or normalizing disk
# sees no difference.
_CASELESS_TWINS = [
    pytest.param(f"{_PREFIX}t/CAFÉ/page", f"{_PREFIX}t/CAFé/page", id="non-ascii-case"),
    pytest.param(
        f"{_PREFIX}t/café/page", f"{_PREFIX}t/cafe{_COMBINING_ACUTE}/page", id="composed-form"
    ),
]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="A readable name keeps letter case and normalization form, which some disks ignore",
)
@pytest.mark.parametrize(("one", "other"), _CASELESS_TWINS)
def test_addresses_that_differ_in_letter_case_or_form_keep_two_saved_paths(one: str, other: str):
    assert _compared(url_to_filename(one)) != _compared(url_to_filename(other))
