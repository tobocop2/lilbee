"""Page titles crafted to add markup to a Sources list, and a check that one shows as text."""

from __future__ import annotations

from markdown_it import MarkdownIt
from markdown_it.token import Token

from lilbee.core.text import collapse_whitespace

CRAFTED_HOST = "forum.example"
CRAFTED_SOURCE = f"_web/{CRAFTED_HOST}/a~0123456789abcdef.md"
CRAFTED_TITLES = (
    "[click](javascript:alert(1))",
    "[click](https://evil.example/x)",
    "<https://evil.example/y>",
    "![px](https://evil.example/p.png)",
    "<img src=x onerror=alert(1)>",
    "`code` and *bold*",
    "*bold* _it_ __strong__",
    "~~struck~~",
    "a\\](https://evil.example/z) b",
    "x](https://evil.example) `[y` <b> \\ end",
    "&lt;img&gt; &amp; &#60;",
    "C++ & Rust: (a) [b] guide",
    "back\\slash \\[x]",
    "line1\nline2 # head",
)
_MARKUP = frozenset({"image", "code_inline", "html_inline", "em_open", "strong_open", "s_open"})


def crafted_label(title: str) -> str:
    """The text a Sources list shows for the crafted page titled *title*."""
    return f"{CRAFTED_HOST} · {collapse_whitespace(title)}"


def shown_inline(parser: MarkdownIt, line: str) -> tuple[list[str], str]:
    """The link targets in one Sources *line* and the text it shows, failing on any other markup."""
    tokens: list[Token] = parser.parseInline(line)[0].children or []
    markup = sorted({token.type for token in tokens} & _MARKUP)
    assert not markup, f"{line!r} renders {markup}"
    hrefs = [str(token.attrGet("href")) for token in tokens if token.type == "link_open"]
    return hrefs, "".join(token.content for token in tokens if token.type == "text")
