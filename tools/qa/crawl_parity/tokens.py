"""Text tokens of markdown and of plain text, and the normalisations applied to them."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from markdown_it import MarkdownIt
from markdown_it.token import Token


class Normalisation(StrEnum):
    """Every way two texts are made comparable; the report names each one."""

    MARKDOWN_SYNTAX = "markdown-syntax"
    UNICODE_NFC = "unicode-nfc"
    WHITESPACE = "whitespace"
    REPLAY_ORIGIN = "replay-origin"


NORMALISATION_NOTES = {
    Normalisation.MARKDOWN_SYNTAX: (
        "markdown is parsed; markers, link targets and symbol runs made only of the table "
        "characters | : - are not tokens"
    ),
    Normalisation.UNICODE_NFC: "tokens are compared in Unicode NFC",
    Normalisation.WHITESPACE: "runs of white space separate tokens and are not tokens",
    Normalisation.REPLAY_ORIGIN: (
        "the origin of the replay server, whose port changes with each run, is written as "
        "http://replay.test"
    ),
}

# One ideograph or kana is a token: these scripts put no space between words.
_UNSPACED = "぀-ヿ㐀-䶿一-鿿豈-﫿"
# A word is a run of letters and digits. An underscore is a symbol: markdown uses it as a
# marker, and a marker that one side's context leaves unparsed must not make a new word.
_TOKEN = re.compile(rf"[{_UNSPACED}]|[^\W_{_UNSPACED}]+|(?:[^\w\s]|_)+")
_TEXT_TYPES = frozenset({"text", "code_inline", "html_inline"})
_TABLE_SYNTAX = re.compile(r"[|:\-]+")
_BLOCK_TEXT_TYPES = frozenset({"fence", "code_block", "html_block"})


@dataclass(frozen=True)
class Tokens:
    """The word tokens and the symbol tokens of one text, in reading order."""

    words: tuple[str, ...]
    symbols: tuple[str, ...]

    def word_counts(self) -> Counter[str]:
        """How many times each word occurs."""
        return Counter(self.words)

    def symbol_counts(self) -> Counter[str]:
        """How many times each symbol run occurs."""
        return Counter(self.symbols)


def markdown_parser() -> MarkdownIt:
    """The dialect structure is read with: CommonMark plus tables and strikethrough."""
    return MarkdownIt("commonmark").enable(["table", "strikethrough"])


def _text_parser() -> MarkdownIt:
    """The dialect text is read with: no tables, because a table parser drops surplus cells."""
    return MarkdownIt("commonmark").enable(["strikethrough"])


def tokenize(text: str) -> Tokens:
    """Split plain text into word tokens and symbol tokens."""
    words: list[str] = []
    symbols: list[str] = []
    for match in _TOKEN.finditer(unicodedata.normalize("NFC", text)):
        token = match.group()
        (words if token[0].isalnum() else symbols).append(token)
    return Tokens(tuple(words), tuple(symbols))


def _inline_text(token: Token) -> list[str]:
    """The text pieces of one inline token, image alternative text included."""
    pieces: list[str] = []
    for child in token.children or []:
        if child.type in _TEXT_TYPES:
            pieces.append(child.content)
        elif child.type == "image":
            # Alternative text is its own words, whatever touches the image.
            pieces.append(f" {child.content} ")
        elif child.type in {"softbreak", "hardbreak"}:
            pieces.append("\n")
    return pieces


def markdown_text(markdown: str) -> str:
    """The text of a markdown document with its syntax parsed away."""
    pieces: list[str] = []
    for token in _text_parser().parse(markdown):
        if token.type == "inline":
            pieces.extend(_inline_text(token))
            pieces.append("\n")
        elif token.type in _BLOCK_TEXT_TYPES:
            pieces.append(token.content)
            pieces.append("\n")
    return "".join(pieces)


def markdown_tokens(markdown: str) -> Tokens:
    """The tokens of a markdown document's text; runs of table syntax are not symbols."""
    tokens = tokenize(markdown_text(markdown))
    symbols = tuple(symbol for symbol in tokens.symbols if not _TABLE_SYNTAX.fullmatch(symbol))
    return Tokens(tokens.words, symbols)


def missing(have: Counter[str], want: Counter[str]) -> Counter[str]:
    """The tokens of *want* that *have* lacks, with their counts."""
    return want - have
