"""The structure signature of a markdown document and the differences between two signatures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from markdown_it.token import Token
from tools.qa.crawl_parity.tokens import markdown_parser, tokenize


class Element(StrEnum):
    """The structural elements a signature counts."""

    HEADING = "heading"
    LINK = "link"
    CODE_BLOCK = "code-block"
    TABLE = "table"
    LIST_ITEM = "list-item"


@dataclass(frozen=True)
class Signature:
    """What structure a document has, as one multiset of descriptions for each element."""

    items: dict[Element, Counter[str]] = field(default_factory=dict)

    def count(self, element: Element) -> int:
        """How many of *element* the document has."""
        return sum(self.items.get(element, Counter()).values())


@dataclass(frozen=True)
class StructureDelta:
    """Descriptions of one element that only one of two documents has."""

    element: Element
    only_reference: Counter[str]
    only_other: Counter[str]


def _words(text: str) -> str:
    return " ".join(tokenize(text).words)


def _inline_words(token: Token) -> str:
    return _words(" ".join(child.content for child in token.children or []))


def _links(inline: Token) -> list[str]:
    """One description for each link of an inline token: its target, then its words."""
    found: list[str] = []
    target: str | None = None
    text: list[str] = []
    for child in inline.children or []:
        if child.type == "link_open":
            target, text = str(child.attrGet("href") or ""), []
        elif child.type == "link_close" and target is not None:
            found.append(f"{target} [{_words(' '.join(text))}]")
            target = None
        elif target is not None:
            text.append(child.content)
    return found


def _table_shape(tokens: list[Token], start: int) -> str:
    """Rows and cells of the table that opens at *start*."""
    rows = cells = 0
    for token in tokens[start + 1 :]:
        if token.type == "table_close":
            break
        rows += token.type == "tr_open"
        cells += token.type in {"th_open", "td_open"}
    return f"{rows} rows, {cells} cells"


def signature(markdown: str) -> Signature:
    """The structure signature of *markdown*."""
    tokens = markdown_parser().parse(markdown)
    items: dict[Element, Counter[str]] = {element: Counter() for element in Element}
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            items[Element.HEADING][f"{token.tag} {_inline_words(tokens[index + 1])}"] += 1
        elif token.type == "inline":
            items[Element.LINK].update(_links(token))
        elif token.type in {"fence", "code_block"}:
            items[Element.CODE_BLOCK][f"{len(tokenize(token.content).words)} words"] += 1
        elif token.type == "table_open":
            items[Element.TABLE][_table_shape(tokens, index)] += 1
        elif token.type == "list_item_open":
            items[Element.LIST_ITEM]["item"] += 1
    return Signature(items)


def compare(reference: Signature, other: Signature) -> list[StructureDelta]:
    """The elements whose descriptions differ between *reference* and *other*."""
    deltas: list[StructureDelta] = []
    for element in Element:
        mine = reference.items.get(element, Counter())
        theirs = other.items.get(element, Counter())
        if mine != theirs:
            deltas.append(StructureDelta(element, mine - theirs, theirs - mine))
    return deltas
