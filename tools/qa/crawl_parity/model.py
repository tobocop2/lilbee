"""The closed sets and the records every part of the harness exchanges."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Side(StrEnum):
    """Which crawler stack produced an output."""

    ORACLE = "oracle"
    CANDIDATE = "candidate"


class Layer(StrEnum):
    """How much of the stack ran, lowest first."""

    CONVERTER = "converter"
    CRAWLER = "crawler"
    LILBEE = "lilbee"


class Mode(StrEnum):
    """How a page is fetched."""

    HTTP = "http"
    BROWSER = "browser"


class Kind(StrEnum):
    """What sort of difference a yardstick found."""

    PAGE_LOST = "page-lost"
    PAGE_EXTRA = "page-extra"
    TEXT_LOST = "text-lost"
    TEXT_ADDED = "text-added"
    STRUCTURE = "structure"
    VISIBLE_MISSED = "visible-missed"
    LEFT_BEHIND = "left-behind"
    SLOWER = "slower"
    RECALL = "recall"
    INVARIANT = "invariant"


class Status(StrEnum):
    """How a difference stands against expected.toml and the thresholds."""

    NEW = "NEW"
    KNOWN = "KNOWN"
    ACCEPTED = "ACCEPTED"
    FIXED = "FIXED"
    NOT_RUN = "NOT-RUN"
    NOT_COUNTED = "NOT-COUNTED"


NO_PAGE = "-"
PAGE_KINDS = frozenset({Kind.PAGE_LOST, Kind.PAGE_EXTRA})


@dataclass(frozen=True)
class Page:
    """One page a driver returned: its markdown, or the reason it has none."""

    url: str
    markdown: str | None
    error: str | None = None
    saved_at: float | None = None


@dataclass(frozen=True)
class Difference:
    """One difference, attributed to a layer and named by a signature.

    ``counts`` is False when the ground truth says the candidate is the side that matches
    what a reader sees; such a difference is reported and never fails a run.
    """

    kind: Kind
    mode: Mode
    layer: Layer
    feature: str
    page: str = NO_PAGE
    amount: float = 1.0
    detail: str = ""
    counts: bool = True

    @property
    def signature(self) -> str:
        """The key an expected.toml entry matches."""
        return f"{self.kind}/{self.mode}/{self.layer}/{self.feature}"

    def same_finding(self, other: Difference) -> bool:
        """Whether *other* is this difference seen at another layer.

        A page that is lost or extra is one finding whatever reason each layer gives.
        """
        if (self.kind, self.page) != (other.kind, other.page):
            return False
        return self.kind in PAGE_KINDS or self.feature == other.feature
