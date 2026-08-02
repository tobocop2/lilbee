"""Per-run counters for what the wiki quality gates did.

One :class:`BuildStats` is threaded through a build, update or synthesize run
and reported in its summary, so a regression in the citation or faithfulness
gate is visible per run instead of only in the logs. Recording never changes a
gate's decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class BuildStatsDict(TypedDict):
    """Serializable snapshot of :class:`BuildStats`, derived rates included."""

    pages_generated: int
    pages_published: int
    pages_drafted: int
    pending_markers: int
    citations_rendered: int
    citations_dropped_unverified: int
    verified_by_page: dict[str, int]
    publish_rate: float
    citation_verify_rate: float


@dataclass
class BuildStats:
    """What one wiki run's quality gates did.

    ``pages_generated`` is every page written to disk: ``pages_published``
    (landed in a content subdir) plus ``pages_drafted`` (routed to ``drafts/``
    by the faithfulness or drift gate). ``pending_markers`` counts sections
    that produced no page at all and left a PENDING marker under ``drafts/``,
    from a parse failure or a concept-slug collision.
    ``citations_dropped_unverified`` counts parsed footnotes that reached no
    page: excerpt not found in the source chunks, or no source to attribute
    them to. Footnotes skipped for citing a wiki page are in neither count.
    ``verified_by_page`` maps a published page's ``wiki_source`` to the number
    of citations it rendered.
    """

    pages_generated: int = 0
    pages_published: int = 0
    pages_drafted: int = 0
    pending_markers: int = 0
    citations_rendered: int = 0
    citations_dropped_unverified: int = 0
    verified_by_page: dict[str, int] = field(default_factory=dict)

    @classmethod
    def ensure(cls, stats: BuildStats | None) -> BuildStats:
        """Return *stats*, or a throwaway collector when the caller passed none."""
        return cls() if stats is None else stats

    @property
    def publish_rate(self) -> float:
        """Fraction of written pages that published rather than drafted."""
        if not self.pages_generated:
            return 0.0
        return self.pages_published / self.pages_generated

    @property
    def citation_verify_rate(self) -> float:
        """Fraction of counted citations that rendered on a page."""
        total = self.citations_rendered + self.citations_dropped_unverified
        if not total:
            return 0.0
        return self.citations_rendered / total

    def record_published(self, wiki_source: str, verified: int) -> None:
        """Count a page that landed in a content subdir with *verified* citations."""
        self.pages_generated += 1
        self.pages_published += 1
        self.verified_by_page[wiki_source] = verified

    def record_drafted(self) -> None:
        """Count a page the faithfulness or drift gate routed to ``drafts/``."""
        self.pages_generated += 1
        self.pages_drafted += 1

    def record_pending_marker(self) -> None:
        """Count a section left as a PENDING marker under ``drafts/``."""
        self.pending_markers += 1

    def record_citations(self, rendered: int, dropped: int) -> None:
        """Count one page's verified and rejected citation records."""
        self.citations_rendered += rendered
        self.citations_dropped_unverified += dropped

    def as_dict(self) -> BuildStatsDict:
        """Snapshot for the summary dicts CLI, HTTP and MCP return."""
        return BuildStatsDict(
            pages_generated=self.pages_generated,
            pages_published=self.pages_published,
            pages_drafted=self.pages_drafted,
            pending_markers=self.pending_markers,
            citations_rendered=self.citations_rendered,
            citations_dropped_unverified=self.citations_dropped_unverified,
            verified_by_page=dict(self.verified_by_page),
            publish_rate=self.publish_rate,
            citation_verify_rate=self.citation_verify_rate,
        )

    def summary_line(self) -> str:
        """One-line human summary of the run."""
        return format_summary_line(self.as_dict())


def format_summary_line(stats: BuildStatsDict) -> str:
    """One-line human summary of a run, for the CLI and ``wiki/log.md``."""
    total = stats["citations_rendered"] + stats["citations_dropped_unverified"]
    return (
        f"{stats['pages_published']} published, {stats['pages_drafted']} drafted, "
        f"{stats['pending_markers']} markers, "
        f"{stats['citations_rendered']}/{total} citations verified"
    )
