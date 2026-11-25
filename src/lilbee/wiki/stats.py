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
    pages_failed: int
    concept_sources: int
    concept_sources_failed: int
    last_model_failure: str
    citations_rendered: int
    citations_dropped_unverified: int
    verified_by_page: dict[str, int]
    publish_rate: float
    citation_verify_rate: float


EMPTY_RESPONSE = "the model returned no text"


class WikiBuildError(RuntimeError):
    """A wiki run wrote no page because every model call failed."""


@dataclass
class BuildStats:
    """What one wiki run's quality gates did.

    ``pages_generated`` is every page written to disk: ``pages_published``
    (landed in a content subdir) plus ``pages_drafted`` (routed to ``drafts/``
    by the faithfulness or drift gate). ``pending_markers`` counts sections
    that produced no page at all and left a PENDING marker under ``drafts/``,
    from a parse failure or a concept-slug collision.
    ``pages_failed`` counts pages not written because the model call that
    would write them failed; a batched call that fails loses every page it
    covered, and those pages also leave PENDING markers.
    ``concept_sources`` counts sources whose batched call also asked for
    concepts, and ``concept_sources_failed`` those whose call failed. A
    concept page is not known before its call, so a failed call is counted
    by source, not by page.
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
    pages_failed: int = 0
    concept_sources: int = 0
    concept_sources_failed: int = 0
    citations_rendered: int = 0
    citations_dropped_unverified: int = 0
    verified_by_page: dict[str, int] = field(default_factory=dict)
    model_calls: int = 0
    model_calls_failed: int = 0
    last_model_failure: str = ""

    @property
    def every_model_call_failed(self) -> bool:
        """True when the run made model calls and none of them gave usable text."""
        return self.model_calls > 0 and self.model_calls_failed == self.model_calls

    def raise_if_nothing_written(self, pages: int) -> None:
        """Raise :class:`WikiBuildError` when *pages* is 0 because every model call failed."""
        if pages == 0 and self.every_model_call_failed:
            raise WikiBuildError(
                f"No wiki pages were written because every model call failed "
                f"({self.model_calls}). Last error: {self.last_model_failure}. "
                "Check that the chat model is available, then run the command again."
            )

    def record_model_call(self, failure: str | None, *, error_type: str = "unknown error") -> None:
        """Count one model call; *failure* is why it gave no text, or None when it did.

        The cause is stored so the run's error can end it with one period:
        surrounding spaces and trailing periods are cut. A cause left blank
        is replaced by *error_type*, so the error always names a cause.
        """
        self.model_calls += 1
        if failure is not None:
            self.model_calls_failed += 1
            self.last_model_failure = failure.strip().rstrip(". ") or error_type

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

    def record_failed_pages(self, count: int) -> None:
        """Count *count* pages lost because the model call that would write them failed."""
        self.pages_failed += count

    def record_concept_call(self, *, failed: bool) -> None:
        """Count one source whose batched call asked for concepts, and whether that call failed."""
        self.concept_sources += 1
        if failed:
            self.concept_sources_failed += 1

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
            pages_failed=self.pages_failed,
            concept_sources=self.concept_sources,
            concept_sources_failed=self.concept_sources_failed,
            last_model_failure=self.last_model_failure,
            citations_rendered=self.citations_rendered,
            citations_dropped_unverified=self.citations_dropped_unverified,
            verified_by_page=dict(self.verified_by_page),
            publish_rate=self.publish_rate,
            citation_verify_rate=self.citation_verify_rate,
        )

    def summary_line(self) -> str:
        """One-line human summary of the run."""
        return format_summary_line(self.as_dict())


def _counted(count: int, noun: str) -> str:
    """*noun* in the number *count* asks for: "1 page", "2 pages"."""
    return noun if count == 1 else f"{noun}s"


def _model_failures(stats: BuildStatsDict) -> list[str]:
    """One sentence per kind of work lost to failed model calls, without the final period.

    The page total is every page the run wrote plus every page it lost, so a
    run that lost all its pages reads "N of N". The source total is every
    source whose call asked for concepts.
    """
    failures = []
    failed = stats["pages_failed"]
    if failed:
        total = stats["pages_generated"] + failed
        failures.append(f"{failed} of {total} {_counted(total, 'page')} could not be written")
    concepts_failed = stats["concept_sources_failed"]
    if concepts_failed:
        sources = stats["concept_sources"]
        failures.append(
            f"Concepts could not be extracted from {concepts_failed} of {sources} "
            f"{_counted(sources, 'source')}"
        )
    return failures


def model_failures_message(stats: BuildStatsDict) -> str | None:
    """What the run lost to failed model calls and the last cause, or None when it lost nothing."""
    failures = _model_failures(stats)
    if not failures:
        return None
    return ".\n".join([*failures, f"Last error: {stats['last_model_failure']}"]) + "."


def format_summary_line(stats: BuildStatsDict) -> str:
    """One-line human summary of a run, for the CLI and ``wiki/log.md``.

    Work lost to failed model calls is added only when there was some.
    """
    total = stats["citations_rendered"] + stats["citations_dropped_unverified"]
    line = (
        f"{stats['pages_published']} published, {stats['pages_drafted']} drafted, "
        f"{stats['pending_markers']} markers, "
        f"{stats['citations_rendered']}/{total} citations verified"
    )
    failures = [failure[0].lower() + failure[1:] for failure in _model_failures(stats)]
    return ", ".join([line, *failures])
