"""Wiki layer commands: build, update, browse, lint, citations, status, prune, drafts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import typer
from rich.table import Table

from lilbee.app.services import get_services
from lilbee.cli import theme
from lilbee.cli.app import (
    apply_overrides,
    console,
    data_dir_option,
    global_option,
)
from lilbee.cli.helpers import json_output
from lilbee.cli.tui import messages as msg
from lilbee.core.config import cfg
from lilbee.core.security import PathTraversalError
from lilbee.runtime.progress import EventType, WikiPageEvent, WikiPhaseEvent
from lilbee.wiki.shared import (
    INVALID_DRAFT_SLUG_ERROR,
    WikiSubdir,
    total_wiki_pages,
)
from lilbee.wiki.stats import format_summary_line

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from lilbee.data.store import CitationRecord
    from lilbee.runtime.progress import DetailedProgressCallback, ProgressEvent
    from lilbee.wiki.generation import WikiEntityCandidate
    from lilbee.wiki.stats import BuildStatsDict


wiki_app = typer.Typer(
    help="Wiki layer commands: generate, browse, lint, citations, status, prune."
)

# Citations table renders excerpts truncated to ``_CITATION_EXCERPT_MAX_CHARS``;
# the ellipsis insertion point is one ``...`` shorter so the visible string never
# exceeds the column width.
_CITATION_EXCERPT_MAX_CHARS = 60
_CITATION_EXCERPT_TRUNCATE_AT = 57

# Dry-run NER output previews the first ``_NER_DRY_RUN_PREVIEW_LIMIT`` sources
# per row, with ``", ..."`` appended when more were dropped.
_NER_DRY_RUN_PREVIEW_LIMIT = 3


def _count_md_files(directory: Path) -> int:
    """Count markdown files in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.md")))


def _print_build_stats(stats: BuildStatsDict) -> None:
    """Print what a build or synthesize run's quality gates did."""
    console.print(f"  Gates: {format_summary_line(stats)}")


def _wiki_progress_line(event_type: EventType, data: ProgressEvent) -> str | None:
    """Progress description for a wiki run event, or None when it carries no line.

    Wording is shared with the TUI task bar so both surfaces name a phase and a
    page the same way.
    """
    if event_type is EventType.WIKI_PHASE and isinstance(data, WikiPhaseEvent):
        return msg.WIKI_BUILD_PHASE.format(phase=data.phase.value)
    if event_type is EventType.WIKI_PAGE and isinstance(data, WikiPageEvent):
        return msg.WIKI_BUILD_PAGE.format(label=data.label, current=data.current, total=data.total)
    return None


@contextmanager
def _wiki_progress() -> Iterator[DetailedProgressCallback]:
    """Render a wiki run's phase and page events on a spinner line.

    A build issues one LLM call per source and runs for hours, so the CLI shows
    the same events the HTTP and TUI surfaces consume. Disabled in json_mode so
    stdout stays a single JSON document.
    """
    from rich.console import Console as RichConsole
    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        transient=True,
        console=RichConsole(stderr=True),
        disable=cfg.json_mode,
    ) as progress:
        task = progress.add_task(msg.WIKI_BUILD_STARTING, total=None)

        def on_progress(event_type: EventType, data: ProgressEvent) -> None:
            line = _wiki_progress_line(event_type, data)
            if line is not None:
                progress.update(task, description=line)

        yield on_progress


def _fail_wiki_disabled() -> NoReturn:
    """Emit the standard wiki-disabled message and exit non-zero."""
    if cfg.json_mode:
        json_output({"error": msg.CMD_WIKI_DISABLED})
    else:
        console.print(msg.CMD_WIKI_DISABLED)
    raise typer.Exit(1)


@wiki_app.command(name="lint")
def wiki_lint(
    wiki_source: str = typer.Argument("", help="Wiki page path (empty = lint all)."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Lint wiki pages for stale citations, missing sources, and unmarked claims.

    Exits 1 when any issue is an error, so a script can gate on the result.
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.lint import IssueSeverity, LintReport, lint_wiki_page
    from lilbee.wiki.lint import lint_all as _lint_all

    store = get_services().store
    report = (
        LintReport(issues=lint_wiki_page(wiki_source, store)) if wiki_source else _lint_all(store)
    )
    issues = report.issues

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_lint",
                "issues": [i.to_dict() for i in issues],
                "total": len(issues),
                "errors": report.error_count,
                "warnings": report.warning_count,
            }
        )
    elif not issues:
        console.print("No issues found.")
    else:
        table = Table(title="Wiki Lint Issues")
        table.add_column("Page", style=theme.ACCENT)
        table.add_column("Severity")
        table.add_column("Message")
        for issue in issues:
            sev_style = theme.ERROR if issue.severity is IssueSeverity.ERROR else theme.WARNING
            sev_text = f"[{sev_style}]{issue.severity.value}[/{sev_style}]"
            table.add_row(issue.wiki_source, sev_text, issue.message)
        console.print(table)

    if report.error_count:
        raise typer.Exit(1)


@wiki_app.command(name="list")
def wiki_list(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List wiki pages with their type, source count, and creation date."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.browse import list_pages

    pages = list_pages(cfg.data_root / cfg.wiki_dir)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_list",
                "pages": [p.to_dict() for p in pages],
                "total": len(pages),
            }
        )
        return

    if not pages:
        console.print("No wiki pages found.")
        return

    table = Table(title=f"Wiki Pages ({len(pages)})")
    table.add_column("Slug", style=theme.ACCENT)
    table.add_column("Title")
    table.add_column("Type", style=theme.MUTED)
    table.add_column("Sources")
    table.add_column("Created", style=theme.MUTED)
    for page in pages:
        table.add_row(
            page.slug, page.title, page.page_type, str(page.source_count), page.created_at
        )
    console.print(table)


@wiki_app.command(name="read")
def wiki_read(
    slug: str = typer.Argument(..., help="Page slug, e.g. entities/chevrolet."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Print a wiki page's markdown."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.browse import read_page

    page = read_page(cfg.data_root / cfg.wiki_dir, slug)
    if page is None:
        message = f"wiki page not found: {slug}"
        if cfg.json_mode:
            json_output({"error": message})
        else:
            console.print(f"[{theme.ERROR}]{message}[/{theme.ERROR}]")
        raise typer.Exit(1)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_read",
                "slug": page.slug,
                "title": page.title,
                "content": page.content,
                "frontmatter": page.frontmatter,
            }
        )
        return
    console.print(page.content)


@wiki_app.command(name="citations")
def wiki_citations(
    wiki_source: str = typer.Argument("", help="Wiki page path, e.g. wiki/summaries/doc.md."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    source: str = typer.Option(
        "",
        "--source",
        help="Reverse lookup: list the wiki pages citing this source document.",
    ),
) -> None:
    """Show a wiki page's citations, or the pages citing a source document."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if bool(wiki_source) == bool(source):
        message = "Pass either a wiki page path or --source, not both."
        if cfg.json_mode:
            json_output({"error": message})
        else:
            console.print(f"[{theme.ERROR}]{message}[/{theme.ERROR}]")
        raise typer.Exit(1)

    store = get_services().store
    if source:
        _render_citations(
            store.get_citations_for_source(source),
            key="source",
            value=source,
            title=f"Pages citing: {source}",
            column_header="Page",
            column_value=lambda rec: rec["wiki_source"],
        )
        return
    _render_citations(
        store.get_citations_for_wiki(wiki_source),
        key="wiki_source",
        value=wiki_source,
        title=f"Citations: {wiki_source}",
        column_header="Source",
        column_value=lambda rec: rec["source_filename"],
    )


def _render_citations(
    records: list[CitationRecord],
    *,
    key: str,
    value: str,
    title: str,
    column_header: str,
    column_value: Callable[[CitationRecord], str],
) -> None:
    """Render citation rows as JSON or a table, in whichever direction was asked.

    The second column differs by direction: the forward lookup names the source
    a page cites, the reverse one names the page citing a source.
    """
    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_citations",
                key: value,
                "citations": [dict(r) for r in records],
                "total": len(records),
            }
        )
        return

    if not records:
        console.print(f"No citations found for [{theme.ACCENT}]{value}[/{theme.ACCENT}]")
        return

    table = Table(title=title)
    table.add_column("Key", style=theme.ACCENT)
    table.add_column(column_header)
    table.add_column("Type", style=theme.MUTED)
    table.add_column("Excerpt", max_width=_CITATION_EXCERPT_MAX_CHARS)
    for rec in records:
        excerpt = (
            rec["excerpt"][:_CITATION_EXCERPT_TRUNCATE_AT] + "..."
            if len(rec["excerpt"]) > _CITATION_EXCERPT_MAX_CHARS
            else rec["excerpt"]
        )
        table.add_row(rec["citation_key"], column_value(rec), rec["claim_type"], excerpt)
    console.print(table)


@wiki_app.command(name="status")
def wiki_status(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show wiki layer status: page counts and lint summary."""
    apply_overrides(data_dir=data_dir, use_global=use_global)

    wiki_root = cfg.data_root / cfg.wiki_dir
    if not cfg.wiki or not wiki_root.exists():
        # A disabled wiki can still have a tree left over from an earlier build;
        # report the disabled state rather than linting it.
        if cfg.json_mode:
            json_output(
                {
                    "wiki_enabled": cfg.wiki,
                    WikiSubdir.SUMMARIES: 0,
                    WikiSubdir.DRAFTS: 0,
                    "pages": 0,
                    "lint_errors": 0,
                    "lint_warnings": 0,
                }
            )
            return
        if not cfg.wiki:
            console.print(f"Wiki: [{theme.ERROR}]disabled[/{theme.ERROR}]")
        else:
            console.print("Wiki directory does not exist yet. Run `lilbee wiki build`.")
        return

    summaries = _count_md_files(wiki_root / WikiSubdir.SUMMARIES)
    drafts = _count_md_files(wiki_root / WikiSubdir.DRAFTS)

    from lilbee.wiki.lint import lint_all as _lint_all

    # Read-only status: lint for counts without appending to the audit log.
    report = _lint_all(get_services().store, record_log=False)

    if cfg.json_mode:
        json_output(
            {
                "wiki_enabled": cfg.wiki,
                WikiSubdir.SUMMARIES: summaries,
                WikiSubdir.DRAFTS: drafts,
                "pages": total_wiki_pages(wiki_root),
                "lint_errors": report.error_count,
                "lint_warnings": report.warning_count,
            }
        )
        return

    console.print(f"Wiki: [{theme.SUCCESS}]enabled[/{theme.SUCCESS}]")
    console.print(f"  Summaries: [{theme.LABEL}]{summaries}[/{theme.LABEL}]")
    console.print(f"  Drafts:    [{theme.LABEL}]{drafts}[/{theme.LABEL}]")
    if report.error_count or report.warning_count:
        console.print(
            f"  Lint: [{theme.ERROR}]{report.error_count} error(s)[/{theme.ERROR}], "
            f"[{theme.WARNING}]{report.warning_count} warning(s)[/{theme.WARNING}]"
        )
    else:
        console.print("  Lint: all clean")


@wiki_app.command(name="synthesize")
def wiki_synthesize(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Generate synthesis pages for concept clusters spanning 3+ sources."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
    from lilbee.wiki import run_full_synthesize

    with _wiki_progress() as on_progress:
        result = run_full_synthesize(cfg, on_progress)

    if cfg.json_mode:
        json_output({"command": "wiki_synthesize", **result})
        return

    paths = result["paths"]
    if paths:
        console.print(
            f"Generated [{theme.LABEL}]{result['count']}[/{theme.LABEL}] synthesis pages:"
        )
        for path in paths:
            console.print(f"  {path}")
    else:
        console.print("No synthesis pages generated (need 3+ sources per cluster).")
    _print_build_stats(result["stats"])


@wiki_app.command(name="prune")
def wiki_prune(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Prune stale and orphaned wiki pages."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
    from lilbee.wiki.prune import prune_wiki

    report = prune_wiki(get_services().store)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_prune",
                "records": [r.to_dict() for r in report.records],
                "archived": report.archived_count,
                "flagged": report.flagged_count,
                "reconciled": report.reconciled_count,
            }
        )
        return

    if not report.records:
        console.print("No pages pruned.")
        return

    table = Table(title="Wiki Prune Results")
    table.add_column("Page", style=theme.ACCENT)
    table.add_column("Action")
    table.add_column("Reason")
    for rec in report.records:
        action_style = theme.ERROR if rec.action.value == "archived" else theme.WARNING
        action_text = f"[{action_style}]{rec.action.value}[/{action_style}]"
        table.add_row(rec.wiki_source, action_text, rec.reason)
    console.print(table)


@wiki_app.command(name="index")
def wiki_index(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Rebuild the browse index of pages the corpus could have.

    Spends no LLM call. A sync refreshes this for you; run it to repair an
    index that was deleted or written by an older version.
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
    from lilbee.wiki.stubs import refresh_stub_index

    stubs = refresh_stub_index(get_services().store)
    if cfg.json_mode:
        json_output({"command": "wiki_index", "entries": len(stubs)})
    else:
        console.print(f"Wiki index: {len(stubs)} page(s) the corpus names")


@wiki_app.command(name="generate")
def wiki_generate(
    slug: str = typer.Argument(..., help="Indexed page slug, as `wiki list` shows it."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Generate one indexed page. Costs a single LLM call and is GPU-heavy."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
    from lilbee.wiki.lazy import UnknownStubError, generate_stub_page

    try:
        with _wiki_progress():
            path = generate_stub_page(slug, get_services().store)
    except UnknownStubError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(str(exc))
        raise typer.Exit(1) from exc

    if path is None:
        message = msg.CMD_WIKI_GENERATE_NO_EVIDENCE.format(slug=slug)
        if cfg.json_mode:
            json_output({"error": message})
        else:
            console.print(message)
        raise typer.Exit(1)

    if cfg.json_mode:
        json_output({"command": "wiki_generate", "slug": slug, "path": str(path)})
    else:
        console.print(f"Wrote {path}")


@wiki_app.command(name="wipe")
def wiki_wipe(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete every generated wiki page and its indexed rows.

    Available with the wiki disabled: turning the setting off stops new pages
    being written but leaves the ones already generated in place.
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.wipe import wipe_wiki

    wiki_root = cfg.data_root / cfg.wiki_dir
    if not yes:
        if cfg.json_mode:
            json_output({"error": msg.CMD_WIKI_WIPE_NEEDS_YES})
            raise typer.Exit(1)
        console.print(msg.CMD_WIKI_WIPE_WARNING.format(path=wiki_root))
        if not typer.confirm("Delete the wiki?", default=False):
            console.print("Aborted.")
            raise typer.Exit(0)

    report = wipe_wiki(get_services().store)
    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_wipe",
                "pages_removed": report.pages_removed,
                "sources_cleared": report.sources_cleared,
                "rows_deleted": report.rows_deleted,
            }
        )
    else:
        console.print(report.summary())
    if not report.rows_deleted:
        raise typer.Exit(1)


@wiki_app.command(name="build")
def wiki_build(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Run extraction only; skip every LLM call. Prints the NER entity candidates. "
            "LLM-curated concept pages require a build call and are not shown in dry-run."
        ),
    ),
) -> None:
    """Build the concept and entity wiki across all ingested sources."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()

    if dry_run:
        from lilbee.wiki.generation import preview_build_entities

        _wiki_build_dry_run_output(preview_build_entities(cfg))
        return

    _run_wiki_build("wiki_build")


def _run_wiki_build(command_name: str) -> None:
    """Run the full build and render its result under *command_name*."""
    from lilbee.wiki import run_full_build

    with _wiki_progress() as on_progress:
        result = run_full_build(cfg, on_progress)

    if cfg.json_mode:
        json_output({"command": command_name, **result})
        return

    pages = result["paths"]
    if pages:
        console.print(
            f"Generated [{theme.LABEL}]{result['count']}[/{theme.LABEL}] "
            f"wiki pages from {result['entities']} extracted records:"
        )
        for path in pages:
            console.print(f"  {path}")
    else:
        console.print("No concept or entity pages generated.")
    _print_build_stats(result["stats"])


def _wiki_build_dry_run_output(rows: list[WikiEntityCandidate]) -> None:
    """Render the extraction result as JSON or table without calling any LLM.

    Concepts come from the per-source batched LLM call, so listing
    them would require the call we are trying to avoid. The dry-run
    surface is NER-entity only, with a trailing note so a user who
    expected concepts in the output knows why they are missing.
    """
    from lilbee.wiki.generation import DRY_RUN_CONCEPT_NOTE

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_build",
                "dry_run": True,
                "entities": rows,
                "count": len(rows),
                "note": DRY_RUN_CONCEPT_NOTE,
            }
        )
        return

    if not rows:
        console.print("No candidate entities extracted. Run sync first.")
        console.print(f"[{theme.MUTED}]{DRY_RUN_CONCEPT_NOTE}[/{theme.MUTED}]")
        return

    table = Table(title=f"Wiki build dry-run ({len(rows)} NER entity candidates)")
    table.add_column("Slug", style=theme.ACCENT)
    table.add_column("Kind", style=theme.MUTED)
    table.add_column("Type")
    table.add_column("Mentions")
    table.add_column("Sources")
    for row in rows:
        sources_list: list[str] = row["sources"]
        table.add_row(
            str(row["slug"]),
            str(row["kind"]),
            str(row["type_hint"]),
            str(row["mentions"]),
            ", ".join(sources_list[:_NER_DRY_RUN_PREVIEW_LIMIT])
            + (", ..." if len(sources_list) > _NER_DRY_RUN_PREVIEW_LIMIT else ""),
        )
    console.print(table)
    console.print(
        f"Dry run: [{theme.LABEL}]{len(rows)}[/{theme.LABEL}] candidate entities. "
        "No LLM calls were made."
    )
    console.print(f"[{theme.MUTED}]{DRY_RUN_CONCEPT_NOTE}[/{theme.MUTED}]")


@wiki_app.command(name="update")
def wiki_update(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Refresh the concept and entity wiki after an ingest.

    A full rebuild: every source is re-extracted and regenerated. The capped
    touched-slug regeneration only runs from the ingest hook.
    """
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
    _run_wiki_build("wiki_update")


drafts_app = typer.Typer(help="Review wiki drafts: list, diff, accept, reject.")
wiki_app.add_typer(drafts_app, name="drafts")


@drafts_app.command(name="list")
def wiki_drafts_list(
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """List pending wiki drafts with drift, faithfulness, and pairing info."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import PendingKind, list_drafts

    wiki_root = cfg.data_root / cfg.wiki_dir
    drafts = list_drafts(wiki_root)

    if cfg.json_mode:
        json_output(
            {
                "command": "wiki_drafts_list",
                "drafts": [d.to_dict() for d in drafts],
                "total": len(drafts),
            }
        )
        return

    if not drafts:
        console.print("No drafts pending review.")
        return

    table = Table(title="Wiki Drafts")
    table.add_column("Slug", style=theme.ACCENT)
    table.add_column("Kind", style=theme.MUTED)
    table.add_column("Drift")
    table.add_column("Faithfulness")
    table.add_column("Published?", style=theme.MUTED)
    for d in drafts:
        kind = d.pending_kind or PendingKind.DRIFT
        drift = f"{d.drift_ratio:.0%}" if d.drift_ratio is not None else "-"
        faith = f"{d.faithfulness_score:.2f}" if d.faithfulness_score is not None else "-"
        published = "yes" if d.published_exists else "no"
        table.add_row(d.slug, kind, drift, faith, published)
    console.print(table)


def _draft_slug_error() -> None:
    """Report a rejected (traversal) draft slug generically, without leaking paths."""
    message = INVALID_DRAFT_SLUG_ERROR
    if cfg.json_mode:
        json_output({"error": message})
    else:
        console.print(f"[{theme.ERROR}]{message}[/{theme.ERROR}]")
    raise typer.Exit(1) from None


@drafts_app.command(name="diff")
def wiki_drafts_diff(
    slug: str = typer.Argument(..., help="Draft slug (e.g. chevrolet)."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Show a unified diff of the draft against its published counterpart."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    from lilbee.wiki.drafts import diff_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        diff = diff_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None
    except PathTraversalError:
        _draft_slug_error()

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_diff", "slug": slug, "diff": diff})
        return
    console.print(diff or "(no differences)")


@drafts_app.command(name="accept")
def wiki_drafts_accept(
    slug: str = typer.Argument(..., help="Draft slug to accept."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Overwrite the published page with the draft and re-index its chunks."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
    from lilbee.wiki.drafts import DraftAcceptError, accept_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        result = accept_draft(slug, wiki_root, get_services().store)
    except (FileNotFoundError, DraftAcceptError) as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None
    except PathTraversalError:
        _draft_slug_error()

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_accept", **result.to_dict()})
        return
    console.print(
        f"Accepted [{theme.ACCENT}]{slug}[/{theme.ACCENT}] -> "
        f"{result.moved_to} ({result.reindexed_chunks} chunks re-indexed)"
    )


@drafts_app.command(name="reject")
def wiki_drafts_reject(
    slug: str = typer.Argument(..., help="Draft slug to reject."),
    data_dir: Path | None = data_dir_option,
    use_global: bool = global_option,
) -> None:
    """Delete the draft file. Does not touch the published page or index."""
    apply_overrides(data_dir=data_dir, use_global=use_global)
    if not cfg.wiki:
        _fail_wiki_disabled()
    from lilbee.wiki.drafts import reject_draft

    wiki_root = cfg.data_root / cfg.wiki_dir
    try:
        reject_draft(slug, wiki_root)
    except FileNotFoundError as exc:
        if cfg.json_mode:
            json_output({"error": str(exc)})
        else:
            console.print(f"[{theme.ERROR}]{exc}[/{theme.ERROR}]")
        raise typer.Exit(1) from None
    except PathTraversalError:
        _draft_slug_error()

    if cfg.json_mode:
        json_output({"command": "wiki_drafts_reject", "slug": slug})
        return
    console.print(f"Rejected [{theme.ACCENT}]{slug}[/{theme.ACCENT}]")
